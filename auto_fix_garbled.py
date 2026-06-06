"""Automated pipeline: find garbled CJK spans → rerun with MinerU → rebuild EPUB.

Usage:
    python3 auto_fix_garbled.py input.pdf --title "书名" --author "作者"

Workflow:
    1. PaddleOCR conversion (if EPUB doesn't exist)
    2. Scan EPUB for garbled CJK text
    3. Map garbled spans back to PDF page numbers via checkpoint search
    4. Run MinerU on those pages (sentence-level patching, no --replace-page)
    5. Rebuild EPUB from updated checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Set
from zipfile import BadZipFile, ZIP_STORED, ZipFile

from paddle_pipeline.config import (
    CHUNK_SIZE as PADDLE_CHUNK_SIZE,
    epub as ebooklib_epub,
)
from paddle_pipeline.ocr_review import find_suspicious_cjk_spans_in_epub
from paddle_pipeline.page_boundary_review import (
    annotate_candidates_with_epub_continuity,
    find_page_boundary_candidates,
)
from paddle_pipeline import mineru_rerun


def _find_work_dir(pdf_path: str) -> str:
    """Locate the existing pipeline work directory for *pdf_path*.

    Must match the derivation in main.py exactly (hash of realpath + mtime + size).
    """
    _stat = os.stat(pdf_path)
    _content_key = f"{os.path.realpath(pdf_path)}:{_stat.st_mtime}:{_stat.st_size}"
    file_hash = hashlib.md5(_content_key.encode("utf-8")).hexdigest()[:8]
    # main.py uses a bare relative name, not os.path.join
    return f"paddle_epub_work_{file_hash}"


def _run_conversion(pdf_path: str, title: str, author: str, language: str,
                    output_epub: str | None = None) -> None:
    """Run pdf2epub_paddle.py as a subprocess."""
    cmd = [
        sys.executable, "pdf2epub_paddle.py", pdf_path,
        "--title", title,
        "--author", author,
        "--auto-toc",
        "--api", "paddle",
        "--language", language,
    ]
    if output_epub:
        cmd += ["--output", output_epub]
    print(f"[*] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _find_garbled_spans(
    epub_path: str,
    *,
    limit: int = 80,
    min_score: float = 0.68,
    min_cjk: int = 16,
) -> Dict[str, List[str]]:
    """Return {epub_filename: [suspicious_cjk_excerpt, ...]}."""
    result: Dict[str, List[str]] = {}
    candidates = find_suspicious_cjk_spans_in_epub(
        epub_path,
        limit=limit,
        min_score=min_score,
        min_cjk=min_cjk,
    )
    for item in candidates:
        result.setdefault(item["file"], []).append(item["excerpt"])
    return result


def _verify_epub_package(epub_path: str) -> Dict[str, Any]:
    """Return lightweight EPUB package validation details."""
    report: Dict[str, Any] = {
        "path": os.path.abspath(epub_path),
        "ok": True,
        "errors": [],
        "warnings": [],
        "entry_count": 0,
        "mimetype_first": False,
        "mimetype_stored": False,
        "has_opf": False,
        "has_nav": False,
        "has_ncx": False,
        "ebooklib_items": None,
    }

    try:
        with ZipFile(epub_path) as archive:
            bad_entry = archive.testzip()
            if bad_entry:
                report["errors"].append(f"zip CRC check failed at {bad_entry}")

            infos = archive.infolist()
            names = archive.namelist()
            name_set = set(names)
            report["entry_count"] = len(infos)
            if not infos:
                report["errors"].append("EPUB zip has no entries")
            else:
                first = infos[0]
                report["mimetype_first"] = first.filename == "mimetype"
                report["mimetype_stored"] = (
                    first.filename == "mimetype"
                    and first.compress_type == ZIP_STORED
                )
                if not report["mimetype_first"]:
                    report["errors"].append("mimetype is not the first zip entry")
                elif not report["mimetype_stored"]:
                    report["errors"].append("mimetype entry is compressed")
                else:
                    content = archive.read("mimetype").decode("ascii", "ignore")
                    if content != "application/epub+zip":
                        report["errors"].append("mimetype content is not application/epub+zip")

            basenames = {os.path.basename(name).lower() for name in names}
            opf_path: str | None = None
            container_path = "META-INF/container.xml"
            if container_path in name_set:
                try:
                    container_xml = archive.read(container_path)
                    container_root = ET.fromstring(container_xml)
                    rootfiles = container_root.findall(".//{*}rootfile")
                    for rootfile in rootfiles:
                        candidate = rootfile.attrib.get("full-path", "").strip()
                        if candidate:
                            opf_path = candidate
                            break
                    if opf_path is None:
                        report["errors"].append(
                            "META-INF/container.xml missing rootfile full-path"
                        )
                except (KeyError, ET.ParseError) as exc:
                    report["errors"].append(f"cannot parse META-INF/container.xml: {exc}")
            else:
                # Keep backward-compatible fallback for non-standard archives.
                if "content.opf" in basenames:
                    opf_path = next(
                        (name for name in names if os.path.basename(name).lower() == "content.opf"),
                        None,
                    )

            report["has_opf"] = bool(opf_path and opf_path in name_set)
            if opf_path and not report["has_opf"]:
                report["errors"].append(f"declared OPF rootfile missing: {opf_path}")
            elif not opf_path:
                report["errors"].append("content.opf missing")

            nav_targets: Set[str] = set()
            ncx_targets: Set[str] = set()
            if report["has_opf"] and opf_path:
                try:
                    opf_xml = archive.read(opf_path)
                    opf_root = ET.fromstring(opf_xml)
                    opf_dir = os.path.dirname(opf_path)
                    for item in opf_root.findall(".//{*}item"):
                        href = item.attrib.get("href", "").strip()
                        if not href:
                            continue
                        normalized_href = os.path.normpath(
                            os.path.join(opf_dir, href)
                        ).replace("\\", "/")
                        properties = item.attrib.get("properties", "")
                        if "nav" in properties.split():
                            nav_targets.add(normalized_href)
                        if item.attrib.get("media-type") == "application/x-dtbncx+xml":
                            ncx_targets.add(normalized_href)
                except (KeyError, ET.ParseError) as exc:
                    report["errors"].append(f"cannot parse OPF manifest: {exc}")

            report["has_nav"] = bool(
                {"nav.xhtml", "nav.html"} & basenames
                or nav_targets & name_set
            )
            report["has_ncx"] = bool("toc.ncx" in basenames or ncx_targets & name_set)
            if not report["has_nav"]:
                report["errors"].append("EPUB nav document missing")
            if not report["has_ncx"]:
                report["warnings"].append("toc.ncx missing")
    except (BadZipFile, OSError) as exc:
        report["errors"].append(f"cannot read EPUB zip: {exc}")

    if ebooklib_epub is None:
        report["warnings"].append("ebooklib is not installed; skipped parse check")
    elif not report["errors"]:
        try:
            book = ebooklib_epub.read_epub(epub_path)
            report["ebooklib_items"] = len(list(book.get_items()))
        except Exception as exc:  # pragma: no cover - exact ebooklib errors vary
            report["errors"].append(f"ebooklib cannot parse EPUB: {exc}")

    report["ok"] = not report["errors"]
    return report


def _print_epub_verification(report: Dict[str, Any]) -> None:
    """Print a compact validation summary for human review."""
    print(f"[*] EPUB package: {report['path']}")
    print(f"    entries={report['entry_count']}")
    print(
        "    mimetype_first={mimetype_first} mimetype_stored={mimetype_stored} "
        "opf={has_opf} nav={has_nav} ncx={has_ncx}".format(**report)
    )
    if report.get("ebooklib_items") is not None:
        print(f"    ebooklib_items={report['ebooklib_items']}")
    for warning in report["warnings"]:
        print(f"    [warning] {warning}")
    for error in report["errors"]:
        print(f"    [error] {error}")


def _write_json_report(path: str | None, report: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[*] Wrote report: {os.path.abspath(path)}")


def _print_page_boundary_candidates(candidates: List[Dict[str, Any]]) -> None:
    """Print a compact list of checkpoint boundary review candidates."""
    print(f"[*] {len(candidates)} page-boundary candidates")
    for item in candidates[:20]:
        reason_text = ",".join(item.get("reasons", []))
        print(
            "    pages={previous_page}->{next_page} score={score:.3f} "
            "reasons={reason_text}".format(reason_text=reason_text, **item)
        )
        print(f"      tail: {item.get('tail', '')}")
        print(f"      head: {item.get('head', '')}")
    if len(candidates) > 20:
        print(f"    ... {len(candidates) - 20} more candidates in JSON report")


def _map_spans_to_pages(work_dir: str, garbled_spans: Dict[str, List[str]]) -> Set[int]:
    """Search checkpoint JSONs for garbled span text → set of 1-based PDF pages."""
    pages: Set[int] = set()
    checkpoint_pages: List[tuple[int, str]] = []

    for json_name in sorted(os.listdir(work_dir)):
        if not json_name.endswith(".json"):
            continue
        # Expect names like chunk_0_5.pdf.json
        base = json_name.replace(".pdf.json", "").replace("chunk_", "")
        try:
            chunk_start = int(base.split("_")[0])
        except (ValueError, IndexError):
            continue

        json_path = os.path.join(work_dir, json_name)
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        layouts = data.get("result", {}).get("layoutParsingResults", [])
        for idx, page in enumerate(layouts):
            md_text = page.get("markdown", {}).get("text", "")
            checkpoint_pages.append((chunk_start + idx + 1, md_text))

    for spans in garbled_spans.values():
        for span_text in spans:
            # Search for meaningful CJK substrings (≥6 chars) and prefer the
            # longest unique run to avoid broad cross-page matches.
            cjk_runs = sorted(
                set(re.findall(r"[㐀-鿿]{6,}", span_text)),
                key=len,
                reverse=True,
            )
            for run in cjk_runs:
                matches = [page_num for page_num, md_text in checkpoint_pages if run in md_text]
                if len(matches) == 1:
                    pages.add(matches[0])
                    break
    return pages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect garbled CJK text and fix via MinerU reruns."
    )
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument("--title", help="Book title")
    parser.add_argument("--author", help="Author name")
    parser.add_argument("--language", default="zh-Hant")
    parser.add_argument("--output", default=None,
                        help="Output EPUB (default: <pdf_basename>.epub)")
    parser.add_argument("--scan-only", action="store_true",
                        help=("Only scan and validate the existing EPUB; "
                              "do not map pages or rerun MinerU"))
    parser.add_argument("--dry-run", action="store_true",
                        help=("Scan and map suspicious spans to PDF pages, "
                              "then stop before MinerU rerun"))
    parser.add_argument("--json-report", default=None,
                        help="Write scan, mapping, and validation details as JSON")
    parser.add_argument("--scan-boundaries", action="store_true",
                        help=("Also scan OCR checkpoints for page-boundary "
                              "missing-sentence candidates"))
    parser.add_argument("--boundary-limit", type=int, default=80,
                        help="Maximum page-boundary candidates to report")
    parser.add_argument("--boundary-min-score", type=float, default=0.70,
                        help="Minimum page-boundary suspicion score, 0-1")
    parser.add_argument("--limit", type=int, default=80,
                        help="Maximum suspicious CJK candidates to consider")
    parser.add_argument("--min-score", type=float, default=0.68,
                        help="Minimum suspicious CJK score, 0-1")
    parser.add_argument("--min-cjk", type=int, default=16,
                        help="Minimum CJK characters in a suspicious candidate")
    args = parser.parse_args(argv)

    pdf_path = os.path.abspath(args.pdf)
    output_epub = args.output or os.path.splitext(os.path.basename(pdf_path))[0] + ".epub"
    work_dir = _find_work_dir(pdf_path)
    run_report: Dict[str, Any] = {
        "pdf": pdf_path,
        "output_epub": os.path.abspath(output_epub),
        "work_dir": os.path.abspath(work_dir),
        "scan_only": args.scan_only,
        "dry_run": args.dry_run,
        "garbled": {},
        "page_boundaries": [],
        "target_pages": [],
        "warnings": [],
        "verification": None,
    }

    needs_initial_conversion = not os.path.exists(output_epub)
    if args.scan_only and needs_initial_conversion:
        parser.error("--scan-only requires an existing EPUB; pass --output if needed")
    if needs_initial_conversion and (not args.title or not args.author):
        parser.error("--title and --author are required when initial conversion is needed")

    # --- Step 1: Initial Paddle conversion ---
    print("=" * 60)
    print("[1/5] PaddleOCR conversion")
    print("=" * 60)
    if not os.path.exists(output_epub):
        _run_conversion(pdf_path, args.title or "", args.author or "", args.language, output_epub)
    else:
        print(f"[*] {output_epub} exists, skip conversion")
    if not os.path.exists(output_epub):
        print("[!] EPUB was not created. Check the conversion output above.")
        sys.exit(1)

    # --- Step 2: Scan for garbled text ---
    print()
    print("=" * 60)
    print("[2/5] Scanning for garbled CJK text")
    print("=" * 60)
    garbled = _find_garbled_spans(
        output_epub,
        limit=args.limit,
        min_score=args.min_score,
        min_cjk=args.min_cjk,
    )
    total = sum(len(s) for s in garbled.values())
    run_report["garbled"] = garbled
    print(f"[*] {total} garbled spans in {len(garbled)} files")
    for fname, spans in garbled.items():
        print(f"    {os.path.basename(fname):50s} {len(spans)} spans")

    boundary_candidates: List[Dict[str, Any]] = []
    if args.scan_boundaries:
        print()
        print("=" * 60)
        print("[boundary] Scanning checkpoint page boundaries")
        print("=" * 60)
        if not os.path.isdir(work_dir):
            warning = f"work directory not found for boundary scan: {work_dir}"
            run_report["warnings"].append(warning)
            print(f"[warning] {warning}")
        else:
            boundary_candidates = find_page_boundary_candidates(
                work_dir,
                min_score=args.boundary_min_score,
                limit=args.boundary_limit,
            )
            boundary_candidates = annotate_candidates_with_epub_continuity(
                boundary_candidates,
                output_epub,
            )
            run_report["page_boundaries"] = boundary_candidates
            _print_page_boundary_candidates(boundary_candidates)

    if args.scan_only or not garbled:
        print()
        print("=" * 60)
        print("[verification] EPUB package checks")
        print("=" * 60)
        verification = _verify_epub_package(output_epub)
        run_report["verification"] = verification
        _print_epub_verification(verification)
        _write_json_report(args.json_report, run_report)
        if not verification["ok"]:
            sys.exit(1)
    if not garbled:
        if boundary_candidates:
            print(
                "[!] No garbled spans found, but page-boundary candidates "
                "need PDF or second-OCR review."
            )
        else:
            print("[*] EPUB is clean, nothing to fix.")
        return
    if args.scan_only:
        return
    if not args.dry_run and (not args.title or not args.author):
        parser.error("--title and --author are required to rebuild when garbled spans are found")

    # --- Step 3: Map spans to PDF pages ---
    print()
    print("=" * 60)
    print("[3/5] Mapping spans to PDF pages")
    print("=" * 60)
    if not os.path.isdir(work_dir):
        print(f"[!] Work directory not found: {work_dir}")
        print("[*] Run the Paddle conversion first to create it.")
        sys.exit(1)
    target_pages = _map_spans_to_pages(work_dir, garbled)
    if not target_pages:
        print("[!] Could not map garbled spans to PDF pages.")
        print("[*] Try running with a fresh conversion first.")
        sys.exit(1)
    sorted_pages = sorted(target_pages)
    run_report["target_pages"] = sorted_pages
    print(f"[*] {len(target_pages)} pages to fix: {sorted_pages}")
    if args.dry_run:
        print("[*] Dry run requested; stopping before MinerU rerun.")
        verification = _verify_epub_package(output_epub)
        run_report["verification"] = verification
        _print_epub_verification(verification)
        _write_json_report(args.json_report, run_report)
        if not verification["ok"]:
            sys.exit(1)
        return

    # --- Step 4: MinerU rerun ---
    print()
    print("=" * 60)
    print("[4/5] MinerU rerun (sentence-level patching)")
    print("=" * 60)
    summaries = mineru_rerun.rerun_mineru_pages(
        pdf_path,
        work_dir,
        pages=target_pages,
        chunk_size=PADDLE_CHUNK_SIZE,
        replace_page=False,
    )
    for s in summaries:
        print(f"    page {s.page_number}: {s.text_preview[:80]}...")

    # --- Step 5: Rebuild EPUB ---
    print()
    print("=" * 60)
    print("[5/5] Rebuilding EPUB")
    print("=" * 60)
    if os.path.exists(output_epub):
        os.remove(output_epub)
    _run_conversion(pdf_path, args.title or "", args.author or "", args.language, output_epub)

    print()
    print("=" * 60)
    print("[verification] EPUB package checks")
    print("=" * 60)
    verification = _verify_epub_package(output_epub)
    run_report["verification"] = verification
    _print_epub_verification(verification)
    _write_json_report(args.json_report, run_report)
    if not verification["ok"]:
        sys.exit(1)
    print()
    print(f"Done → {os.path.abspath(output_epub)}")


if __name__ == "__main__":
    main()
