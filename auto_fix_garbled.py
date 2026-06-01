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

from typing import Dict, List, Set

from paddle_pipeline.config import CHUNK_SIZE as PADDLE_CHUNK_SIZE, EPUB_STRUCTURAL_FILES
# Private helpers needed because find_garbled_cjk_in_epub reports counts
# but not span text; we need actual garbled substrings for checkpoint search.
from paddle_pipeline.ocr_noise import (
    _build_cjk_bigram_model,
    _scan_text_for_garbled_cjk_spans,
    _strip_html_tags,
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


def _find_garbled_spans(epub_path: str) -> Dict[str, List[str]]:
    """Return {epub_filename: [garbled_cjk_span, ...]}."""
    structural = EPUB_STRUCTURAL_FILES

    # Build full-book bigram model once
    import zipfile
    all_cjk: List[str] = []
    file_texts: Dict[str, str] = {}
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if os.path.basename(lower) in structural:
                continue
            if not lower.endswith((".xhtml", ".html")):
                continue
            plain = _strip_html_tags(archive.read(name).decode("utf-8", "ignore"))
            file_texts[name] = plain
            all_cjk.extend(re.findall(r"[一-鿿]", plain))

    bigram_freq = _build_cjk_bigram_model(all_cjk)

    result: Dict[str, List[str]] = {}
    for name, text in file_texts.items():
        spans = _scan_text_for_garbled_cjk_spans(text, bigram_freq)
        if spans:
            result[name] = spans
    return result


def _map_spans_to_pages(work_dir: str, garbled_spans: Dict[str, List[str]]) -> Set[int]:
    """Search checkpoint JSONs for garbled span text → set of 1-based PDF pages."""
    pages: Set[int] = set()

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
            for spans in garbled_spans.values():
                for span_text in spans:
                    # Search for a meaningful CJK substring (≥6 chars) to avoid
                    # false matches on tiny fragments
                    cjk_runs = re.findall(r"[一-鿿]{6,}", span_text)
                    for run in cjk_runs:
                        if run in md_text:
                            pages.add(chunk_start + idx + 1)  # 1-based
                            break
    return pages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect garbled CJK text and fix via MinerU reruns."
    )
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument("--title", required=True, help="Book title")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--language", default="zh-Hant")
    parser.add_argument("--output", default=None,
                        help="Output EPUB (default: <pdf_basename>.epub)")
    args = parser.parse_args(argv)

    pdf_path = os.path.abspath(args.pdf)
    output_epub = args.output or os.path.splitext(os.path.basename(pdf_path))[0] + ".epub"
    work_dir = _find_work_dir(pdf_path)

    # --- Step 1: Initial Paddle conversion ---
    print("=" * 60)
    print("[1/5] PaddleOCR conversion")
    print("=" * 60)
    if not os.path.exists(output_epub):
        _run_conversion(pdf_path, args.title, args.author, args.language, output_epub)
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
    garbled = _find_garbled_spans(output_epub)
    total = sum(len(s) for s in garbled.values())
    print(f"[*] {total} garbled spans in {len(garbled)} files")
    for fname, spans in garbled.items():
        print(f"    {os.path.basename(fname):50s} {len(spans)} spans")

    if not garbled:
        print("[*] EPUB is clean, nothing to fix.")
        return

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
    print(f"[*] {len(target_pages)} pages to fix: {sorted(target_pages)}")

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
    _run_conversion(pdf_path, args.title, args.author, args.language, output_epub)
    print()
    print(f"Done → {os.path.abspath(output_epub)}")


if __name__ == "__main__":
    main()
