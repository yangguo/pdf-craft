#!/usr/bin/env python3
"""Move interleaved EPUB footnote blocks to the end of each chapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree


XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"
XHTML = f"{{{XHTML_NS}}}"
EPUB_TYPE = f"{{{EPUB_NS}}}type"
CHAPTER_NOTES_CSS = b"""

.chapter-notes {
  page-break-before: always;
  break-before: page;
  margin-top: 2em;
}
.chapter-notes > h2 { margin-top: 0; }
.chapter-notes .footnotes {
  border-top: 0;
  margin: .75em 0;
  padding-top: 0;
}
"""

ElementTree.register_namespace("", XHTML_NS)
ElementTree.register_namespace("epub", EPUB_NS)


def _is_footnotes_aside(element: ElementTree.Element) -> bool:
    classes = element.attrib.get("class", "").split()
    return (
        element.tag == f"{XHTML}aside"
        and (
            "footnotes" in classes
            or element.attrib.get(EPUB_TYPE) == "footnotes"
        )
    )


def _serialize_xhtml(root: ElementTree.Element) -> bytes:
    xml = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    declaration, document = xml.split(b"\n", 1)
    return declaration + b"\n<!DOCTYPE html>\n" + document


def relocate_notes_in_xhtml(content: bytes) -> tuple[bytes, int]:
    """Collect direct chapter footnote asides into one endnotes section."""
    root = ElementTree.fromstring(content)
    body = root.find(f"{XHTML}body")
    if body is None:
        return content, 0

    notes = [child for child in list(body) if _is_footnotes_aside(child)]
    if not notes:
        return content, 0

    endnotes = ElementTree.Element(
        f"{XHTML}section",
        {"class": "chapter-notes", EPUB_TYPE: "endnotes"},
    )
    heading = ElementTree.SubElement(endnotes, f"{XHTML}h2")
    heading.text = "注釋"

    for note in notes:
        body.remove(note)
        endnotes.append(note)
    body.append(endnotes)
    return _serialize_xhtml(root), len(notes)


def _with_chapter_notes_css(content: bytes) -> bytes:
    if b".chapter-notes" in content:
        return content
    return content.rstrip() + CHAPTER_NOTES_CSS


def _stored_mimetype_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("mimetype", source.date_time)
    info.comment = source.comment
    info.extra = source.extra
    info.internal_attr = source.internal_attr
    info.external_attr = source.external_attr
    info.create_system = source.create_system
    info.compress_type = zipfile.ZIP_STORED
    return info


def relocate_epub_notes(input_path: Path, output_path: Path) -> dict[str, int]:
    """Write a copy of an EPUB with each chapter's notes collected at its end."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output EPUB paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten_members: dict[str, bytes] = {}
    chapters_changed = 0
    notes_moved = 0

    with zipfile.ZipFile(input_path) as source:
        infos = source.infolist()
        try:
            mimetype_info = source.getinfo("mimetype")
        except KeyError as exc:
            raise ValueError("input is missing the EPUB mimetype member") from exc
        if source.read("mimetype") != b"application/epub+zip":
            raise ValueError("input has an invalid EPUB mimetype member")

        for info in infos:
            if not info.filename.lower().endswith((".xhtml", ".html")):
                continue
            rewritten, moved = relocate_notes_in_xhtml(source.read(info.filename))
            if moved:
                rewritten_members[info.filename] = rewritten
                chapters_changed += 1
                notes_moved += moved

        ordered_infos = [mimetype_info] + [
            info for info in infos if info.filename != "mimetype"
        ]
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        try:
            with zipfile.ZipFile(temporary_path, "w") as destination:
                for info in ordered_infos:
                    content = rewritten_members.get(
                        info.filename,
                        source.read(info.filename),
                    )
                    if chapters_changed and info.filename.lower().endswith(".css"):
                        content = _with_chapter_notes_css(content)
                    target_info = (
                        _stored_mimetype_info(info)
                        if info.filename == "mimetype"
                        else info
                    )
                    destination.writestr(target_info, content)

            with zipfile.ZipFile(temporary_path) as candidate:
                if candidate.testzip() is not None:
                    raise ValueError("rewritten EPUB failed ZIP integrity testing")
            os.chmod(temporary_path, input_path.stat().st_mode & 0o777)
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    return {
        "chapters_changed": chapters_changed,
        "notes_moved": notes_moved,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move interleaved EPUB footnotes to the end of each chapter.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = relocate_epub_notes(args.input, args.output)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        **stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
