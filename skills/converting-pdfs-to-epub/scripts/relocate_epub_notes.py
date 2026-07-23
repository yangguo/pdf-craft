#!/usr/bin/env python3
"""Move interleaved EPUB footnote blocks to the end of each chapter."""

from __future__ import annotations

import argparse
import json
import os
import re
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
INLINE_NOTE_CSS = b"""

.inline-note-marker {
  font-size: .72em;
  line-height: 0;
  vertical-align: super;
}
p > .page-anchor {
  display: inline;
}
"""

ElementTree.register_namespace("", XHTML_NS)
ElementTree.register_namespace("epub", EPUB_NS)

INLINE_NOTE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$", re.DOTALL)
NUMBERED_NOTE_FRAGMENT_RE = re.compile(r"^\d{1,3}\.\s")
SENTENCE_ENDINGS = frozenset("。！？!?；;")


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


def _join_fragments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and re.search(r"[A-Za-z]-$", left) and re.match(
        r"^[A-Za-z]", right
    ):
        return left[:-1] + right
    if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right):
        return left + " " + right
    return left + right


def _append_text(element: ElementTree.Element, text: str) -> None:
    if not text:
        return
    if len(element):
        last = element[-1]
        last.tail = _join_fragments(last.tail or "", text)
    else:
        element.text = _join_fragments(element.text or "", text)


def _append_contents(
    target: ElementTree.Element,
    source: ElementTree.Element,
) -> None:
    _append_text(target, source.text or "")
    for child in list(source):
        source.remove(child)
        target.append(child)


def _is_page_anchor(element: ElementTree.Element) -> bool:
    return (
        element.tag == f"{XHTML}span"
        and "page-anchor" in element.attrib.get("class", "").split()
    )


def _append_marker(
    paragraph: ElementTree.Element,
    number: str,
    remainder: str,
) -> None:
    marker = ElementTree.SubElement(
        paragraph,
        f"{XHTML}sup",
        {"class": "inline-note-marker"},
    )
    marker.text = f"[{number}]"
    marker.tail = remainder


def repair_inline_note_markers_in_xhtml(
    content: bytes,
) -> tuple[bytes, int, int]:
    """Turn false blockquote note markers back into inline paragraph content."""
    root = ElementTree.fromstring(content)
    body = root.find(f"{XHTML}body")
    if body is None:
        return content, 0, 0

    repaired = 0
    joined = 0
    for marker_block in list(body):
        if marker_block.tag != f"{XHTML}blockquote":
            continue
        match = INLINE_NOTE_RE.fullmatch(
            "".join(marker_block.itertext()).strip()
        )
        if match is None:
            continue

        children = list(body)
        index = children.index(marker_block)
        remainder = match.group(2).strip()

        preceding_anchor = None
        if index > 0 and children[index - 1].tag == f"{XHTML}p":
            paragraph = children[index - 1]
        elif (
            index > 1
            and _is_page_anchor(children[index - 1])
            and children[index - 2].tag == f"{XHTML}p"
        ):
            paragraph = children[index - 2]
            preceding_anchor = children[index - 1]
        else:
            paragraph = marker_block

        following_anchor = None
        continuation = None
        if index + 1 < len(children) and children[index + 1].tag == f"{XHTML}p":
            continuation = children[index + 1]
        elif (
            index + 2 < len(children)
            and _is_page_anchor(children[index + 1])
            and children[index + 2].tag == f"{XHTML}p"
        ):
            following_anchor = children[index + 1]
            continuation = children[index + 2]

        if paragraph is marker_block:
            tail = marker_block.tail
            marker_block.clear()
            marker_block.tag = f"{XHTML}p"
            marker_block.tail = tail
        else:
            if preceding_anchor is not None:
                body.remove(preceding_anchor)
                preceding_anchor.tail = None
                paragraph.append(preceding_anchor)
            body.remove(marker_block)

        _append_marker(paragraph, match.group(1), remainder)

        if (
            remainder
            and remainder[-1] not in SENTENCE_ENDINGS
            and continuation is not None
        ):
            if following_anchor is not None:
                body.remove(following_anchor)
                following_anchor.tail = None
                paragraph.append(following_anchor)
            _append_contents(paragraph, continuation)
            body.remove(continuation)
            joined += 1
        repaired += 1

    if not repaired:
        return content, 0, 0
    return _serialize_xhtml(root), repaired, joined


def repair_leading_paragraph_markers_in_xhtml(
    content: bytes,
) -> tuple[bytes, int]:
    """Style direct body paragraphs that begin with a footnote marker."""
    root = ElementTree.fromstring(content)
    body = root.find(f"{XHTML}body")
    if body is None:
        return content, 0

    repaired = 0
    for paragraph in list(body):
        if paragraph.tag != f"{XHTML}p" or not paragraph.text:
            continue
        match = re.match(r"^\s*\[(\d+)\]\s*", paragraph.text)
        if match is None:
            continue
        remainder = paragraph.text[match.end() :]
        paragraph.text = None
        marker = ElementTree.Element(
            f"{XHTML}sup",
            {"class": "inline-note-marker"},
        )
        marker.text = f"[{match.group(1)}]"
        marker.tail = remainder
        paragraph.insert(0, marker)
        repaired += 1

    if not repaired:
        return content, 0
    return _serialize_xhtml(root), repaired


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


def relocate_interposed_note_fragments_in_xhtml(
    content: bytes,
) -> tuple[bytes, int]:
    """Move page-bottom fragments that split an unfinished body sentence."""
    root = ElementTree.fromstring(content)
    body = root.find(f"{XHTML}body")
    if body is None:
        return content, 0

    endnotes = next(
        (
            child
            for child in body
            if child.tag == f"{XHTML}section"
            and "chapter-notes" in child.attrib.get("class", "").split()
        ),
        None,
    )
    moved = 0
    for marker_block in list(body):
        if marker_block.tag != f"{XHTML}blockquote":
            continue
        marker_match = INLINE_NOTE_RE.fullmatch(
            "".join(marker_block.itertext()).strip()
        )
        if marker_match is None:
            continue
        remainder = marker_match.group(2).strip()
        if not remainder or remainder[-1] in SENTENCE_ENDINGS:
            continue

        children = list(body)
        index = children.index(marker_block)
        if index + 2 >= len(children):
            continue
        fragment = children[index + 1]
        if not _is_page_anchor(children[index + 2]):
            continue

        fragment_text = "".join(fragment.itertext()).strip()
        is_numbered_heading = (
            fragment.tag == f"{XHTML}h3"
            and NUMBERED_NOTE_FRAGMENT_RE.match(fragment_text) is not None
        )
        is_layout_quote = (
            fragment.tag == f"{XHTML}blockquote"
            and INLINE_NOTE_RE.fullmatch(fragment_text) is None
        )
        if not (is_numbered_heading or is_layout_quote):
            continue

        if endnotes is None:
            endnotes = ElementTree.Element(
                f"{XHTML}section",
                {"class": "chapter-notes", EPUB_TYPE: "endnotes"},
            )
            heading = ElementTree.SubElement(endnotes, f"{XHTML}h2")
            heading.text = "注釋"
            body.append(endnotes)

        body.remove(fragment)
        aside = ElementTree.SubElement(
            endnotes,
            f"{XHTML}aside",
            {"class": "footnotes relocated-fragment", EPUB_TYPE: "footnotes"},
        )
        fragment_children = list(fragment)
        if (
            not (fragment.text or "").strip()
            and len(fragment_children) == 1
            and fragment_children[0].tag == f"{XHTML}p"
        ):
            paragraph = fragment_children[0]
            fragment.remove(paragraph)
            paragraph.tail = None
            aside.append(paragraph)
        else:
            paragraph = ElementTree.SubElement(aside, f"{XHTML}p")
            _append_contents(paragraph, fragment)
        moved += 1

    if not moved:
        return content, 0
    return _serialize_xhtml(root), moved


def reorder_displaced_prefixes_in_xhtml(
    content: bytes,
    corrections: list[dict[str, str]],
) -> tuple[bytes, int]:
    """Apply explicit, PDF-verified reading-order fixes around inline markers."""
    if not corrections:
        return content, 0

    root = ElementTree.fromstring(content)
    applied = 0
    for correction in corrections:
        marker_text = f"[{correction['marker']}]"
        remainder = correction["remainder"]
        continuation_starts = correction["continuation_starts"]
        marker = next(
            (
                candidate
                for candidate in root.iter(f"{XHTML}sup")
                if candidate.text == marker_text
                and "inline-note-marker"
                in candidate.attrib.get("class", "").split()
            ),
            None,
        )
        if marker is None:
            continue

        tail = marker.tail or ""
        if not tail.startswith(remainder):
            continue
        continuation_index = tail.find(continuation_starts, len(remainder))
        if continuation_index <= len(remainder):
            continue
        displaced_prefix = tail[len(remainder) : continuation_index]

        parent = next(
            (candidate for candidate in root.iter() if marker in candidate),
            None,
        )
        if parent is None:
            continue
        siblings = list(parent)
        marker_index = siblings.index(marker)
        if marker_index:
            previous = siblings[marker_index - 1]
            previous.tail = (previous.tail or "") + displaced_prefix
        else:
            parent.text = (parent.text or "") + displaced_prefix
        marker.tail = remainder + tail[continuation_index:]
        applied += 1

    if not applied:
        return content, 0
    return _serialize_xhtml(root), applied


def _with_repair_css(
    content: bytes,
    *,
    include_chapter_notes: bool,
    include_inline_markers: bool,
) -> bytes:
    result = content.rstrip()
    if include_chapter_notes and b".chapter-notes" not in result:
        result += CHAPTER_NOTES_CSS
    if include_inline_markers and b".inline-note-marker" not in result:
        result += INLINE_NOTE_CSS
    return result


def _stored_mimetype_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("mimetype", source.date_time)
    info.comment = source.comment
    info.extra = source.extra
    info.internal_attr = source.internal_attr
    info.external_attr = source.external_attr
    info.create_system = source.create_system
    info.compress_type = zipfile.ZIP_STORED
    return info


def relocate_epub_notes(
    input_path: Path,
    output_path: Path,
    reading_order_corrections: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, int]:
    """Write a copy of an EPUB with each chapter's notes collected at its end."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output EPUB paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten_members: dict[str, bytes] = {}
    chapters_changed = 0
    notes_moved = 0
    note_fragments_relocated = 0
    inline_markers_repaired = 0
    leading_markers_repaired = 0
    continuations_joined = 0
    reading_order_corrections_applied = 0
    expected_corrections = sum(
        len(corrections)
        for corrections in (reading_order_corrections or {}).values()
    )

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
            content = source.read(info.filename)
            rewritten, moved = relocate_notes_in_xhtml(content)
            rewritten, fragments_moved = (
                relocate_interposed_note_fragments_in_xhtml(rewritten)
            )
            rewritten, repaired, joined = repair_inline_note_markers_in_xhtml(
                rewritten
            )
            rewritten, leading_repaired = (
                repair_leading_paragraph_markers_in_xhtml(rewritten)
            )
            rewritten, corrections_applied = reorder_displaced_prefixes_in_xhtml(
                rewritten,
                (reading_order_corrections or {}).get(info.filename, []),
            )
            if (
                moved
                or fragments_moved
                or repaired
                or leading_repaired
                or corrections_applied
            ):
                rewritten_members[info.filename] = rewritten
                chapters_changed += 1
            notes_moved += moved
            note_fragments_relocated += fragments_moved
            inline_markers_repaired += repaired
            leading_markers_repaired += leading_repaired
            continuations_joined += joined
            reading_order_corrections_applied += corrections_applied

        if reading_order_corrections_applied != expected_corrections:
            raise ValueError(
                "not all reading-order corrections matched: "
                f"{reading_order_corrections_applied}/{expected_corrections}"
            )

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
                        content = _with_repair_css(
                            content,
                            include_chapter_notes=bool(notes_moved),
                            include_inline_markers=bool(
                                inline_markers_repaired
                                or leading_markers_repaired
                            ),
                        )
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
        "note_fragments_relocated": note_fragments_relocated,
        "inline_markers_repaired": inline_markers_repaired,
        "leading_markers_repaired": leading_markers_repaired,
        "continuations_joined": continuations_joined,
        "reading_order_corrections_applied": (
            reading_order_corrections_applied
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move interleaved EPUB footnotes to the end of each chapter.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reading-order-corrections",
        type=Path,
        help="JSON mapping of EPUB member names to PDF-verified marker fixes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corrections = None
    if args.reading_order_corrections is not None:
        corrections = json.loads(
            args.reading_order_corrections.read_text(encoding="utf-8")
        )
        if not isinstance(corrections, dict):
            raise ValueError("reading-order corrections must be a JSON object")
    stats = relocate_epub_notes(args.input, args.output, corrections)
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
