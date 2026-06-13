"""Footnote extraction, linking, and HTML rendering."""

import html
import re

from typing import Any, Dict, List

from .config import (
    FOOTNOTE_LABELS,
    FOOTNOTE_MARKER_PATTERN,
    INLINE_FOOTNOTE_MARKER_PATTERN,
)
from .ocr_noise import clean_ocr_noise


def extract_page_footnotes(page_res: Dict[str, Any]) -> List[str]:
    """Extract OCR footnote blocks omitted from Paddle's markdown text."""
    footnotes: List[str] = []
    blocks = page_res.get("prunedResult", {}).get("parsing_res_list", [])
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("block_label") not in FOOTNOTE_LABELS:
            continue
        content = str(block.get("block_content") or "").strip()
        if not content:
            continue
        cleaned = clean_ocr_noise(content)
        if cleaned:
            footnotes.append(cleaned)
    return _infer_missing_footnote_markers(page_res, footnotes)


def _infer_missing_footnote_markers(
    page_res: Dict[str, Any],
    footnotes: List[str],
) -> List[str]:
    inline_numbers = [
        match.group(1)
        for match in INLINE_FOOTNOTE_MARKER_PATTERN.finditer(
            page_res.get("markdown", {}).get("text", "")
        )
    ]
    if not inline_numbers:
        return footnotes

    explicit_counts: Dict[str, int] = {}
    parsed_numbers: List[str | None] = []
    for footnote in footnotes:
        number, _body = _split_footnote_marker(footnote)
        parsed_numbers.append(number)
        if number is not None:
            explicit_counts[number] = explicit_counts.get(number, 0) + 1

    available_numbers: List[str] = []
    seen_counts: Dict[str, int] = {}
    for number in inline_numbers:
        seen_counts[number] = seen_counts.get(number, 0) + 1
        if seen_counts[number] > explicit_counts.get(number, 0):
            available_numbers.append(number)

    if not available_numbers:
        return footnotes

    inferred: List[str] = []
    available_index = 0
    for footnote, number in zip(footnotes, parsed_numbers):
        if number is None and available_index < len(available_numbers):
            inferred.append(f"$ ^{{{available_numbers[available_index]}}} $ {footnote}")
            available_index += 1
        else:
            inferred.append(footnote)
    return inferred


def _split_footnote_marker(footnote: str) -> tuple[str | None, str]:
    marker_match = FOOTNOTE_MARKER_PATTERN.match(footnote)
    if marker_match:
        return marker_match.group(1), marker_match.group(2).strip()
    return None, footnote


def _footnote_anchor_id(prefix: str, page_number: int, number: str, occurrence: int) -> str:
    suffix = f"-{occurrence}" if occurrence > 1 else ""
    return f"{prefix}-p{page_number}-{number}{suffix}"


def _build_page_footnote_refs(
    footnotes: List[str],
    page_number: int,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for footnote in footnotes:
        number, _body = _split_footnote_marker(footnote)
        if number is None:
            refs.append({})
            continue
        occurrence = counts.get(number, 0) + 1
        counts[number] = occurrence
        refs.append(
            {
                "number": number,
                "note_id": _footnote_anchor_id("fn", page_number, number, occurrence),
                "ref_id": _footnote_anchor_id("fnref", page_number, number, occurrence),
                "linked": False,
            }
        )
    return refs


def link_page_footnote_references(
    markdown_text: str,
    footnotes: List[str],
    page_number: int,
) -> tuple[str, List[Dict[str, Any]]]:
    """Turn inline OCR footnote markers into EPUB noteref links when possible."""
    refs = _build_page_footnote_refs(footnotes, page_number)
    refs_by_number: Dict[str, List[Dict[str, Any]]] = {}
    for ref in refs:
        number = ref.get("number")
        if number:
            refs_by_number.setdefault(str(number), []).append(ref)

    used_by_number: Dict[str, int] = {}

    def replace_marker(match: re.Match[str]) -> str:
        number = match.group(1)
        used = used_by_number.get(number, 0)
        numbered_refs = refs_by_number.get(number, [])
        if used < len(numbered_refs):
            ref = numbered_refs[used]
            used_by_number[number] = used + 1
            ref["linked"] = True
            return (
                f'<sup id="{html.escape(ref["ref_id"])}" class="footnote-ref">'
                f'<a epub:type="noteref" href="#{html.escape(ref["note_id"])}">'
                f"{html.escape(number)}</a></sup>"
            )
        return (
            '<sup class="unlinked-footnote-marker">'
            + html.escape(number)
            + "</sup>"
        )

    return INLINE_FOOTNOTE_MARKER_PATTERN.sub(replace_marker, markdown_text), refs


def _format_single_footnote_html(
    footnote: str,
    footnote_ref: Dict[str, Any] | None = None,
) -> str:
    number, body = _split_footnote_marker(footnote)
    linked = bool(footnote_ref and footnote_ref.get("linked"))
    attrs = ""
    backlink = ""
    if linked and footnote_ref is not None:
        attrs = f' id="{html.escape(str(footnote_ref["note_id"]))}"'
        backlink = (
            f' <a class="footnote-backlink" epub:type="backlink" '
            f'href="#{html.escape(str(footnote_ref["ref_id"]))}">&#8617;</a>'
        )
    if number is not None:
        return (
            f'<p{attrs} class="footnote"><sup>'
            + html.escape(number)
            + "</sup> "
            + html.escape(body)
            + backlink
            + "</p>"
        )
    return f'<p{attrs} class="footnote">' + html.escape(footnote) + backlink + "</p>"


def format_page_footnotes_html(
    footnotes: List[str],
    page_number: int,
    footnote_refs: List[Dict[str, Any]] | None = None,
) -> str:
    """Render page footnotes as EPUB-friendly HTML appended near their source page."""
    if not footnotes:
        return ""
    if footnote_refs is None:
        footnote_refs = [{} for _footnote in footnotes]
    items = "\n".join(
        _format_single_footnote_html(
            footnote,
            footnote_refs[index] if index < len(footnote_refs) else {},
        )
        for index, footnote in enumerate(footnotes)
    )
    return (
        f'<section class="page-footnotes" epub:type="footnotes" '
        f'data-source-page="{page_number}">\n'
        f"{items}\n"
        "</section>"
    )

