"""Heading detection, garbled heading cleanup, and bibliography title removal."""

import html
import os
import re

from typing import Any, Dict

from .config import (
    KNOWN_GARBLED_HEADINGS,
    NUMBERED_TOC_LABEL_PATTERN,
    PART_TOC_KINDS,
)
from .epub_href import (
    _epub_spine_xhtml_order,
    _extract_numbered_toc_marker,
    _is_chapter_number_text,
    _iter_xhtml_text_elements,
    _normalize_numbered_marker,
    _plain_text_from_html,
    _resolve_epub_file_href,
    _resolve_epub_fragment_href,
)


def _normalize_heading_preview_text(value: str) -> str:
    value = _normalize_numbered_marker(value)
    value = value.replace("强", "強")
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]", "", value)


def _has_numbered_toc_marker(elements: list[Dict[str, Any]]) -> bool:
    return any(
        NUMBERED_TOC_LABEL_PATTERN.match(element["text"].strip()) for element in elements
    )


def _only_closing_markup_after_last_element(xhtml_text: str, last_end: int) -> bool:
    trailer = xhtml_text[last_end:]
    trailer = re.sub(r"(?is)</?(?:body|html)\b[^>]*>", "", trailer)
    return not trailer.strip()


def _normalized_preview_marker_and_suffix(
    elements: list[Dict[str, Any]],
) -> tuple[str, str] | None:
    if not elements:
        return None
    marker_info = _extract_numbered_toc_marker(elements[0]["text"])
    if not marker_info:
        return None

    marker = _normalize_heading_preview_text(marker_info[0])
    preview = _normalize_heading_preview_text(
        "".join(element["text"] for element in elements)
    )
    if not marker or not preview.startswith(marker):
        return None
    return marker, preview[len(marker):]


def _heading_preview_matches_next(
    candidate_elements: list[Dict[str, Any]],
    next_elements: list[Dict[str, Any]],
) -> bool:
    candidate = _normalize_heading_preview_text(
        "".join(element["text"] for element in candidate_elements)
    )
    next_prefix = _normalize_heading_preview_text(
        "".join(element["text"] for element in next_elements[:8])
    )
    if candidate and candidate in next_prefix:
        return True

    marker_and_suffix = _normalized_preview_marker_and_suffix(candidate_elements)
    if not marker_and_suffix:
        return False

    marker, suffix = marker_and_suffix
    if len(suffix) < 6:
        return False

    for index in range(min(4, len(next_elements))):
        next_marker_and_suffix = _normalized_preview_marker_and_suffix(
            next_elements[index:index + 8]
        )
        if not next_marker_and_suffix:
            continue
        next_marker, next_suffix = next_marker_and_suffix
        if next_marker == marker and suffix in next_suffix:
            return True
    return False


def _is_orphaned_part_preview(candidate_elements: list[Dict[str, Any]]) -> bool:
    if len(candidate_elements) < 2 or len(candidate_elements) > 5:
        return False

    marker_info = _extract_numbered_toc_marker(candidate_elements[0]["text"])
    if not marker_info or marker_info[1] not in PART_TOC_KINDS:
        return False

    preview_text = " ".join(element["text"] for element in candidate_elements)
    preview_norm = _normalize_heading_preview_text(preview_text)
    marker_norm = _normalize_heading_preview_text(marker_info[0])
    if len(preview_norm) > 80 or len(preview_norm) <= len(marker_norm):
        return False

    return all(
        len(_normalize_heading_preview_text(element["text"])) <= 40
        for element in candidate_elements[1:]
    )


def _is_orphaned_chapter_marker_preview(candidate_elements: list[Dict[str, Any]]) -> bool:
    if len(candidate_elements) != 1:
        return False

    text = candidate_elements[0]["text"].strip()
    marker_info = _extract_numbered_toc_marker(text)
    return bool(marker_info and marker_info[1] == "章" and text == marker_info[0])


def _remove_trailing_next_heading_preview(prev_xhtml: str, next_xhtml: str) -> tuple[str, bool]:
    prev_elements = _iter_xhtml_text_elements(prev_xhtml)
    next_elements = _iter_xhtml_text_elements(next_xhtml)
    if not prev_elements or not next_elements:
        return prev_xhtml, False
    if not _only_closing_markup_after_last_element(prev_xhtml, prev_elements[-1]["end"]):
        return prev_xhtml, False

    best_start_index = None
    last_index = len(prev_elements) - 1
    first_candidate_index = max(0, len(prev_elements) - 8)
    for index in range(last_index, first_candidate_index - 1, -1):
        candidate_elements = prev_elements[index:last_index + 1]
        if not _has_numbered_toc_marker(candidate_elements):
            continue

        if _heading_preview_matches_next(
            candidate_elements,
            next_elements,
        ) or _is_orphaned_part_preview(
            candidate_elements
        ) or _is_orphaned_chapter_marker_preview(candidate_elements):
            best_start_index = index
            continue
        if best_start_index is not None:
            break

    if best_start_index is None:
        return prev_xhtml, False

    remove_start = prev_elements[best_start_index]["start"]
    remove_end = prev_elements[last_index]["end"]
    return prev_xhtml[:remove_start] + prev_xhtml[remove_end:], True


def _remove_trailing_next_heading_previews(entries: Dict[str, bytes]) -> list[str]:
    changed_files = []
    spine = _epub_spine_xhtml_order(entries)
    for prev_name, next_name in zip(spine, spine[1:]):
        if prev_name not in entries or next_name not in entries:
            continue
        prev_xhtml = entries[prev_name].decode("utf-8", "ignore")
        next_xhtml = entries[next_name].decode("utf-8", "ignore")
        patched, changed = _remove_trailing_next_heading_preview(prev_xhtml, next_xhtml)
        if changed:
            entries[prev_name] = patched.encode("utf-8")
            changed_files.append(prev_name)
    return changed_files


def _chapter_label_from_elements(elements: list[Dict[str, Any]]) -> str | None:
    if not elements:
        return None
    marker_info = _extract_numbered_toc_marker(elements[0]["text"])
    if not marker_info or marker_info[1] != "章":
        return None

    label_parts = [marker_info[0]]
    for element in elements[1:3]:
        text = element["text"].strip()
        if not text or NUMBERED_TOC_LABEL_PATTERN.match(text):
            break
        label_parts.append(text)
        break
    return " ".join(label_parts)


def _clean_known_garbled_heading_xhtml(xhtml_text: str) -> tuple[str, str | None]:
    elements = _iter_xhtml_text_elements(xhtml_text)
    if len(elements) < 2 or elements[0]["text"].strip() not in KNOWN_GARBLED_HEADINGS:
        return xhtml_text, None

    label = _chapter_label_from_elements(elements[1:])
    if not label:
        return xhtml_text, None

    first = elements[0]
    updated = xhtml_text[:first["start"]] + xhtml_text[first["end"]:]

    def replace_title(match: re.Match) -> str:
        if _plain_text_from_html(match.group(2)) not in KNOWN_GARBLED_HEADINGS:
            return match.group(0)
        return match.group(1) + html.escape(label) + match.group(3)

    updated = re.sub(
        r"(?is)(<title\b[^>]*>)(.*?)(</title>)",
        replace_title,
        updated,
        count=1,
    )
    return updated, label


def _label_for_epub_file(entries: Dict[str, bytes], filename: str) -> str | None:
    if filename not in entries:
        return None
    elements = _iter_xhtml_text_elements(entries[filename].decode("utf-8", "ignore"))
    return _chapter_label_from_elements(elements)


def _clean_known_garbled_toc_labels(
    entries: Dict[str, bytes],
    file_labels: Dict[str, str],
) -> list[str]:
    changed_files = []
    for name, data in list(entries.items()):
        basename = os.path.basename(name.lower())
        if basename not in {"nav.xhtml", "nav.html", "toc.ncx"}:
            continue

        text = data.decode("utf-8", "ignore")

        def replacement_label(source_name: str, href: str, current_label: str) -> str | None:
            if _plain_text_from_html(current_label) not in KNOWN_GARBLED_HEADINGS:
                return None
            target = _resolve_epub_file_href(source_name, href)
            if not target:
                resolved = _resolve_epub_fragment_href(source_name, href)
                target = resolved[0] if resolved else None
            if not target:
                return None
            return file_labels.get(target) or _label_for_epub_file(entries, target)

        if basename in {"nav.xhtml", "nav.html"}:

            def replace_nav_link(match: re.Match) -> str:
                label = replacement_label(name, match.group(3), match.group(5))
                if not label:
                    return match.group(0)
                return (
                    match.group(1)
                    + match.group(2)
                    + match.group(3)
                    + match.group(2)
                    + match.group(4)
                    + html.escape(label)
                    + match.group(6)
                )

            patched = re.sub(
                r"(?is)(<a\b[^>]*\bhref\s*=\s*)(['\"])(.*?)\2([^>]*>)(.*?)(</a>)",
                replace_nav_link,
                text,
            )
        else:

            def replace_ncx_label(match: re.Match) -> str:
                label = replacement_label(name, match.group(5), match.group(2))
                if not label:
                    return match.group(0)
                return (
                    match.group(1)
                    + html.escape(label)
                    + match.group(3)
                    + match.group(4)
                    + match.group(5)
                    + match.group(4)
                )

            patched = re.sub(
                r"(?is)(<navLabel>\s*<text[^>]*>)(.*?)(</text>\s*</navLabel>\s*"
                r"<content\b[^>]*\bsrc\s*=\s*)(['\"])(.*?)\4",
                replace_ncx_label,
                text,
            )

        if patched != text:
            entries[name] = patched.encode("utf-8")
            changed_files.append(name)
    return changed_files


def _clean_known_garbled_headings(entries: Dict[str, bytes]) -> list[str]:
    changed_files = []
    file_labels: Dict[str, str] = {}
    for name, data in list(entries.items()):
        if not name.lower().endswith((".xhtml", ".html", ".htm")):
            continue
        if os.path.basename(name.lower()) in {"nav.xhtml", "nav.html"}:
            continue

        text = data.decode("utf-8", "ignore")
        patched, label = _clean_known_garbled_heading_xhtml(text)
        if label:
            file_labels[name] = label
        if patched != text:
            entries[name] = patched.encode("utf-8")
            changed_files.append(name)

    changed_files.extend(_clean_known_garbled_toc_labels(entries, file_labels))
    return changed_files


def _remove_in_file_previous_chapter_bibliography_title(xhtml_text: str) -> tuple[str, bool]:
    elements = _iter_xhtml_text_elements(xhtml_text)
    remove_ranges = []
    for index, element in enumerate(elements):
        if not re.match(r"h[1-6]$", element["tag"]):
            continue

        marker_info = _extract_numbered_toc_marker(element["text"])
        if not marker_info or marker_info[1] != "章":
            continue
        if element["text"].strip() == marker_info[0]:
            continue

        lookahead = elements[index + 1:index + 8]
        if not any(_is_chapter_number_text(candidate["text"]) for candidate in lookahead):
            continue

        previous_markup = xhtml_text[max(0, element["start"] - 1200):element["start"]]
        if "footnote" not in previous_markup and "OCR_FOOTNOTES_END" not in previous_markup:
            continue
        remove_ranges.append((element["start"], element["end"]))

    if not remove_ranges:
        return xhtml_text, False

    updated = xhtml_text
    for start, end in reversed(remove_ranges):
        updated = updated[:start] + updated[end:]
    return updated, True


def _remove_in_file_previous_chapter_bibliography_titles(entries: Dict[str, bytes]) -> list[str]:
    changed_files = []
    for name, data in list(entries.items()):
        if not name.lower().endswith((".xhtml", ".html", ".htm")):
            continue
        if os.path.basename(name.lower()) in {"nav.xhtml", "nav.html"}:
            continue
        text = data.decode("utf-8", "ignore")
        patched, changed = _remove_in_file_previous_chapter_bibliography_title(text)
        if changed:
            entries[name] = patched.encode("utf-8")
            changed_files.append(name)
    return changed_files


