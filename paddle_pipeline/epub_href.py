"""EPUB href/fragment resolution and XHTML text element parsing."""

import html
import os
import posixpath
import re
import urllib.parse

from typing import Any, Dict, List

from .config import NUMBERED_TOC_LABEL_PATTERN, XHTML_TEXT_ELEMENT_PATTERN


def _resolve_epub_fragment_href(base_name: str, href: str) -> tuple[str, str] | None:
    """Resolve a TOC href to an EPUB member and fragment id."""
    href = html.unescape(href.strip())
    if not href or re.match(r"(?i)^[a-z][a-z0-9+.-]*:", href):
        return None

    file_part, separator, fragment = href.partition("#")
    if not separator or not fragment:
        return None

    fragment = urllib.parse.unquote(fragment)
    if not file_part:
        return base_name.lstrip("/"), fragment

    target = urllib.parse.unquote(file_part)
    target = posixpath.normpath(posixpath.join(posixpath.dirname(base_name), target))
    return target.lstrip("/"), fragment


def _resolve_epub_file_href(base_name: str, href: str) -> str | None:
    """Resolve a TOC href to an EPUB member, allowing hrefs without fragments."""
    href = html.unescape(href.strip())
    if not href or href.startswith("#") or re.match(r"(?i)^[a-z][a-z0-9+.-]*:", href):
        return None

    file_part = href.partition("#")[0]
    if not file_part:
        return None

    target = urllib.parse.unquote(file_part)
    target = posixpath.normpath(posixpath.join(posixpath.dirname(base_name), target))
    return target.lstrip("/")


def _relative_epub_href(source_name: str, target_name: str, fragment: str) -> str:
    relative = posixpath.relpath(target_name, posixpath.dirname(source_name) or ".")
    escaped_fragment = urllib.parse.quote(fragment, safe="A-Za-z0-9_.:-")
    return f"{relative}#{escaped_fragment}"


def _collect_toc_fragment_targets(entries: Dict[str, bytes]) -> Dict[str, set[str]]:
    """Collect file/id targets referenced by EPUB TOC documents."""
    targets: Dict[str, set[str]] = {}
    for name, data in entries.items():
        lower_name = name.lower()
        if os.path.basename(lower_name) not in {"nav.xhtml", "nav.html", "toc.ncx"}:
            continue

        text = data.decode("utf-8", "ignore")
        for match in re.finditer(r"(?is)\b(?:href|src)\s*=\s*(['\"])(.*?)\1", text):
            resolved = _resolve_epub_fragment_href(name, match.group(2))
            if not resolved:
                continue
            target_file, fragment = resolved
            targets.setdefault(target_file, set()).add(fragment)
    return targets


def _plain_text_from_html(value: str) -> str:
    text = html.unescape(re.sub(r"(?is)<[^>]+>", "", value))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_numbered_marker(value: str) -> str:
    return value.replace("编", "編").strip()


def _extract_numbered_toc_marker(label: str) -> tuple[str, str] | None:
    label_text = _plain_text_from_html(label)
    match = NUMBERED_TOC_LABEL_PATTERN.match(label_text)
    if not match:
        return None
    return match.group(1), match.group(2)


def _get_start_tag_attr(start_tag: str, attr_name: str) -> str | None:
    match = re.search(
        rf"(?is)\b{re.escape(attr_name)}\s*=\s*(['\"])(.*?)\1",
        start_tag,
    )
    if not match:
        return None
    return html.unescape(match.group(2))


def _existing_fragment_ids(xhtml_text: str) -> set[str]:
    return {
        html.unescape(match.group(2))
        for match in re.finditer(
            r"(?is)\b(?:id|name)\s*=\s*(['\"])(.*?)\1",
            xhtml_text,
        )
    }


def _unique_fragment_id(xhtml_text: str, base: str, suffix: str) -> str:
    safe_base = re.sub(r"[^A-Za-z0-9_.:-]+", "-", base).strip("-") or "toc-auto"
    candidate = f"{safe_base}-{suffix}"
    existing = _existing_fragment_ids(xhtml_text)
    if candidate not in existing:
        return candidate

    index = 2
    while f"{candidate}-{index}" in existing:
        index += 1
    return f"{candidate}-{index}"


def _add_id_to_text_element(xhtml_text: str, element: Dict[str, Any], new_id: str) -> str:
    start_tag = element["start_tag"]
    if _get_start_tag_attr(start_tag, "id") or _get_start_tag_attr(start_tag, "name"):
        return xhtml_text

    insert_at = element["start"] + len(start_tag) - 1
    return (
        xhtml_text[:insert_at]
        + f' id="{html.escape(new_id, quote=True)}"'
        + xhtml_text[insert_at:]
    )


def _iter_xhtml_text_elements(xhtml_text: str) -> list[Dict[str, Any]]:
    elements = []
    for match in XHTML_TEXT_ELEMENT_PATTERN.finditer(xhtml_text):
        start_tag = match.group(1)
        elements.append({
            "start": match.start(),
            "end": match.end(),
            "start_tag": start_tag,
            "tag": match.group("tag").lower(),
            "text": _plain_text_from_html(match.group("body")),
            "id": _get_start_tag_attr(start_tag, "id"),
            "name": _get_start_tag_attr(start_tag, "name"),
        })
    return elements


def _is_chapter_number_text(value: str) -> bool:
    match = NUMBERED_TOC_LABEL_PATTERN.match(value.strip())
    return bool(match and match.group(2) == "章" and value.strip() == match.group(1))


def _move_part_block_before_previous_chapter_marker(
    xhtml_text: str,
    elements: list[Dict[str, Any]],
    target_index: int,
) -> str:
    if target_index <= 0:
        return xhtml_text

    previous = elements[target_index - 1]
    if not _is_chapter_number_text(previous["text"]):
        return xhtml_text

    block_start = elements[target_index]["start"]
    block_end = elements[target_index]["end"]
    for element in elements[target_index + 1:target_index + 4]:
        if element["tag"] != "p" or element.get("id") or element.get("name"):
            break
        if NUMBERED_TOC_LABEL_PATTERN.match(element["text"]):
            break
        block_end = element["end"]

    return (
        xhtml_text[:previous["start"]]
        + xhtml_text[block_start:block_end]
        + xhtml_text[previous["start"]:block_start]
        + xhtml_text[block_end:]
    )


def _resolve_opf_href(opf_name: str, href: str) -> str:
    href = urllib.parse.unquote(html.unescape(href)).partition("#")[0]
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(opf_name), href)
    ).lstrip("/")


def _epub_spine_xhtml_order(entries: Dict[str, bytes]) -> list[str]:
    opf_name = next((name for name in entries if name.lower().endswith(".opf")), None)
    if not opf_name:
        return []

    text = entries[opf_name].decode("utf-8", "ignore")
    manifest: Dict[str, str] = {}
    for match in re.finditer(r"(?is)<item\b[^>]*>", text):
        item_tag = match.group(0)
        item_id = _get_start_tag_attr(item_tag, "id")
        href = _get_start_tag_attr(item_tag, "href")
        media_type = (_get_start_tag_attr(item_tag, "media-type") or "").lower()
        if not item_id or not href:
            continue
        resolved = _resolve_opf_href(opf_name, href)
        if media_type == "application/xhtml+xml" or resolved.lower().endswith(
            (".xhtml", ".html", ".htm")
        ):
            manifest[item_id] = resolved

    spine = []
    for match in re.finditer(r"(?is)<itemref\b[^>]*>", text):
        idref = _get_start_tag_attr(match.group(0), "idref")
        if idref and idref in manifest and manifest[idref] in entries:
            spine.append(manifest[idref])
    return spine


