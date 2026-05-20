"""TOC retargeting, page start marking, and EPUB TOC post-processing."""

import html
import os
import re
import tempfile
import urllib.parse
import zipfile

from typing import Any, Dict

from .config import (
    NUMBERED_TOC_LABEL_PATTERN,
    PART_TOC_KINDS,
    TOC_PAGE_START_CLASS,
    TOC_PAGE_START_CSS,
)
from .epub_href import (
    _add_id_to_text_element,
    _collect_toc_fragment_targets,
    _epub_spine_xhtml_order,
    _extract_numbered_toc_marker,
    _get_start_tag_attr,
    _is_chapter_number_text,
    _iter_xhtml_text_elements,
    _move_part_block_before_previous_chapter_marker,
    _normalize_numbered_marker,
    _plain_text_from_html,
    _relative_epub_href,
    _resolve_epub_file_href,
    _resolve_epub_fragment_href,
    _unique_fragment_id,
)
from .heading_cleanup import (
    _chapter_label_from_elements,
    _clean_known_garbled_headings,
    _heading_preview_matches_next,
    _has_numbered_toc_marker,
    _is_orphaned_chapter_marker_preview,
    _is_orphaned_part_preview,
    _label_for_epub_file,
    _normalize_heading_preview_text,
    _only_closing_markup_after_last_element,
    _remove_in_file_previous_chapter_bibliography_titles,
    _remove_trailing_next_heading_preview,
    _remove_trailing_next_heading_previews,
)


def _is_toc_like_content_file(filename: str) -> bool:
    basename = os.path.basename(filename.lower())
    if basename in {"nav.xhtml", "nav.html", "cover.xhtml"}:
        return True
    return bool(re.match(r"content_\d+\.xhtml$", basename))


def _toc_marker_search_order(entries: Dict[str, bytes], preferred_file: str | None) -> list[str]:
    ordered = []
    if preferred_file and preferred_file in entries and not _is_toc_like_content_file(preferred_file):
        ordered.append(preferred_file)
    for filename in _epub_spine_xhtml_order(entries):
        if filename not in entries or filename in ordered or _is_toc_like_content_file(filename):
            continue
        ordered.append(filename)
    for filename in entries:
        if filename in ordered or _is_toc_like_content_file(filename):
            continue
        if filename.lower().endswith((".xhtml", ".html", ".htm")):
            ordered.append(filename)
    return ordered


def _find_numbered_toc_target(
    entries: Dict[str, bytes],
    preferred_file: str | None,
    marker: str,
    kind: str,
    label_text: str,
) -> tuple[str, str] | None:
    marker_norm = _normalize_numbered_marker(marker)
    title_hint = _normalize_heading_preview_text(label_text[len(marker):])
    best: tuple[int, str, int, Dict[str, Any], str] | None = None

    for filename in _toc_marker_search_order(entries, preferred_file):
        xhtml_text = entries[filename].decode("utf-8", "ignore")
        elements = _iter_xhtml_text_elements(xhtml_text)
        for index, element in enumerate(elements):
            text = element["text"].strip()
            if kind == "章":
                marker_matches = (
                    _normalize_numbered_marker(text) == marker_norm
                    or _normalize_numbered_marker(text).startswith(marker_norm)
                )
            else:
                marker_matches = _normalize_numbered_marker(text).startswith(marker_norm)
            if not marker_matches:
                continue

            context = _normalize_heading_preview_text(
                "".join(candidate["text"] for candidate in elements[index:index + 3])
            )
            score = 0
            if filename == preferred_file:
                score += 2
            if re.match(r"h[1-6]$", element["tag"]):
                score += 2
            if text == marker:
                score += 2
            if title_hint and title_hint in context:
                score += 6
            candidate = (score, filename, index, element, xhtml_text)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None

    _score, filename, _index, element, xhtml_text = best
    fragment = element.get("id") or element.get("name")
    if fragment:
        return filename, fragment

    new_id = _unique_fragment_id(xhtml_text, "toc-auto", "start")
    entries[filename] = _add_id_to_text_element(xhtml_text, element, new_id).encode("utf-8")
    return filename, new_id


def _retarget_numbered_toc_fragment(
    xhtml_text: str,
    target_fragment: str,
    marker: str,
    kind: str,
    label_text: str,
) -> tuple[str, str]:
    elements = _iter_xhtml_text_elements(xhtml_text)
    target_index = next(
        (
            index
            for index, element in enumerate(elements)
            if target_fragment in {element.get("id"), element.get("name")}
        ),
        None,
    )
    if target_index is None:
        return xhtml_text, target_fragment

    target = elements[target_index]
    if _normalize_numbered_marker(target["text"]).startswith(
        _normalize_numbered_marker(marker)
    ):
        if kind in PART_TOC_KINDS:
            return (
                _move_part_block_before_previous_chapter_marker(
                    xhtml_text,
                    elements,
                    target_index,
                ),
                target_fragment,
            )
        return xhtml_text, target_fragment

    normalized_marker = _normalize_numbered_marker(marker)
    for element in reversed(elements[max(0, target_index - 8):target_index]):
        if _normalize_numbered_marker(element["text"]) != normalized_marker:
            continue

        existing_id = element.get("id") or element.get("name")
        if existing_id:
            return xhtml_text, existing_id

        new_id = _unique_fragment_id(xhtml_text, target_fragment, "start")
        return _add_id_to_text_element(xhtml_text, element, new_id), new_id

    if kind in PART_TOC_KINDS:
        new_id = _unique_fragment_id(xhtml_text, target_fragment, "part")
        remainder = label_text[len(marker):].strip()
        remainder = re.sub(r"^[\s:：,，、]+", "", remainder)
        inserted = f'<p id="{new_id}">{html.escape(marker)}</p>'
        if remainder:
            inserted += f"<p>{html.escape(remainder)}</p>"
        return xhtml_text[:target["start"]] + inserted + xhtml_text[target["start"]:], new_id

    if kind != "章":
        return xhtml_text, target_fragment

    tag = target["tag"] if re.match(r"h[1-6]$", target["tag"]) else "h2"
    new_id = _unique_fragment_id(xhtml_text, target_fragment, "chapter")
    inserted = f'<{tag} id="{new_id}">{html.escape(marker)}</{tag}>'
    return xhtml_text[:target["start"]] + inserted + xhtml_text[target["start"]:], new_id


def _replace_fragment_in_href(href: str, fragment: str) -> str:
    file_part = href.partition("#")[0]
    escaped_fragment = urllib.parse.quote(fragment, safe="A-Za-z0-9_.:-")
    return f"{file_part}#{escaped_fragment}"


def _retarget_numbered_toc_hrefs(entries: Dict[str, bytes]) -> int:
    changed_count = 0

    def retarget_href(source_name: str, href: str, label: str) -> str:
        nonlocal changed_count
        label_text = _plain_text_from_html(label)
        marker_info = _extract_numbered_toc_marker(label_text)
        if not marker_info:
            return href

        marker, kind = marker_info
        resolved = _resolve_epub_fragment_href(source_name, href)
        if not resolved:
            preferred_file = _resolve_epub_file_href(source_name, href)
            found = _find_numbered_toc_target(
                entries,
                preferred_file,
                marker,
                kind,
                label_text,
            )
            if not found:
                return href
            changed_count += 1
            target_file, target_fragment = found
            return _relative_epub_href(source_name, target_file, target_fragment)

        target_file, target_fragment = resolved
        if target_file not in entries:
            return href

        xhtml_text = entries[target_file].decode("utf-8", "ignore")
        updated_text, updated_fragment = _retarget_numbered_toc_fragment(
            xhtml_text,
            target_fragment,
            marker,
            kind,
            label_text,
        )
        content_changed = updated_text != xhtml_text
        if updated_text != xhtml_text:
            entries[target_file] = updated_text.encode("utf-8")
        if updated_fragment == target_fragment:
            if content_changed:
                changed_count += 1
            return href

        changed_count += 1
        return _replace_fragment_in_href(href, updated_fragment)

    for name, data in list(entries.items()):
        lower_name = name.lower()
        basename = os.path.basename(lower_name)
        if basename in {"nav.xhtml", "nav.html"}:
            text = data.decode("utf-8", "ignore")

            def replace_nav_link(match: re.Match) -> str:
                href = match.group(3)
                label = _plain_text_from_html(match.group(5))
                new_href = retarget_href(name, href, label)
                return (
                    match.group(1)
                    + match.group(2)
                    + html.escape(new_href, quote=True)
                    + match.group(2)
                    + match.group(4)
                    + match.group(5)
                    + match.group(6)
                )

            patched = re.sub(
                r"(?is)(<a\b[^>]*\bhref\s*=\s*)(['\"])(.*?)\2([^>]*>)(.*?)(</a>)",
                replace_nav_link,
                text,
            )
            if patched != text:
                entries[name] = patched.encode("utf-8")
        elif basename == "toc.ncx":
            text = data.decode("utf-8", "ignore")

            def replace_ncx_content(match: re.Match) -> str:
                label = _plain_text_from_html(match.group(2))
                href = match.group(5)
                new_href = retarget_href(name, href, label)
                return (
                    match.group(1)
                    + match.group(2)
                    + match.group(3)
                    + match.group(4)
                    + html.escape(new_href, quote=True)
                    + match.group(4)
                )

            patched = re.sub(
                r"(?is)(<navLabel>\s*<text[^>]*>)(.*?)(</text>\s*</navLabel>\s*"
                r"<content\b[^>]*\bsrc\s*=\s*)(['\"])(.*?)\4",
                replace_ncx_content,
                text,
            )
            if patched != text:
                entries[name] = patched.encode("utf-8")

    return changed_count


def _add_class_to_start_tag(start_tag: str, class_name: str) -> str:
    class_match = re.search(r"(?is)\bclass\s*=\s*(['\"])(.*?)\1", start_tag)
    if not class_match:
        return f'{start_tag} class="{class_name}"'

    classes = class_match.group(2).split()
    if class_name in classes:
        return start_tag

    classes.append(class_name)
    return (
        start_tag[:class_match.start(2)]
        + " ".join(classes)
        + start_tag[class_match.end(2):]
    )


def _mark_toc_targets_as_page_starts(xhtml_text: str, target_ids: set[str]) -> str:
    """Add a page-start class to elements used as TOC fragment targets."""
    updated = xhtml_text
    for target_id in sorted(target_ids, key=len, reverse=True):
        pattern = re.compile(
            r"(<[A-Za-z][\w:.-]*\b"
            r"(?=[^>]*\b(?:id|name)\s*=\s*['\"]"
            + re.escape(target_id)
            + r"['\"])"
            r"[^>]*?)(\s*/?>)",
            re.IGNORECASE | re.DOTALL,
        )
        updated = pattern.sub(
            lambda match: _add_class_to_start_tag(match.group(1), TOC_PAGE_START_CLASS)
            + match.group(2),
            updated,
            count=1,
        )
    return updated


def _remove_toc_page_start_classes(xhtml_text: str) -> str:
    def replace_class_attr(match: re.Match) -> str:
        classes = [
            class_name
            for class_name in match.group(2).split()
            if class_name != TOC_PAGE_START_CLASS
        ]
        if not classes:
            return ""
        return f' class={match.group(1)}{" ".join(classes)}{match.group(1)}'

    return re.sub(
        r"(?is)\s+class\s*=\s*(['\"])(.*?)\1",
        replace_class_attr,
        xhtml_text,
    )


def _append_toc_page_start_css(css_text: str) -> str:
    if f".{TOC_PAGE_START_CLASS}" in css_text:
        return css_text
    return css_text.rstrip() + "\n\n" + TOC_PAGE_START_CSS.strip() + "\n"


def ensure_toc_targets_start_pages(epub_path: str) -> Dict[str, Any]:
    """Patch an EPUB so TOC fragment targets begin on a fresh reader page."""
    epub_path = os.fspath(epub_path)
    with zipfile.ZipFile(epub_path, "r") as archive:
        original_entries = [(info, archive.read(info.filename)) for info in archive.infolist()]

    entry_map = {info.filename: data for info, data in original_entries}
    preclean_files = []
    preclean_files.extend(_clean_known_garbled_headings(entry_map))
    preclean_files.extend(_remove_in_file_previous_chapter_bibliography_titles(entry_map))
    retargeted_links = _retarget_numbered_toc_hrefs(entry_map)
    toc_targets = _collect_toc_fragment_targets(entry_map)

    changed = bool(preclean_files or retargeted_links)
    xhtml_files = list(dict.fromkeys(preclean_files))
    css_files = []
    patched_entries = []
    for info, _data in original_entries:
        data = entry_map.get(info.filename, _data)
        lower_name = info.filename.lower()
        new_data = data

        if toc_targets and lower_name.endswith((".xhtml", ".html", ".htm")):
            text = data.decode("utf-8", "ignore")
            text = _remove_toc_page_start_classes(text)
            patched = text
            if info.filename in toc_targets:
                patched = _mark_toc_targets_as_page_starts(
                    text,
                    toc_targets[info.filename],
                )
            if patched.encode("utf-8") != data:
                changed = True
                xhtml_files.append(info.filename)
                new_data = patched.encode("utf-8")

        if toc_targets and lower_name.endswith(".css"):
            text = new_data.decode("utf-8", "ignore")
            patched = _append_toc_page_start_css(text)
            if patched != text:
                changed = True
                css_files.append(info.filename)
                new_data = patched.encode("utf-8")

        patched_entries.append((info, new_data))

    patched_entry_map = {info.filename: data for info, data in patched_entries}
    preview_files = _remove_trailing_next_heading_previews(patched_entry_map)
    if preview_files:
        changed = True
        for filename in preview_files:
            if filename not in xhtml_files:
                xhtml_files.append(filename)
        patched_entries = [
            (info, patched_entry_map.get(info.filename, data)) for info, data in patched_entries
        ]

    if not changed:
        return {
            "targets": sum(len(ids) for ids in toc_targets.values()),
            "retargeted_links": retargeted_links,
            "xhtml_files": xhtml_files,
            "css_files": css_files,
        }

    output_dir = os.path.dirname(os.path.abspath(epub_path))
    output_name = os.path.basename(epub_path)
    fd, temp_output = tempfile.mkstemp(
        prefix=f".{output_name}.",
        suffix=".epub",
        dir=output_dir,
    )
    os.close(fd)

    try:
        with zipfile.ZipFile(temp_output, "w") as archive:
            mimetype_entries = [
                (info, data) for info, data in patched_entries if info.filename == "mimetype"
            ]
            other_entries = [
                (info, data) for info, data in patched_entries if info.filename != "mimetype"
            ]
            for info, data in mimetype_entries + other_entries:
                if info.filename == "mimetype":
                    info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, data)
        os.replace(temp_output, epub_path)
    except Exception:
        try:
            os.remove(temp_output)
        except FileNotFoundError:
            pass
        raise

    return {
        "targets": sum(len(ids) for ids in toc_targets.values()),
        "retargeted_links": retargeted_links,
        "xhtml_files": xhtml_files,
        "css_files": css_files,
    }


