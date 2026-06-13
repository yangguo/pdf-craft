"""Restructure EPUBs for WeRead AI reading chapter detection.

The WeRead AI reader appears to mix formal TOC entries, XHTML headings, and
spine document boundaries.  This module narrows all three signals to the same
front-matter/part/chapter list.
"""

from __future__ import annotations

import argparse
import copy
import html
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

XHTML_NS_URI = "http://www.w3.org/1999/xhtml"
EPUB_NS_URI = "http://www.idpf.org/2007/ops"
XHTML_NS = f"{{{XHTML_NS_URI}}}"
OPF_NS = "{http://www.idpf.org/2007/opf}"
NCX_NS = "{http://www.daisy.org/z3986/2005/ncx/}"

ET.register_namespace("", XHTML_NS_URI)
ET.register_namespace("epub", EPUB_NS_URI)

DEFAULT_FRONT_MATTER = {
    "地圖目錄",
    "出版者言",
    "原著者中文版序",
    "郭序",
    "第六版序",
    "第六版序(英文版)",
    "第一版序",
    "第一版序(英文版)",
    "歷代紀元表",
    "貨幣及度量衡折算表",
    "索引",
}

DEFAULT_UNWANTED_TEXTS = {
    "阿美士德使團，1816年",
    "戰爭爆發",
    "新文化運動的展開",
}

REQUIRED_MEMBERS = {
    "mimetype",
    "META-INF/container.xml",
    "EPUB/content.opf",
    "EPUB/nav.xhtml",
    "EPUB/toc.ncx",
}

BOUNDARY_RE = re.compile(r"^第[零〇一二三四五六七八九十百兩0-9]+[編章]\b")


@dataclass(frozen=True)
class ReadingBoundary:
    """A selected front-matter, part, or chapter boundary from the EPUB TOC."""

    title: str
    source_href: str
    fragment: str
    order: int


@dataclass
class ReadingSegment:
    """A generated XHTML spine item corresponding to one reading boundary."""

    title: str
    anchor: str
    file_name: str
    chunks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChapterizeResult:
    """Summary of an EPUB chapterization run."""

    output_path: Path
    segment_count: int
    backup_path: Path | None = None


def _norm(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("，", ",").replace("–", "-").replace("—", "-")
    value = value.replace("「", "").replace("」", "").replace("　", " ")
    value = value.replace("闕", "閥").replace("内", "內")
    return re.sub(r"\s+", "", value)


def _visible_text_from_fragment(value: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value))).strip()


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _direct_children(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [child for child in list(element) if child.tag == XHTML_NS + local_name]


def _walk_nav_ol(ol: ET.Element, depth: int = 1) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []
    for li in _direct_children(ol, "li"):
        anchor = next((child for child in list(li) if child.tag == XHTML_NS + "a"), None)
        if anchor is not None:
            entries.append((depth, _element_text(anchor), anchor.attrib.get("href", "")))
        for nested in _direct_children(li, "ol"):
            entries.extend(_walk_nav_ol(nested, depth + 1))
    return entries


def _parse_nav_entries(payloads: dict[str, bytes]) -> list[tuple[int, str, str]]:
    root = ET.fromstring(payloads["EPUB/nav.xhtml"])
    nav = next(element for element in root.iter() if element.tag == XHTML_NS + "nav")
    ol = next(child for child in list(nav) if child.tag == XHTML_NS + "ol")
    return _walk_nav_ol(ol)


def _is_reading_boundary(title: str, front_matter: set[str]) -> bool:
    stripped = title.strip()
    return stripped in front_matter or BOUNDARY_RE.match(stripped) is not None


def _collect_boundaries(
    payloads: dict[str, bytes],
    front_matter: set[str],
) -> list[ReadingBoundary]:
    boundaries: list[ReadingBoundary] = []
    seen: set[tuple[str, str]] = set()
    for order, (_depth, title, href) in enumerate(_parse_nav_entries(payloads), start=1):
        if not _is_reading_boundary(title, front_matter):
            continue
        source_href, separator, fragment = html.unescape(href).partition("#")
        if not separator or not fragment:
            continue
        key = (source_href, fragment)
        if key in seen:
            continue
        seen.add(key)
        boundaries.append(
            ReadingBoundary(
                title=title,
                source_href=source_href,
                fragment=fragment,
                order=order,
            )
        )
    return boundaries


def _collect_spine_hrefs(payloads: dict[str, bytes]) -> list[str]:
    root = ET.fromstring(payloads["EPUB/content.opf"])
    manifest = {
        item.attrib["id"]: item.attrib["href"]
        for item in root.findall(f".//{OPF_NS}manifest/{OPF_NS}item")
    }
    return [
        manifest[item.attrib["idref"]]
        for item in root.findall(f".//{OPF_NS}spine/{OPF_NS}itemref")
        if item.attrib.get("idref") in manifest
    ]


def _body_children(payloads: dict[str, bytes], href: str) -> list[ET.Element]:
    member = href if href.startswith("EPUB/") else f"EPUB/{href}"
    root = ET.fromstring(payloads[member])
    body = next((element for element in root.iter() if element.tag == XHTML_NS + "body"), None)
    return list(body) if body is not None else []


def _ids_for_element(element: ET.Element) -> set[str]:
    ids: set[str] = set()
    for node in element.iter():
        ident = node.attrib.get("id")
        if ident:
            ids.add(ident)
    return ids


def _serialize_element(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode", method="xml")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_visible_text(element: ET.Element) -> str:
    return _visible_text_from_fragment(_serialize_element(element))


def _should_skip_duplicate_title(element: ET.Element, title: str) -> bool:
    text = _element_visible_text(element)
    if not text or len(text) > 48:
        return False
    title_norm = _norm(title)
    text_norm = _norm(text)
    return bool(text_norm and text_norm in title_norm)


def _canonical_h1(boundary: ReadingBoundary) -> str:
    anchor = html.escape(boundary.fragment, quote=True)
    title = html.escape(boundary.title)
    return f'<h1 id="{anchor}">{title}</h1>'


def _remove_duplicate_ids(element: ET.Element, fragment: str) -> None:
    for node in element.iter():
        if node.attrib.get("id") == fragment:
            del node.attrib["id"]


def _strip_leading_duplicate_titles(element: ET.Element, title: str) -> None:
    for child in list(element):
        if _should_skip_duplicate_title(child, title):
            element.remove(child)
            continue
        break


def _boundary_remainder_chunks(
    element: ET.Element,
    boundary: ReadingBoundary,
) -> list[str]:
    if _should_skip_duplicate_title(element, boundary.title):
        return []
    if _local_name(element.tag) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return []

    clone = copy.deepcopy(element)
    _remove_duplicate_ids(clone, boundary.fragment)
    _strip_leading_duplicate_titles(clone, boundary.title)
    if not _element_visible_text(clone):
        return []
    return [_serialize_element(clone)]


def _safe_file_name(index: int) -> str:
    return f"weread_{index:03d}.xhtml"


def _build_segments(
    payloads: dict[str, bytes],
    boundaries: list[ReadingBoundary],
) -> list[ReadingSegment]:
    by_href: dict[str, list[ReadingBoundary]] = {}
    for boundary in boundaries:
        by_href.setdefault(boundary.source_href, []).append(boundary)

    boundary_index = {boundary: idx for idx, boundary in enumerate(boundaries, start=1)}
    segments: list[ReadingSegment] = []
    pending_prefix: list[str] = []

    for href in _collect_spine_hrefs(payloads):
        if href in {"nav.xhtml", "cover.xhtml"} or not href.endswith(".xhtml"):
            continue
        children = _body_children(payloads, href)
        if not children:
            continue

        child_ids = [_ids_for_element(child) for child in children]
        file_boundaries: list[tuple[int, ReadingBoundary]] = []
        for boundary in by_href.get(href, []):
            matches = [idx for idx, ids in enumerate(child_ids) if boundary.fragment in ids]
            if not matches:
                raise RuntimeError(
                    f"{href}: boundary id not found: {boundary.fragment} {boundary.title}"
                )
            file_boundaries.append((matches[0], boundary))
        file_boundaries.sort(key=lambda item: (item[0], item[1].order))

        for idx in range(1, len(file_boundaries)):
            if file_boundaries[idx][0] == file_boundaries[idx - 1][0]:
                details = ", ".join(
                    f"{boundary.title}@{position}" for position, boundary in file_boundaries
                )
                raise RuntimeError(f"{href}: multiple reading boundaries in one element: {details}")

        if not file_boundaries:
            serialized = [_serialize_element(child) for child in children]
            if segments:
                segments[-1].chunks.extend(serialized)
            else:
                pending_prefix.extend(serialized)
            continue

        first_start = file_boundaries[0][0]
        prefix = [_serialize_element(child) for child in children[:first_start]]
        if prefix:
            if segments:
                segments[-1].chunks.extend(prefix)
            else:
                pending_prefix.extend(prefix)

        for offset, (start, boundary) in enumerate(file_boundaries):
            end = file_boundaries[offset + 1][0] if offset + 1 < len(file_boundaries) else len(children)
            segment_index = boundary_index[boundary]
            segment = ReadingSegment(
                title=boundary.title,
                anchor=boundary.fragment,
                file_name=_safe_file_name(segment_index),
            )
            if pending_prefix:
                segment.chunks.extend(pending_prefix)
                pending_prefix = []
            segment.chunks.append(_canonical_h1(boundary))
            segment.chunks.extend(_boundary_remainder_chunks(children[start], boundary))

            following = children[start + 1:end]
            skip_budget = 4
            for child in following:
                if skip_budget > 0 and _should_skip_duplicate_title(child, boundary.title):
                    skip_budget -= 1
                    continue
                skip_budget = 0
                segment.chunks.append(_serialize_element(child))
            segments.append(segment)

    if pending_prefix:
        if not segments:
            raise RuntimeError("No reading boundaries found; cannot attach leading content")
        segments[0].chunks = pending_prefix + segments[0].chunks

    if len(segments) != len(boundaries):
        raise RuntimeError(
            f"segment count mismatch: segments={len(segments)} boundaries={len(boundaries)}"
        )
    return segments


def _build_xhtml_document(title: str, chunks: list[str]) -> bytes:
    body = "\n".join(chunks)
    document = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f'<html xmlns="{XHTML_NS_URI}" xmlns:epub="{EPUB_NS_URI}" '
        'lang="zh-Hant" xml:lang="zh-Hant">\n'
        "<head>\n"
        f"  <title>{html.escape(title)}</title>\n"
        '  <link href="style/nav.css" rel="stylesheet" type="text/css" />\n'
        "</head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>\n"
    )
    return document.encode("utf-8")


def _build_nav_document(book_title: str, segments: list[ReadingSegment]) -> bytes:
    items = "\n".join(
        f'        <li><a href="{html.escape(segment.file_name)}#{html.escape(segment.anchor)}">'
        f"{html.escape(segment.title)}</a></li>"
        for segment in segments
    )
    document = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<!DOCTYPE html>\n"
        f'<html xmlns="{XHTML_NS_URI}" xmlns:epub="{EPUB_NS_URI}" '
        'lang="zh-Hant" xml:lang="zh-Hant">\n'
        "<head>\n"
        f"  <title>{html.escape(book_title)}</title>\n"
        "</head>\n"
        "<body>\n"
        '  <nav epub:type="toc" id="toc" role="doc-toc">\n'
        f'    <p class="section-title">{html.escape(book_title)}</p>\n'
        "    <ol>\n"
        f"{items}\n"
        "    </ol>\n"
        "  </nav>\n"
        "</body>\n"
        "</html>\n"
    )
    return document.encode("utf-8")


def _build_ncx_document(book_title: str, uid: str, segments: list[ReadingSegment]) -> bytes:
    points = []
    for index, segment in enumerate(segments, start=1):
        points.append(
            f'    <navPoint id="navPoint-{index}" playOrder="{index}">\n'
            "      <navLabel>\n"
            f"        <text>{html.escape(segment.title)}</text>\n"
            "      </navLabel>\n"
            f'      <content src="{html.escape(segment.file_name)}#{html.escape(segment.anchor)}"/>\n'
            "    </navPoint>"
        )
    document = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        "  <head>\n"
        f'    <meta content="{html.escape(uid, quote=True)}" name="dtb:uid"/>\n'
        '    <meta content="1" name="dtb:depth"/>\n'
        '    <meta content="0" name="dtb:totalPageCount"/>\n'
        '    <meta content="0" name="dtb:maxPageNumber"/>\n'
        "  </head>\n"
        "  <docTitle>\n"
        f"    <text>{html.escape(book_title)}</text>\n"
        "  </docTitle>\n"
        "  <navMap>\n"
        + "\n".join(points)
        + "\n  </navMap>\n"
        "</ncx>\n"
    )
    return document.encode("utf-8")


def _opf_text_metadata(opf: bytes) -> tuple[str, str]:
    root = ET.fromstring(opf)
    dc = {"dc": "http://purl.org/dc/elements/1.1/"}
    title = _element_text(root.find(".//dc:title", dc)) or "Untitled"
    uid = ""
    package_uid = root.attrib.get("unique-identifier")
    if package_uid:
        ident = root.find(f".//{{http://purl.org/dc/elements/1.1/}}identifier[@id='{package_uid}']")
        uid = _element_text(ident)
    if not uid:
        ident = root.find(".//dc:identifier", dc)
        uid = _element_text(ident) or title
    return title, uid


def _rewrite_content_opf(opf: bytes, segments: list[ReadingSegment]) -> bytes:
    root = ET.fromstring(opf)
    manifest = root.find(f"{OPF_NS}manifest")
    spine = root.find(f"{OPF_NS}spine")
    if manifest is None or spine is None:
        raise RuntimeError("content.opf is missing manifest or spine")

    nav_id = ""
    ncx_id = spine.attrib.get("toc", "")
    for item in manifest.findall(f"{OPF_NS}item"):
        media_type = item.attrib.get("media-type", "")
        href = item.attrib.get("href", "")
        properties = set(item.attrib.get("properties", "").split())
        if not nav_id and ("nav" in properties or href == "nav.xhtml"):
            nav_id = item.attrib.get("id", "")
        if not ncx_id and media_type == "application/x-dtbncx+xml":
            ncx_id = item.attrib.get("id", "")

    original_nav_in_spine = any(
        itemref.attrib.get("idref") == nav_id
        for itemref in spine.findall(f"{OPF_NS}itemref")
    )

    for item in list(manifest):
        if item.tag != OPF_NS + "item":
            continue
        media_type = item.attrib.get("media-type")
        href = item.attrib.get("href", "")
        properties = set(item.attrib.get("properties", "").split())
        keep = "nav" in properties or href == "nav.xhtml" or href == "cover.xhtml"
        if media_type == "application/xhtml+xml" and not keep:
            manifest.remove(item)

    for index, segment in enumerate(segments, start=1):
        ET.SubElement(
            manifest,
            OPF_NS + "item",
            {
                "href": segment.file_name,
                "id": f"weread_{index:03d}",
                "media-type": "application/xhtml+xml",
            },
        )

    if ncx_id:
        spine.set("toc", ncx_id)
    elif "toc" in spine.attrib:
        del spine.attrib["toc"]

    for itemref in list(spine):
        spine.remove(itemref)
    if nav_id and original_nav_in_spine:
        ET.SubElement(spine, OPF_NS + "itemref", {"idref": nav_id})
    for index, _segment in enumerate(segments, start=1):
        ET.SubElement(spine, OPF_NS + "itemref", {"idref": f"weread_{index:03d}"})

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _is_old_content_xhtml(member: str) -> bool:
    if not member.startswith("EPUB/") or not member.endswith(".xhtml"):
        return False
    return member not in {"EPUB/nav.xhtml", "EPUB/cover.xhtml"}


def _collect_payloads(path: Path) -> tuple[list[zipfile.ZipInfo], dict[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        payloads = {info.filename: archive.read(info.filename) for info in infos}
    return infos, payloads


def _write_rebuilt_epub(
    output_path: Path,
    infos: list[zipfile.ZipInfo],
    payloads: dict[str, bytes],
    replacements: dict[str, bytes],
) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".chapterized-tmp")
    with zipfile.ZipFile(tmp_path, "w") as zout:
        for info in infos:
            if _is_old_content_xhtml(info.filename):
                continue
            data = replacements.pop(info.filename, payloads[info.filename])
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.comment = info.comment
            zi.extra = info.extra
            zi.internal_attr = info.internal_attr
            zi.external_attr = info.external_attr
            zi.create_system = info.create_system
            zi.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
            zout.writestr(zi, data)
        for member, data in sorted(replacements.items()):
            zi = zipfile.ZipInfo(member)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, data)
    tmp_path.replace(output_path)


def chapterize_epub_for_weread(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    backup_path: str | Path | None = None,
    front_matter: set[str] | None = None,
) -> ChapterizeResult:
    """Rewrite an EPUB so formal TOC, spine, and h1 boundaries match.

    If *output_path* is omitted, the input EPUB is patched in place.  When
    patching in place, pass *backup_path* to preserve the pre-patch file.
    """

    source = Path(input_path)
    output = Path(output_path) if output_path is not None else source
    selected_front_matter = set(front_matter or DEFAULT_FRONT_MATTER)

    backup: Path | None = None
    if output == source and backup_path is not None:
        backup = Path(backup_path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)

    infos, payloads = _collect_payloads(source)
    boundaries = _collect_boundaries(payloads, selected_front_matter)
    if not boundaries:
        raise RuntimeError(f"{source}: no reading boundaries selected")
    segments = _build_segments(payloads, boundaries)
    book_title, uid = _opf_text_metadata(payloads["EPUB/content.opf"])

    replacements: dict[str, bytes] = {
        "EPUB/content.opf": _rewrite_content_opf(payloads["EPUB/content.opf"], segments),
        "EPUB/nav.xhtml": _build_nav_document(book_title, segments),
        "EPUB/toc.ncx": _build_ncx_document(book_title, uid, segments),
    }
    for segment in segments:
        replacements[f"EPUB/{segment.file_name}"] = _build_xhtml_document(
            segment.title,
            segment.chunks,
        )

    _write_rebuilt_epub(output, infos, payloads, replacements)
    return ChapterizeResult(output_path=output, segment_count=len(segments), backup_path=backup)


def _collect_nav_entries_from_zip(archive: zipfile.ZipFile) -> list[tuple[int, str, str]]:
    root = ET.fromstring(archive.read("EPUB/nav.xhtml"))
    nav = next(element for element in root.iter() if element.tag == XHTML_NS + "nav")
    ol = next(child for child in list(nav) if child.tag == XHTML_NS + "ol")
    return _walk_nav_ol(ol)


def _collect_ncx_labels(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("EPUB/toc.ncx"))
    labels: list[str] = []
    for point in root.findall(f".//{NCX_NS}navPoint"):
        text = point.find(f"{NCX_NS}navLabel/{NCX_NS}text")
        labels.append(_element_text(text) if text is not None else "")
    return labels


def _collect_spine_from_zip(archive: zipfile.ZipFile) -> tuple[list[str], list[str]]:
    opf = ET.fromstring(archive.read("EPUB/content.opf"))
    manifest = {
        item.attrib["id"]: item.attrib["href"]
        for item in opf.findall(f".//{OPF_NS}manifest/{OPF_NS}item")
    }
    hrefs: list[str] = []
    missing_idrefs: list[str] = []
    for item in opf.findall(f".//{OPF_NS}spine/{OPF_NS}itemref"):
        idref = item.attrib.get("idref", "")
        if idref in manifest:
            hrefs.append(manifest[idref])
        else:
            missing_idrefs.append(idref)
    return hrefs, missing_idrefs


def _collect_h1_texts(archive: zipfile.ZipFile, member: str) -> list[str]:
    data = archive.read(member).decode("utf-8", errors="replace")
    return [
        _visible_text_from_fragment(match.group(1))
        for match in re.finditer(r"<h1\b[^>]*>(.*?)</h1>", data, re.S)
    ]


def verify_weread_chapterized_epub(
    path: str | Path,
    *,
    front_matter: set[str] | None = None,
    unwanted_texts: set[str] | None = None,
) -> list[str]:
    """Return structural issues that can confuse WeRead AI chapter detection."""

    selected_front_matter = set(front_matter or DEFAULT_FRONT_MATTER)
    selected_unwanted = set(unwanted_texts or DEFAULT_UNWANTED_TEXTS)
    issues: list[str] = []

    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            issues.append(f"ZIP corrupt member: {corrupt}")

        infos = archive.infolist()
        if not infos or infos[0].filename != "mimetype":
            issues.append("mimetype is not the first ZIP member")
        mimetype_info = next((info for info in infos if info.filename == "mimetype"), None)
        if mimetype_info is not None and mimetype_info.compress_type != zipfile.ZIP_STORED:
            issues.append("mimetype must be stored without compression")

        names = set(archive.namelist())
        for required in sorted(REQUIRED_MEMBERS):
            if required not in names:
                issues.append(f"missing required member: {required}")
        if not REQUIRED_MEMBERS.issubset(names):
            return issues

        nav_entries = _collect_nav_entries_from_zip(archive)
        ncx_labels = _collect_ncx_labels(archive)
        spine, missing_idrefs = _collect_spine_from_zip(archive)
        for idref in missing_idrefs:
            issues.append(f"spine idref missing from manifest: {idref}")
        content_spine = [
            href
            for href in spine
            if href.endswith(".xhtml") and href not in {"nav.xhtml", "cover.xhtml"}
        ]

        if len(ncx_labels) != len(nav_entries):
            issues.append(f"NCX/nav count mismatch: ncx={len(ncx_labels)} nav={len(nav_entries)}")

        if len(content_spine) != len(nav_entries):
            issues.append(
                f"spine/nav count mismatch: spine_content={len(content_spine)} nav={len(nav_entries)}"
            )

        for index, (depth, text, href) in enumerate(nav_entries, start=1):
            if depth != 1:
                issues.append(f"nav entry {index} is nested at depth {depth}: {text}")
            if not _is_reading_boundary(text, selected_front_matter):
                issues.append(f"nav entry {index} is not a reading boundary: {text}")
            if any(_norm(text) == _norm(bad) for bad in selected_unwanted):
                issues.append(f"unwanted nav entry remains: {text}")
            filename, separator, fragment = href.partition("#")
            if not separator or not fragment:
                issues.append(f"nav entry has no fragment: {text} -> {href}")
                continue
            member = filename if filename.startswith("EPUB/") else f"EPUB/{filename}"
            if member not in names:
                issues.append(f"nav target file missing: {text} -> {href}")
                continue
            data = archive.read(member).decode("utf-8", errors="replace")
            if f'id="{fragment}"' not in data:
                issues.append(f"nav target id missing: {text} -> {href}")

        for label in ncx_labels:
            if not _is_reading_boundary(label, selected_front_matter):
                issues.append(f"NCX label is not a reading boundary: {label}")
            if any(_norm(label) == _norm(bad) for bad in selected_unwanted):
                issues.append(f"unwanted NCX label remains: {label}")

        for href in content_spine:
            member = href if href.startswith("EPUB/") else f"EPUB/{href}"
            if member not in names:
                issues.append(f"spine file missing from zip: {href}")
                continue
            h1s = _collect_h1_texts(archive, member)
            if len(h1s) != 1:
                issues.append(f"{href}: expected exactly one h1, found {len(h1s)}")
            for heading in h1s:
                if not _is_reading_boundary(heading, selected_front_matter):
                    issues.append(f"{href}: h1 is not a reading boundary: {heading}")
                if any(_norm(heading) == _norm(bad) for bad in selected_unwanted):
                    issues.append(f"{href}: unwanted h1 remains: {heading}")

    return issues


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten and re-spine an EPUB for WeRead AI reading chapter detection.",
    )
    parser.add_argument("epub", help="Path to the EPUB to inspect or rewrite")
    parser.add_argument(
        "-o",
        "--output",
        help="Write to this EPUB instead of patching the input in place",
    )
    parser.add_argument(
        "--backup",
        help="Backup path when patching in place",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only validate the EPUB structure; do not rewrite it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    epub_path = Path(args.epub)

    if args.verify_only:
        issues = verify_weread_chapterized_epub(epub_path)
        for issue in issues:
            print(f"- {issue}")
        print(f"TOTAL_ISSUES={len(issues)}")
        return 1 if issues else 0

    result = chapterize_epub_for_weread(
        epub_path,
        Path(args.output) if args.output else None,
        backup_path=Path(args.backup) if args.backup else None,
    )
    issues = verify_weread_chapterized_epub(result.output_path)
    print(f"{result.output_path}: rebuilt {result.segment_count} reading documents")
    if result.backup_path is not None:
        print(f"backup: {result.backup_path}")
    print(f"TOTAL_ISSUES={len(issues)}")
    for issue in issues:
        print(f"- {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
