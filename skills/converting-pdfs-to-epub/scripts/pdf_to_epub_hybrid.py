#!/usr/bin/env python3
"""Inspect, build, and verify hybrid text+visual EPUBs from PDFs."""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import re
import statistics
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

try:
    import fitz
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyMuPDF is required: python3 -m pip install pymupdf") from exc

try:
    from ebooklib import epub
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("EbookLib is required: python3 -m pip install ebooklib") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Pillow is required: python3 -m pip install pillow") from exc


LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")

CSS = """
body {
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.45;
  margin: 0 5%;
  color: #111;
}
h1 {
  font-size: 1.45em;
  line-height: 1.2;
  margin: 1.5em 0 1em;
}
section.pdf-page {
  margin: 0 0 1.8em;
}
p {
  margin: 0 0 0.8em;
  text-indent: 1.15em;
}
p.noindent, p.index-entry {
  text-indent: 0;
}
p.index-entry {
  margin-bottom: 0.35em;
  font-size: 0.95em;
}
figure.visual-snapshot {
  margin: 1.1em 0 1.5em;
  text-align: center;
  page-break-inside: avoid;
}
figure.visual-snapshot img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
}
figure.visual-snapshot figcaption {
  font-size: 0.85em;
  color: #555;
  margin-top: 0.35em;
  text-indent: 0;
}
section.endnotes {
  margin-top: 2.5em;
  border-top: 1px solid #999;
  page-break-before: always;
}
section.endnotes h2 {
  font-size: 1.2em;
  margin: 1.2em 0 0.8em;
}
section.endnotes ol {
  padding-left: 1.6em;
}
section.endnotes li {
  margin: 0 0 0.65em;
  text-indent: 0;
}
"""


@dataclass(frozen=True)
class Section:
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class CaptionedVisual:
    """A figure or table region paired with its immediately following title."""

    rect: fitz.Rect
    caption_rect: fitz.Rect
    label: str
    caption: str


@dataclass
class VisualAsset:
    item: epub.EpubImage
    caption: str | None = None
    caption_rect: fitz.Rect | None = None
    source_rect: fitz.Rect | None = None


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = "".join(character for character in text if character >= " " or character in "\n\t")
    for source, target in LIGATURES.items():
        text = text.replace(source, target)
    return unicodedata.normalize("NFC", text)


def parse_page_ranges(raw: str | None) -> set[int]:
    pages: set[int] = set()
    if not raw:
        return pages
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def parse_sections(raw_sections: list[str], page_count: int) -> list[Section]:
    sections: list[Section] = []
    for raw in raw_sections:
        if ":" not in raw or "-" not in raw.rsplit(":", 1)[1]:
            raise ValueError(f"section must be 'Title:start-end': {raw}")
        title, raw_range = raw.rsplit(":", 1)
        start_s, end_s = raw_range.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start < 1 or end > page_count or end < start:
            raise ValueError(f"section range out of bounds: {raw}")
        sections.append(Section(title.strip(), start, end))
    return sections


def default_sections(doc: fitz.Document) -> list[Section]:
    toc = doc.get_toc(simple=True)
    # Duplicate bookmark starts must not create empty ranges; bookmarks may
    # also omit the cover, title page, and other front matter.
    starts: dict[int, str] = {}
    for level, title, page in toc:
        if level == 1 and 1 <= page <= doc.page_count:
            starts.setdefault(page, title)
    level_one = [(title, page) for page, title in sorted(starts.items())]
    if level_one:
        sections = []
        if level_one[0][1] > 1:
            sections.append(Section("Front Matter", 1, level_one[0][1] - 1))
        for index, (title, start) in enumerate(level_one):
            next_start = level_one[index + 1][1] if index + 1 < len(level_one) else doc.page_count + 1
            sections.append(Section(normalize_text(title), start, next_start - 1))
        return sections

    chunk_size = 50
    sections = []
    for start in range(1, doc.page_count + 1, chunk_size):
        end = min(start + chunk_size - 1, doc.page_count)
        sections.append(Section(f"Pages {start}-{end}", start, end))
    return sections


def safe_filename(index: int, title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:48] or f"section_{index:02d}"
    return f"{index:02d}_{slug}.xhtml"


def clean_lines(text: str) -> list[str]:
    return [line.strip() for line in normalize_text(text).splitlines() if line.strip()]


def join_pdf_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") and not out.endswith(("–", "—")):
            out = out[:-1] + line
        elif re.match(r"^[,.;:?!)]", line):
            out += line
        else:
            out += " " + line
    return re.sub(r"\s+", " ", out).strip()


def _join_reflowed_text(left: str, right: str) -> str:
    if not left:
        return right
    if left.endswith("-") and re.search(r"[A-Za-z]-$", left) and re.match(r"^[A-Za-z]", right):
        return left[:-1] + right
    if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right):
        return left + " " + right
    return left + right


def reflow_text_blocks(blocks: list[tuple]) -> list[str]:
    raw = [(block, clean_lines(block[4])) for block in blocks]
    raw = [(block, lines) for block, lines in raw if lines]
    combined = "".join("".join(lines) for _, lines in raw)
    compact = re.sub(r"\s+", "", combined)
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", compact))
    if not compact or cjk_count / len(compact) < 0.2:
        return [join_pdf_lines(lines) for _, lines in raw]

    prepared = []
    for block, lines in raw:
        text = ""
        for line in lines:
            text = _join_reflowed_text(text, line)
        first_line = block[4].splitlines()[0] if block[4].splitlines() else ""
        explicit_indent = len(first_line) - len(first_line.lstrip(" \t\u3000\u2002\u2003")) >= 2
        prepared.append((block, text, explicit_indent))

    base_x = min(float(block[0]) for block, _, _ in prepared)
    paragraphs: list[str] = []
    current = ""
    previous_bottom: float | None = None
    previous_x: float | None = None
    for block, text, explicit_indent in prepared:
        gap = 0.0 if previous_bottom is None else float(block[1]) - previous_bottom
        x = float(block[0])
        # Some PDFs expose every line in an inset paragraph as its own text
        # block. Treat only the transition into an inset line as indentation;
        # otherwise each following line would become a separate paragraph.
        geometric_indent = (
            x >= base_x + 12
            and previous_x is not None
            and previous_x < base_x + 12
            and previous_bottom is not None
            and float(block[1]) >= previous_bottom
        )
        indented = explicit_indent or geometric_indent
        if current and (indented or gap > 8):
            paragraphs.append(current)
            current = ""
        current = _join_reflowed_text(current, text)
        previous_bottom = float(block[3])
        previous_x = x
    if current:
        paragraphs.append(current)
    return paragraphs


XHTML_NS = "http://www.w3.org/1999/xhtml"


def page_continues_paragraph(
    previous: fitz.Page, current: fitz.Page, *, drop_headers: bool
) -> bool:
    """Conservatively recognize a CJK paragraph continuing over a page turn.

    Require a full-width unfinished last line near the bottom, a same-size
    unindented first line near the top, and matching page dimensions. This is
    deliberately inconclusive for columns, poetry, and short final lines.
    """
    if abs(previous.rect.width - current.rect.width) > 1:
        return False
    before = _page_text_records(previous, 0, drop_headers)
    after = _page_text_records(current, 0, drop_headers)
    if not before or not after:
        return False
    last, last_size = before[-1]
    first, first_size = after[0]
    body_size = statistics.median(size for _, size in before)
    if abs(last_size - body_size) > 0.5 or abs(first_size - body_size) > 0.5:
        return False
    base_x = min(block[0] for block, size in before if abs(size - body_size) <= 0.5)
    right_x = max(block[2] for block, size in before if abs(size - body_size) <= 0.5)
    # Block bounds include earlier lines; consult the last line itself.
    source = previous.get_text("dict")["blocks"][last[5]]
    last_line = source["lines"][-1]["bbox"]
    if (
        last_line[3] < previous.rect.height * 0.8
        or first[1] > current.rect.height * 0.2
        or abs(first[0] - base_x) > 3
        or abs(last_line[2] - right_x) > 6
        or right_x - base_x < previous.rect.width * 0.3
    ):
        return False
    first_raw = first[4].splitlines()[0]
    if len(first_raw) - len(first_raw.lstrip(" \t\u3000\u2002\u2003")) >= 2:
        return False
    left_text = normalize_text(last[4]).rstrip().rstrip('”’"」』）》】）]')
    if not left_text or left_text[-1] in "。！？!?…；;.:：":
        return False
    text = re.sub(r"\s+", "", last[4] + first[4])
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    return bool(text) and cjk / len(text) >= 0.2


def flow_page_sections(root: ElementTree.Element, *, continuation_pages: set[int]) -> None:
    """Flatten generated page wrappers without losing inline markup or IDs.

    Only the first paragraph of a marked page may join the immediately
    preceding paragraph. Figures, headings, and other elements are barriers.
    """
    tag = lambda local: f"{{{XHTML_NS}}}{local}"
    body = root.find(tag("body"))
    if body is None:
        return
    original = list(body)
    for child in original:
        body.remove(child)
    for child in original:
        if child.tag != tag("section") or child.get("class") != "pdf-page":
            body.append(child)
            continue
        page_id = child.get("id", "")
        match = re.fullmatch(r"p(\d+)", page_id)
        anchor = ElementTree.Element(tag("span"), {"class": "source-page-anchor"})
        if page_id:
            anchor.set("id", page_id)
        contents = list(child)
        previous = body[-1] if len(body) else None
        join = (
            match is not None and int(match.group(1)) in continuation_pages
            and previous is not None and previous.tag == tag("p")
            and bool(contents) and contents[0].tag == tag("p")
        )
        if join:
            paragraph = contents.pop(0)
            previous.append(anchor)
            if paragraph.get("id"):
                paragraph_anchor = ElementTree.SubElement(previous, tag("span"), {"id": paragraph.get("id")})
            else:
                paragraph_anchor = anchor
            left = "".join(previous.itertext()).rstrip()
            right = paragraph.text or ""
            separator = " " if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right) else ""
            paragraph_anchor.tail = separator + right
            for inline in list(paragraph):
                previous.append(inline)
        else:
            body.append(anchor)
        for content in contents:
            body.append(content)


def is_header_or_footer(block: tuple, drop_headers: bool, page_height: float = 792.0) -> bool:
    if not drop_headers:
        return False
    _, y0, _, _, text = block[:5]
    compact = " ".join(clean_lines(text))
    if not compact:
        return True
    header_threshold = page_height * 0.065
    footer_threshold = page_height * 0.9
    return y0 < header_threshold or y0 > footer_threshold


def is_layout_marker_span(span: dict) -> bool:
    """Ignore InDesign emphasis dots misencoded as the digit ``4``."""

    font = str(span.get("font", "")).casefold()
    return "kenten" in font and bool(re.fullmatch(r"4+", span.get("text", "").strip()))


def _page_text_records(
    page: fitz.Page,
    page_no: int,
    drop_headers: bool,
    *,
    excluded_rects: list[fitz.Rect] | None = None,
    column_split: float | None = None,
) -> list[tuple[tuple, float]]:
    records = []
    for index, source_block in enumerate(page.get_text("dict")["blocks"]):
        if source_block["type"] != 0:
            continue
        spans = [
            span
            for line in source_block["lines"]
            for span in line["spans"]
            if span["text"].strip() and not is_layout_marker_span(span)
        ]
        if not spans:
            continue
        text = "\n".join(
            "".join(
                span["text"]
                for span in line["spans"]
                if not is_layout_marker_span(span)
            )
            for line in source_block["lines"]
        )
        block = (*source_block["bbox"], text, index, 0)
        records.append((block, statistics.median(span["size"] for span in spans)))

    if column_split is not None:
        records = sorted(records, key=lambda record: (record[0][0] >= column_split, record[0][1], record[0][0]))
    else:
        records = sorted(records, key=lambda record: (round(record[0][1], 1), round(record[0][0], 1)))

    text_records = []
    page_height = page.rect.height
    for block, font_size in records:
        if is_header_or_footer(block, drop_headers, page_height):
            continue
        rect = fitz.Rect(block[:4])
        if excluded_rects and any(rect.intersects(excluded) for excluded in excluded_rects):
            continue
        text_records.append((block, font_size))
    return text_records


def is_readable_footnote(text: str) -> bool:
    """Reject hidden OCR/layout debris that masquerades as a small footnote."""

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    # Digits are legitimate prose/citations. Known misencoded emphasis glyphs
    # are removed by font in _page_text_records, never by numeric pattern.
    corrupted = compact.count("\ufffd")
    return corrupted / len(compact) < 0.08


def page_body_and_footnotes(
    page: fitz.Page,
    page_no: int,
    drop_headers: bool,
    *,
    excluded_rects: list[fitz.Rect] | None = None,
    column_split: float | None = None,
) -> tuple[list[str], list[str]]:
    """Return reflowed body text and smaller, bottom-of-page source notes."""

    records = _page_text_records(
        page, page_no, drop_headers, excluded_rects=excluded_rects, column_split=column_split
    )
    page_height = page.rect.height
    body_sizes = [
        font_size
        for block, font_size in records
        if page_height * 0.1 <= block[1] < page_height * 0.72 and len(block[4].strip()) >= 20
    ]
    if not body_sizes:
        body_sizes = [
            font_size
            for block, font_size in records
            if page_height * 0.1 <= block[1] < page_height * 0.72
        ]
    body_size = statistics.median(body_sizes) if body_sizes else None
    note_records = []
    body_records = []
    for block, font_size in records:
        is_footnote = (
            body_size is not None
            and page_height * 0.72 <= block[1] < page_height * 0.9
            and font_size <= body_size - 1.0
        )
        if is_footnote:
            if is_readable_footnote(block[4]):
                note_records.append(block)
        else:
            body_records.append(block)
    return reflow_text_blocks(body_records), reflow_text_blocks(note_records)


def page_text_blocks(
    page: fitz.Page,
    page_no: int,
    drop_headers: bool,
    *,
    excluded_rects: list[fitz.Rect] | None = None,
    column_split: float | None = None,
) -> list[str]:
    body, _ = page_body_and_footnotes(
        page,
        page_no,
        drop_headers,
        excluded_rects=excluded_rects,
        column_split=column_split,
    )
    return body


def remove_duplicate_section_heading(
    blocks: list[str], title: str, *, is_section_start: bool
) -> list[str]:
    if not is_section_start or not blocks:
        return blocks
    compact = lambda value: re.sub(r"\s+", "", normalize_text(value)).casefold()
    if compact(blocks[0]) == compact(title):
        return blocks[1:]
    return blocks


def paragraph_html(text: str, *, first: bool, index_page: bool) -> str:
    if index_page:
        cls = ' class="index-entry"'
    elif first or len(text) < 90 or re.match(r"^(Table|Figure)\s+\d", text):
        cls = ' class="noindent"'
    else:
        cls = ""
    return f"<p{cls}>{escape(text)}</p>"


def _substantive_image_regions(page: fitz.Page, *, min_side: float = 30.0) -> list[fitz.Rect]:
    """Return image placements large enough to be actual page content.

    This intentionally ignores the tiny repeated raster marks used by the
    source PDF for registration/crop furniture.  Those marks were the reason
    a page containing normal prose could previously be mistaken for a visual
    page.
    """

    regions = []
    for info in page.get_image_info(xrefs=True):
        rect = fitz.Rect(info["bbox"])
        if rect.width >= min_side and rect.height >= min_side:
            regions.append(rect)
    return regions


def _vector_content_regions(page: fitz.Page) -> list[fitz.Rect]:
    """Find interior vector content while excluding page furniture."""

    page_rect = page.rect
    safe_rect = fitz.Rect(
        page_rect.x0 + 12,
        page_rect.y0 + 12,
        page_rect.x1 - 12,
        page_rect.y1 - 12,
    )
    area_regions = []
    line_regions = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if not rect.is_valid or not safe_rect.contains(rect.tl) or not safe_rect.contains(rect.br):
            continue
        if rect.width > 2 and rect.height > 2:
            area_regions.append(rect)
        elif max(rect.width, rect.height) >= 20:
            # PyMuPDF represents a stroked line with zero thickness; give it
            # a tiny extent so rectangle union includes all table rules.
            if rect.height <= 2:
                rect = fitz.Rect(rect.x0, rect.y0 - 1, rect.x1, rect.y1 + 1)
            elif rect.width <= 2:
                rect = fitz.Rect(rect.x0 - 1, rect.y0, rect.x1 + 1, rect.y1)
            line_regions.append(rect)

    # A single line is normally an underline or a rule, not a figure.  A
    # group of interior lines is useful for vector-only tables in this PDF.
    if area_regions:
        return area_regions + line_regions
    if len(line_regions) >= 3:
        return line_regions
    return []


def _nearby_caption_regions(page: fitz.Page, crop: fitz.Rect) -> list[fitz.Rect]:
    """Include a nearby figure/table title for vector-only content."""

    caption_pattern = re.compile(r"^(?:图|圖|表|Figure|Table)\s*[0-9０-９]", re.IGNORECASE)
    captions = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        text = normalize_text(block[4]).strip()
        rect = fitz.Rect(block[:4])
        if not caption_pattern.match(text):
            continue
        horizontal_overlap = min(crop.x1, rect.x1) - max(crop.x0, rect.x0)
        vertical_gap = crop.y0 - rect.y1
        if horizontal_overlap > 0 and 0 <= vertical_gap <= 16:
            captions.append(rect)
    return captions


CAPTION_PATTERN = re.compile(
    r"^\s*(?P<kind>图|圖|表|Figure|Table)\s*(?P<chapter>\d+)\s*[-－—]\s*(?P<index>\d+)",
    re.IGNORECASE,
)


def captioned_visual_regions(
    page: fitz.Page,
    *,
    max_caption_gap: float = 32.0,
) -> list[CaptionedVisual]:
    """Pair each real figure/table title with the visual directly above it.

    References in body prose (for example, ``图7-1 和图7-2 显示``) are
    deliberately excluded: a real title must begin its text block and have a
    nearby visual region immediately above it.
    """

    regions = _substantive_image_regions(page) or _vector_content_regions(page)
    if not regions:
        return []

    captions: list[tuple[fitz.Rect, str, str]] = []
    for block in page.get_text("blocks"):
        if len(block) >= 7 and block[6] != 0:
            continue
        text = " ".join(clean_lines(block[4]))
        match = CAPTION_PATTERN.match(text)
        if not match:
            continue
        marker = match.group("kind")
        label = f"{marker} {match.group('chapter')}-{match.group('index')}"
        captions.append((fitz.Rect(block[:4]), label, text))

    paired: list[CaptionedVisual] = []
    used_regions: set[int] = set()
    for caption_rect, label, caption in sorted(captions, key=lambda item: item[0].y0):
        candidates = []
        for index, region in enumerate(regions):
            if index in used_regions or region.y1 > caption_rect.y0:
                continue
            gap = caption_rect.y0 - region.y1
            overlap = min(region.x1, caption_rect.x1) - max(region.x0, caption_rect.x0)
            if gap <= max_caption_gap and overlap > 0:
                candidates.append((gap, -overlap, index, region))
        if not candidates:
            continue
        _, _, index, region = min(candidates)
        used_regions.add(index)
        paired.append(CaptionedVisual(region, caption_rect, label, caption))
    return paired


def visual_crop_rect(page: fitz.Page, *, margin: float = 8.0) -> fitz.Rect | None:
    # When an actual image exists, do not union it with unrelated vector
    # furniture.  The surrounding prose remains selectable text in XHTML.
    regions = _substantive_image_regions(page)
    if not regions:
        regions = _vector_content_regions(page)
    if not regions:
        return None

    crop = fitz.Rect(regions[0])
    for region in regions[1:]:
        crop |= region
    if not _substantive_image_regions(page):
        for caption in _nearby_caption_regions(page, crop):
            crop |= caption
    page_rect = page.rect
    return fitz.Rect(
        max(page_rect.x0, crop.x0 - margin),
        max(page_rect.y0, crop.y0 - margin),
        min(page_rect.x1, crop.x1 + margin),
        min(page_rect.y1, crop.y1 + margin),
    )


def render_page_jpeg(
    page: fitz.Page,
    *,
    zoom: float,
    quality: int,
    crop_visual: bool = False,
) -> bytes:
    if crop_visual:
        clip = visual_crop_rect(page) or page.rect
    else:
        rect = page.rect
        clip_x0 = max(rect.x0, 30)
        clip_y0 = max(rect.y0, 24)
        clip_x1 = max(rect.x0 + 1, rect.x1 - 30)
        clip_y1 = max(rect.y0 + 1, rect.y1 - 24)
        clip = fitz.Rect(clip_x0, clip_y0, clip_x1, clip_y1)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def render_visual_jpeg(page: fitz.Page, rect: fitz.Rect, *, zoom: float, quality: int) -> bytes:
    """Render one already-identified visual region without surrounding prose."""

    page_rect = page.rect
    clip = fitz.Rect(
        max(page_rect.x0, rect.x0 - 8),
        max(page_rect.y0, rect.y0 - 8),
        min(page_rect.x1, rect.x1 + 8),
        min(page_rect.y1, rect.y1 + 8),
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def render_cover_jpeg(page: fitz.Page) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def candidate_visual_pages(doc: fitz.Document, *, skip_page_rasters: bool = False) -> set[int]:
    pages: set[int] = set()
    drawing_counts = [len(page.get_drawings()) for page in doc]
    drawing_baseline = statistics.median(drawing_counts) if drawing_counts else 0
    drawing_threshold = max(5, math.ceil(drawing_baseline * 1.5))
    caption_pattern = re.compile(
        r"^[ \t]*(?:(?:Figure|Table)[ \t]*\d+(?:\.\d+)?|(?:图|圖|表)[ \t]*\d+(?:\.\d+)?)",
        re.IGNORECASE | re.MULTILINE,
    )
    for index, page in enumerate(doc, start=1):
        text = normalize_text(page.get_text("text"))
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        if index == 1 and (images or drawings) and len(text.strip()) < 40:
            continue
        image_regions = _substantive_image_regions(page)
        if image_regions:
            image_bounds = fitz.Rect(image_regions[0])
            for region in image_regions[1:]:
                image_bounds |= region
            image_coverage = image_bounds.get_area() / max(page.rect.get_area(), 1)
            has_caption = bool(caption_pattern.search(text))
            # Area alone cannot distinguish a scan from full-page artwork.
            # Suppress page rasters only when the caller reviewed the source.
            if not skip_page_rasters or image_coverage < 0.45 or has_caption or len(image_regions) > 1:
                pages.add(index)
        if len(drawings) >= drawing_threshold:
            pages.add(index)
        interior = fitz.Rect(
            page.rect.width * 0.1,
            page.rect.height * 0.1,
            page.rect.width * 0.9,
            page.rect.height * 0.9,
        )
        interior_drawings = sum(
            1
            for drawing in drawings
            if interior.contains(
                fitz.Point(
                    (drawing["rect"].x0 + drawing["rect"].x1) / 2,
                    (drawing["rect"].y0 + drawing["rect"].y1) / 2,
                )
            )
        )
        vector_regions = _vector_content_regions(page)
        vector_area = sum(region.get_area() for region in vector_regions)
        if (
            len(text.strip()) < 80
            and interior_drawings >= 5
            and (len(vector_regions) >= 3 or vector_area >= 400)
        ):
            pages.add(index)
        if caption_pattern.search(text) and (drawings or images):
            pages.add(index)
    return pages


def inspect_pdf(args: argparse.Namespace) -> int:
    doc = fitz.open(args.pdf)
    pages = []
    text_lengths = []
    image_pages = []
    drawing_pages = []
    for index, page in enumerate(doc, start=1):
        text_len = len(page.get_text("text").strip())
        image_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        text_lengths.append(text_len)
        if image_count:
            image_pages.append({"page": index, "images": image_count, "text_len": text_len})
        if drawing_count >= args.drawing_threshold:
            drawing_pages.append({"page": index, "drawings": drawing_count, "text_len": text_len})
        if index <= args.sample_pages:
            pages.append(
                {
                    "page": index,
                    "text_len": text_len,
                    "images": image_count,
                    "drawings": drawing_count,
                    "sample": normalize_text(page.get_text("text")).strip().replace("\n", " ")[:300],
                }
            )

    nonempty = sum(1 for length in text_lengths if length > 40)
    summary = {
        "pdf": str(Path(args.pdf).resolve()),
        "page_count": doc.page_count,
        "metadata": doc.metadata,
        "text_layer_ratio": round(nonempty / max(doc.page_count, 1), 3),
        "image_pages": image_pages,
        "drawing_pages": drawing_pages,
        "candidate_visual_pages": sorted(candidate_visual_pages(doc)),
        "sample_pages": pages,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_epub(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf)
    out_path = Path(args.output)
    doc = fitz.open(pdf_path)

    if not (1 <= args.cover_page <= doc.page_count):
        print(f"[!] Error: cover_page must be between 1 and {doc.page_count}, got {args.cover_page}")
        return 1

    title = args.title or normalize_text(doc.metadata.get("title") or "") or pdf_path.stem
    author = args.author or normalize_text(doc.metadata.get("author") or "") or "Unknown"
    sections = parse_sections(args.section, doc.page_count) if args.section else default_sections(doc)
    visual_pages = parse_page_ranges(args.visual_pages) if args.visual_pages else candidate_visual_pages(doc, skip_page_rasters=args.skip_page_rasters)
    visual_pages = {page for page in visual_pages if 1 <= page <= doc.page_count}

    print(f"[*] Building EPUB from {doc.page_count} PDF pages")
    print(f"[*] Visual pages: {sorted(visual_pages)}")

    book = epub.EpubBook()
    identifier = args.identifier or (f"urn:isbn:{args.isbn}" if args.isbn else f"{pdf_path.stem}-hybrid-epub")
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language(args.language)
    book.add_author(author)
    for index, translator in enumerate(args.translator, 1):
        contributor_id = f"translator_{index}"
        book.add_metadata("DC", "contributor", translator, {"id": contributor_id})
        book.add_metadata(None, "meta", "trl", {"refines": f"#{contributor_id}", "property": "role", "scheme": "marc:relators"})
    if args.publisher:
        book.add_metadata("DC", "publisher", args.publisher)
    if args.published:
        book.add_metadata("DC", "date", args.published)
    if args.description:
        book.add_metadata("DC", "description", args.description)
    book.set_cover("cover.jpg", render_cover_jpeg(doc[args.cover_page - 1]))

    stylesheet = CSS
    if args.layout == "pages":
        stylesheet += "\nsection.pdf-page { page-break-after: always; }\n"
    if args.language.lower().startswith(("zh", "ja", "ko")):
        stylesheet += "\nbody { line-height: 1.8; } p { text-indent: 2em; }\n"
    css = epub.EpubItem(uid="book_css", file_name="style/book.css", media_type="text/css", content=stylesheet.encode("utf-8"))
    book.add_item(css)

    visual_items: dict[int, list[VisualAsset]] = {}
    for page_no in sorted(visual_pages):
        page = doc[page_no - 1]
        captioned_visuals = captioned_visual_regions(page)
        if captioned_visuals:
            assets = []
            for visual_index, visual in enumerate(captioned_visuals, 1):
                data = render_visual_jpeg(
                    page,
                    visual.rect,
                    zoom=args.zoom,
                    quality=args.jpeg_quality,
                )
                name = f"images/visual_p{page_no:04d}_f{visual_index:02d}.jpg"
                item = epub.EpubImage(
                    uid=f"visual_p{page_no:04d}_f{visual_index:02d}",
                    file_name=name,
                    media_type="image/jpeg",
                    content=data,
                )
                book.add_item(item)
                assets.append(VisualAsset(item, visual.caption, visual.caption_rect, visual.rect))
                im = Image.open(BytesIO(data))
                print(f"    {visual.label}: {len(data)} bytes {im.size}")
            visual_items[page_no] = assets
            continue

        fallback_crop = visual_crop_rect(page)
        if page_no != args.cover_page and fallback_crop is None:
            print(f"    visual page {page_no}: skipped (no visual region detected)")
            continue
        data = render_page_jpeg(
            page,
            zoom=args.zoom,
            quality=args.jpeg_quality,
            crop_visual=True,
        )
        name = f"images/visual_p{page_no:04d}.jpg"
        item = epub.EpubImage(uid=f"visual_p{page_no:04d}", file_name=name, media_type="image/jpeg", content=data)
        book.add_item(item)
        visual_items[page_no] = [VisualAsset(item, source_rect=fallback_crop)]
        im = Image.open(BytesIO(data))
        print(f"    visual page {page_no}: {len(data)} bytes {im.size}")

    chapters = []
    for index, section in enumerate(sections, 1):
        print(f"    section {index:02d}: {section.title} pages {section.start}-{section.end}")
        parts = [
            "<?xml version='1.0' encoding='utf-8'?>",
            "<!DOCTYPE html>",
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">',
            "<head>",
            f"<title>{escape(section.title)}</title>",
            '<link href="style/book.css" rel="stylesheet" type="text/css"/>',
            "</head><body>",
            f"<h1>{escape(section.title)}</h1>",
        ]
        chapter_footnotes: list[str] = []
        continuation_pages: set[int] = set()
        for page_no in range(section.start, section.end + 1):
            page_assets = visual_items.get(page_no, [])
            excluded_rects = [
                rect
                for asset in page_assets
                for rect in (asset.source_rect, asset.caption_rect)
                if rect is not None
            ]
            blocks, footnotes = page_body_and_footnotes(
                doc[page_no - 1],
                page_no,
                args.drop_headers,
                excluded_rects=excluded_rects,
                column_split=args.column_split,
            )
            if (
                args.layout == "flow" and args.join_page_continuations
                and args.column_split is None and page_no > section.start
                and "index" not in section.title.lower()
                and not page_assets and not visual_items.get(page_no - 1)
                and page_continues_paragraph(doc[page_no - 2], doc[page_no - 1], drop_headers=args.drop_headers)
            ):
                continuation_pages.add(page_no)
            chapter_footnotes.extend(footnotes)
            blocks = remove_duplicate_section_heading(
                blocks,
                section.title,
                is_section_start=(page_no == section.start),
            )
            if not blocks and page_no not in visual_items:
                continue
            parts.append(f'<section class="pdf-page" id="p{page_no:04d}">')
            for block_index, text in enumerate(blocks):
                parts.append(
                    paragraph_html(
                        text,
                        first=(block_index == 0),
                        index_page=("index" in section.title.lower()),
                    )
                )
            for asset in page_assets:
                src = asset.item.file_name
                alt_text = asset.caption or f"Visual content from source page {page_no}"
                figcaption = f"<figcaption>{escape(asset.caption)}</figcaption>" if asset.caption else ""
                parts.append(
                    '<figure class="visual-snapshot">'
                    f'<img src="{escape(src)}" alt="{escape(alt_text)}"/>'
                    f"{figcaption}</figure>"
                )
            parts.append("</section>")
        if chapter_footnotes:
            parts.append('<section class="endnotes"><h2>注释</h2><ol>')
            parts.extend(f"<li>{escape(note)}</li>" for note in chapter_footnotes)
            parts.append("</ol></section>")
        parts.append("</body></html>")

        chapter_content = "\n".join(parts).encode("utf-8")
        if args.layout == "flow":
            root = ElementTree.fromstring(chapter_content)
            flow_page_sections(root, continuation_pages=continuation_pages)
            ElementTree.register_namespace("", XHTML_NS)
            chapter_content = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)

        chapter = epub.EpubHtml(
            title=section.title,
            file_name=safe_filename(index, section.title),
            lang=args.language,
            content=chapter_content,
        )
        chapter.add_item(css)
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = tuple(epub.Link(ch.file_name, ch.title, f"section_{i:02d}") for i, ch in enumerate(chapters, 1))
    book.spine = ["nav"] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(out_path), book, {})
    print(f"[*] Wrote {out_path.resolve()} {out_path.stat().st_size} bytes")
    return 0


def verify_epub(args: argparse.Namespace) -> int:
    path = Path(args.epub)
    expected_pages = parse_page_ranges(args.expected_visual_pages)
    failures = []

    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                failures.append(f"bad zip member: {bad}")
            names = archive.namelist()
            infos = archive.infolist()
            if not infos or infos[0].filename != "mimetype":
                failures.append("mimetype must be the first ZIP member")
            try:
                mimetype_info = archive.getinfo("mimetype")
                if mimetype_info.compress_type != zipfile.ZIP_STORED:
                    failures.append("mimetype must be stored without compression")
                if archive.read("mimetype") != b"application/epub+zip":
                    failures.append("mimetype content must be application/epub+zip")
            except KeyError:
                failures.append("mimetype member is missing")

            name_set = set(names)
            missing_references: set[str] = set()
            for name in names:
                if not name.lower().endswith((".xhtml", ".html", ".opf", ".ncx")):
                    continue
                try:
                    root = ElementTree.fromstring(archive.read(name))
                except ElementTree.ParseError as exc:
                    failures.append(f"invalid XML in {name}: {exc}")
                    continue
                for element in root.iter():
                    for attribute, raw_reference in element.attrib.items():
                        if attribute.rsplit("}", 1)[-1] not in {"href", "src"}:
                            continue
                        parsed = urlsplit(raw_reference)
                        if parsed.scheme or parsed.netloc or not parsed.path:
                            continue
                        resolved = posixpath.normpath(
                            posixpath.join(posixpath.dirname(name), unquote(parsed.path))
                        )
                        if resolved not in name_set:
                            missing_references.add(f"{name} -> {raw_reference}")
            if missing_references:
                failures.append(
                    "missing local references: " + ", ".join(sorted(missing_references))
                )

            images = [n for n in names if n.lower().endswith(IMAGE_EXTENSIONS)]
            visuals = sorted(n for n in images if "visual_p" in n)
            visual_pages = {
                int(match.group(1))
                for name in visuals
                if (match := re.search(r"visual_p(\d+)(?:_f\d+)?\.(?:jpg|jpeg|png|webp)$", name, re.IGNORECASE))
            }
            html = "\n".join(
                archive.read(n).decode("utf-8", "ignore")
                for n in names
                if n.lower().endswith((".xhtml", ".html", ".opf"))
            )
            remote_image_refs = re.findall(r'<img[^>]+src=["\']https?://', html, flags=re.IGNORECASE)
            missing_labels = [label for label in args.expected_label if label not in html]
            missing_pages = sorted(expected_pages - visual_pages)
            extra_pages = sorted(visual_pages - expected_pages) if expected_pages else []
            if "cover" not in html.lower():
                failures.append("cover metadata/markup not detected")
            if remote_image_refs:
                failures.append(f"remote image references detected: {len(remote_image_refs)}")
            if missing_pages:
                failures.append(f"missing visual pages: {missing_pages}")
            if missing_labels:
                failures.append(f"missing labels: {missing_labels}")
            print(
                json.dumps(
                    {
                        "epub": str(path.resolve()),
                        "size": path.stat().st_size,
                        "image_count": len(images),
                        "visual_count": len(visuals),
                        "visual_pages": sorted(visual_pages),
                        "missing_visual_pages": missing_pages,
                        "extra_visual_pages": extra_pages,
                        "missing_labels": missing_labels,
                        "remote_image_refs": len(remote_image_refs),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except zipfile.BadZipFile:
        failures.append("not a valid zip/epub archive")

    if failures:
        print("FAIL:", "; ".join(failures), file=sys.stderr)
        return 1
    print("OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect PDF structure")
    inspect_parser.add_argument("pdf")
    inspect_parser.add_argument("--sample-pages", type=int, default=8)
    inspect_parser.add_argument("--drawing-threshold", type=int, default=5)
    inspect_parser.set_defaults(func=inspect_pdf)

    convert_parser = subparsers.add_parser("convert", help="build a hybrid text+visual EPUB")
    convert_parser.add_argument("pdf")
    convert_parser.add_argument("output")
    convert_parser.add_argument("--title")
    convert_parser.add_argument("--author")
    convert_parser.add_argument("--translator", action="append", default=[], help="repeatable translator contributor")
    convert_parser.add_argument("--layout", choices=("flow", "pages"), default="flow", help="continuous prose (default) or preserve PDF page breaks")
    convert_parser.add_argument("--join-page-continuations", action="store_true", help="join CJK page-turn continuations when geometry supports it; review the result")
    convert_parser.add_argument("--language", default="en")
    convert_parser.add_argument("--identifier")
    convert_parser.add_argument("--isbn", help="ISBN used as the identifier when --identifier is omitted")
    convert_parser.add_argument("--publisher")
    convert_parser.add_argument("--published", help="publication date, preferably YYYY-MM-DD")
    convert_parser.add_argument("--description")
    convert_parser.add_argument("--cover-page", type=int, default=1)
    convert_parser.add_argument("--visual-pages", help="comma-separated page numbers/ranges")
    convert_parser.add_argument("--skip-page-rasters", action="store_true", help="exclude uncaptioned large raster pages after manual scan/artwork review")
    convert_parser.add_argument("--column-split", type=float, help="read left column then right column, split at this PDF x coordinate in points")
    convert_parser.add_argument("--section", action="append", default=[], help="repeat: 'Title:start-end'")
    convert_parser.add_argument("--zoom", type=float, default=1.75)
    convert_parser.add_argument("--jpeg-quality", type=int, default=83)
    convert_parser.add_argument("--keep-headers", dest="drop_headers", action="store_false")
    convert_parser.set_defaults(func=build_epub, drop_headers=True)

    verify_parser = subparsers.add_parser("verify", help="verify EPUB archive and expected content")
    verify_parser.add_argument("epub")
    verify_parser.add_argument("--expected-visual-pages", default="")
    verify_parser.add_argument("--expected-label", action="append", default=[])
    verify_parser.set_defaults(func=verify_epub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
