#!/usr/bin/env python3
"""Inspect, build, and verify hybrid text+visual EPUBs from PDFs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path

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
  page-break-after: always;
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
"""


@dataclass(frozen=True)
class Section:
    title: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
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
    level_one = [(title, page) for level, title, page in toc if level == 1 and 1 <= page <= doc.page_count]
    if level_one:
        sections = []
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


def is_header_or_footer(block: tuple, drop_headers: bool, page_height: float = 792.0) -> bool:
    if not drop_headers:
        return False
    _, y0, _, _, text = block[:5]
    compact = " ".join(clean_lines(text))
    if not compact:
        return True
    header_threshold = page_height * 0.065
    footer_threshold = page_height * 0.732
    return y0 < header_threshold or y0 > footer_threshold


def page_text_blocks(page: fitz.Page, page_no: int, drop_headers: bool) -> list[str]:
    blocks = [b for b in page.get_text("blocks") if len(b) < 7 or b[6] == 0]
    if page_no >= 300:
        blocks = sorted(blocks, key=lambda b: (0 if b[0] < 200 else 1, b[1], b[0]))
    else:
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

    text_blocks = []
    page_height = page.rect.height
    for block in blocks:
        if is_header_or_footer(block, drop_headers, page_height):
            continue
        text = join_pdf_lines(clean_lines(block[4]))
        if text:
            text_blocks.append(text)
    return text_blocks


def paragraph_html(text: str, *, first: bool, index_page: bool) -> str:
    if index_page:
        cls = ' class="index-entry"'
    elif first or len(text) < 90 or re.match(r"^(Table|Figure)\s+\d", text):
        cls = ' class="noindent"'
    else:
        cls = ""
    return f"<p{cls}>{escape(text)}</p>"


def render_page_jpeg(page: fitz.Page, *, zoom: float, quality: int) -> bytes:
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


def render_cover_jpeg(page: fitz.Page) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def candidate_visual_pages(doc: fitz.Document) -> set[int]:
    pages: set[int] = set()
    caption_pattern = re.compile(r"\b(?:Figure|Table)\s+\d+(?:\.\d+)?", re.IGNORECASE)
    for index, page in enumerate(doc, start=1):
        text = normalize_text(page.get_text("text"))
        if index == 1 and page.get_images(full=True):
            continue
        if page.get_images(full=True):
            pages.add(index)
        if len(page.get_drawings()) >= 5:
            pages.add(index)
        if caption_pattern.search(text) and (page.get_drawings() or "Table " in text or "Figure " in text):
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
    visual_pages = parse_page_ranges(args.visual_pages) if args.visual_pages else candidate_visual_pages(doc)
    visual_pages = {page for page in visual_pages if 1 <= page <= doc.page_count}

    print(f"[*] Building EPUB from {doc.page_count} PDF pages")
    print(f"[*] Visual pages: {sorted(visual_pages)}")

    book = epub.EpubBook()
    book.set_identifier(args.identifier or f"{pdf_path.stem}-hybrid-epub")
    book.set_title(title)
    book.set_language(args.language)
    book.add_author(author)
    book.set_cover("cover.jpg", render_cover_jpeg(doc[args.cover_page - 1]))

    css = epub.EpubItem(uid="book_css", file_name="style/book.css", media_type="text/css", content=CSS.encode("utf-8"))
    book.add_item(css)

    visual_items = {}
    for page_no in sorted(visual_pages):
        data = render_page_jpeg(doc[page_no - 1], zoom=args.zoom, quality=args.jpeg_quality)
        name = f"images/visual_p{page_no:04d}.jpg"
        item = epub.EpubImage(uid=f"visual_p{page_no:04d}", file_name=name, media_type="image/jpeg", content=data)
        book.add_item(item)
        visual_items[page_no] = item
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
        for page_no in range(section.start, section.end + 1):
            blocks = page_text_blocks(doc[page_no - 1], page_no, args.drop_headers)
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
            if page_no in visual_items:
                src = visual_items[page_no].file_name
                caption = f"PDF page {page_no} figure/table visual snapshot"
                parts.append(
                    '<figure class="visual-snapshot">'
                    f'<img src="{escape(src)}" alt="{escape(caption)}"/>'
                    f"<figcaption>{escape(caption)}</figcaption>"
                    "</figure>"
                )
            parts.append("</section>")
        parts.append("</body></html>")

        chapter = epub.EpubHtml(
            title=section.title,
            file_name=safe_filename(index, section.title),
            lang=args.language,
            content="\n".join(parts).encode("utf-8"),
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
            images = [n for n in names if n.lower().endswith(IMAGE_EXTENSIONS)]
            visuals = sorted(n for n in images if "visual_p" in n)
            visual_pages = {
                int(match.group(1))
                for name in visuals
                if (match := re.search(r"visual_p(\d+)\.(?:jpg|jpeg|png|webp)$", name, re.IGNORECASE))
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
    convert_parser.add_argument("--language", default="en")
    convert_parser.add_argument("--identifier")
    convert_parser.add_argument("--cover-page", type=int, default=1)
    convert_parser.add_argument("--visual-pages", help="comma-separated page numbers/ranges")
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
