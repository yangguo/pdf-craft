#!/usr/bin/env python3
"""Build Markdown + EPUB from per-page OCR results (PaddleOCR MCP).

This pipeline assumes you already have page images rendered to disk and a
`bird_ocr_results.json` produced by running PaddleOCR via the Codex MCP tool.

Defaults are tuned for a readable, smaller EPUB:
- Skip embedding full-page images for text-heavy pages.
- Crop white margins for embedded images.
- Clean OCR text and convert Markdown headings to EPUB headings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipInfo, ZipFile

from PIL import Image as PILImage

from epub_generator import (
    BookMeta,
    Chapter,
    EpubData,
    Image,
    TextBlock,
    TextKind,
    TocItem,
    generate_epub,
)

_RE_HTML_DIV_OPEN = re.compile(r"<div[^>]*>")
_RE_HTML_DIV_CLOSE = re.compile(r"</div>")
_RE_HTML_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_RE_HTML_TABLE = re.compile(r"</?(table|tr|td|tbody|thead|th)\b[^>]*>", re.IGNORECASE)
_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_RE_WHITESPACE_ONLY = re.compile(r"^\s+$")

_EXTRA_CSS = """
body {
  line-height: 1.7;
}

p {
  margin: 0.35em 0;
}

h1 {
  margin: 1.2em 0 0.6em;
  font-size: 1.6em;
}

h2 {
  margin: 1.0em 0 0.5em;
  font-size: 1.3em;
}

h3 {
  margin: 0.9em 0 0.4em;
  font-size: 1.15em;
}

div.asset {
  margin: 1.1em 0;
}

div.alt-wrapper {
  padding-bottom: 1.25em;
}

img {
  max-width: 100%;
  height: auto;
}
""".strip()


def load_skeleton(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pages", [])


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(path: Path, pages: list[dict]) -> None:
    path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line == "❌ No document content detected":
            continue
        line = _RE_HTML_DIV_OPEN.sub("", line)
        line = _RE_HTML_DIV_CLOSE.sub("", line)
        line = _RE_HTML_IMG.sub("", line)
        line = _RE_HTML_TABLE.sub("", line)
        line = _RE_MD_IMAGE.sub("", line).strip()
        if _RE_WHITESPACE_ONLY.match(line):
            continue
        lines.append(line)

    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line == "":
            if blank:
                continue
            blank = True
            cleaned.append("")
            continue
        blank = False
        # Drop standalone digit-only lines (page numbers / scan pagination)
        if re.fullmatch(r"[0-9]+", line):
            continue
        cleaned.append(line)

    out = "\n".join(cleaned).strip()
    if re.fullmatch(r"[0-9 ]+", out):
        return ""
    return out


def is_blank_image(
    path: Path,
    *,
    threshold: int = 250,
    blank_ratio: float = 0.995,
    downsample: int = 10,
) -> bool:
    """Return True if the image is effectively blank (almost all white)."""
    img = PILImage.open(path).convert("L")
    width, height = img.size
    small = img.resize((max(1, width // downsample), max(1, height // downsample)))
    pixels = list(small.getdata())
    if not pixels:
        return True
    white = sum(1 for p in pixels if p >= threshold)
    return (white / len(pixels)) >= blank_ratio


def should_include_page_image(
    page: dict,
    *,
    policy: str = "auto",
    short_text_threshold: int = 120,
) -> bool:
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy != "auto":
        raise ValueError(f"Unknown image policy: {policy}")

    if page.get("force_image"):
        return True

    raw_text = page.get("raw_text") or page.get("text") or ""
    if "No document content detected" in raw_text:
        return True
    if _RE_HTML_IMG.search(raw_text):
        return True

    text = clean_text(raw_text)
    if not text:
        return True

    if short_text_threshold <= 0:
        return False
    return len(text) <= short_text_threshold


def crop_image_whitespace(
    src_path: Path,
    dst_path: Path,
    *,
    threshold: int = 250,
    margin: int = 10,
) -> None:
    img = PILImage.open(src_path).convert("RGB")
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()

    if bbox is None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        return

    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.size[0], right + margin)
    bottom = min(img.size[1], bottom + margin)

    cropped = img.crop((left, top, right, bottom))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst_path)


def patch_epub_css(epub_path: Path, extra_css: str) -> None:
    tmp_path = epub_path.with_suffix(epub_path.suffix + ".tmp")

    with ZipFile(epub_path, "r") as zin, ZipFile(tmp_path, "w") as zout:
        mimetype = zin.read("mimetype")
        mime_info = ZipInfo("mimetype")
        mime_info.compress_type = ZIP_STORED
        zout.writestr(mime_info, mimetype)

        for info in zin.infolist():
            name = info.filename
            if name == "mimetype":
                continue
            data = zin.read(name)
            if name == "OEBPS/styles/style.css":
                css = data.decode("utf-8", errors="replace").rstrip()
                if extra_css.strip() and extra_css not in css:
                    css = f"{css}\n\n/* pdf-craft overrides */\n{extra_css.strip()}\n"
                data = css.encode("utf-8")

            new_info = ZipInfo(name)
            new_info.date_time = info.date_time
            new_info.compress_type = ZIP_DEFLATED
            new_info.external_attr = info.external_attr
            new_info.internal_attr = info.internal_attr
            zout.writestr(new_info, data)

    tmp_path.replace(epub_path)


def _strip_leading_title_line(text: str, title: str) -> str:
    if not text or not title:
        return text
    lines = text.split("\n")
    if not lines:
        return text
    first = lines[0].strip()
    first_norm = re.sub(r"^#+\s+", "", first).strip()
    if first_norm != title:
        return text
    rest = "\n".join(lines[1:])
    return rest.lstrip("\n").strip()


def merge_pages_into_chapters(pages: list[dict]) -> list[dict]:
    chapters: list[dict] = []
    current: dict = {"title": "表紙", "pages": []}

    for page in pages:
        raw_text = page.get("raw_text") or page.get("text") or ""
        text = clean_text(raw_text)
        page = {**page, "raw_text": raw_text, "text": text}

        if not text:
            current["pages"].append(page)
            continue

        first_line = text.split("\n", 1)[0].strip()
        title_candidate: str | None = None
        is_chapter_start = False

        if first_line.startswith("#"):
            match = re.match(r"^(#+)\s+(.*)$", first_line)
            if match and len(match.group(1)) == 1:
                title_candidate = match.group(2).strip() or None
                is_chapter_start = title_candidate is not None
        elif first_line in ("目次", "はじめに"):
            title_candidate = first_line
            is_chapter_start = True

        if is_chapter_start and title_candidate and title_candidate != current.get("title"):
            if current.get("pages"):
                chapters.append(current)
            current = {"title": title_candidate, "pages": []}

        page["text"] = _strip_leading_title_line(page.get("text") or "", current.get("title") or "")
        current["pages"].append(page)

    if current.get("pages"):
        chapters.append(current)

    return chapters


def _iter_text_blocks(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    cleaned = clean_text(text)
    if not cleaned:
        return blocks

    lines = cleaned.split("\n")
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if not para:
            return
        paragraph = " ".join(para).strip()
        if paragraph:
            blocks.append(TextBlock(kind=TextKind.BODY, level=0, content=[paragraph]))
        para = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_para()
            continue

        if line.startswith(">"):
            flush_para()
            content = line[1:].strip()
            if content:
                blocks.append(TextBlock(kind=TextKind.QUOTE, level=0, content=[content]))
            continue

        if line.startswith("#"):
            flush_para()
            match = re.match(r"^(#+)\s+(.*)$", line)
            if match:
                level = min(max(len(match.group(1)) - 1, 0), 5)
                title = match.group(2).strip()
                if title:
                    blocks.append(TextBlock(kind=TextKind.HEADLINE, level=level, content=[title]))
                continue

        if line.startswith(("•", "-", "・")):
            flush_para()
            blocks.append(TextBlock(kind=TextKind.BODY, level=0, content=[line]))
            continue

        para.append(line)

    flush_para()
    return blocks


def build_markdown(
    chapters: list[dict],
    *,
    image_policy: str = "auto",
    short_text_threshold: int = 120,
) -> str:
    lines: list[str] = []
    for chapter in chapters:
        title = chapter.get("title") or "Chapter"
        lines.append(f"# {title}")
        for page in chapter.get("pages", []):
            raw_text = page.get("raw_text") or ""
            include_image = should_include_page_image(
                page, policy=image_policy, short_text_threshold=short_text_threshold
            )

            page_has_output = False
            if include_image:
                image_path = page.get("file")
                if image_path:
                    img_file = Path(image_path)
                    if not (img_file.exists() and is_blank_image(img_file)):
                        lines.append(f"![]({image_path})")
                        page_has_output = True

            text = clean_text(page.get("text") or "")
            if text:
                lines.append(text)
                page_has_output = True

            if page_has_output:
                lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_epub(
    chapters: list[dict],
    output_path: Path,
    *,
    title: str,
    author: str,
    cover_image: Path | None,
    image_policy: str = "auto",
    short_text_threshold: int = 120,
    crop_images: bool = True,
    crop_threshold: int = 250,
    crop_margin: int = 10,
    lan: str = "en",
) -> None:
    meta = BookMeta(title=title, authors=[author])
    toc_items: list[TocItem] = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        for chapter in chapters:
            chapter_title = chapter.get("title") or "Chapter"
            elements: list[TextBlock | Image] = [
                TextBlock(kind=TextKind.HEADLINE, level=0, content=[chapter_title])
            ]

            for page in chapter.get("pages", []):
                page_text = clean_text(page.get("text") or "")
                include_img = should_include_page_image(
                    page,
                    policy=image_policy,
                    short_text_threshold=short_text_threshold,
                )

                if include_img and page.get("file"):
                    src = Path(page["file"]).resolve()
                    if not src.exists():
                        include_img = False
                    elif is_blank_image(src):
                        include_img = False
                    else:
                        img_path = src
                        if crop_images:
                            # Use a per-source subdirectory to avoid basename collisions
                            # when two pages reference different files named identically
                            # (e.g. a/page.png and b/page.png).
                            src_hash = hashlib.md5(str(src).encode()).hexdigest()[:8]
                            dst = td_path / src_hash / src.name
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            crop_image_whitespace(
                                src,
                                dst,
                                threshold=crop_threshold,
                                margin=crop_margin,
                            )
                            img_path = dst
                        # Avoid noisy per-page captions by default.
                        elements.append(Image(path=img_path, caption=[]))

                elements.extend(_iter_text_blocks(page_text))

            toc_items.append(
                TocItem(
                    title=chapter_title,
                    get_chapter=lambda elems=elements: Chapter(elements=elems, footnotes=[]),
                    children=[],
                )
            )

        epub_data = EpubData(meta=meta, chapters=toc_items, cover_image_path=cover_image)
        generate_epub(epub_data=epub_data, epub_file_path=output_path, lan=lan)  # type: ignore[arg-type]

    patch_epub_css(output_path, _EXTRA_CSS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton", default="bird_ocr_skeleton.json")
    parser.add_argument("--results", default="bird_ocr_results.json")
    parser.add_argument("--markdown", default="bird.md")
    parser.add_argument("--epub", default="bird.epub")
    parser.add_argument("--title", default="僕には鳥の言葉がわかる")
    parser.add_argument("--author", default="鈴木俊貴")
    parser.add_argument("--cover", default="bird_pages/page_001.png")
    parser.add_argument("--skip-ocr", action="store_true")

    parser.add_argument("--image-policy", choices=["auto", "all", "none"], default="auto")
    # Default 0: keep page images only for illustration pages (inline <img> artifacts)
    # and for pages with no OCR text. Set >0 to also include scanned page images for
    # very short OCR pages.
    parser.add_argument("--short-text-threshold", type=int, default=0)
    parser.add_argument("--no-crop-images", action="store_true")
    parser.add_argument("--crop-threshold", type=int, default=250)
    parser.add_argument("--crop-margin", type=int, default=10)

    # epub_generator i18n only supports zh/en, and unknown values render blank labels.
    parser.add_argument("--lan", choices=["en", "zh"], default="en")

    args = parser.parse_args()

    if not args.skip_ocr:
        raise RuntimeError(
            "OCR must be run via the Codex PaddleOCR MCP tool. "
            "Generate the results JSON first, then re-run with --skip-ocr."
        )

    pages = load_results(Path(args.results))
    if not pages:
        results_path = Path(args.results)
        if not results_path.exists():
            raise SystemExit(f"[error] OCR results file not found: {results_path}")
        if results_path.stat().st_size == 0:
            raise SystemExit(f"[error] OCR results file is empty: {results_path}")
        raise SystemExit(
            f"[error] OCR results file contains no pages: {results_path}\n"
            "If OCR was interrupted, re-run without --skip-ocr to regenerate it."
        )

    chapters = merge_pages_into_chapters(pages)

    markdown = build_markdown(
        chapters,
        image_policy=args.image_policy,
        short_text_threshold=args.short_text_threshold,
    )
    Path(args.markdown).write_text(markdown, encoding="utf-8")

    _cover_candidate = Path(args.cover).resolve() if args.cover else None
    cover_image = _cover_candidate if (_cover_candidate and _cover_candidate.exists()) else None
    if args.cover and cover_image is None:
        print(f"[!] Cover image not found, skipping: {args.cover}")
    build_epub(
        chapters,
        Path(args.epub),
        title=args.title,
        author=args.author,
        cover_image=cover_image,
        image_policy=args.image_policy,
        short_text_threshold=args.short_text_threshold,
        crop_images=not args.no_crop_images,
        crop_threshold=args.crop_threshold,
        crop_margin=args.crop_margin,
        lan=args.lan,
    )


if __name__ == "__main__":
    main()
