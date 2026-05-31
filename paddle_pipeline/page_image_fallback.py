"""Fallback rendering for visual pages missed by Paddle layout extraction."""

import html
import os
import re

from typing import Any, Dict, List

from .config import fitz  # Optional dependency

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None  # type: ignore[assignment]


_VISUAL_PAGE_TITLE_PATTERN = re.compile(
    r"(家\s*系\s*[圖图]|家\s*[譜谱]|系\s*[圖图]|地\s*[圖图]|"
    r"示\s*意\s*[圖图]|關\s*係\s*[圖图]|关\s*系\s*[图圖])"
)
_VISIBLE_MARK_CHANNEL_MAX = 210
_VISIBLE_MARK_MIN_RATIO = 0.005


def _plain_text(markdown_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", markdown_text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", "", text)


def _alt_text(markdown_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", markdown_text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _is_sparse_visual_page(markdown_text: str, images: Dict[str, Any]) -> bool:
    if images:
        return False
    text = _plain_text(markdown_text)
    if not text or len(text) > 40:
        return False
    return bool(_VISUAL_PAGE_TITLE_PATTERN.search(markdown_text))


def _dark_pixel_ratio(pix: Any) -> float:
    samples = getattr(pix, "samples", b"")
    channels = max(1, int(getattr(pix, "n", 1) or 1))
    if not samples:
        return 0.0

    width = int(getattr(pix, "width", 0) or 0)
    height = int(getattr(pix, "height", 0) or 0)
    expected = width * height * channels
    if np is not None and expected > 0:
        arr = np.frombuffer(samples, dtype=np.uint8)
        if arr.size >= expected:
            arr = arr[:expected].reshape((height, width, channels))
            color_channels = min(3, channels)
            dark = (
                arr[:, :, :color_channels].min(axis=2)
                < _VISIBLE_MARK_CHANNEL_MAX
            )
            return float(dark.mean())

    total = len(samples) // channels
    if total <= 0:
        return 0.0

    dark = 0
    color_channels = min(3, channels)
    for offset in range(0, total * channels, channels):
        pixel = samples[offset:offset + color_channels]
        if pixel and min(pixel) < _VISIBLE_MARK_CHANNEL_MAX:
            dark += 1
    return dark / total


def _pdf_page_has_visible_marks(doc: Any | None, page_number: int,
                                zoom: float = 0.25) -> bool:
    if fitz is None or doc is None:
        return False

    page = doc[page_number - 1]
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    return _dark_pixel_ratio(pix) >= _VISIBLE_MARK_MIN_RATIO


def _is_empty_ocr_visible_page(doc: Any | None, page_number: int,
                               markdown_text: str,
                               images: Dict[str, Any]) -> bool:
    if images or _plain_text(markdown_text):
        return False
    return _pdf_page_has_visible_marks(doc, page_number)


def _has_missing_image_assets(images: Dict[str, Any], image_dir: str) -> bool:
    for rel_path in images:
        if rel_path.startswith("imgs/page_fallback_"):
            continue
        if not os.path.exists(os.path.join(image_dir, rel_path)):
            return True
    return False


def _rotation_for_page(markdown_text: str, page: Any) -> int:
    text = _plain_text(markdown_text)
    rect = getattr(page, "rect", None)
    width = getattr(rect, "width", 0)
    height = getattr(rect, "height", 0)
    if re.search(r"家\s*系\s*[圖图]|家\s*[譜谱]|系\s*[圖图]", text) and height > width:
        return 90
    return 0


def _render_page_png(doc: Any | None, page_number: int, output_path: str,
                     markdown_text: str, zoom: float = 2.0) -> None:
    if fitz is None or doc is None:
        raise RuntimeError("pymupdf is required for page image fallback rendering")

    page = doc[page_number - 1]
    matrix = fitz.Matrix(zoom, zoom)
    rotation = _rotation_for_page(markdown_text, page)
    if rotation:
        matrix = matrix.prerotate(rotation)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pix.save(output_path)


def apply_page_image_fallbacks(pdf_path: str, results: List[Dict[str, Any]],
                               image_dir: str) -> int:
    """Render whole-page images for visual pages missing OCR image assets."""
    rendered = 0
    global_page = 0
    doc = fitz.open(pdf_path) if fitz is not None else None

    try:
        for result in results:
            layout_results = result.get("result", {}).get("layoutParsingResults", [])
            for page_res in layout_results:
                global_page += 1
                markdown = page_res.setdefault("markdown", {})
                markdown_text = markdown.get("text", "")
                images = markdown.setdefault("images", {})

                if not (
                    _is_sparse_visual_page(markdown_text, images)
                    or _is_empty_ocr_visible_page(
                        doc,
                        global_page,
                        markdown_text,
                        images,
                    )
                    or _has_missing_image_assets(images, image_dir)
                ):
                    continue

                rel_path = f"imgs/page_fallback_{global_page:04d}.png"
                local_path = os.path.join(image_dir, rel_path)

                if rel_path not in images or not os.path.exists(local_path):
                    _render_page_png(doc, global_page, local_path, markdown_text)
                    images[rel_path] = ""

                if rel_path not in markdown_text:
                    alt_text = _alt_text(markdown_text) or "Page image"
                    escaped_alt = html.escape(alt_text, quote=True)
                    markdown["text"] = (
                        markdown_text.rstrip()
                        + "\n\n"
                        + f'<div style="text-align: center;"><img src="{rel_path}" '
                        + f'alt="{escaped_alt}" width="100%" /></div>'
                    )

                rendered += 1
    finally:
        if doc is not None:
            doc.close()

    if rendered:
        print(f"[*] Rendered {rendered} visual page fallback image(s)")
    return rendered
