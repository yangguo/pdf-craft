"""Fallback rendering for visual pages missed by Paddle layout extraction."""

import html
import os
import re

from typing import Any, Dict, List

from .config import fitz  # Optional dependency


_VISUAL_PAGE_TITLE_PATTERN = re.compile(
    r"(家\s*系\s*[圖图]|家\s*[譜谱]|系\s*[圖图]|地\s*[圖图]|"
    r"示\s*意\s*[圖图]|關\s*係\s*[圖图]|关\s*系\s*[图圖])"
)


def _plain_text(markdown_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", markdown_text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", "", text)


def _is_sparse_visual_page(markdown_text: str, images: Dict[str, Any]) -> bool:
    if images:
        return False
    text = _plain_text(markdown_text)
    if not text or len(text) > 40:
        return False
    return bool(_VISUAL_PAGE_TITLE_PATTERN.search(markdown_text))


def _rotation_for_page(markdown_text: str, page: Any) -> int:
    text = _plain_text(markdown_text)
    rect = getattr(page, "rect", None)
    width = getattr(rect, "width", 0)
    height = getattr(rect, "height", 0)
    if re.search(r"家\s*系\s*[圖图]|家\s*[譜谱]|系\s*[圖图]", text) and height > width:
        return 90
    return 0


def _render_page_png(pdf_path: str, page_number: int, output_path: str,
                     markdown_text: str, zoom: float = 2.0) -> None:
    if fitz is None:
        raise RuntimeError("pymupdf is required for page image fallback rendering")

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        matrix = fitz.Matrix(zoom, zoom)
        rotation = _rotation_for_page(markdown_text, page)
        if rotation:
            matrix = matrix.prerotate(rotation)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pix.save(output_path)
    finally:
        doc.close()


def apply_page_image_fallbacks(pdf_path: str, results: List[Dict[str, Any]],
                               image_dir: str) -> int:
    """Render whole-page images for sparse visual pages missing OCR image assets."""
    rendered = 0
    global_page = 0

    for result in results:
        layout_results = result.get("result", {}).get("layoutParsingResults", [])
        for page_res in layout_results:
            global_page += 1
            markdown = page_res.setdefault("markdown", {})
            markdown_text = markdown.get("text", "")
            images = markdown.setdefault("images", {})

            if not _is_sparse_visual_page(markdown_text, images):
                continue

            rel_path = f"imgs/page_fallback_{global_page:04d}.png"
            local_path = os.path.join(image_dir, rel_path)

            if rel_path not in images or not os.path.exists(local_path):
                _render_page_png(pdf_path, global_page, local_path, markdown_text)
                images[rel_path] = ""

            if rel_path not in markdown_text:
                alt_text = (
                    re.sub(r"^#{1,6}\s*", "", markdown_text.strip()).strip()
                    or "Page image"
                )
                escaped_alt = html.escape(alt_text, quote=True)
                markdown["text"] = (
                    markdown_text.rstrip()
                    + "\n\n"
                    + f'<div style="text-align: center;"><img src="{rel_path}" '
                    + f'alt="{escaped_alt}" width="100%" /></div>'
                )

            rendered += 1

    if rendered:
        print(f"[*] Rendered {rendered} sparse visual page fallback image(s)")
    return rendered
