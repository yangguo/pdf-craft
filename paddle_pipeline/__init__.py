"""Paddle OCR PDF-to-EPUB conversion pipeline.

Usage:
    python -m paddle_pipeline.main input.pdf
    # or
    from paddle_pipeline import main
    main()
"""

from .main import main

# Re-export public functions used by tests
from .config import (
    API_TIMEOUT_SECONDS,
    CHUNK_SIZE,
    DEFAULT_COVER_JPEG_QUALITY,
    DEFAULT_COVER_MAX_EDGE,
    TOC_PAGE_START_CSS,
)
from .epub_href import _resolve_epub_fragment_href
from .epub_validate import (
    scan_epub_for_ocr_noise,
    validate_epub_no_ocr_noise,
    write_validated_epub,
)
from .epub_builder import create_epub
from .footnotes import (
    extract_page_footnotes,
    format_page_footnotes_html,
    link_page_footnote_references,
)
from .ocr_noise import clean_ocr_noise
from .paddle_api import check_dependencies
from .toc_retarget import ensure_toc_targets_start_pages

__all__ = [
    "main",
    "check_dependencies",
    "clean_ocr_noise",
    "create_epub",
    "ensure_toc_targets_start_pages",
    "extract_page_footnotes",
    "format_page_footnotes_html",
    "link_page_footnote_references",
    "scan_epub_for_ocr_noise",
    "validate_epub_no_ocr_noise",
]
