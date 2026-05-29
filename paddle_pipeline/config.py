"""Configuration constants, environment variables, and regex patterns for the Paddle OCR pipeline."""

import os
import re

# Optional third-party dependencies – checked at runtime via check_dependencies()
try:
    import requests
    import fitz  # PyMuPDF
    from ebooklib import epub
    from dotenv import load_dotenv
    from tqdm import tqdm
    load_dotenv()
except ImportError:
    requests = None  # type: ignore[assignment]
    fitz = None  # type: ignore[assignment]
    epub = None  # type: ignore[assignment]
    tqdm = None  # type: ignore[assignment]


# --- Configuration ---
API_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL_VERSION = "PaddleOCR-VL-1.6"
# Environment variable for API token
API_TOKEN = os.getenv("PADDLE_API_TOKEN", "")

# MinerU API configuration
MINERU_API_URL = os.getenv("MINERU_API_URL", "https://mineru.net/api/v4/extract/task")
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")

DEFAULT_EPUB_LANGUAGE = os.getenv("EPUB_LANGUAGE", "zh-Hant")
MAX_DAILY_PAGES = 3000


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Parse integer env vars without making import fail on bad local config."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return default
    return parsed


def _env_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse float env vars without making import fail on bad local config."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    """Parse bool-like env vars without making import fail on bad local config."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


CHUNK_SIZE = _env_int("PADDLE_CHUNK_SIZE", 5, minimum=1)
API_TIMEOUT_SECONDS = _env_int("PADDLE_API_TIMEOUT_SECONDS", 600, minimum=1)
PADDLE_POLL_INTERVAL = _env_int("PADDLE_POLL_INTERVAL", 5, minimum=1)
PADDLE_MAX_POLL_TIME = _env_int("PADDLE_MAX_POLL_TIME", 1800, minimum=1)
PADDLE_LAYOUT_THRESHOLD = _env_float("PADDLE_LAYOUT_THRESHOLD", 0.35, minimum=0.0, maximum=1.0)
PADDLE_TEMPERATURE = _env_float("PADDLE_TEMPERATURE", 0.1, minimum=0.0, maximum=2.0)
PADDLE_REPETITION_PENALTY = _env_float("PADDLE_REPETITION_PENALTY", 1.05, minimum=0.0)
PADDLE_TOP_P = _env_float("PADDLE_TOP_P", 0.75, minimum=0.0, maximum=1.0)
PADDLE_PAGE_TOP_PADDING_RATIO = _env_float(
    "PADDLE_PAGE_TOP_PADDING_RATIO", 0.05, minimum=0.0, maximum=1.0
)
PADDLE_PAGE_BOTTOM_PADDING_RATIO = _env_float(
    "PADDLE_PAGE_BOTTOM_PADDING_RATIO", 0.05, minimum=0.0, maximum=1.0
)
MINERU_POLL_INTERVAL = _env_int("MINERU_POLL_INTERVAL", 5, minimum=1)
MINERU_MAX_POLL_TIME = _env_int("MINERU_MAX_POLL_TIME", 1800, minimum=1)
MINERU_CHUNK_SIZE = _env_int("MINERU_CHUNK_SIZE", 20, minimum=1)
MINERU_MODEL_VERSION = os.getenv("MINERU_MODEL_VERSION", "vlm")
MINERU_LANGUAGE = os.getenv("MINERU_LANGUAGE", "ch_tra")
MINERU_ENABLE_TABLE = _env_bool("MINERU_ENABLE_TABLE", False)
MINERU_PAGE_LEFT_MARGIN_POINTS = _env_float("MINERU_PAGE_LEFT_MARGIN_POINTS", 8.0, minimum=0.0)
MINERU_PAGE_TOP_PADDING_RATIO = _env_float(
    "MINERU_PAGE_TOP_PADDING_RATIO", 0.05, minimum=0.0, maximum=1.0
)
MINERU_PAGE_BOTTOM_PADDING_RATIO = _env_float(
    "MINERU_PAGE_BOTTOM_PADDING_RATIO", 0.05, minimum=0.0, maximum=1.0
)
DEFAULT_COVER_MAX_EDGE = _env_int("EPUB_COVER_MAX_EDGE", 2000, minimum=1)
DEFAULT_COVER_JPEG_QUALITY = _env_int("EPUB_COVER_JPEG_QUALITY", 82, minimum=1)

DOWNARROW_PROSE_SEPARATOR_PATTERN = re.compile(
    r"([㐀-鿿])\s*\$\s*\\downarrow\s*\$\s*([㐀-鿿])"
)
HTML_TABLE_PATTERN = re.compile(r"(?is)<table\b.*?</table>")
DOTTED_NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d.])")
OCR_NOISE_PATTERNS = (
    ("\\underset{\\cdot}", re.compile(r"\\underset\{\\cdot\}")),
    ("CJK $ \\downarrow $ separator", DOWNARROW_PROSE_SEPARATOR_PATTERN),
    (
        "20 $ \\frac{1}{2} $7年",
        re.compile(r"20\s*\$\s*\\frac\{1\}\{2\}\s*\$\s*7年"),
    ),
)
EPUB_STRUCTURAL_FILES = {
    "content.opf",
    "toc.ncx",
    "nav.xhtml",
    "nav.html",
}
FOOTNOTE_LABELS = {"footnote", "vision_footnote"}
FOOTNOTE_MARKER_PATTERN = re.compile(
    r"^\s*(?:\$\s*\^\{)?(\d{1,3})(?:\}\s*\$)?\s*(.*)$",
    re.DOTALL,
)
INLINE_FOOTNOTE_MARKER_PATTERN = re.compile(r"\$\s*\^\{?(\d{1,3})\}?\s*\$")
KNOWN_GARBLED_HEADINGS = {"扙艶倣麠邉薬棩盩"}
TOC_PAGE_START_CLASS = "toc-page-start"
TOC_PAGE_START_CSS = f"""
.{TOC_PAGE_START_CLASS} {{
    break-before: page;
    page-break-before: always;
    margin-top: 0 !important;
    padding-top: 0 !important;
}}
body > .{TOC_PAGE_START_CLASS}:first-child {{
    break-before: auto;
    page-break-before: auto;
}}
p.{TOC_PAGE_START_CLASS} + p + h1.{TOC_PAGE_START_CLASS},
p.{TOC_PAGE_START_CLASS} + p + h2.{TOC_PAGE_START_CLASS} {{
    break-before: auto;
    page-break-before: auto;
}}
"""
NUMBERED_TOC_LABEL_PATTERN = re.compile(
    r"^(第[零〇一二三四五六七八九十百千0-9]+([章編编篇部卷]))"
)
PART_TOC_KINDS = {"編", "编", "篇", "部", "卷"}
XHTML_TEXT_ELEMENT_PATTERN = re.compile(
    r"(?is)(<(?P<tag>h[1-6]|p|div|span)\b(?P<attrs>[^>]*)>)"
    r"(?P<body>.*?)</(?P=tag)>"
)
