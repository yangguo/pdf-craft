"""Configuration constants, environment variables, and regex patterns for the Paddle OCR pipeline."""

import os
import re

from typing import List, Dict, Any

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
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_opt_int(name: str, minimum: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if minimum is not None and parsed < minimum:
        return None
    return parsed


def _env_opt_float(
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _env_opt_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _env_opt_str(name: str, allowed: set[str] | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if allowed is not None and value not in allowed:
        return None
    return value


CHUNK_SIZE = _env_int("PADDLE_CHUNK_SIZE", 5, minimum=1)
API_TIMEOUT_SECONDS = _env_int("PADDLE_API_TIMEOUT_SECONDS", 600, minimum=1)
PADDLE_POLL_INTERVAL = _env_int("PADDLE_POLL_INTERVAL", 5, minimum=1)
PADDLE_MAX_POLL_TIME = _env_int("PADDLE_MAX_POLL_TIME", 1800, minimum=1)
MINERU_POLL_INTERVAL = _env_int("MINERU_POLL_INTERVAL", 5, minimum=1)
MINERU_MAX_POLL_TIME = _env_int("MINERU_MAX_POLL_TIME", 1800, minimum=1)
MINERU_CHUNK_SIZE = _env_int("MINERU_CHUNK_SIZE", 20, minimum=1)
MINERU_MODEL_VERSION = os.getenv("MINERU_MODEL_VERSION", "vlm")
MINERU_LANGUAGE = os.getenv("MINERU_LANGUAGE", "ch_server")
DEFAULT_COVER_MAX_EDGE = _env_int("EPUB_COVER_MAX_EDGE", 2000, minimum=1)
DEFAULT_COVER_JPEG_QUALITY = _env_int("EPUB_COVER_JPEG_QUALITY", 82, minimum=1)

MODEL_VERSION = os.getenv("PADDLE_MODEL_VERSION", MODEL_VERSION)
PADDLE_USE_DOC_ORIENTATION_CLASSIFY = _env_bool("PADDLE_USE_DOC_ORIENTATION_CLASSIFY", True)
PADDLE_USE_DOC_UNWARPING = _env_bool("PADDLE_USE_DOC_UNWARPING", True)
PADDLE_USE_CHART_RECOGNITION = _env_bool("PADDLE_USE_CHART_RECOGNITION", False)
PADDLE_USE_LAYOUT_DETECTION = _env_opt_bool("PADDLE_USE_LAYOUT_DETECTION")
PADDLE_LAYOUT_THRESHOLD = _env_float("PADDLE_LAYOUT_THRESHOLD", 0.5, minimum=0.0, maximum=1.0)
PADDLE_LAYOUT_NMS = _env_opt_bool("PADDLE_LAYOUT_NMS")
PADDLE_LAYOUT_UNCLIP_RATIO = _env_opt_float("PADDLE_LAYOUT_UNCLIP_RATIO", minimum=0.0)
PADDLE_LAYOUT_MERGE_BBOXES_MODE = _env_opt_str(
    "PADDLE_LAYOUT_MERGE_BBOXES_MODE",
    allowed={"large", "small", "union"},
)
PADDLE_LAYOUT_SHAPE_MODE = _env_opt_str(
    "PADDLE_LAYOUT_SHAPE_MODE",
    allowed={"rect", "quad", "poly", "auto"},
)
PADDLE_PROMPT_LABEL = _env_opt_str(
    "PADDLE_PROMPT_LABEL",
    allowed={"ocr", "formula", "table", "chart"},
)
PADDLE_REPETITION_PENALTY = _env_float("PADDLE_REPETITION_PENALTY", 1.2, minimum=0.0)
PADDLE_TEMPERATURE = _env_float("PADDLE_TEMPERATURE", 0.2, minimum=0.0)
PADDLE_TOP_P = _env_float("PADDLE_TOP_P", 0.85, minimum=0.0, maximum=1.0)
PADDLE_MIN_PIXELS = _env_opt_int("PADDLE_MIN_PIXELS", minimum=1)
PADDLE_MAX_PIXELS = _env_opt_int("PADDLE_MAX_PIXELS", minimum=1)
PADDLE_PRETTIFY_MARKDOWN = _env_opt_bool("PADDLE_PRETTIFY_MARKDOWN")
PADDLE_VISUALIZE = _env_opt_bool("PADDLE_VISUALIZE")

PADDLE_FORCE_ROTATE = _env_int("PADDLE_FORCE_ROTATE", 0, minimum=0)
PADDLE_PAGE_PADDING_X = _env_float("PADDLE_PAGE_PADDING_X", 0.0, minimum=0.0)
PADDLE_PAGE_PADDING_TOP = _env_float("PADDLE_PAGE_PADDING_TOP", 0.0, minimum=0.0)
PADDLE_PAGE_PADDING_BOTTOM = _env_float("PADDLE_PAGE_PADDING_BOTTOM", 0.05, minimum=0.0)

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
