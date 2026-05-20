"""OCR noise table detection and cleanup functions."""

import html
import re

from .config import (
    DOTTED_NUMERIC_TOKEN_PATTERN,
    DOWNARROW_PROSE_SEPARATOR_PATTERN,
    HTML_TABLE_PATTERN,
)


def is_numeric_only_ocr_table(table_html: str) -> bool:
    """Detect blank-page OCR tables made only from repeated year-like numbers."""
    row_count = len(re.findall(r"(?i)<tr\b", table_html))
    if row_count < 12:
        return False

    text = html.unescape(re.sub(r"(?is)<[^>]+>", " ", table_html))
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if len(numbers) < 30:
        return False

    non_numeric = re.sub(r"[\d\s.,;:|+\-–—/\\年月]+", "", text)
    if non_numeric.strip():
        return False

    year_like = [value for value in numbers if 1900 <= value <= 2200]
    if len(year_like) / len(numbers) < 0.8:
        return False

    return len(set(year_like)) >= 12 and (max(year_like) - min(year_like)) >= 12


def is_dotted_numeric_ocr_table(table_html: str) -> bool:
    """Detect blank-page OCR tables made from repeated outline-like numbers."""
    cell_count = len(re.findall(r"(?i)<t[dh]\b", table_html))
    if cell_count < 16:
        return False

    text = html.unescape(re.sub(r"(?is)<[^>]+>", " ", table_html))
    dotted_tokens = DOTTED_NUMERIC_TOKEN_PATTERN.findall(text)
    if len(dotted_tokens) < 16:
        return False
    if len(dotted_tokens) / cell_count < 0.75:
        return False

    prefixes: Dict[str, int] = {}
    tails = []
    for token in dotted_tokens:
        prefix, tail = token.rsplit(".", 1)
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        tails.append(int(tail))

    if max(prefixes.values()) / len(dotted_tokens) < 0.8:
        return False
    if len(set(tails)) < 12 or (max(tails) - min(tails)) < 12:
        return False

    residue = DOTTED_NUMERIC_TOKEN_PATTERN.sub(" ", text)
    residue = re.sub(r"[\d\s.,;:|+\-–—/\\年月·•、。．]+", "", residue)
    return len(residue.strip()) <= 12


def is_ocr_noise_table(table_html: str) -> bool:
    """Detect OCR table artifacts that should not be carried into EPUB output."""
    return (
        is_numeric_only_ocr_table(table_html)
        or is_dotted_numeric_ocr_table(table_html)
    )


def remove_numeric_only_ocr_tables(markdown_text: str) -> str:
    """Strip numeric-like HTML tables commonly hallucinated from blank pages."""
    return HTML_TABLE_PATTERN.sub(
        lambda match: "" if is_ocr_noise_table(match.group(0)) else match.group(0),
        markdown_text,
    )


def clean_ocr_noise(markdown_text: str) -> str:
    """Remove common PaddleOCR layout artifacts while preserving prose."""
    if not markdown_text:
        return ""

    cleaned = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = remove_numeric_only_ocr_tables(cleaned)

    # False inline math around emphasized Chinese characters.
    cleaned = re.sub(
        r"\$\s*\\underset\{\\cdot\}\{([^{}]+)\}\s*\$",
        r"\1",
        cleaned,
    )
    # Known OCR split: 2017 rendered as 20 $ \frac{1}{2} $7.
    cleaned = re.sub(
        r"20\s*\$\s*\\frac\{1\}\{2\}\s*\$\s*7年",
        "2017年",
        cleaned,
    )
    # False arrow used as a list separator in CJK prose.
    cleaned = DOWNARROW_PROSE_SEPARATOR_PATTERN.sub(r"\1、\2", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


