"""OCR noise table detection and cleanup functions."""

import html
import re

from collections import Counter
from typing import Dict, List

from .config import (
    DOTTED_NUMERIC_TOKEN_PATTERN,
    DOWNARROW_PROSE_SEPARATOR_PATTERN,
    HTML_TABLE_PATTERN,
)


# --- Garbled CJK text detection (self-calibrating bigram model) ---
# Detects OCR-hallucinated Chinese text by finding character spans whose
# bigrams are almost entirely singletons (appearing nowhere else in the book).
# Natural text re-uses common character transitions; garbled text does not.

_GARBLED_WINDOW_SIZE = 24
_GARBLED_MIN_SINGLETONS = 22  # singletons out of (window_size - 1) bigrams
_GARBLED_MIN_CONSECUTIVE = 10  # consecutive flagged windows to form a span


def _build_cjk_bigram_model(cjk_chars: List[str]) -> Dict[str, int]:
    """Build a character-bigram frequency Counter from a list of CJK characters."""
    bigrams = [
        cjk_chars[i] + cjk_chars[i + 1]
        for i in range(len(cjk_chars) - 1)
    ]
    return Counter(bigrams)


def _scan_text_for_garbled_cjk_spans(
    plain_text: str,
    bigram_freq: Dict[str, int],
) -> List[str]:
    """Return garbled-CJK spans found in *plain_text* using a pre-built bigram model.

    A sliding window flags positions where nearly all character bigrams are
    singletons (frequency == 1).  Contiguous flagged windows are merged into
    spans and returned when the run is long enough.
    """
    chars = re.findall(r"[一-鿿]", plain_text)
    w = _GARBLED_WINDOW_SIZE
    if len(chars) < w:
        return []

    # Flag each window position.
    flagged = []
    for i in range(len(chars) - w + 1):
        window = chars[i : i + w]
        window_bigrams = [
            window[j] + window[j + 1] for j in range(len(window) - 1)
        ]
        singletons = sum(
            1 for bg in window_bigrams if bigram_freq.get(bg, 0) == 1
        )
        flagged.append(singletons >= _GARBLED_MIN_SINGLETONS)

    # Merge consecutive flagged windows into spans.
    spans: List[str] = []
    run_start: int | None = None
    for i, f in enumerate(flagged):
        if f and run_start is None:
            run_start = i
        elif not f and run_start is not None:
            run_len = i - run_start
            if run_len >= _GARBLED_MIN_CONSECUTIVE:
                spans.append("".join(chars[run_start : i + w - 1]))
            run_start = None
    if run_start is not None:
        run_len = len(flagged) - run_start
        if run_len >= _GARBLED_MIN_CONSECUTIVE:
            spans.append("".join(chars[run_start:]))

    return spans


def _strip_html_tags(html_text: str) -> str:
    """Remove HTML tags and decode entities, returning plain text."""
    plain = re.sub(r"(?is)<[^>]+>", " ", html_text)
    plain = html.unescape(plain)
    return plain


def find_garbled_cjk_in_epub(
    epub_path: str,
    structural_files: set,
) -> List[Dict]:
    """Scan an EPUB for garbled CJK text spans using self-calibrating bigram analysis.

    Builds a character-bigram frequency model from the full book text, then
    scans each content file for spans where nearly all bigrams are singletons
    (appearing nowhere else in the book).  Such spans are likely OCR
    hallucinations.

    Returns a list of finding dicts with keys ``file``, ``token``, and ``count``,
    suitable for consumption by the OCR-noise validation pipeline.
    """
    import os
    import zipfile

    # First pass: collect all CJK text from content files.
    all_cjk: List[str] = []
    file_texts: Dict[str, str] = {}
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if os.path.basename(lower) in structural_files:
                continue
            if not lower.endswith((".xhtml", ".html")):
                continue
            html_content = archive.read(name).decode("utf-8", "ignore")
            plain = _strip_html_tags(html_content)
            file_texts[name] = plain
            all_cjk.extend(re.findall(r"[一-鿿]", plain))

    if not all_cjk:
        return []

    # Build self-calibrating bigram model from the full book text.
    bigram_freq = _build_cjk_bigram_model(all_cjk)

    # Second pass: scan each file for garbled spans.
    findings: List[Dict] = []
    for name, text in file_texts.items():
        spans = _scan_text_for_garbled_cjk_spans(text, bigram_freq)
        if spans:
            findings.append({
                "file": name,
                "token": (
                    f"potential garbled CJK text ({len(spans)} span"
                    + ("s" if len(spans) > 1 else "")
                    + ")"
                ),
                "count": len(spans),
            })

    return findings


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

