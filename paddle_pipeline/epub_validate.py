"""EPUB post-processing validation and rewriting."""

import html
import os
import re
import tempfile
import zipfile

from typing import Any, Dict, List

from .config import EPUB_STRUCTURAL_FILES, HTML_TABLE_PATTERN, OCR_NOISE_PATTERNS
from .config import epub  # Optional dependency, checked at runtime
from .ocr_noise import (
    find_garbled_cjk_in_epub,
    is_dotted_numeric_ocr_table,
    is_numeric_only_ocr_table,
)
from .toc_retarget import ensure_toc_targets_start_pages


_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_XHTML_HEADING_RE = re.compile(r"(?is)<h([1-6])\b[^>]*>(.*?)</h\1>")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_BODY_HEADING_SENTENCE_ENDINGS = (
    ".", "!", "?", ":", ";",
    "。", "！", "？", "：", "；", "）", "】", "〉", "》", "」", "』",
)
_STRUCTURAL_HEADING_RE = re.compile(
    r"^(?:Chapter|Part|Lecture|Preface|Intro|Appendix|Prologue|Epilogue|"
    r"Conclusion|Book|Acknowledgements|Contents|Abstract|序|前言|导论|目錄|目录|"
    r"第[零一二三四五六七八九十百千0-9]+[篇章講讲節节编編部卷]).*"
)


def _plain_text_from_html_fragment(html_text: str) -> str:
    plain = _HTML_TAG_RE.sub(" ", html_text)
    plain = html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def _compact_cjk(value: str) -> str:
    return "".join(_CJK_RE.findall(value))


def _has_compact_overlap(value: str, candidate: str, min_overlap: int = 8) -> bool:
    if not value or not candidate:
        return False
    if value in candidate or candidate in value:
        return True
    if len(candidate) < min_overlap or len(value) < min_overlap:
        return False
    shorter, longer = (
        (candidate, value) if len(candidate) <= len(value) else (value, candidate)
    )
    for index in range(0, len(shorter) - min_overlap + 1):
        if shorter[index:index + min_overlap] in longer:
            return True
    return False


def _find_suspicious_body_headings(
    xhtml_text: str,
    garbled_spans: list[str],
) -> list[str]:
    """Return heading texts that look like OCR-promoted body prose."""
    garbled_compact = [_compact_cjk(span) for span in garbled_spans]
    headings: list[str] = []
    for match in _XHTML_HEADING_RE.finditer(xhtml_text):
        text = _plain_text_from_html_fragment(match.group(2))
        if not text or _STRUCTURAL_HEADING_RE.match(text):
            continue

        compact = _compact_cjk(text)
        cjk_len = len(compact)
        if cjk_len < 10:
            continue

        sentence_like = text.endswith(_BODY_HEADING_SENTENCE_ENDINGS)
        contains_garbled_span = any(
            _has_compact_overlap(compact, span)
            for span in garbled_compact
        )
        if sentence_like or contains_garbled_span:
            headings.append(text)
    return headings


def scan_epub_for_ocr_noise(epub_path: str) -> List[Dict[str, Any]]:
    """Return suspicious OCR LaTeX artifacts found in EPUB text files."""
    findings: List[Dict[str, Any]] = []
    content_entries: dict[str, str] = {}
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower_name = name.lower()
            if os.path.basename(lower_name) in EPUB_STRUCTURAL_FILES:
                continue
            if not lower_name.endswith((".xhtml", ".html")):
                continue
            text = archive.read(name).decode("utf-8", "ignore")
            content_entries[name] = text
            for token, pattern in OCR_NOISE_PATTERNS:
                count = len(pattern.findall(text))
                if count:
                    findings.append({"file": name, "token": token, "count": count})
            numeric_table_count = sum(
                1
                for match in HTML_TABLE_PATTERN.finditer(text)
                if is_numeric_only_ocr_table(match.group(0))
            )
            if numeric_table_count:
                findings.append({
                    "file": name,
                    "token": "numeric-only OCR table",
                    "count": numeric_table_count,
                })
            dotted_table_count = sum(
                1
                for match in HTML_TABLE_PATTERN.finditer(text)
                if (
                    not is_numeric_only_ocr_table(match.group(0))
                    and is_dotted_numeric_ocr_table(match.group(0))
                )
            )
            if dotted_table_count:
                findings.append({
                    "file": name,
                    "token": "dotted numeric OCR table",
                    "count": dotted_table_count,
                })

    # Garbled CJK text detection (self-calibrating bigram model).
    garbled_findings = find_garbled_cjk_in_epub(
        epub_path, EPUB_STRUCTURAL_FILES,
    )
    garbled_spans_by_file = {
        finding["file"]: finding.get("spans", [])
        for finding in garbled_findings
    }
    for finding in garbled_findings:
        findings.append({
            "file": finding["file"],
            "token": f"potential garbled CJK text ({len(finding['spans'])} span{'s' if len(finding['spans']) > 1 else ''})",
            "count": len(finding["spans"]),
            "examples": finding["spans"][:3],
        })

    for name, text in content_entries.items():
        suspicious_headings = _find_suspicious_body_headings(
            text,
            garbled_spans_by_file.get(name, []),
        )
        if suspicious_headings:
            findings.append({
                "file": name,
                "token": "suspicious body heading",
                "count": len(suspicious_headings),
                "examples": suspicious_headings[:3],
            })

    return findings


def validate_epub_no_ocr_noise(epub_path: str, strict: bool = False) -> List[Dict[str, Any]]:
    """Report suspicious OCR artifacts found in the EPUB, optionally failing."""
    findings = scan_epub_for_ocr_noise(epub_path)
    if not findings:
        return []

    examples = ", ".join(
        f"{item['file']}:{item['token']} x{item['count']}"
        for item in findings[:8]
    )
    truncated = ""
    if len(findings) > 8:
        truncated = f" (showing 8 of {len(findings)})"
    message = (
        "OCR noise tokens remain in generated EPUB. "
        f"Inspect and clean these artifacts before using the file: {examples}{truncated}"
    )
    if strict:
        raise RuntimeError(message)
    print(f"[!] {message}")
    return findings


def write_validated_epub(
    book: Any,
    output_file: str,
    strict_ocr_validation: bool = False,
) -> None:
    """Write EPUB to a same-directory temp file and publish only after validation."""
    output_dir = os.path.dirname(os.path.abspath(output_file))
    output_name = os.path.basename(output_file)
    fd, temp_output = tempfile.mkstemp(
        prefix=f".{output_name}.",
        suffix=".epub",
        dir=output_dir,
    )
    os.close(fd)

    try:
        epub.write_epub(temp_output, book, {})
        if zipfile.is_zipfile(temp_output):
            ensure_toc_targets_start_pages(temp_output)
        validate_epub_no_ocr_noise(temp_output, strict=strict_ocr_validation)
        os.replace(temp_output, output_file)
    except Exception:
        try:
            os.remove(temp_output)
        except FileNotFoundError:
            pass
        raise
