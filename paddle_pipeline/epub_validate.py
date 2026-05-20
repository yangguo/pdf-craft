"""EPUB post-processing validation and rewriting."""

import os
import tempfile
import zipfile

from typing import Any, Dict, List

from .config import EPUB_STRUCTURAL_FILES, HTML_TABLE_PATTERN, OCR_NOISE_PATTERNS
from .config import epub  # Optional dependency, checked at runtime
from .ocr_noise import is_dotted_numeric_ocr_table, is_numeric_only_ocr_table
from .toc_retarget import ensure_toc_targets_start_pages


def scan_epub_for_ocr_noise(epub_path: str) -> List[Dict[str, Any]]:
    """Return suspicious OCR LaTeX artifacts found in EPUB text files."""
    findings: List[Dict[str, Any]] = []
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower_name = name.lower()
            if os.path.basename(lower_name) in EPUB_STRUCTURAL_FILES:
                continue
            if not lower_name.endswith((".xhtml", ".html")):
                continue
            text = archive.read(name).decode("utf-8", "ignore")
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


