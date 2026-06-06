"""Targeted MinerU re-OCR helpers for selected PDF pages."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, List

from . import mineru_api
from .config import MINERU_API_TOKEN, MINERU_CHUNK_SIZE, fitz
from .ocr_review import _cjk_chars, _score_fragment


@dataclass(frozen=True)
class MineruRerunSummary:
    page_number: int
    checkpoint_path: str
    page_index: int
    text_preview: str


def parse_page_ranges(spec: str) -> list[int]:
    """Parse 1-based page numbers such as ``"71,75-79,82"``."""
    pages: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start.strip())
            end = int(raw_end.strip())
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError(f"Invalid page number: {part}")
            pages.add(page)
    return sorted(pages)


def _chunk_bounds(
    page_number: int,
    chunk_size: int,
    total_pages: int | None = None,
) -> tuple[int, int, int]:
    zero_based = page_number - 1
    start = (zero_based // chunk_size) * chunk_size
    end = start + chunk_size
    if total_pages is not None:
        end = min(end, total_pages)
    return start, end, zero_based - start


def _checkpoint_path(
    work_dir: str,
    page_number: int,
    chunk_size: int,
    total_pages: int | None = None,
) -> tuple[str, int, int]:
    start, end, index = _chunk_bounds(page_number, chunk_size, total_pages)
    return os.path.join(work_dir, f"chunk_{start}_{end}.pdf.json"), index, end - start


def _write_single_page_pdf(source: Any, page_number: int, output_pdf: str) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for MinerU partial reruns")

    if page_number < 1 or page_number > len(source):
        raise ValueError(
            f"Page {page_number} is outside PDF page range 1-{len(source)}"
        )
    single = fitz.open()
    try:
        page_index = page_number - 1
        single.insert_pdf(source, from_page=page_index, to_page=page_index)
        for page in single:
            mineru_api._add_ocr_guard_bands(page)
        single.save(output_pdf)
    finally:
        single.close()


def _first_layout_page(result: dict[str, Any]) -> dict[str, Any]:
    layouts = result.get("result", {}).get("layoutParsingResults", [])
    if not layouts:
        raise RuntimeError("MinerU rerun returned no layoutParsingResults")
    return layouts[0]


_CHAPTERISH_RE = re.compile(r"第[零一二三四五六七八九十百千0-9]+[章節编編篇卷]")
_CJK_CHAR_RE = re.compile(r"[㐀-鿿]")


def _strip_single_page_running_header(
    replacement: dict[str, Any],
    original: dict[str, Any],
) -> dict[str, Any]:
    """Remove a MinerU single-page running H1 when the original page lacked it."""
    text = replacement.get("markdown", {}).get("text")
    if not isinstance(text, str):
        return replacement
    match = re.match(r"^\s*#\s+([^\n]{1,30})\n{2,}(.*)$", text, re.DOTALL)
    if not match:
        return replacement

    header = match.group(1).strip()

    # Always preserve chapter-like headings regardless of whether the original
    # OCR was readable enough to contain the heading text for confirmation.
    if _CHAPTERISH_RE.search(header):
        return replacement

    original_text = str(original.get("markdown", {}).get("text", ""))
    original_start = re.sub(r"\s+", "", original_text[:120])
    normalized_header = re.sub(r"\s+", "", header)
    if normalized_header and normalized_header in original_start:
        return replacement

    patched = dict(replacement)
    markdown = dict(patched.get("markdown", {}))
    markdown["text"] = match.group(2).strip()
    patched["markdown"] = markdown
    return patched


def _build_checkpoint_cjk_model(
    layouts: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    chars: list[str] = []
    for page in layouts:
        text = page.get("markdown", {}).get("text", "")
        if isinstance(text, str):
            chars.extend(_cjk_chars(text))
    char_freq = Counter(chars)
    bigram_freq = Counter(
        chars[index] + chars[index + 1]
        for index in range(len(chars) - 1)
    )
    return char_freq, bigram_freq


def _diff_hunks(
    original_text: str,
    replacement_text: str,
) -> list[tuple[int, int, int, int]]:
    """Return changed text hunks, merging tiny equal islands inside a change."""
    opcodes = SequenceMatcher(
        None, original_text, replacement_text, autojunk=False,
    ).get_opcodes()
    hunks: list[tuple[int, int, int, int]] = []
    current: list[int] | None = None

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            equal_cjk_len = len(_cjk_chars(original_text[i1:i2]))
            if current is not None and equal_cjk_len < 8:
                current[1] = i2
                current[3] = j2
            elif current is not None:
                hunks.append((current[0], current[1], current[2], current[3]))
                current = None
            continue

        if current is None:
            current = [i1, i2, j1, j2]
        else:
            current[1] = i2
            current[3] = j2

    if current is not None:
        hunks.append((current[0], current[1], current[2], current[3]))

    return hunks


def _should_patch_hunk(
    original_span: str,
    replacement_span: str,
    char_freq: Counter[str],
    bigram_freq: Counter[str],
) -> bool:
    original_cjk_len = len(_cjk_chars(original_span))
    replacement_cjk_len = len(_cjk_chars(replacement_span))
    if original_cjk_len < 10 or replacement_cjk_len < 8:
        return False

    original_score, _ = _score_fragment(original_span, char_freq, bigram_freq)
    if original_score < 0.68:
        return False

    replacement_score, replacement_reasons = _score_fragment(
        replacement_span,
        char_freq,
        bigram_freq,
    )
    if replacement_score >= 0.68:
        return False

    replacement_chars = _cjk_chars(replacement_span)
    replacement_bigrams = [
        replacement_chars[index] + replacement_chars[index + 1]
        for index in range(len(replacement_chars) - 1)
    ]
    unseen_bigram_ratio = (
        sum(1 for bigram in replacement_bigrams if bigram_freq.get(bigram, 0) == 0)
        / max(1, len(replacement_bigrams))
    )
    if (
        unseen_bigram_ratio >= 0.85
        and replacement_reasons.get("rare_char_ratio", 0.0) >= 0.90
        and replacement_reasons.get("common_char_ratio", 1.0) <= 0.25
    ):
        return False
    return True


def _patch_suspicious_ocr_spans(
    original_page: dict[str, Any],
    replacement_page: dict[str, Any],
    layouts: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Patch only suspicious original spans, preserving the rest of the page."""
    original_text = original_page.get("markdown", {}).get("text", "")
    replacement_text = replacement_page.get("markdown", {}).get("text", "")
    if not isinstance(original_text, str) or not isinstance(replacement_text, str):
        return original_page, 0

    char_freq, bigram_freq = _build_checkpoint_cjk_model(layouts)
    output: list[str] = []
    last_original_index = 0
    patched_count = 0

    for i1, i2, j1, j2 in _diff_hunks(original_text, replacement_text):
        original_span = original_text[i1:i2]
        replacement_span = replacement_text[j1:j2]
        if _should_patch_hunk(original_span, replacement_span, char_freq, bigram_freq):
            output.append(original_text[last_original_index:i1])
            output.append(replacement_span)
            last_original_index = i2
            patched_count += 1

    if patched_count == 0:
        return original_page, 0

    output.append(original_text[last_original_index:])
    patched = dict(original_page)
    markdown = dict(patched.get("markdown", {}))
    markdown["text"] = "".join(output)
    patched["markdown"] = markdown
    return patched, patched_count


def _cjk_index_to_text_index(text: str, cjk_index: int) -> int:
    if cjk_index <= 0:
        return 0

    seen = 0
    for index, char in enumerate(text):
        if not _CJK_CHAR_RE.match(char):
            continue
        if seen == cjk_index:
            return index
        seen += 1
    return len(text)


def _find_delayed_page_start_alignment(
    original_cjk: str,
    replacement_cjk: str,
    *,
    min_prefix_cjk: int = 8,
    max_original_drop_cjk: int = 12,
    min_match_cjk: int = 14,
    match_cjk: int = 28,
    min_similarity: float = 0.84,
) -> tuple[int, int, float] | None:
    """Return (replacement_offset, original_offset, similarity)."""
    original_head = original_cjk[:160]
    replacement_head = replacement_cjk[:260]
    best: tuple[int, int, float] | None = None

    for original_offset in range(0, min(max_original_drop_cjk, len(original_head)) + 1):
        target_len = min(match_cjk, len(original_head) - original_offset)
        if target_len < min_match_cjk:
            continue
        target = original_head[original_offset:original_offset + target_len]
        upper = len(replacement_head) - target_len + 1
        for replacement_offset in range(min_prefix_cjk, max(min_prefix_cjk, upper)):
            sample = replacement_head[replacement_offset:replacement_offset + target_len]
            similarity = SequenceMatcher(
                None, target, sample, autojunk=False,
            ).ratio()
            if similarity < min_similarity:
                continue
            if best is None or (
                similarity,
                replacement_offset,
                -original_offset,
            ) > (
                best[2],
                best[0],
                -best[1],
            ):
                best = (replacement_offset, original_offset, round(similarity, 3))

    return best


def _patch_missing_page_start(
    original_page: dict[str, Any],
    replacement_page: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Insert a MinerU-detected page-start prefix into a checkpoint page."""
    original_text = original_page.get("markdown", {}).get("text", "")
    replacement_text = replacement_page.get("markdown", {}).get("text", "")
    if not isinstance(original_text, str) or not isinstance(replacement_text, str):
        return original_page, 0

    original_cjk = "".join(_cjk_chars(original_text))
    replacement_cjk = "".join(_cjk_chars(replacement_text))
    if len(original_cjk) < 20 or len(replacement_cjk) < 30:
        return original_page, 0

    alignment = _find_delayed_page_start_alignment(original_cjk, replacement_cjk)
    if alignment is None:
        return original_page, 0

    replacement_offset, original_offset, _similarity = alignment
    replacement_cut = _cjk_index_to_text_index(replacement_text, replacement_offset)
    original_cut = _cjk_index_to_text_index(original_text, original_offset)
    prefix = replacement_text[:replacement_cut].rstrip()
    suffix = original_text[original_cut:].lstrip()
    if len(_cjk_chars(prefix)) < 8 or not suffix:
        return original_page, 0

    patched = dict(original_page)
    markdown = dict(patched.get("markdown", {}))
    markdown["text"] = prefix + suffix
    patched["markdown"] = markdown
    return patched, 1


def _replace_checkpoint_page(
    checkpoint_path: str,
    page_index: int,
    expected_page_count: int,
    page_number: int,
    rerun_result: dict[str, Any],
    *,
    replace_page: bool = False,
    patch_page_start: bool = False,
) -> MineruRerunSummary:
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    layouts = checkpoint.get("result", {}).get("layoutParsingResults", [])
    if len(layouts) != expected_page_count:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has {len(layouts)} parsed pages, "
            f"which does not match physical chunk length {expected_page_count}; "
            "cannot safely map PDF page numbers to checkpoint indexes"
        )
    if page_index < 0 or page_index >= len(layouts):
        raise IndexError(
            f"Checkpoint {checkpoint_path} has {len(layouts)} pages; "
            f"cannot replace index {page_index}"
        )

    original_page = layouts[page_index]
    rerun_page = _strip_single_page_running_header(
        _first_layout_page(rerun_result),
        original_page,
    )
    if replace_page:
        replacement = rerun_page
        patched_span_count: int | None = None
        mode = "replace_page"
    elif patch_page_start:
        replacement, patched_span_count = _patch_missing_page_start(
            original_page,
            rerun_page,
        )
        mode = "patch_missing_page_start"
    else:
        replacement, patched_span_count = _patch_suspicious_ocr_spans(
            original_page,
            rerun_page,
            layouts,
        )
        mode = "patch_suspicious_spans"
    layouts[page_index] = replacement

    original_zip_url = checkpoint.pop("_mineru_zip_url", None)
    if original_zip_url and "_mineru_original_zip_url" not in checkpoint:
        checkpoint["_mineru_original_zip_url"] = original_zip_url

    zip_url = rerun_result.get("_mineru_zip_url")
    reruns = [
        item for item in checkpoint.get("_mineru_partial_reruns", [])
        if item.get("page") != page_number
    ]
    entry = {
        "page": page_number,
        "mode": mode,
    }
    if patched_span_count is not None:
        entry["patched_spans"] = patched_span_count
    if zip_url:
        entry["zip_url"] = zip_url
    reruns.append(entry)
    checkpoint["_mineru_partial_reruns"] = reruns

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    with tempfile.NamedTemporaryFile(
        "w", dir=checkpoint_dir, suffix=".tmp", delete=False, encoding="utf-8"
    ) as tf:
        tmp_name = tf.name
        try:
            json.dump(checkpoint, tf, ensure_ascii=False, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    os.replace(tmp_name, checkpoint_path)

    text = replacement.get("markdown", {}).get("text", "")
    preview = " ".join(str(text).split())[:120]
    return MineruRerunSummary(
        page_number=page_number,
        checkpoint_path=checkpoint_path,
        page_index=page_index,
        text_preview=preview,
    )


def rerun_mineru_pages(
    pdf_path: str,
    work_dir: str,
    *,
    pages: Iterable[int],
    token: str | None = None,
    chunk_size: int | None = None,
    replace_page: bool = False,
    patch_page_start: bool = False,
) -> List[MineruRerunSummary]:
    """Re-run MinerU for selected 1-based PDF pages and update checkpoints."""
    if token is None:
        token = MINERU_API_TOKEN
    if not token:
        raise RuntimeError("MINERU_API_TOKEN is not set")
    if chunk_size is None:
        chunk_size = MINERU_CHUNK_SIZE
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if replace_page and patch_page_start:
        raise ValueError("replace_page and patch_page_start are mutually exclusive")

    if fitz is None:
        raise RuntimeError("PyMuPDF is required for MinerU partial reruns")

    page_numbers = sorted(set(int(page) for page in pages))
    summaries: list[MineruRerunSummary] = []
    with tempfile.TemporaryDirectory(prefix="mineru_rerun_") as td:
        source = fitz.open(pdf_path)
        try:
            total_pages = len(source)
            for page_number in page_numbers:
                if page_number <= 0:
                    raise ValueError(f"Invalid page number: {page_number}")
                checkpoint_path, page_index, expected_page_count = _checkpoint_path(
                    work_dir, page_number, chunk_size, total_pages
                )
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"Checkpoint not found for page {page_number}: {checkpoint_path}"
                    )

                single_page_pdf = os.path.join(td, f"page_{page_number}.pdf")
                _write_single_page_pdf(source, page_number, single_page_pdf)
                rerun_result = mineru_api.parse_pdf_chunk(single_page_pdf, token)
                if rerun_result is None:
                    raise RuntimeError(f"MinerU rerun failed for page {page_number}")

                summaries.append(
                    _replace_checkpoint_page(
                        checkpoint_path,
                        page_index,
                        expected_page_count,
                        page_number,
                        rerun_result,
                        replace_page=replace_page,
                        patch_page_start=patch_page_start,
                    )
                )
        finally:
            source.close()

    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-run MinerU OCR for selected PDF pages and update checkpoints."
    )
    parser.add_argument("pdf", help="Original input PDF")
    parser.add_argument("--work-dir", required=True, help="Existing pipeline work dir")
    parser.add_argument("--pages", required=True,
                        help="1-based pages, e.g. 71,75-79,82")
    parser.add_argument("--chunk-size", type=int, default=MINERU_CHUNK_SIZE,
                        help=f"Original MinerU chunk size (default: {MINERU_CHUNK_SIZE})")
    parser.add_argument("--replace-page", action="store_true",
                        help="Replace the whole checkpoint page instead of only suspicious OCR spans")
    parser.add_argument("--patch-page-start", action="store_true",
                        help="Patch missing page-start text from MinerU while preserving the rest of the page")
    args = parser.parse_args(argv)

    pages = parse_page_ranges(args.pages)
    summaries = rerun_mineru_pages(
        args.pdf,
        args.work_dir,
        pages=pages,
        chunk_size=args.chunk_size,
        replace_page=args.replace_page,
        patch_page_start=args.patch_page_start,
    )
    for item in summaries:
        print(
            f"[*] Updated PDF page {item.page_number} "
            f"in {Path(item.checkpoint_path).name} index {item.page_index}"
        )
        if item.text_preview:
            print(f"    {item.text_preview}")


if __name__ == "__main__":
    main()
