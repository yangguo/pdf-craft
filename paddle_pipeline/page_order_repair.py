"""Repair adjacent scanned-page inversions using printed page numbers."""

import re

from collections import Counter
from typing import Any, Dict, List, Optional, cast

from .config import fitz  # Optional dependency


_PRINTED_PAGE_PATTERNS = (
    re.compile(r"[•.·]\s*(\d{1,3})\s*[•.·]"),
    re.compile(r"\b(\d{1,3})\b\s*[•.·]"),
    re.compile(r"[•.·]\s*(\d{1,3})\b"),
)


def _extract_printed_page_number(text: str) -> Optional[int]:
    head = re.sub(r"\s+", " ", text[:300])
    for pattern in _PRINTED_PAGE_PATTERNS:
        for match in pattern.finditer(head):
            number = int(match.group(1))
            if 1 <= number <= 500:
                return number
    return None


def _printed_page_numbers(pdf_path: str) -> List[Optional[int]]:
    if fitz is None:
        return []

    doc = fitz.open(pdf_path)
    try:
        return [
            _extract_printed_page_number(cast(Any, doc[index]).get_text())
            for index in range(doc.page_count)
        ]
    finally:
        doc.close()


def _flatten_layout_pages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages = []
    for result in results:
        pages.extend(result.get("result", {}).get("layoutParsingResults", []))
    return pages


def _write_flattened_pages(results: List[Dict[str, Any]],
                           pages: List[Dict[str, Any]]) -> None:
    offset = 0
    for result in results:
        layout_results = result.get("result", {}).get("layoutParsingResults", [])
        count = len(layout_results)
        layout_results[:] = pages[offset:offset + count]
        offset += count


def _adjacent_inversion_indices(numbers: List[int | None]) -> List[int]:
    indices = []
    for index in range(len(numbers) - 1):
        current = numbers[index]
        next_number = numbers[index + 1]
        if current is None or next_number is None:
            continue
        if current == next_number + 1:
            indices.append(index)
    return indices


def _infer_systematic_pair_inversions(
        numbers: List[int | None],
        inversion_indices: List[int]) -> set[int]:
    """Infer missing 3/2, 5/4-style swaps from detected odd/even pairs."""
    start_candidates = []
    matching_currents_by_start: Dict[int, List[int]] = {}
    for index in inversion_indices:
        current = numbers[index]
        next_number = numbers[index + 1]
        if current is None or next_number is None:
            continue
        if current < 3 or current % 2 == 0 or next_number != current - 1:
            continue

        pair_start = index - (current - 3)
        if pair_start < 0:
            continue

        start_candidates.append(pair_start)
        matching_currents_by_start.setdefault(pair_start, []).append(current)

    if len(start_candidates) < 2:
        return set()

    pair_start, evidence_count = Counter(start_candidates).most_common(1)[0]
    if evidence_count < 2 or evidence_count / len(start_candidates) < 0.6:
        return set()

    max_current = max(matching_currents_by_start[pair_start])
    last_pair_start = min(len(numbers) - 2, pair_start + max_current - 3)
    return set(range(pair_start, last_pair_start + 1, 2))


def _non_overlapping_swap_indices(indices: set[int], page_count: int) -> List[int]:
    selected = []
    previous = -2
    for index in sorted(indices):
        if index < 0 or index + 1 >= page_count:
            continue
        if index <= previous + 1:
            continue
        selected.append(index)
        previous = index
    return selected


def repair_page_order_by_printed_numbers(pdf_path: str,
                                         results: List[Dict[str, Any]]) -> int:
    """Swap adjacent OCR pages when printed page numbers show a one-page inversion."""
    pages = _flatten_layout_pages(results)
    numbers = _printed_page_numbers(pdf_path)[:len(pages)]
    if len(numbers) < 2:
        return 0

    inversion_indices = _adjacent_inversion_indices(numbers)
    swap_indices = set(inversion_indices)
    swap_indices.update(_infer_systematic_pair_inversions(numbers, inversion_indices))
    selected_swaps = _non_overlapping_swap_indices(swap_indices, len(pages))

    for index in selected_swaps:
        pages[index], pages[index + 1] = pages[index + 1], pages[index]
        numbers[index], numbers[index + 1] = numbers[index + 1], numbers[index]

    swaps = len(selected_swaps)

    if swaps:
        _write_flattened_pages(results, pages)
        print(f"[*] Repaired {swaps} adjacent scanned page-order inversion(s)")

    return swaps
