"""Review OCR checkpoint page boundaries for likely missing page-start text."""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import xml.etree.ElementTree as ET

from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable, List
from zipfile import BadZipFile, ZipFile

from .ocr_review import _cjk_chars


_CHECKPOINT_RE = re.compile(r"^chunk_(\d+)_(\d+)\.pdf\.json$")
_CJK_RUN_RE = re.compile(r"[㐀-鿿]+")
_IMAGE_TAG_RE = re.compile(r"(?is)<img\b")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_TRAILING_NOTE_RE = re.compile(r"[\s\d０-９¹²³⁴⁵⁶⁷⁸⁹⁰]+$")
_TERMINAL_PUNCT = set("。！？!?；;")
_CLOSING_PUNCT = set("」』”’）》】〕〉》")
_OPEN_END_CHARS = set(
    "的之其此該将將把被為爲在向對与與和及同跟從由到以可會能要想令讓使"
    "仍就又但而或因於于給给受成有無无不未正若如當当並并則则"
)
_OPEN_TAIL_SUFFIXES = (
    "長相",
    "即令",
    "即使",
    "雖然",
    "因為",
    "由於",
    "關於",
    "對於",
    "表示",
    "指出",
    "認為",
    "主張",
    "要求",
    "不但",
    "但是",
    "可是",
)
_NEW_SENTENCE_STARTERS = set(
    "蔣毛孫宋史羅斯美中蘇俄日英法德國共黨他她其這那同可但由因在"
)


def _checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    match = _CHECKPOINT_RE.match(os.path.basename(path))
    if not match:
        return (10**9, 10**9, path)
    return (int(match.group(1)), int(match.group(2)), path)


def _first_cjk_index(text: str) -> int:
    match = _CJK_RUN_RE.search(text or "")
    return match.start() if match else -1


def _blank_match(match: re.Match[str]) -> str:
    return " " * (match.end() - match.start())


def _visible_text_without_markup(text: str) -> str:
    text = _MARKDOWN_IMAGE_RE.sub(_blank_match, text or "")
    return _HTML_TAG_RE.sub(_blank_match, text)


def _page_starts_with_image(text: str) -> bool:
    """Return True when image markup appears before the first CJK text."""
    text = text or ""
    image_starts = [match.start() for match in _IMAGE_TAG_RE.finditer(text)]
    image_starts.extend(match.start() for match in _MARKDOWN_IMAGE_RE.finditer(text))
    if not image_starts:
        return False
    first_cjk = _first_cjk_index(_visible_text_without_markup(text))
    return first_cjk < 0 or min(image_starts) < first_cjk


def iter_checkpoint_pages(work_dir: str) -> Iterable[dict[str, Any]]:
    """Yield page records from Paddle/MinerU checkpoint JSON files."""
    json_paths = [
        os.path.join(work_dir, name)
        for name in os.listdir(work_dir)
        if _CHECKPOINT_RE.match(name)
    ]
    for json_path in sorted(json_paths, key=_checkpoint_sort_key):
        match = _CHECKPOINT_RE.match(os.path.basename(json_path))
        if not match:
            continue
        chunk_start = int(match.group(1))
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        layouts = data.get("result", {}).get("layoutParsingResults", [])
        for index, page in enumerate(layouts):
            markdown = page.get("markdown", {})
            if not isinstance(markdown, dict):
                markdown = {}
            text = markdown.get("text", "")
            if not isinstance(text, str):
                text = ""
            images = markdown.get("images", {})
            has_images = (
                isinstance(images, dict) and bool(images)
            ) or bool(_IMAGE_TAG_RE.search(text) or _MARKDOWN_IMAGE_RE.search(text))
            yield {
                "page_number": chunk_start + index + 1,
                "checkpoint_path": json_path,
                "page_index": index,
                "text": text,
                "has_images": has_images,
                "starts_with_image": _page_starts_with_image(text),
            }


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _significant_tail(text: str) -> str:
    tail = _normalized_text(text)
    tail = _TRAILING_NOTE_RE.sub("", tail)
    while tail and tail[-1] in _CLOSING_PUNCT:
        tail = tail[:-1]
        tail = _TRAILING_NOTE_RE.sub("", tail)
    return tail


def _text_excerpt(text: str, *, head: bool, length: int = 42) -> str:
    text = _visible_text_without_markup(text)
    compact = re.sub(r"\s+", " ", html.unescape(text)).strip()
    if len(compact) <= length:
        return compact
    if head:
        return compact[:length]
    return compact[-length:]


def _first_cjk(text: str) -> str | None:
    chars = _cjk_chars(text)
    return chars[0] if chars else None


def _last_cjk(text: str) -> str | None:
    chars = _cjk_chars(text)
    return chars[-1] if chars else None


def _checkpoint_bigram_freq(pages: list[dict[str, Any]]) -> Counter[str]:
    chars: list[str] = []
    for page in pages:
        chars.extend(_cjk_chars(str(page.get("text", ""))))
    return Counter(chars[index] + chars[index + 1] for index in range(len(chars) - 1))


def _page_head_singleton_bigram_ratio(
    text: str,
    bigram_freq: Counter[str],
    *,
    max_chars: int = 18,
) -> float:
    """Return singleton-bigram ratio for the first CJK run on a page.

    Short semantic garbage immediately after an image is a common failure mode
    on vertical OCR pages: the characters may be common individually, but the
    local bigrams are unique in the book.
    """
    match = _CJK_RUN_RE.search(_visible_text_without_markup(text))
    if not match:
        return 0.0
    chars = list(match.group(0)[:max_chars])
    if len(chars) < 8:
        return 0.0
    bigrams = [chars[index] + chars[index + 1] for index in range(len(chars) - 1)]
    if not bigrams:
        return 0.0
    singletons = sum(1 for bigram in bigrams if bigram_freq.get(bigram, 0) <= 1)
    return round(singletons / len(bigrams), 3)


def _score_boundary(
    previous_text: str,
    next_text: str,
    bigram_freq: Counter[str],
    *,
    previous_has_images: bool = False,
    next_has_images: bool = False,
    next_starts_with_image: bool = False,
) -> tuple[float, list[str]]:
    tail = _significant_tail(previous_text)
    if not tail:
        return 0.0, []

    if tail[-1] in _TERMINAL_PUNCT:
        return 0.0, []

    previous_cjk = _last_cjk(tail)
    next_cjk = _first_cjk(next_text)
    if previous_cjk is None or next_cjk is None:
        return 0.0, []

    score = 0.40
    reasons = ["open_tail_no_terminal"]

    if any(tail.endswith(suffix) for suffix in _OPEN_TAIL_SUFFIXES):
        score += 0.30
        reasons.append("open_tail_suffix")
    elif previous_cjk in _OPEN_END_CHARS:
        score += 0.15
        reasons.append("open_tail_char")

    bridge = previous_cjk + next_cjk
    bridge_count = bigram_freq.get(bridge, 0)
    if bridge_count <= 1:
        score += 0.15
        reasons.append("rare_boundary_bigram")

    if next_cjk in _NEW_SENTENCE_STARTERS:
        score += 0.05
        reasons.append("new_sentence_like_head")

    image_adjacent = previous_has_images or next_has_images
    if image_adjacent:
        reasons.append("image_adjacent_boundary")
        score += 0.05

    if next_starts_with_image:
        score += 0.15
        reasons.append("next_page_starts_with_image")

    head_singleton_ratio = _page_head_singleton_bigram_ratio(
        next_text,
        bigram_freq,
    )
    if next_starts_with_image and head_singleton_ratio >= 0.75:
        score += 0.25
        reasons.append("garbled_page_head")
    elif image_adjacent and head_singleton_ratio >= 0.90:
        score += 0.15
        reasons.append("garbled_page_head")

    return min(round(score, 3), 1.0), reasons


def find_page_boundary_candidates(
    work_dir: str,
    *,
    min_score: float = 0.70,
    limit: int = 80,
) -> List[dict[str, Any]]:
    """Return ranked checkpoint page boundaries that need visual/OCR review.

    The scanner is intentionally conservative about changing text: it only
    ranks boundaries where the previous page ends without sentence punctuation.
    Human review, rendered PDF evidence, or a second OCR pass is still required
    before patching prose.
    """
    pages = list(iter_checkpoint_pages(work_dir))
    if len(pages) < 2:
        return []

    bigram_freq = _checkpoint_bigram_freq(pages)
    candidates: list[dict[str, Any]] = []
    for previous, current in zip(pages, pages[1:]):
        if current["page_number"] != previous["page_number"] + 1:
            continue
        score, reasons = _score_boundary(
            previous["text"],
            current["text"],
            bigram_freq,
            previous_has_images=bool(previous.get("has_images")),
            next_has_images=bool(current.get("has_images")),
            next_starts_with_image=bool(current.get("starts_with_image")),
        )
        if score < min_score:
            continue
        candidates.append(
            {
                "previous_page": previous["page_number"],
                "next_page": current["page_number"],
                "score": score,
                "reasons": reasons,
                "tail": _text_excerpt(previous["text"], head=False),
                "head": _text_excerpt(current["text"], head=True),
                "previous_checkpoint": previous["checkpoint_path"],
                "next_checkpoint": current["checkpoint_path"],
                "previous_page_index": previous["page_index"],
                "next_page_index": current["page_index"],
                "previous_has_images": bool(previous.get("has_images")),
                "next_has_images": bool(current.get("has_images")),
                "next_starts_with_image": bool(current.get("starts_with_image")),
                "head_singleton_bigram_ratio": _page_head_singleton_bigram_ratio(
                    current["text"],
                    bigram_freq,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (-item["score"], item["previous_page"], item["next_page"])
    )
    return candidates[:limit]


def _cjk_compact(text: str) -> str:
    return "".join(_CJK_RUN_RE.findall(text or ""))


def compare_page_start_ocr(
    checkpoint_head: str,
    second_ocr_text: str,
    *,
    window: int = 24,
    max_search: int = 140,
    min_offset: int = 8,
    min_similarity: float = 0.72,
) -> dict[str, Any]:
    """Compare checkpoint page start against a second OCR page reading.

    If the best fuzzy match for the checkpoint head starts well after the
    second OCR's beginning, the checkpoint likely omitted page-start text.
    """
    target = _cjk_compact(checkpoint_head)[:window]
    observed = _cjk_compact(second_ocr_text)[:max_search]
    if not target or not observed:
        return {
            "possible_omission": False,
            "second_ocr_offset": -1,
            "similarity": 0.0,
            "window": target,
            "second_ocr_head": observed[:window],
        }

    best_offset = 0
    best_similarity = -1.0
    best_window = target
    min_window = min(len(target), max(8, window // 2))
    for window_len in range(len(target), min_window - 1, -1):
        target_window = target[:window_len]
        upper = max(1, min(len(observed), max_search) - window_len + 1)
        for offset in range(upper):
            sample = observed[offset:offset + window_len]
            similarity = SequenceMatcher(
                None, target_window, sample, autojunk=False,
            ).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_offset = offset
                best_window = target_window

    best_similarity = round(best_similarity, 3)
    return {
        "possible_omission": (
            best_offset >= min_offset and best_similarity >= min_similarity
        ),
        "second_ocr_offset": best_offset,
        "similarity": best_similarity,
        "window": best_window,
        "second_ocr_head": observed[:window],
    }


def _extract_xhtml_text(raw: bytes) -> str:
    text = raw.decode("utf-8-sig", "ignore")
    try:
        root = ET.fromstring(text)
        return "".join(root.itertext())
    except ET.ParseError:
        return html.unescape(re.sub(r"<[^>]+>", " ", text))


def _resolve_posix_path(base_path: str, href: str) -> str:
    base_dir = posixpath.dirname(base_path)
    return posixpath.normpath(posixpath.join(base_dir, href))


def _epub_spine_documents(archive: ZipFile) -> list[str]:
    names = set(archive.namelist())
    opf_path: str | None = None
    container_path = "META-INF/container.xml"
    if container_path in names:
        try:
            root = ET.fromstring(archive.read(container_path))
            rootfile = root.find(".//{*}rootfile")
            if rootfile is not None:
                opf_path = rootfile.attrib.get("full-path")
        except ET.ParseError:
            opf_path = None

    if not opf_path:
        opf_path = next((name for name in archive.namelist() if name.endswith(".opf")), None)
    if not opf_path or opf_path not in names:
        return []

    try:
        opf_root = ET.fromstring(archive.read(opf_path))
    except ET.ParseError:
        return []

    manifest: dict[str, str] = {}
    for item in opf_root.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = _resolve_posix_path(opf_path, href)

    ordered: list[str] = []
    for itemref in opf_root.findall(".//{*}spine/{*}itemref"):
        href = manifest.get(itemref.attrib.get("idref", ""))
        if href and href in names:
            ordered.append(href)
    return ordered


def _epub_content_documents(archive: ZipFile) -> list[str]:
    spine_docs = _epub_spine_documents(archive)
    if spine_docs:
        return [
            name for name in spine_docs
            if os.path.basename(name).lower() not in {"nav.xhtml", "nav.html", "toc.ncx"}
        ]

    return sorted(
        name for name in archive.namelist()
        if name.lower().endswith((".xhtml", ".html"))
        and os.path.basename(name).lower() not in {"nav.xhtml", "nav.html"}
    )


def _epub_cjk_text(epub_path: str) -> str:
    with ZipFile(epub_path) as archive:
        texts = [
            _extract_xhtml_text(archive.read(name))
            for name in _epub_content_documents(archive)
        ]
    return _cjk_compact("".join(texts))


def annotate_candidates_with_epub_continuity(
    candidates: List[dict[str, Any]],
    epub_path: str,
    *,
    window: int = 18,
) -> List[dict[str, Any]]:
    """Annotate whether each checkpoint boundary join appears in the EPUB.

    A present checkpoint join only proves the EPUB contains what the checkpoint
    contains. It does not prove the PDF page start was recognized correctly.
    """
    try:
        epub_text = _epub_cjk_text(epub_path)
        epub_error = None
    except (BadZipFile, OSError, KeyError) as exc:
        epub_text = ""
        epub_error = str(exc)

    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        tail_probe = _cjk_compact(str(item.get("tail", "")))[-window:]
        head_probe = _cjk_compact(str(item.get("head", "")))[:window]
        probe = tail_probe + head_probe
        item["epub_probe_tail"] = tail_probe
        item["epub_probe_head"] = head_probe
        item["epub_probe"] = probe

        if epub_error is not None:
            item["epub_status"] = "epub_unavailable"
            item["epub_error"] = epub_error
            item["epub_boundary_present"] = False
            item["epub_match_index"] = -1
            item["epub_boundary_window"] = ""
        elif len(tail_probe) < 4 or len(head_probe) < 4:
            item["epub_status"] = "insufficient_boundary_text"
            item["epub_boundary_present"] = False
            item["epub_match_index"] = -1
            item["epub_boundary_window"] = ""
        else:
            match_index = epub_text.find(probe)
            item["epub_match_index"] = match_index
            item["epub_boundary_present"] = match_index >= 0
            if match_index >= 0:
                start = max(0, match_index - window)
                end = min(len(epub_text), match_index + len(probe) + window)
                item["epub_status"] = "checkpoint_boundary_present"
                item["epub_boundary_window"] = epub_text[start:end]
            else:
                item["epub_status"] = "checkpoint_boundary_missing_from_epub"
                item["epub_boundary_window"] = ""

        annotated.append(item)

    return annotated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank OCR checkpoint page boundaries that may hide missing text."
    )
    parser.add_argument("work_dir", help="Pipeline work directory with chunk_*.pdf.json")
    parser.add_argument("--limit", type=int, default=80, help="Maximum candidates")
    parser.add_argument("--min-score", type=float, default=0.70,
                        help="Minimum boundary suspicion score, 0-1")
    parser.add_argument("--epub", default=None,
                        help="Annotate whether checkpoint joins appear in this EPUB")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    candidates = find_page_boundary_candidates(
        args.work_dir,
        min_score=args.min_score,
        limit=args.limit,
    )
    if args.epub:
        candidates = annotate_candidates_with_epub_continuity(candidates, args.epub)
    if args.json:
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        return

    if not candidates:
        print("[*] No suspicious page-boundary candidates found.")
        return

    for index, item in enumerate(candidates, 1):
        print(
            f"{index:>3}. score={item['score']:.3f} "
            f"pages={item['previous_page']}->{item['next_page']} "
            f"reasons={','.join(item['reasons'])}"
        )
        print(f"     tail: {item['tail']}")
        print(f"     head: {item['head']}")
        if args.epub:
            print(f"     epub: {item['epub_status']}")


if __name__ == "__main__":
    main()
