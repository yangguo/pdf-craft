"""High-recall OCR review helpers for suspicious CJK text candidates."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import zipfile

from collections import Counter
from typing import Any, Dict, Iterable, List

from .config import EPUB_STRUCTURAL_FILES


_CJK_RE = re.compile(r"[㐀-鿿]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_TAG_RE = re.compile(r"(?is)<[^>]+>")

# A compact common-character set used only as one feature in a review score.
# It intentionally mixes simplified/traditional variants because OCR output can
# contain either form in otherwise traditional Chinese books.
_COMMON_CJK = frozenset(
    "的一是在不了有和人這这中大為为上個个國国我以要他時时來来用們们生到作地於于出就"
    "分對对成會会可主發发年動动同工也能下過过子說说產产種种面而方後后多定行學学"
    "法所民得經经十三之進进著着等部度家電电力裡里如水化高自二理起小物現现实加"
    "量都兩两體体制機机當当使點点從从業业本去把性好應应開开它合還还因由其些然"
    "前外天政四日那社義义事平形相全表間间樣样與与關关各重新線线內内數数正心反"
    "明看原又麼么利比或但質质氣气第向道命此變变條条只沒没結结解問问意建月公無"
    "无系軍军很情者最立代想已通並并提直題题黨党程展五果料象員员革位入常文總总"
    "次品式活設设及管特件長长求老頭头基資资邊边流路級级少圖图山統统接知較较將"
    "将組组見见計计別别她手角期根論论運运農农指幾几九區区強强放決决西被干做必"
    "戰战先回則则任取據据處处隊队南給给色光門门即保治北造百規规熱热領领七海口"
    "東东導导器壓压志世金增爭争濟济階阶油思術术極极交受聯联什認认六共權权收證"
    "证改清己美再採采轉转更單单風风切打白教速花帶带安場场身車车例真務务具萬万"
    "每目至達达走積积示議议聲声報报鬥斗完類类八離离華华名確确才科張张信馬马節"
    "节話话米整空元況况今集溫温傳传土許许步群廣广石記记需段研界拉林律叫且究觀"
    "观越織织裝装影算低持音眾众書书布復复容兒儿須须際际商非驗验連连斷断深難难"
    "近礦矿千周委素技備备半辦办青省列習习響响約约支般史感勞劳便團团往歷历市"
)


def _plain_text_from_html(html_text: str) -> str:
    plain = _TAG_RE.sub(" ", html_text)
    plain = html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def _iter_epub_texts(
    epub_path: str,
    structural_files: set[str],
) -> Iterable[tuple[str, str]]:
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if os.path.basename(lower) in structural_files:
                continue
            if not lower.endswith((".xhtml", ".html")):
                continue
            content = archive.read(name).decode("utf-8", "ignore")
            yield name, _plain_text_from_html(content)


def _cjk_chars(text: str) -> list[str]:
    return _CJK_RE.findall(text)


def _review_fragments(
    text: str,
    max_cjk: int,
) -> Iterable[str]:
    """Yield sentence-sized review fragments, falling back to CJK windows."""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        chars = _cjk_chars(sentence)
        if len(chars) <= max_cjk:
            yield sentence
            continue

        # Long paragraphs without useful punctuation still need review.  Use
        # overlapping CJK-only windows so the caller can rank suspicious spans.
        step = max(12, max_cjk // 2)
        for start in range(0, len(chars), step):
            window = chars[start:start + max_cjk]
            if len(window) < 12:
                break
            yield "".join(window)


def _longest_singleton_run(bigrams: list[str], bigram_freq: Counter[str]) -> int:
    longest = 0
    current = 0
    for bigram in bigrams:
        if bigram_freq.get(bigram, 0) == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _repetition_ratio(chars: list[str]) -> float:
    """Return the ratio of characters that are identical to their predecessor.

    A high ratio (≥ 0.80) with ≥ 8 CJK chars signals OCR hallucination —
    typically the same character repeated dozens or thousands of times
    (e.g. ``三三三三三三…``), which normal bigram analysis misses because
    ``三三`` is a perfectly common bigram in a history book.
    """
    if len(chars) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i - 1])
    return repeats / (len(chars) - 1)


def _score_fragment(
    fragment: str,
    char_freq: Counter[str],
    bigram_freq: Counter[str],
) -> tuple[float, Dict[str, float]]:
    chars = _cjk_chars(fragment)
    if len(chars) < 2:
        return 0.0, {}

    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    bigram_count = max(1, len(bigrams))
    singleton_ratio = (
        sum(1 for bigram in bigrams if bigram_freq.get(bigram, 0) == 1)
        / bigram_count
    )
    rare_char_ratio = (
        sum(1 for char in chars if char_freq.get(char, 0) <= 2)
        / len(chars)
    )
    common_char_ratio = sum(1 for char in chars if char in _COMMON_CJK) / len(chars)
    anchor_ratio = (
        sum(1 for bigram in bigrams if bigram_freq.get(bigram, 0) >= 3)
        / bigram_count
    )
    singleton_run_ratio = (
        _longest_singleton_run(bigrams, bigram_freq) / bigram_count
    )
    repetition_ratio = _repetition_ratio(chars)

    score = (
        0.44 * singleton_ratio
        + 0.22 * rare_char_ratio
        + 0.20 * singleton_run_ratio
        + 0.14 * (1.0 - common_char_ratio)
        - 0.25 * anchor_ratio
    )
    # Short-span singleton-density boost: a compact span (8-25 CJK chars)
    # with a very high singleton-bigram ratio is a strong garbled signal
    # even when the characters themselves are common (e.g. OCR hallucination
    # fragments like 居之者忽音但一通乃一祔一筆八).  The bonus scales
    # from 0 at 75 % singleton ratio to ≈ 0.30 at 100 %.
    cjk_len = len(chars)
    if 8 <= cjk_len <= 25 and singleton_ratio >= 0.75:
        density_bonus = (singleton_ratio - 0.75) * 1.2
        score += density_bonus
    # Repetition boost: a span where ≥ 80 % of characters are identical to
    # their predecessor is almost certainly OCR noise (e.g. ``三三三三…``).
    # Normal Chinese never has this many consecutive repeats.  The bonus
    # scales from 0 at 80 % repetition to 0.50 at 100 %.
    if cjk_len >= 8 and repetition_ratio >= 0.80:
        rep_bonus = (repetition_ratio - 0.80) * 2.5
        score += rep_bonus
    # Dedicated catch for highly-repetitive common-character spans: even
    # if ``三`` is common and ``三三`` is a valid bigram, a span of ≥ 20
    # consecutive same-char repeats is NEVER legitimate Chinese text.
    if cjk_len >= 20 and repetition_ratio >= 0.95:
        score += 0.30
    score = max(0.0, min(1.0, score))
    reasons = {
        "singleton_bigram_ratio": round(singleton_ratio, 3),
        "rare_char_ratio": round(rare_char_ratio, 3),
        "singleton_run_ratio": round(singleton_run_ratio, 3),
        "common_char_ratio": round(common_char_ratio, 3),
        "anchor_bigram_ratio": round(anchor_ratio, 3),
        "repetition_ratio": round(repetition_ratio, 3),
    }
    return score, reasons


def find_suspicious_cjk_spans_in_epub(
    epub_path: str,
    *,
    structural_files: set[str] | None = None,
    limit: int = 50,
    min_score: float = 0.70,
    min_cjk: int = 10,
    max_cjk: int = 90,
) -> List[Dict[str, Any]]:
    """Return ranked suspicious CJK OCR candidates for human review.

    This scanner is intentionally separate from strict EPUB validation.  It is
    a high-recall review tool: candidates should be checked visually or by a
    second OCR pass before changing book text.
    """
    if structural_files is None:
        structural_files = EPUB_STRUCTURAL_FILES

    file_texts = list(_iter_epub_texts(epub_path, structural_files))
    all_chars: list[str] = []
    for _, text in file_texts:
        all_chars.extend(_cjk_chars(text))
    if len(all_chars) < min_cjk:
        return []

    char_freq = Counter(all_chars)
    bigram_freq: Counter[str] = Counter()
    for index in range(len(all_chars) - 1):
        bigram_freq[all_chars[index] + all_chars[index + 1]] += 1

    candidates_by_key: dict[tuple[str, str], Dict[str, Any]] = {}
    for name, text in file_texts:
        for fragment in _review_fragments(text, max_cjk=max_cjk):
            chars = _cjk_chars(fragment)
            if len(chars) < min_cjk:
                continue
            score, reasons = _score_fragment(fragment, char_freq, bigram_freq)
            if score < min_score:
                continue
            excerpt = re.sub(r"\s+", " ", fragment).strip()
            key = (name, excerpt)
            candidate = {
                "file": name,
                "token": "suspicious CJK OCR candidate",
                "score": round(score, 3),
                "cjk_len": len(chars),
                "excerpt": excerpt,
                "reasons": reasons,
            }
            previous = candidates_by_key.get(key)
            if previous is None or candidate["score"] > previous["score"]:
                candidates_by_key[key] = candidate

    candidates = sorted(
        candidates_by_key.values(),
        key=lambda item: (-item["score"], item["file"], item["excerpt"]),
    )
    return candidates[:limit]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank suspicious CJK OCR spans in an EPUB for human review."
    )
    parser.add_argument("epub", help="Path to EPUB file")
    parser.add_argument("--limit", type=int, default=50, help="Maximum candidates")
    parser.add_argument("--min-score", type=float, default=0.70,
                        help="Minimum suspicion score, 0-1")
    parser.add_argument("--min-cjk", type=int, default=10,
                        help="Minimum CJK characters in a candidate")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    candidates = find_suspicious_cjk_spans_in_epub(
        args.epub,
        limit=args.limit,
        min_score=args.min_score,
        min_cjk=args.min_cjk,
    )
    if args.json:
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        return

    if not candidates:
        print("[*] No suspicious CJK OCR candidates found.")
        return

    for index, item in enumerate(candidates, 1):
        print(f"{index:>3}. score={item['score']:.3f} file={item['file']}")
        print(f"     {item['excerpt']}")
        print(f"     reasons={item['reasons']}")


if __name__ == "__main__":
    main()
