"""Semantic OCR review: detect garbled CJK text that statistical tools miss.

Default mode is offline: generates semantic-garble candidates from an EPUB
using sliding-window statistical analysis tuned for recall over precision.

LLM review is opt-in via ``--llm`` and uses an OpenAI-compatible chat
completions endpoint (configured from ``.env`` or environment variables).

``--deep`` adds sentence-level candidates that bypass statistical scoring
entirely, catching Tier-3 semantic garble where every individual bigram is
common but the sentence is nonsense (e.g. ``安事變凸顯出壽、罰之罰均固人關系…``).

Usage::

    # Offline candidate generation
    python3 -m paddle_pipeline.semantic_ocr_review jiang1.epub --json candidates.json

    # With LLM review (auto-lowers min_score to 0.30)
    python3 -m paddle_pipeline.semantic_ocr_review jiang1.epub --llm --json review.json

    # Deep scan: statistical windows + sentence-level chunks + LLM filter
    python3 -m paddle_pipeline.semantic_ocr_review jiang1.epub --llm --deep --only-garbled --json deep.json

    # Limit reviewed candidates; only show LLM positives
    python3 -m paddle_pipeline.semantic_ocr_review jiang1.epub --llm --limit 50 --only-garbled
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import zipfile

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .config import EPUB_STRUCTURAL_FILES
from .ocr_review import _cjk_chars, _score_fragment, _plain_text_from_html

_TAG_RE = re.compile(r"(?is)<[^>]+>")


# ---------------------------------------------------------------------------
# EPUB text extraction
# ---------------------------------------------------------------------------

def _iter_epub_texts(
    epub_path: str,
    structural_files: set[str],
) -> Iterable[tuple[str, str]]:
    """Yield ``(filename, plain_text)`` for every XHTML/HTML content file."""
    with zipfile.ZipFile(epub_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if os.path.basename(lower) in structural_files:
                continue
            if not lower.endswith((".xhtml", ".html")):
                continue
            content = archive.read(name).decode("utf-8", "ignore")
            yield name, _plain_text_from_html(content)


def _extract_file_texts(
    epub_path: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return ``[(filename, plain_text), …]`` and the full flattened CJK char list."""
    file_texts: list[tuple[str, str]] = []
    all_chars: list[str] = []
    for name, text in _iter_epub_texts(epub_path, EPUB_STRUCTURAL_FILES):
        file_texts.append((name, text))
        all_chars.extend(_cjk_chars(text))
    return file_texts, all_chars


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[。！？\n][」』）\)」]*")

_SENTENCE_END_RE = re.compile(r"[。！？]")

_CJK_RE = re.compile(r"[㐀-鿿]")


def _cjk_char_positions(text: str) -> list[tuple[str, int]]:
    """Return ``(char, byte_index)`` for every CJK character in *text*."""
    return [(m.group(0), m.start()) for m in _CJK_RE.finditer(text)]


def _iter_cjk_sentences(text: str, min_cjk: int = 10) -> Iterable[tuple[str, int, int]]:
    """Yield ``(sentence_text, byte_start, byte_end)`` for each CJK sentence.

    A "sentence" is a segment bounded by 。！？or \\n\\n that contains at least
    *min_cjk* CJK characters after stripping.
    """
    # Split by sentence-ending punctuation
    raw_parts = _SENTENCE_SPLIT_RE.split(text)
    # Also split long segments by double-newline
    parts: list[str] = []
    for part in raw_parts:
        if "\n\n" in part:
            parts.extend(p for p in part.split("\n\n") if p.strip())
        else:
            parts.append(part)

    byte_pos = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            byte_pos = text.find(part, byte_pos) + len(part) if part else byte_pos
            continue
        cjk_count = len(_CJK_RE.findall(stripped))
        if cjk_count < min_cjk:
            byte_pos = text.find(part, byte_pos) + len(part) if part else byte_pos
            continue
        # Find the segment in the original text
        idx = text.find(stripped, byte_pos)
        if idx < 0:
            byte_pos = text.find(part, byte_pos) + len(part) if part else byte_pos
            continue
        yield stripped, idx, idx + len(stripped)
        byte_pos = idx + len(stripped)


def _sentence_context(
    text: str,
    start_index: int,
    end_index: int,
    context_chars: int = 200,
) -> tuple[str, str]:
    """Return ``(context_before, context_after)`` for a sentence candidate."""
    before = text[max(0, start_index - context_chars) : start_index].strip()
    after = text[end_index : min(len(text), end_index + context_chars)].strip()
    return before, after


_FALSE_POSITIVE_MARKERS = (
    "University Press",
    "http://",
    "https://",
    "ISBN",
    "譯者",
)


def _looks_like_reference(text: str) -> bool:
    """Return True if *text* is likely a citation or reference entry."""
    return any(marker in text for marker in _FALSE_POSITIVE_MARKERS)


def _looks_like_punctuation_list(text: str) -> bool:
    """Return True if *text* is primarily a list of book titles or proper names."""
    # High ratio of 《》「」（） punctuation to regular text
    bracket_chars = sum(1 for ch in text if ch in "《》「」（）〝〞")
    cjk = len(_CJK_RE.findall(text))
    if cjk == 0:
        return True
    return bracket_chars / max(cjk, 1) > 0.3


def _candidate_context(
    text: str,
    start_index: int,
    end_index: int,
    before_chars: int = 80,
    after_chars: int = 80,
) -> tuple[str, str]:
    """Return ``(context_before, context_after)`` around *text* indices."""
    before = text[max(0, start_index - before_chars) : start_index].strip()
    after = text[end_index : min(len(text), end_index + after_chars)].strip()
    return before, after


def generate_sentence_candidates(
    epub_path: str,
    *,
    limit: int = 200,
    min_cjk: int = 10,
) -> list[dict[str, Any]]:
    """Return sentence-level candidates from *epub_path* for LLM review.

    These candidates bypass statistical scoring entirely — every paragraph
    or sentence with enough CJK characters is a candidate.  They only make
    sense when reviewed by an LLM afterwards (``--llm``).

    Candidates are collected from **all** files first, then shuffled and
    trimmed to *limit*, so every chapter contributes candidates.
    """
    file_texts, _ = _extract_file_texts(epub_path)
    per_file: list[list[dict[str, Any]]] = []
    seen = set()

    for filename, text in file_texts:
        file_candidates: list[dict[str, Any]] = []
        for sentence, byte_start, byte_end in _iter_cjk_sentences(text, min_cjk=min_cjk):
            if _looks_like_punctuation_list(sentence):
                continue
            if _looks_like_reference(sentence):
                continue

            s_clean = sentence.strip()
            if len(s_clean) < 6:
                continue

            key = (filename, s_clean[:60])
            if key in seen:
                continue
            seen.add(key)

            context_before, context_after = _sentence_context(
                text, byte_start, byte_end
            )
            cjk_count = len(_CJK_RE.findall(s_clean))
            file_candidates.append({
                "file": filename,
                "excerpt": re.sub(r"\s+", " ", s_clean),
                "llm_excerpt": re.sub(r"\s+", " ", s_clean),
                "context_before": context_before,
                "context_after": context_after,
                "score": 0.0,
                "features": {"source": "sentence_candidate"},
                "cjk_len": cjk_count,
                "source": "sentence_candidate",
            })
        per_file.append(file_candidates)

    # Round-robin interleave from each file so every chapter is represented,
    # then trim to limit.
    candidates: list[dict[str, Any]] = []
    max_per_file = max((len(pf) for pf in per_file), default=0)
    for i in range(max_per_file):
        for pf in per_file:
            if i < len(pf):
                candidates.append(pf[i])
                if len(candidates) >= limit:
                    return candidates

    return candidates


def generate_semantic_candidates(
    epub_path: str,
    *,
    limit: int = 100,
    min_score: float = 0.50,
    min_cjk: int = 14,
    window_sizes: Iterable[int] = (16, 24, 32),
) -> list[dict[str, Any]]:
    """Return ranked semantic-garble candidates from *epub_path*.

    Uses sliding CJK windows with scoring *tuned for recall*.
    These candidates need further review — either by an LLM or
    a human — before being treated as confirmed garble.
    """
    file_texts, all_chars = _extract_file_texts(epub_path)
    if len(all_chars) < min_cjk:
        return []

    char_freq = Counter(all_chars)
    bigram_freq = Counter(
        all_chars[i] + all_chars[i + 1] for i in range(len(all_chars) - 1)
    )

    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for filename, text in file_texts:
        positions = _cjk_char_positions(text)
        chars = [ch for ch, _ in positions]
        text_len = len(chars)

        for window_size in window_sizes:
            if text_len < window_size:
                continue
            step = max(4, window_size // 2)
            for start in range(0, text_len - window_size + 1, step):
                frag = "".join(chars[start : start + window_size])
                score, reasons = _score_fragment(frag, char_freq, bigram_freq)
                if score < min_score:
                    continue

                byte_start = positions[start][1]
                byte_end = positions[start + window_size - 1][1] + 1
                context_before, context_after = _candidate_context(
                    text, byte_start, byte_end, before_chars=120, after_chars=120
                )

                excerpt = re.sub(r"\s+", " ", frag).strip()
                llm_excerpt = re.sub(
                    r"\s+", " ", text[byte_start:byte_end],
                ).strip()
                if _looks_like_reference(frag) and not any(
                    ch not in "《》「」（）、，。；：？！" and ch > "鿿"
                    for ch in frag[:12]
                ):
                    continue

                key = (filename, excerpt)
                candidate = {
                    "file": filename,
                    "excerpt": excerpt,
                    "llm_excerpt": llm_excerpt or excerpt,
                    "context_before": context_before,
                    "context_after": context_after,
                    "score": round(score, 3),
                    "features": reasons,
                    "cjk_len": len(frag),
                    "source": "semantic_candidate",
                }
                previous = candidates_by_key.get(key)
                if previous is None or candidate["score"] > previous["score"]:
                    candidates_by_key[key] = candidate

    candidates = sorted(
        candidates_by_key.values(),
        key=lambda item: (-item["score"], item["file"], item["excerpt"]),
    )
    return candidates[:limit]


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> dict[str, str]:
    """Load key=value pairs from *path* (mimics the project's manual .env usage)."""
    env: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def _openai_config_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Resolve OpenAI-compatible configuration."""
    dotenv = _load_dotenv()
    return {
        "api_key": args.openai_api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or dotenv.get("OPENAI_API_KEY", ""),
        "base_url": args.openai_base_url
        or os.environ.get("OPENAI_BASE_URL", "")
        or dotenv.get("OPENAI_BASE_URL", ""),
        "model": args.openai_model
        or os.environ.get("OPENAI_MODEL", "")
        or dotenv.get("OPENAI_MODEL", ""),
    }


# ---------------------------------------------------------------------------
# LLM review (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "你是繁体中文 OCR 品質審查器。\n\n"
    "判斷候選片段是否為 OCR 語義亂碼。\n"
    "不要改寫，不要補全文。只输出 JSON。\n\n"
    "亂碼標準：\n"
    "- 字符大多是漢字，但組合不符合中文語法或語義。\n"
    "- 句子無法解釋為文言、引文、詩句、書名、外文音譯、專名、注釋。\n"
    "- 與前後文明顯不連貫。\n\n"
    "以下情況 **不要** 判定為亂碼：\n"
    "- 文言文\n"
    "- 詩詞歌謠\n"
    "- 歷史引文\n"
    "- 書名、文獻標題列表\n"
    "- 軍事口號\n"
    "- 暴行敘述列表\n"
    "- 注釋、參考文獻條目\n"
    "- 古典成語\n\n"
    "輸出 JSON：\n"
    '{"is_garbled":bool,"confidence":0-1,"category":"semantic_garble|normal_text|needs_human_review",'
    '"suspicious_span":"…","reason":"…","needs_source_ocr":bool}'
)

_USER_PROMPT_TEMPLATE = (
    "前文：\n{context_before}\n\n"
    "候選片段：\n{excerpt}\n\n"
    "後文：\n{context_after}\n\n"
    '請判斷以上「候選片段」是否為 OCR 語義亂碼。只輸出 JSON。'
)

_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ocr_review",
        "schema": {
            "type": "object",
            "properties": {
                "is_garbled": {"type": "boolean"},
                "confidence": {"type": "number"},
                "category": {
                    "type": "string",
                    "enum": ["semantic_garble", "normal_text", "needs_human_review"],
                },
                "suspicious_span": {"type": "string"},
                "reason": {"type": "string"},
                "needs_source_ocr": {"type": "boolean"},
            },
            "required": [
                "is_garbled",
                "confidence",
                "category",
                "suspicious_span",
                "reason",
                "needs_source_ocr",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _build_llm_prompt(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a chat-completion message list for *candidate*."""
    user_text = _USER_PROMPT_TEMPLATE.format(
        context_before=candidate.get("context_before", ""),
        excerpt=candidate.get("llm_excerpt") or candidate.get("excerpt", ""),
        context_after=candidate.get("context_after", ""),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _call_openai_chat(
    messages: list[dict[str, Any]],
    config: dict[str, str],
    timeout: int = 120,
) -> dict[str, Any] | None:
    """Send a chat completion request to an OpenAI-compatible endpoint.

    Returns the parsed JSON result, or *None* on failure.
    """
    try:
        import urllib.request
    except ImportError:
        return None

    payload = json.dumps(
        {
            "model": config["model"],
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    url = config["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"HTTP error: {exc}"}

    try:
        raw_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"error": f"unexpected response shape: {json.dumps(body)[:200]}"}

    # Tolerate fenced JSON blocks.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        raw_text = fence_match.group(1)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": f"invalid JSON: {raw_text[:200]}"}


def _is_retryable_llm_error(error: str) -> bool:
    """Decide whether an LLM error message looks transient.

    ``HTTP error:`` covers network/DNS/TLS/timeout/5xx — those are worth
    retrying. ``unexpected response shape`` and ``invalid JSON`` are
    deterministic failures from the model itself; retrying just multiplies
    cost without changing the outcome.
    """
    if not error:
        return True  # treat empty/no-response as transient
    return error.startswith("HTTP error:")


def _review_candidate_with_llm(
    candidate: dict[str, Any],
    config: dict[str, str],
    timeout: int = 120,
    max_attempts: int = 8,
    *,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Review one candidate; returns ``{llm_review: …}`` or ``{llm_error: …}``.

    Retries up to *max_attempts* times on transient errors (network, no
    response, 5xx). Deterministic failures (malformed JSON, unexpected
    response shape) return immediately without retrying.
    """
    messages = _build_llm_prompt(candidate)
    last_error = "no response"
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        result = _call_openai_chat(messages, config, timeout=timeout)
        if result is None:
            last_error = "no response"
            error_text = last_error
        elif "error" in result:
            last_error = result["error"]
            error_text = last_error
        else:
            return {"llm_review": result}

        if not _is_retryable_llm_error(error_text):
            # Deterministic failure — don't burn retries on it.
            return {"llm_error": last_error}

        if attempt < attempts - 1:
            # Exponential backoff capped at 8 s. Keeps a flaky endpoint
            # from being hammered while still recovering quickly.
            sleep(min(2 ** attempt, 8))

    return {"llm_error": last_error}


def _slice_candidates(
    candidates: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    """Return candidates starting at zero-based *start_index*."""
    if start_index <= 0:
        return candidates
    return candidates[start_index:]


def _write_review_progress(
    candidates: list[dict[str, Any]],
    progress_path: str | None,
) -> None:
    """Atomically write current LLM review progress when requested."""
    if not progress_path:
        return
    directory = os.path.dirname(progress_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{progress_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, progress_path)


def review_candidates_with_llm(
    candidates: list[dict[str, Any]],
    config: dict[str, str],
    *,
    strict: bool = False,
    timeout: int = 120,
    progress_path: str | None = None,
) -> list[dict[str, Any]]:
    """Augment *candidates* in place with LLM review results.

    Each candidate receives ``llm_review`` or ``llm_error``.
    Returns the modified list.
    """
    for i, candidate in enumerate(candidates):
        print(
            f"  [{i + 1}/{len(candidates)}] reviewing: {candidate['excerpt'][:40]}...",
            file=sys.stderr,
        )
        result = _review_candidate_with_llm(candidate, config, timeout=timeout)
        candidate.update(result)
        _write_review_progress(candidates, progress_path)
        if "llm_error" in result and strict:
            raise RuntimeError(
                f"LLM review failed for candidate {i}: {result['llm_error']}"
            )
    return candidates


# ---------------------------------------------------------------------------
# Source-text lookup from MinerU/PaddleOCR checkpoints
# ---------------------------------------------------------------------------

def _find_source_for_candidates(
    candidates: list[dict[str, Any]],
    work_dir: str,
) -> None:
    """Augment each candidate with ``source_text`` from checkpoint JSON files.

    Searches ``work_dir/chunk_*.json`` for each candidate's excerpt keywords.
    """
    import glob as _glob

    checkpoint_files = sorted(_glob.glob(os.path.join(work_dir, "chunk_*.json")))
    if not checkpoint_files:
        print(f"  [!] No chunk_*.json files found in {work_dir}", file=sys.stderr)
        return

    for i, candidate in enumerate(candidates):
        excerpt = candidate.get("excerpt", "")
        # Use 4-char slices as search keys
        keys = [excerpt[j:j+5] for j in range(0, min(len(excerpt)-4, 25), 5)]
        if not keys:
            continue

        found = None
        for cpath in checkpoint_files:
            try:
                with open(cpath, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            for page in data.get("result", {}).get("layoutParsingResults", []):
                md = page.get("markdown", {}).get("text", "")
                for key in keys:
                    if key and key in md:
                        # Extract surrounding context
                        idx = md.find(key)
                        ctx = md[max(0, idx - 40):idx + len(excerpt) + 120]
                        found = ctx.strip()
                        break
                if found:
                    break
            if found:
                break

        if found:
            candidate["source_text"] = found
            print(f"  [{i+1}/{len(candidates)}] found source", file=sys.stderr)
        else:
            print(f"  [{i+1}/{len(candidates)}] no match", file=sys.stderr)


def _patch_epub_from_source(
    epub_path: str,
    candidates: list[dict[str, Any]],
) -> None:
    """Apply source_text replacements directly to *epub_path*.

    Only candidates with non-empty ``source_text`` are processed.
    """
    import shutil as _shutil
    import tempfile as _tempfile
    import zipfile as _zipfile

    patches: dict[str, list[tuple[str, str]]] = {}
    for c in candidates:
        source = c.get("source_text", "")
        excerpt = c.get("excerpt", "")
        if not source or not excerpt or excerpt in source:
            continue
        fname = c["file"]
        patches.setdefault(fname, []).append((excerpt, source))

    if not patches:
        print("  [!] No patchable candidates.", file=sys.stderr)
        return

    contents: dict[str, str] = {}
    with _zipfile.ZipFile(epub_path) as zf:
        for fname in patches:
            contents[fname] = zf.read(fname).decode("utf-8")

    applied = 0
    for fname, replacements in patches.items():
        for old, new in replacements:
            if old in contents[fname]:
                contents[fname] = contents[fname].replace(old, new, 1)
                applied += 1

    if not applied:
        print("  [!] No replacements matched in EPUB XHTML.", file=sys.stderr)
        return

    td = _tempfile.mkdtemp()
    with _zipfile.ZipFile(epub_path) as zf:
        zf.extractall(td)
    for fname, content in contents.items():
        path = os.path.join(td, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    os.remove(epub_path)
    with _zipfile.ZipFile(epub_path, "w", _zipfile.ZIP_DEFLATED) as zf:
        mt = os.path.join(td, "mimetype")
        if os.path.exists(mt):
            zf.write(mt, "mimetype", compress_type=_zipfile.ZIP_STORED)
        for root, _, files in os.walk(td):
            for file in files:
                if file == "mimetype":
                    continue
                fp = os.path.join(root, file)
                an = os.path.relpath(fp, td).replace(os.sep, "/")
                zf.write(fp, an)
    _shutil.rmtree(td)

    print(f"  [*] Patched {applied} occurrence(s) in {epub_path}.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate semantic OCR review candidates for an EPUB."
    )
    parser.add_argument("epub", help="Path to EPUB file")
    parser.add_argument(
        "--json", dest="json_path", help="Write JSON report to this path"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum candidates (default 100)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many generated candidates before review/output. "
             "Use with --limit to resume or run LLM review in slices.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum suspicion score for window candidates "
             "(default: 0.50 without --llm, 0.30 with --llm)",
    )
    parser.add_argument(
        "--min-cjk",
        type=int,
        default=14,
        help="Minimum CJK characters in a statistical window (default 14)",
    )
    parser.add_argument(
        "--min-cjk-sentence",
        type=int,
        default=10,
        help="Minimum CJK characters per sentence candidate (default 10)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable OpenAI-compatible LLM review",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Add sentence-level candidates that bypass statistical scoring. "
             "Recommended with --llm to catch Tier-3 semantic garble.",
    )
    parser.add_argument(
        "--only-garbled",
        action="store_true",
        help="Only emit candidates where LLM says is_garbled=true",
    )
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help="Exit nonzero if any LLM request fails",
    )
    parser.add_argument("--openai-api-key", help="OpenAI-compatible API key")
    parser.add_argument("--openai-base-url", help="OpenAI-compatible base URL")
    parser.add_argument(
        "--openai-model", help="OpenAI-compatible model identifier"
    )
    parser.add_argument(
        "--find-source",
        metavar="WORK_DIR",
        help="Search a MinerU/PaddleOCR work directory for correct source text of each candidate",
    )
    parser.add_argument(
        "--patch-from-source",
        metavar="WORK_DIR",
        help="Like --find-source, but also apply found corrections to the EPUB (DANGER: overwrites EPUB)",
    )
    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")

    # Resolve min_score default based on LLM mode
    if args.min_score is None:
        args.min_score = 0.30 if args.llm else 0.50

    # 1. Generate candidates offline -------------------------------------------------
    print(f"[*] Scanning {args.epub} for semantic OCR candidates...", file=sys.stderr)

    # Statistical window candidates
    candidates = generate_semantic_candidates(
        args.epub,
        limit=args.limit,
        min_score=args.min_score,
        min_cjk=args.min_cjk,
    )
    print(f"[*] Statistical windows: {len(candidates)} candidates (min_score={args.min_score:.2f}).",
          file=sys.stderr)

    # Sentence-level candidates (deep mode — bypasses statistical scoring)
    if args.deep:
        sentence_candidates = generate_sentence_candidates(
            args.epub,
            limit=args.limit,
            min_cjk=args.min_cjk_sentence,
        )
        print(f"[*] Sentence chunks:    {len(sentence_candidates)} candidates.",
              file=sys.stderr)
        # Merge: deduplicate by (file, excerpt_prefix)
        stat_keys = {(c["file"], c["excerpt"][:40]) for c in candidates}
        for sc in sentence_candidates:
            key = (sc["file"], sc["excerpt"][:40])
            if key not in stat_keys:
                candidates.append(sc)
        # Sort statistical candidates by score desc, keep sentence candidates
        # in their original round-robin order (already interleaved across files).
        stat_cands = [c for c in candidates if c["source"] == "semantic_candidate"]
        sent_cands = [c for c in candidates if c["source"] == "sentence_candidate"]
        stat_cands.sort(key=lambda c: (-c["score"], c["file"], c["excerpt"]))
        candidates = stat_cands + sent_cands
        candidates = candidates[:args.limit]
        print(f"[*] Combined:           {len(candidates)} candidates after dedup.",
              file=sys.stderr)

    if args.start_index:
        before_count = len(candidates)
        candidates = _slice_candidates(candidates, args.start_index)
        print(
            f"[*] Start index:        {args.start_index} "
            f"({len(candidates)} of {before_count} candidates remain).",
            file=sys.stderr,
        )

    # 2. Optional LLM review ---------------------------------------------------------
    if args.llm:
        config = _openai_config_from_args(args)
        missing = [k for k, v in config.items() if not v]
        if missing:
            print(
                f"[!] Missing OpenAI config: {', '.join(missing)}. "
                "Set via .env or --openai-* flags.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"[*] Reviewing with LLM: model={config['model']} "
            f"base_url={config['base_url']}",
            file=sys.stderr,
        )
        review_candidates_with_llm(
            candidates,
            config,
            strict=args.strict_llm,
            progress_path=args.json_path,
        )

    # 3. Optional source search from MinerU/PaddleOCR checkpoints --------------------
    if args.find_source or args.patch_from_source:
        work_dir = args.find_source or args.patch_from_source
        print(f"[*] Searching checkpoints in {work_dir}...", file=sys.stderr)
        _find_source_for_candidates(candidates, work_dir)
        if args.patch_from_source:
            print(f"[*] Patching {args.epub} from source...", file=sys.stderr)
            _patch_epub_from_source(args.epub, candidates)

    # 4. Output ----------------------------------------------------------------------
    if args.only_garbled and args.llm:
        candidates = [
            c
            for c in candidates
            if c.get("llm_review", {}).get("is_garbled", False)
        ]
        print(
            f"[*] Filtered to {len(candidates)} LLM-positive candidates.",
            file=sys.stderr,
        )

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        print(f"[*] Wrote {args.json_path}", file=sys.stderr)
    else:
        if not candidates:
            print("[*] No semantic OCR candidates found.")
            return

        for i, item in enumerate(candidates, 1):
            print(f"\n{'─' * 60}")
            print(f"{i:>3}. score={item['score']:.3f}  file={item['file']}")
            print(f"    excerpt: {item['excerpt']}")
            if item.get("context_before"):
                print(f"    before:  {item['context_before'][-80:]}")
            if item.get("context_after"):
                print(f"    after:   {item['context_after'][:80]}")
            if item.get("llm_review"):
                r = item["llm_review"]
                print(
                    f"    LLM: is_garbled={r.get('is_garbled')} "
                    f"confidence={r.get('confidence')} "
                    f"category={r.get('category')}"
                )
                print(f"    LLM reason: {r.get('reason', '')[:120]}")
            elif item.get("llm_error"):
                print(f"    LLM ERROR: {item['llm_error']}")


if __name__ == "__main__":
    main()
