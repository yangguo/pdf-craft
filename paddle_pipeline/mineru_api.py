"""MinerU API integration — file upload, batch polling, result download.

Flow (per https://mineru.net/doc/docs/):
  1. POST /api/v4/file-urls/batch → get signed PUT URLs + batch_id
  2. PUT chunk file to signed URL → task auto-submitted
  3. GET /api/v4/extract-results/batch/{batch_id} → poll until done
  4. Download result ZIP from full_zip_url → parse markdown
"""

import io
import json
import os
import re
import time
import zipfile

from typing import Any, Dict, List

from .config import (
    MINERU_API_URL,
    MINERU_API_TOKEN,
    MINERU_CHUNK_SIZE,
    MINERU_LANGUAGE,
    MINERU_MAX_POLL_TIME,
    MINERU_MODEL_VERSION,
    MINERU_POLL_INTERVAL,
    fitz,
    requests,
)

_VERIFY_SSL = os.getenv("MINERU_VERIFY_SSL", "1") not in ("0", "false", "no", "off")
if not _VERIFY_SSL:
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message="Unverified HTTPS request")
_API_BASE = re.sub(r"/api/v\d+/.*$", "", MINERU_API_URL)


def _add_ocr_guard_bands(page: Any) -> None:
    """Add small white guard bands for MinerU OCR edge readability."""
    rect = page.rect
    # Keep this small to avoid confusing the VLM layout model while still
    # protecting characters near the book spine and bottom page edge.
    left_margin = 8
    new_h = rect.height * 1.05
    page.set_mediabox(fitz.Rect(-left_margin, 0, rect.width, new_h))
    page.draw_rect(
        fitz.Rect(-left_margin, 0, 0, new_h),
        color=None, fill=(1, 1, 1),
    )
    page.draw_rect(
        fitz.Rect(0, rect.height, rect.width, new_h),
        color=None, fill=(1, 1, 1),
    )


def split_pdf(file_path: str, chunk_size: int | None = None) -> List[str]:
    """Split a PDF into chunks for the MinerU API (default 20 pages per chunk)."""
    import tempfile

    if chunk_size is None:
        chunk_size = MINERU_CHUNK_SIZE

    doc = fitz.open(file_path)
    total_pages = len(doc)
    print(f"[*] Total pages: {total_pages}")

    chunk_paths = []
    temp_dir = tempfile.mkdtemp(prefix="mineru_chunks_")

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

        for page in chunk_doc:
            _add_ocr_guard_bands(page)

        chunk_filename = os.path.join(temp_dir, f"chunk_{start_page}_{end_page}.pdf")
        chunk_doc.save(chunk_filename)
        chunk_doc.close()
        chunk_paths.append(chunk_filename)

    doc.close()
    return chunk_paths


def _api_headers(token: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _download_zip_with_retry(zip_url: str, token: str) -> bytes | None:
    """Download a completed MinerU result ZIP without resubmitting the batch."""
    max_retries = 5
    for attempt in range(max_retries):
        if attempt > 0:
            wait = (2 ** (attempt - 1)) * 5
            print(f"    ...retrying ZIP download in {wait}s...")
            time.sleep(wait)
        try:
            zr = requests.get(
                zip_url, headers=_api_headers(token), timeout=120, verify=_VERIFY_SSL,
            )
            if zr.status_code == 200:
                return zr.content
            print(f"[!] ZIP download HTTP {zr.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[!] ZIP download error: {e}")

    return None


def parse_pdf_chunk(chunk_path: str, token: str | None = None) -> Dict[str, Any] | None:
    """Process a PDF chunk through the MinerU v4 API.

    1. Request signed upload URL + PUT file
    2. Poll GET /api/v4/extract-results/batch/{batch_id} until done
    3. Download and parse the result ZIP

    Returns a dict compatible with the epub_builder pipeline, or None on failure.
    """
    if token is None:
        token = MINERU_API_TOKEN

    if not token:
        print("[!] MINERU_API_TOKEN is not set.")
        return None

    file_name = os.path.basename(chunk_path)
    print(f"[*] MinerU: uploading {file_name}")

    # Step 1: Get signed upload URL (with retry)
    data = None
    batch_id = ""
    upload_url = ""
    for attempt in range(5):
        if attempt > 0:
            wait = (2 ** (attempt - 1)) * 5
            print(f"    ...retrying in {wait}s...")
            time.sleep(wait)
        try:
            resp = requests.post(
                f"{_API_BASE}/api/v4/file-urls/batch",
                json={
                    "files": [{"name": file_name}],
                    "model_version": MINERU_MODEL_VERSION,
                    "is_ocr": True,
                    "enable_formula": False,
                    "enable_table": True,
                    "language": MINERU_LANGUAGE,
                },
                headers=_api_headers(token),
                timeout=60,
                verify=_VERIFY_SSL,
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    print("[!] Upload URL response was not valid JSON")
                    if attempt == 4:
                        return None
                    continue
                if data.get("code") == 0:
                    payload = data.get("data")
                    if not isinstance(payload, dict):
                        print("[!] Upload URL response missing data object")
                        if attempt == 4:
                            return None
                        continue
                    candidate_batch_id = payload.get("batch_id")
                    file_urls = payload.get("file_urls")
                    candidate_upload_url = (
                        file_urls[0]
                        if isinstance(file_urls, list) and file_urls else None
                    )
                    if (
                        isinstance(candidate_batch_id, str)
                        and candidate_batch_id
                        and isinstance(candidate_upload_url, str)
                        and candidate_upload_url
                    ):
                        batch_id = candidate_batch_id
                        upload_url = candidate_upload_url
                        break
                    print("[!] Upload URL response missing batch_id/file_urls")
                    if attempt == 4:
                        return None
                    continue
                print(f"[!] Upload URL request failed: {data.get('msg')}")
                if attempt == 4:
                    return None
            else:
                print(f"[!] Upload URL request HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[!] Upload URL request error: {e}")
            if attempt == 4:
                return None

    if not data or not batch_id or not upload_url:
        return None

    # Step 2: PUT file (with retry)
    for attempt in range(5):
        if attempt > 0:
            wait = (2 ** (attempt - 1)) * 5
            print(f"    ...retrying in {wait}s...")
            time.sleep(wait)
        try:
            with open(chunk_path, "rb") as f:
                put_resp = requests.put(upload_url, data=f, timeout=300, verify=_VERIFY_SSL)
            if put_resp.status_code in (200, 204):
                break
            print(f"[!] File upload HTTP {put_resp.status_code}")
            if attempt == 4:
                return None
        except requests.exceptions.RequestException as e:
            print(f"[!] File upload error: {e}")
            if attempt == 4:
                return None

    print(f"[*] MinerU: uploaded, batch {batch_id}")

    # Step 3: Poll batch results
    batch_url = f"{_API_BASE}/api/v4/extract-results/batch/{batch_id}"
    start_time = time.time()
    first_poll = True

    while True:
        elapsed = time.time() - start_time
        if elapsed > MINERU_MAX_POLL_TIME:
            print(f"[!] Timed out waiting for batch {batch_id}")
            return None

        if not first_poll:
            time.sleep(MINERU_POLL_INTERVAL)
        first_poll = False

        try:
            r = requests.get(
                batch_url, headers=_api_headers(token), timeout=30, verify=_VERIFY_SSL,
            )
            if r.status_code == 401:
                print("[!] MinerU API: Unauthorized — check MINERU_API_TOKEN")
                return None
            if r.status_code in (403, 404):
                print(f"[!] MinerU API: Batch {batch_id} HTTP {r.status_code}")
                return None
            if r.status_code != 200:
                continue

            try:
                result = r.json()
            except ValueError:
                print("[!] Poll response was not valid JSON")
                continue
            if result.get("code") != 0:
                continue

            payload = result.get("data")
            if not isinstance(payload, dict):
                print("[!] Poll response missing data object")
                continue
            results_list = payload.get("extract_result", [])
            if not results_list:
                continue

            first = results_list[0]
            state = first.get("state", "")

            if state == "done":
                zip_url = first.get("full_zip_url", "")
                if not zip_url:
                    print("[!] No zip URL in result")
                    return None

                # Step 4: Download and parse ZIP. The batch is already
                # complete, so retry the ZIP read here instead of resubmitting.
                zip_data = _download_zip_with_retry(zip_url, token)
                if zip_data is None:
                    return None

                parsed = _parse_zip(zip_data)
                if parsed:
                    # Stash the ZIP URL so the checkpoint can re-parse
                    # later when the parsing logic changes.
                    parsed["_mineru_zip_url"] = zip_url
                    parsed["_mineru_reparsed"] = True
                    print(f"[*] MinerU: batch {batch_id} done")
                    return parsed
                return None

            elif state == "failed":
                print(f"[!] Batch {batch_id} failed: {first.get('err_msg', 'unknown')}")
                return None

            # else: still processing — continue polling

        except requests.exceptions.RequestException as e:
            print(f"[!] Poll error: {e}")


# ── MinerU markdown cleaning ────────────────────────────────────────────
# MinerU's VLM model generates image classification labels and AI descriptions
# (e.g. "natural_image", "Full-body photo of a person wearing...") that end
# up in the EPUB as noise.  Strip them here before the text enters the common
# EPUB-builder pipeline.

# Known MinerU VLM image classification labels (lowercase).
_MINERU_IMAGE_LABELS = frozenset({
    "natural_image", "natural image", "text_image", "text image",
    "photo", "photograph", "figure", "illustration", "drawing",
    "painting", "portrait", "landscape", "screenshot", "diagram",
    "chart", "graph", "logo", "icon", "symbol", "sign",
    "badge", "emblem", "map", "poster", "banner",
})

# Subset of _MINERU_IMAGE_LABELS that commonly appear as
# "label: description text" headers (not just standalone).
_MINERU_DESC_LABEL_PREFIXES = (
    "natural_image", "natural image", "photo", "photograph", "figure",
)

# Parenthetical image-content notes — a signature of AI-generated
# descriptions, e.g. "(no visible text or symbols)", "(with text)".
_MINERU_IMAGE_NOTE = re.compile(
    r"\((no\s+visible|without|with|text)\s"
    r"(visible\s+)?(text|symbols?|people|person|background|logo|"
    r"writing|lettering|signage|branding|face|body)",
    re.IGNORECASE,
)

# Image-description starter phrases ("Full-body photo of...", etc.).
_MINERU_DESC_START = re.compile(
    r"^(Full.body|Close.up|Aerial|Overhead|Side.view|Front.view|"
    r"Back.view|Top.view|Bottom.view|Wide.angle|Extreme\s+close.up|"
    r"Medium\s+shot|Long\s+shot|Cropped|Blurred|"
    r"A\s+(photo|picture|shot|view|close.up|portrait)\s+of\b)",
    re.IGNORECASE,
)


def _clean_mineru_markdown(md_text: str) -> str:
    """Remove MinerU VLM-generated image classification labels and descriptions."""

    # 0. Strip <details><summary>image_label</summary>...</details> blocks
    #    that MinerU VLM injects around every image.  These contain nothing
    #    but classification labels and AI-generated descriptions.
    md_text = re.sub(
        r"<details>\s*<summary>[^<]*</summary>.*?</details>",
        "",
        md_text,
        flags=re.DOTALL,
    )

    # 1. Strip standalone "natural_image"-style labels on their own line.
    lines = md_text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped in _MINERU_IMAGE_LABELS:
            continue
        # Also strip lines that start with a label followed by a long
        # ASCII description (e.g. "natural_image: Full-body photo...")
        # Require a colon or full-width colon after the label to avoid
        # false positives on words like "photography" or "Figure 1.".
        for label in _MINERU_DESC_LABEL_PREFIXES:
            colon_pos = len(label)
            if len(stripped) > colon_pos + 15 and (
                stripped.startswith(f"{label}:")
                or stripped.startswith(f"{label}：")
            ):
                rest = stripped[len(label):].lstrip(":： \t")
                ascii_cnt = sum(1 for c in rest if ord(c) < 128)
                if len(rest) > 20 and ascii_cnt / len(rest) >= 0.85:
                    # It is an AI image description — skip this line
                    break
        else:
            filtered.append(line)
            continue
        # (line was skipped because the inner loop broke)
    md_text = "\n".join(filtered)

    # 2. Strip AI-generated alt text from markdown image references.
    def _clean_alt(m: re.Match) -> str:
        alt = m.group(1)
        path = m.group(2).strip()
        if not alt:
            return m.group(0)  # already empty — leave alone
        # Known classification label → nuke alt text
        if alt.strip().lower() in _MINERU_IMAGE_LABELS:
            return f"[]({path})"
        # Long (>50 chars) mostly-ASCII alt text → AI-generated description
        ascii_cnt = sum(1 for c in alt if ord(c) < 128)
        if len(alt) > 50 and ascii_cnt / len(alt) >= 0.85:
            return f"[]({path})"
        return m.group(0)

    md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _clean_alt, md_text)

    # 3. Remove standalone paragraphs that are AI image descriptions.
    #    A line that is long, mostly ASCII, and either starts with a
    #    description phrase or contains a parenthetical image-content note
    #    is almost certainly generated by the VLM.
    lines = md_text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 40:
            ascii_cnt = sum(1 for c in stripped if ord(c) < 128)
            if ascii_cnt / len(stripped) >= 0.80:
                if _MINERU_IMAGE_NOTE.search(stripped) or _MINERU_DESC_START.search(stripped):
                    continue
        result.append(line)

    md_text = "\n".join(result)

    # 3.5 Repair OCR reading-order errors where a short sentence-ending
    #     fragment appears before a sentence-starting fragment.
    #     Example: "話！\n\n就在這時刻...我要講幾句" →
    #              "就在這時刻...我要講幾句話！"
    terminal_set = frozenset({"。", "！", "？", ".", "!", "?", "」", "』"})
    lines = md_text.split("\n")
    repaired = list(lines)
    i = 0
    while i < len(repaired) - 2:
        a = repaired[i].strip()
        # Find the next non-empty line after a (skipping blanks)
        j = i + 1
        while j < len(repaired) and not repaired[j].strip():
            j += 1
        if j >= len(repaired):
            break
        b = repaired[j].strip()
        # a is very short and ends with terminal punctuation —
        # it's likely the tail of the sentence starting on b.
        if (len(a) <= 5 and a and a[-1] in terminal_set
            and len(b) > 10 and b[-1] not in terminal_set):
            # Merge: b becomes the full sentence, a is removed
            repaired[j] = b + a
            repaired[i] = ""
            i = j + 1
        else:
            i = j

    md_text = "\n".join(repaired)

    # 3.6 Repair OCR mid-line truncation where text near the page spine
    #     is missed, splitting one sentence across two lines.
    #     Example: "...採用「一國兩制」方針\n\n國在香港既有的利益..."
    #              → "...採用「一國兩制」方針國在香港既有的利益..."
    #     The first line ends mid-sentence (no terminal punctuation), and
    #     the next non-blank line starts with a character that doesn't
    #     look like the beginning of a new sentence.
    _sentence_starters = frozenset(
        "第這那如但可因爲所在其此該本而若則又並且或雖然何當從對與以"
        "一二三四五六七八九十"
    )
    _cjk_punct = frozenset("，。！？、：；」「『』（）【】《》…—")
    lines = md_text.split("\n")
    repaired = list(lines)
    i = 0
    while i < len(repaired) - 2:
        a = repaired[i].strip()
        if not a or a.startswith("#") or a.startswith("!"):
            i += 1
            continue
        # Find next non-empty line
        j = i + 1
        while j < len(repaired) and not repaired[j].strip():
            j += 1
        if j >= len(repaired):
            break
        b = repaired[j].strip()
        if not b or b.startswith("#") or b.startswith("!"):
            i = j
            continue
        # Line a ends without terminal punctuation, and line b starts
        # with a CJK character that doesn't look like a sentence start.
        if (len(a) > 15 and a[-1] not in terminal_set
            and a[-1] not in _cjk_punct
            and len(b) > 5 and b[0] not in _sentence_starters
            and b[0] not in _cjk_punct
            and not b[0].isdigit()
            and not b[0].isascii()):
            # Merge: a and b are one sentence split by OCR truncation
            repaired[i] = a + b
            repaired[j] = ""
            i = j + 1
        else:
            i = j

    return "\n".join(repaired)


def _parse_zip(zip_data: bytes) -> Dict[str, Any] | None:
    """Parse a MinerU result ZIP into pipeline-compatible format.

    Extracts markdown and converts bundled images to data URIs.
    Splits combined markdown at repeated page-header headings so the
    downstream EPUB builder gets per-page granularity for chapter detection.
    """
    import base64

    _MIME = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp",
    }

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            file_list = zf.namelist()

            md_files = sorted([f for f in file_list if f.endswith(".md")])
            image_files = [f for f in file_list if f.lower().endswith(
                tuple(_MIME.keys())
            )]

            if not md_files:
                print("[!] No .md files in result ZIP")
                return None

            # Pre-encode all images to data URIs
            all_images: Dict[str, str] = {}
            for img_name in image_files:
                try:
                    img_bytes = zf.read(img_name)
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")
                    ext = os.path.splitext(img_name)[1].lower()
                    mime = _MIME.get(ext, "image/jpeg")
                    all_images[img_name] = f"data:{mime};base64,{img_b64}"
                except Exception as e:
                    print(f"[!] Failed to read image {img_name}: {e}")

            layout_results = []
            for md_name in md_files:
                md_text = zf.read(md_name).decode("utf-8", errors="replace")

                # Split combined markdown at repeated page-header headings.
                # MinerU bundles an entire chunk into one .md file where
                # each physical page starts with an H1 that repeats on
                # every page (e.g. book title).
                pages = _split_by_page_headers(md_text)

                for page_md in pages:
                    if not page_md.strip():
                        continue
                    # Drop fragments that are just page numbers or
                    # a lone header (ghost pages).  Real pages have
                    # at least ~100 chars of body text.
                    if len(page_md.strip()) < 80:
                        continue
                    page_md = _clean_mineru_markdown(page_md)
                    if len(page_md.strip()) < 80:
                        continue
                    layout_results.append({
                        "markdown": {"text": page_md, "images": dict(all_images)}
                    })

            return {"result": {"layoutParsingResults": layout_results}}

    except zipfile.BadZipFile:
        print("[!] Corrupted result ZIP")
        return None


def _detect_page_headers(md_lines: List[str]) -> tuple[set, set]:
    """Find headings that are page numbers or repeated across pages.

    Returns (markdown_headers, plain_text_headers) where markdown_headers
    are ``# ...`` lines to split on and plain_text_headers are the text
    without the ``# `` prefix for body-stripping.  Also detects plain-text
    lines that repeat 3+ times in the same chunk — these are running page
    headers that MinerU outputs without markdown markup.
    """
    from collections import Counter
    h1_headings = []
    chapterish = re.compile(r"第[零一二三四五六七八九十百千0-9]+[章節编編篇卷]")
    for line in md_lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            h1_headings.append(stripped)
    counts = Counter(h1_headings)

    md_headers = set()
    for h, c in counts.items():
        text = h[2:]  # strip "# " prefix
        # Standalone page numbers (1-4 digits)
        if text.isdigit() and len(text) <= 4:
            md_headers.add(h)
        # Repeated headings (book title as running header)
        elif c >= 2:
            md_headers.add(h)

    plain_headers = {h[2:] for h in md_headers}

    # Also detect plain-text lines that repeat — these are running page
    # headers / chapter titles that MinerU outputs without markdown markup.
    _titleish = re.compile(r"^[「『\"][^」』\"]{1,20}[」』\"]$")
    plain_lines = []
    for line in md_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Skip very short or number-only lines (page numbers)
        if len(s) < 3 or s.isdigit():
            continue
        plain_lines.append(s)
    plain_counts = Counter(plain_lines)
    for line_text, c in plain_counts.items():
        if c >= 3:
            # Accept lines that look like chapter titles, are in bracket
            # quotes, or are short enough to be page headers (not body text
            # that coincidentally repeats)
            if (chapterish.match(line_text) or len(line_text) <= 15
                    or _titleish.match(line_text)):
                plain_headers.add(line_text)
        elif c >= 2:
            # Lines appearing 2+ times: accept if short, chapterish,
            # or in title brackets (e.g. book title as running header)
            if (len(line_text) <= 10
                    or (chapterish.match(line_text) and len(line_text) <= 25)
                    or _titleish.match(line_text)):
                plain_headers.add(line_text)

    return md_headers, plain_headers


def _split_by_page_headers(md_text: str) -> list:
    """Split markdown at repeated page-header boundaries.

    Auto-detects H1 headings that repeat 2+ times as page headers,
    splits at each occurrence, strips the header line and any
    plain-text header repetition from the start of every segment.
    Also strips standalone page numbers.
    """
    lines = md_text.split("\n")
    md_headers, plain_headers = _detect_page_headers(lines)

    # Collect split points from H1 headers first, then fall back to
    # plain-text headers when MinerU outputs them without markdown markup.
    split_indices = []
    for i, line in enumerate(lines):
        if line.strip() in md_headers:
            split_indices.append(i)

    if not split_indices and plain_headers:
        # No H1-anchored split points — use plain-text repeated headers
        # as page boundaries so the chunk does not flow through as one
        # monolithic block (which would let truncation repair merge
        # across page boundaries).
        for i, line in enumerate(lines):
            if line.strip() in plain_headers:
                split_indices.append(i)

    if not split_indices:
        return [md_text]

    segments = []
    for j, start_idx in enumerate(split_indices):
        content_start = start_idx + 1  # skip header line
        content_end = split_indices[j + 1] if j + 1 < len(split_indices) else len(lines)
        segment_lines = lines[content_start:content_end]

        # Strip plain-text header repetition from the top of the segment
        # (MinerU sometimes outputs the header as the first text line too)
        segment_lines = _strip_leading_headers(segment_lines, plain_headers)
        # Strip leading standalone page numbers
        segment_lines = _strip_leading_page_numbers(segment_lines)

        segment = "\n".join(segment_lines).strip()
        if segment:
            segments.append(segment)

    # Content before the first page header (e.g. cover page lead-in)
    if split_indices[0] > 0:
        prefix = "\n".join(lines[:split_indices[0]]).strip()
        if prefix:
            segments.insert(0, prefix)

    return segments


def _strip_leading_headers(lines: list, plain_headers: set) -> list:
    """Remove page-header text when it appears as leading or trailing lines."""
    # Strip from the top
    while lines and lines[0].strip() in plain_headers:
        lines = lines[1:]
    # Also handle "HEADER HEADER" (doubled OCR repetition)
    if lines:
        first = lines[0].strip()
        for h in plain_headers:
            if first == f"{h} {h}":
                lines = lines[1:]
                break
            # "HEADER real content" → strip prefix only
            if first.startswith(h + " ") and len(first) > len(h) + 1:
                lines[0] = lines[0].replace(h + " ", "", 1).strip()
                break
            if first.startswith(h + " ") and len(first) == len(h) + 1:
                lines = lines[1:]
                break
    # Strip from the bottom
    while lines and lines[-1].strip() in plain_headers:
        lines = lines[:-1]
    return lines


def _strip_leading_page_numbers(lines: list) -> list:
    """Remove standalone page numbers from the top of a segment."""
    while lines:
        stripped = lines[0].strip()
        if stripped.isdigit() and len(stripped) <= 4:
            lines = lines[1:]
        else:
            break
    return lines


def reparse_checkpoint(checkpoint: Dict[str, Any],
                       token: str | None = None) -> Dict[str, Any]:
    """Re-parse a MinerU checkpoint using the current parsing logic.

    When the checkpoint contains ``_mineru_zip_url`` (stashed by
    ``parse_pdf_chunk``), this re-downloads the ZIP and applies the
    latest version of ``_parse_zip``.  If the URL is no longer valid
    or the field is absent, the checkpoint is returned as-is.

    After a successful reparse the ``_mineru_reparsed`` sentinel is set so
    that subsequent runs skip re-downloading the ZIP.
    """
    if checkpoint.get("_mineru_reparsed"):
        return checkpoint

    zip_url = checkpoint.get("_mineru_zip_url")
    if not zip_url:
        return checkpoint

    if token is None:
        token = MINERU_API_TOKEN
    if not token:
        return checkpoint

    zip_data = _download_zip_with_retry(zip_url, token)
    if zip_data is None:
        print("[!] Re-download ZIP failed, using cached result")
        return checkpoint

    reparsed = _parse_zip(zip_data)
    if reparsed:
        reparsed["_mineru_zip_url"] = zip_url
        reparsed["_mineru_reparsed"] = True
        return reparsed

    return checkpoint


def strip_checkpoint_data_uris(res: Dict[str, Any]) -> bool:
    """Replace base64 data URI image values with empty strings in-place.

    Keeps the image keys (so ``_is_sparse_visual_page`` still sees a
    non-empty dict for pages that have images) while removing the bulky
    base64 payload that would otherwise bloat checkpoint files into the
    gigabyte range.

    Returns ``True`` if any URIs were stripped, ``False`` otherwise.
    """
    stripped = False
    for page_res in res.get("result", {}).get("layoutParsingResults", []):
        images_map = page_res.get("markdown", {}).get("images", {})
        for rel_path, img_url in list(images_map.items()):
            if isinstance(img_url, str) and img_url.startswith("data:"):
                images_map[rel_path] = ""
                stripped = True
    return stripped
