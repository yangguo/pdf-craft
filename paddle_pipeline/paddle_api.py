"""PDF chunking, Paddle OCR API calls, and dependency checking."""

import json
import os
import sys
import tempfile
import time

from typing import Any, Dict, List

from .config import (
    API_TIMEOUT_SECONDS,
    API_URL,
    CHUNK_SIZE,
    MAX_DAILY_PAGES,
    MODEL_VERSION,
    PADDLE_MAX_POLL_TIME,
    PADDLE_POLL_INTERVAL,
    PADDLE_FORCE_ROTATE,
    PADDLE_LAYOUT_MERGE_BBOXES_MODE,
    PADDLE_LAYOUT_NMS,
    PADDLE_LAYOUT_SHAPE_MODE,
    PADDLE_LAYOUT_THRESHOLD,
    PADDLE_LAYOUT_UNCLIP_RATIO,
    PADDLE_MAX_PIXELS,
    PADDLE_MIN_PIXELS,
    PADDLE_PAGE_PADDING_BOTTOM,
    PADDLE_PAGE_PADDING_TOP,
    PADDLE_PAGE_PADDING_X,
    PADDLE_PRETTIFY_MARKDOWN,
    PADDLE_PROMPT_LABEL,
    PADDLE_REPETITION_PENALTY,
    PADDLE_TEMPERATURE,
    PADDLE_TOP_P,
    PADDLE_USE_CHART_RECOGNITION,
    PADDLE_USE_DOC_ORIENTATION_CLASSIFY,
    PADDLE_USE_DOC_UNWARPING,
    PADDLE_USE_LAYOUT_DETECTION,
    PADDLE_VISUALIZE,
    epub,      # Optional dependency
    fitz,      # Optional dependency
    requests,  # Optional dependency
)


def check_dependencies():
    """Checks if required libraries are installed."""
    missing = []
    if requests is None:
        missing.append("requests")
    if fitz is None:
        missing.append("pymupdf")
    if epub is None:
        missing.append("EbookLib")

    if missing:
        print(f"[!] Missing dependencies: {', '.join(missing)}")
        print(f"    Please run: pip install {' '.join(missing)}")
        return False
    return True


def split_pdf(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    options: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Splits a PDF into chunks of `chunk_size` pages.
    Returns a list of paths to the temporary chunk files.
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    print(f"[*] Total pages: {total_pages}")

    if total_pages > MAX_DAILY_PAGES:
        print(
            f"[!] WARNING: This document ({total_pages} pages) exceeds the daily API limit of {MAX_DAILY_PAGES} pages."
        )
        print("    Processing may fail or get blocked if you exceed your quota.")

    chunk_paths = []
    temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_")

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        # Create a new PDF for this chunk
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

        for page in chunk_doc:
            force_rotate = PADDLE_FORCE_ROTATE
            pad_x_ratio = PADDLE_PAGE_PADDING_X
            pad_top_ratio = PADDLE_PAGE_PADDING_TOP
            pad_bottom_ratio = PADDLE_PAGE_PADDING_BOTTOM

            if options:
                if isinstance(options.get("force_rotate"), int):
                    force_rotate = options["force_rotate"]
                if isinstance(options.get("padding_x"), (int, float)):
                    pad_x_ratio = float(options["padding_x"])
                if isinstance(options.get("padding_top"), (int, float)):
                    pad_top_ratio = float(options["padding_top"])
                if isinstance(options.get("padding_bottom"), (int, float)):
                    pad_bottom_ratio = float(options["padding_bottom"])

            if force_rotate:
                page.set_rotation((page.rotation + force_rotate) % 360)

            rect = page.rect
            pad_x = rect.width * max(0.0, pad_x_ratio)
            pad_top = rect.height * max(0.0, pad_top_ratio)
            pad_bottom = rect.height * max(0.0, pad_bottom_ratio)

            if pad_x or pad_top or pad_bottom:
                page.set_mediabox(
                    fitz.Rect(
                        -pad_x,
                        -pad_top,
                        rect.width + pad_x,
                        rect.height + pad_bottom,
                    )
                )

                if pad_x:
                    page.draw_rect(
                        fitz.Rect(-pad_x, -pad_top, 0, rect.height + pad_bottom),
                        color=None,
                        fill=(1, 1, 1),
                    )
                    page.draw_rect(
                        fitz.Rect(rect.width, -pad_top, rect.width + pad_x, rect.height + pad_bottom),
                        color=None,
                        fill=(1, 1, 1),
                    )
                if pad_top:
                    page.draw_rect(
                        fitz.Rect(0, -pad_top, rect.width, 0),
                        color=None,
                        fill=(1, 1, 1),
                    )
                if pad_bottom:
                    page.draw_rect(
                        fitz.Rect(0, rect.height, rect.width, rect.height + pad_bottom),
                        color=None,
                        fill=(1, 1, 1),
                    )

        chunk_filename = os.path.join(temp_dir, f"chunk_{start_page}_{end_page}.pdf")
        chunk_doc.save(chunk_filename)
        chunk_doc.close()
        chunk_paths.append(chunk_filename)

    doc.close()
    return chunk_paths


def build_paddle_optional_payload(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "useDocOrientationClassify": PADDLE_USE_DOC_ORIENTATION_CLASSIFY,
        "useDocUnwarping": PADDLE_USE_DOC_UNWARPING,
        "useChartRecognition": PADDLE_USE_CHART_RECOGNITION,
        "layoutThreshold": PADDLE_LAYOUT_THRESHOLD,
        "temperature": PADDLE_TEMPERATURE,
        "repetitionPenalty": PADDLE_REPETITION_PENALTY,
        "topP": PADDLE_TOP_P,
    }

    def set_opt(key: str, value: Any) -> None:
        if value is not None:
            payload[key] = value

    set_opt("useLayoutDetection", PADDLE_USE_LAYOUT_DETECTION)
    set_opt("layoutNms", PADDLE_LAYOUT_NMS)
    set_opt("layoutUnclipRatio", PADDLE_LAYOUT_UNCLIP_RATIO)
    set_opt("layoutMergeBboxesMode", PADDLE_LAYOUT_MERGE_BBOXES_MODE)
    set_opt("layoutShapeMode", PADDLE_LAYOUT_SHAPE_MODE)
    set_opt("promptLabel", PADDLE_PROMPT_LABEL)
    set_opt("minPixels", PADDLE_MIN_PIXELS)
    set_opt("maxPixels", PADDLE_MAX_PIXELS)
    set_opt("prettifyMarkdown", PADDLE_PRETTIFY_MARKDOWN)
    set_opt("visualize", PADDLE_VISUALIZE)

    if overrides:
        payload.update(overrides)

    return payload


def parse_pdf_chunk(
    chunk_path: str,
    token: str,
    optional_payload_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    Sends a PDF chunk to the PaddleOCR async job API and returns the parsed result.
    Submits a job, polls until completion, then downloads and aggregates JSONL results.
    """
    print(f"[*] uploading chunk: {os.path.basename(chunk_path)}")

    headers = {"Authorization": f"Bearer {token}"}

    optional_payload = build_paddle_optional_payload(optional_payload_overrides)

    data = {
        "model": MODEL_VERSION,
        "optionalPayload": json.dumps(optional_payload),
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": f}
                job_response = requests.post(
                    API_URL, headers=headers, data=data, files=files,
                    timeout=API_TIMEOUT_SECONDS,
                )
            job_response.raise_for_status()
            job_result = job_response.json()

            if job_result.get("errorCode") and job_result["errorCode"] != 0:
                print(
                    f"[!] Job submission error for chunk {os.path.basename(chunk_path)}: "
                    f"{job_result.get('errorMsg', 'unknown error')}"
                )
                return None

            job_id = job_result["data"]["jobId"]
            print(f"[*] job submitted: {job_id}")

            # Poll for completion
            poll_start = time.time()
            while True:
                if time.time() - poll_start > PADDLE_MAX_POLL_TIME:
                    print(f"[!] Timed out waiting for job {job_id}")
                    return None
                time.sleep(PADDLE_POLL_INTERVAL)
                poll_response = requests.get(
                    f"{API_URL}/{job_id}", headers=headers, timeout=API_TIMEOUT_SECONDS,
                )
                poll_response.raise_for_status()
                poll_result = poll_response.json()
                state = poll_result["data"]["state"]

                if state == "pending":
                    print("    job pending...")
                elif state == "running":
                    try:
                        progress = poll_result["data"]["extractProgress"]
                        print(
                            f"    running: {progress['extractedPages']}/{progress['totalPages']} pages"
                        )
                    except KeyError:
                        print("    running...")
                elif state == "done":
                    extracted = poll_result["data"]["extractProgress"]["extractedPages"]
                    print(f"    done: {extracted} pages extracted")
                    jsonl_url = poll_result["data"]["resultUrl"]["jsonUrl"]
                    break
                elif state == "failed":
                    error_msg = poll_result["data"].get("errorMsg", "unknown error")
                    print(f"[!] Job failed: {error_msg}")
                    return None

            # Download and parse JSONL results
            jsonl_response = requests.get(jsonl_url, timeout=API_TIMEOUT_SECONDS)
            jsonl_response.raise_for_status()

            layout_results = []
            for line in jsonl_response.text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    page_result = json.loads(line).get("result", {})
                except json.JSONDecodeError:
                    print(f"[!] Skipping malformed JSONL line")
                    continue
                for page_res in page_result.get("layoutParsingResults", []):
                    layout_results.append(page_res)

            print(f"[*] chunk result: {len(layout_results)} pages")
            return {"result": {"layoutParsingResults": layout_results}}

        except (requests.exceptions.RequestException, ConnectionError) as e:
            wait_time = (2 ** attempt) * 5
            print(f"[!] API Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(
                    f"[!] Permanently failed processing chunk: {os.path.basename(chunk_path)}"
                )
                return None
        except Exception as e:
            print(
                f"[!] Unexpected error processing chunk {os.path.basename(chunk_path)}: {e}"
            )
            return None
