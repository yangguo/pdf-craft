"""PDF chunking, Paddle OCR API calls, and dependency checking."""

import json
import os
import tempfile
import time

from typing import Any, Dict, List

from .config import (
    API_TIMEOUT_SECONDS,
    API_URL,
    CHUNK_SIZE,
    MAX_DAILY_PAGES,
    MODEL_VERSION,
    PADDLE_LAYOUT_THRESHOLD,
    PADDLE_MAX_POLL_TIME,
    PADDLE_PAGE_BOTTOM_PADDING_RATIO,
    PADDLE_PAGE_TOP_PADDING_RATIO,
    PADDLE_POLL_INTERVAL,
    PADDLE_REPETITION_PENALTY,
    PADDLE_TEMPERATURE,
    PADDLE_TOP_P,
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


def _build_optional_payload() -> Dict[str, Any]:
    """Return PaddleOCR optional payload tuned for vertical/traditional books."""
    return {
        "useDocOrientationClassify": True,
        "useDocUnwarping": True,
        "useChartRecognition": False,
        "layoutThreshold": PADDLE_LAYOUT_THRESHOLD,
        "temperature": PADDLE_TEMPERATURE,
        "repetitionPenalty": PADDLE_REPETITION_PENALTY,
        "topP": PADDLE_TOP_P,
    }


def split_pdf(file_path: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
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

        # Add vertical padding to reduce cropping for scanned books whose
        # first/last vertical-text characters sit close to the page edge.
        for page in chunk_doc:
            rect = page.rect
            top_padding = rect.height * PADDLE_PAGE_TOP_PADDING_RATIO
            bottom_padding = rect.height * PADDLE_PAGE_BOTTOM_PADDING_RATIO
            page.set_mediabox(
                fitz.Rect(0, -top_padding, rect.width, rect.height + bottom_padding)
            )
            page.draw_rect(
                fitz.Rect(0, -top_padding, rect.width, 0),
                color=None, fill=(1, 1, 1),
            )
            page.draw_rect(
                fitz.Rect(0, rect.height, rect.width, rect.height + bottom_padding),
                color=None, fill=(1, 1, 1),
            )

        chunk_filename = os.path.join(temp_dir, f"chunk_{start_page}_{end_page}.pdf")
        chunk_doc.save(chunk_filename)
        chunk_doc.close()
        chunk_paths.append(chunk_filename)

    doc.close()
    return chunk_paths


def parse_pdf_chunk(chunk_path: str, token: str) -> Dict[str, Any] | None:
    """
    Sends a PDF chunk to the PaddleOCR async job API and returns the parsed result.
    Submits a job, polls until completion, then downloads and aggregates JSONL results.
    """
    print(f"[*] uploading chunk: {os.path.basename(chunk_path)}")

    headers = {"Authorization": f"Bearer {token}"}

    data = {
        "model": MODEL_VERSION,
        "optionalPayload": json.dumps(_build_optional_payload()),
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
                    print("[!] Skipping malformed JSONL line")
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
