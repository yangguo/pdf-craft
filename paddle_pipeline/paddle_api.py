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
    PADDLE_BOTTOM_PADDING_PERCENT,
    PADDLE_PAGE_MARGIN_PT,
    PADDLE_POLL_INTERVAL,
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


def _pad_page_for_ocr(
    page: Any,
    side_margin_pt: int = PADDLE_PAGE_MARGIN_PT,
    bottom_padding_percent: int = PADDLE_BOTTOM_PADDING_PERCENT,
) -> None:
    """Add white OCR guard bands around text close to page edges."""
    rect = page.rect
    side_margin = max(0, side_margin_pt)
    bottom_padding = max(0, bottom_padding_percent)
    new_h = rect.height * (1 + bottom_padding / 100)
    page.set_mediabox(fitz.Rect(-side_margin, 0, rect.width + side_margin, new_h))

    if side_margin:
        page.draw_rect(
            fitz.Rect(-side_margin, 0, 0, new_h),
            color=None, fill=(1, 1, 1),
        )
        page.draw_rect(
            fitz.Rect(rect.width, 0, rect.width + side_margin, new_h),
            color=None, fill=(1, 1, 1),
        )

    if new_h > rect.height:
        page.draw_rect(
            fitz.Rect(-side_margin, rect.height, rect.width + side_margin, new_h),
            color=None, fill=(1, 1, 1),
        )


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

        # Add white guard bands so the OCR model does not crop or mangle
        # edge-adjacent text, especially vertical CJK columns.
        for page in chunk_doc:
            _pad_page_for_ocr(page)

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

    optional_payload = {
        "useDocOrientationClassify": True,
        "useDocUnwarping": True,
        "useChartRecognition": False,
        "layoutThreshold": 0.5,
        "temperature": 0.2,
        "repetitionPenalty": 1.2,
        "topP": 0.85,
    }

    data = {
        "model": MODEL_VERSION,
        "optionalPayload": json.dumps(optional_payload),
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            try:
                with open(chunk_path, "rb") as f:
                    files = {"file": f}
                    job_response = requests.post(
                        API_URL, headers=headers, data=data, files=files,
                        timeout=API_TIMEOUT_SECONDS,
                    )
            except (requests.exceptions.RequestException, ConnectionError) as e:
                # Submission timeout/transport failures are ambiguous for
                # non-idempotent async jobs; retrying can create duplicates.
                print(f"[!] Job submission transport failure: {e}")
                print(
                    f"[!] Permanently failed processing chunk: {os.path.basename(chunk_path)}"
                )
                return None
            job_response.raise_for_status()
            job_result = job_response.json()

            if job_result.get("errorCode") and job_result["errorCode"] != 0:
                print(
                    f"[!] Job submission error for chunk {os.path.basename(chunk_path)}: "
                    f"{job_result.get('errorMsg', 'unknown error')}"
                )
                return None

            job_data = job_result.get("data")
            job_id = job_data.get("jobId") if isinstance(job_data, dict) else None
            if not isinstance(job_id, str) or not job_id:
                wait_time = (2 ** attempt) * 5
                print(
                    f"[!] Malformed job submission response "
                    f"(attempt {attempt + 1}/{max_retries}): missing data.jobId"
                )
                if attempt < max_retries - 1:
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(
                    f"[!] Permanently failed processing chunk: {os.path.basename(chunk_path)}"
                )
                return None
            print(f"[*] job submitted: {job_id}")

            # Poll for completion
            poll_start = time.time()
            poll_error_count = 0
            skip_interval = False
            while True:
                if time.time() - poll_start > PADDLE_MAX_POLL_TIME:
                    print(f"[!] Timed out waiting for job {job_id}")
                    return None
                if not skip_interval:
                    time.sleep(PADDLE_POLL_INTERVAL)
                skip_interval = False
                try:
                    poll_response = requests.get(
                        f"{API_URL}/{job_id}", headers=headers, timeout=API_TIMEOUT_SECONDS,
                    )
                    poll_response.raise_for_status()
                    poll_error_count = 0
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    poll_error_count += 1
                    wait_time = (2 ** (poll_error_count - 1)) * 5
                    print(
                        f"[!] Job poll failed "
                        f"(attempt {poll_error_count}/{max_retries}): {e}"
                    )
                    if poll_error_count < max_retries:
                        print(f"    Retrying job poll in {wait_time}s...")
                        time.sleep(wait_time)
                        skip_interval = True
                        continue
                    print(
                        f"[!] Permanently failed polling submitted job {job_id}; "
                        "not resubmitting chunk."
                    )
                    return None
                poll_result = poll_response.json()
                poll_data = poll_result.get("data")
                state = poll_data.get("state") if isinstance(poll_data, dict) else None
                if not isinstance(state, str) or not state:
                    poll_error_count += 1
                    wait_time = (2 ** (poll_error_count - 1)) * 5
                    print(
                        f"[!] Malformed job poll response "
                        f"(attempt {poll_error_count}/{max_retries}): missing data.state"
                    )
                    if poll_error_count < max_retries:
                        print(f"    Retrying job poll in {wait_time}s...")
                        time.sleep(wait_time)
                        skip_interval = True
                        continue
                    print(
                        f"[!] Permanently failed polling submitted job {job_id}; "
                        "not resubmitting chunk."
                    )
                    return None

                if state == "pending":
                    print("    job pending...")
                elif state == "running":
                    try:
                        progress = poll_data["extractProgress"]
                        print(
                            f"    running: {progress['extractedPages']}/{progress['totalPages']} pages"
                        )
                    except KeyError:
                        print("    running...")
                elif state == "done":
                    progress = poll_data.get("extractProgress", {})
                    extracted = (
                        progress.get("extractedPages")
                        if isinstance(progress, dict) else None
                    )
                    if extracted is None:
                        print("    done: pages extracted")
                    else:
                        print(f"    done: {extracted} pages extracted")
                    result_url = poll_data.get("resultUrl")
                    jsonl_url = result_url.get("jsonUrl") if isinstance(result_url, dict) else None
                    if not isinstance(jsonl_url, str) or not jsonl_url:
                        poll_error_count += 1
                        wait_time = (2 ** (poll_error_count - 1)) * 5
                        print(
                            f"[!] Malformed job poll response "
                            f"(attempt {poll_error_count}/{max_retries}): missing data.resultUrl.jsonUrl"
                        )
                        if poll_error_count < max_retries:
                            print(f"    Retrying job poll in {wait_time}s...")
                            time.sleep(wait_time)
                            skip_interval = True
                            continue
                        print(
                            f"[!] Permanently failed polling submitted job {job_id}; "
                            "not resubmitting chunk."
                        )
                        return None
                    break
                elif state == "failed":
                    error_msg = poll_data.get("errorMsg", "unknown error")
                    print(f"[!] Job failed: {error_msg}")
                    return None

            # Download and parse JSONL results. This is separate from job
            # submission so a transient object-storage read failure does not
            # resubmit a completed OCR job.
            jsonl_text = ""
            for download_attempt in range(max_retries):
                try:
                    jsonl_response = requests.get(
                        jsonl_url, timeout=API_TIMEOUT_SECONDS
                    )
                    jsonl_response.raise_for_status()
                    jsonl_text = jsonl_response.text
                    break
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    wait_time = (2 ** download_attempt) * 5
                    print(
                        f"[!] JSON result download failed "
                        f"(attempt {download_attempt + 1}/{max_retries}): {e}"
                    )
                    if download_attempt < max_retries - 1:
                        print(f"    Retrying JSON download in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(
                            f"[!] Permanently failed downloading result for job {job_id}; "
                            "not resubmitting chunk."
                        )
                        return None

            layout_results = []
            for line in jsonl_text.strip().split("\n"):
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
            print(f"[!] API Request failed: {e}")
            print(
                f"[!] Permanently failed processing chunk: {os.path.basename(chunk_path)}"
            )
            return None
        except Exception as e:
            print(
                f"[!] Unexpected error processing chunk {os.path.basename(chunk_path)}: {e}"
            )
            return None
