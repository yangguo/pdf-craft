"""PDF chunking, Paddle OCR API calls, and dependency checking."""

import base64
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

        # Add bottom padding to prevent OCR model from cropping text at
        # the page edge (observed on scanned books where the last line
        # sits very close to the physical page bottom).
        for page in chunk_doc:
            rect = page.rect
            new_h = rect.height * 1.05
            page.set_mediabox(fitz.Rect(0, 0, rect.width, new_h))
            page.draw_rect(
                fitz.Rect(0, rect.height, rect.width, new_h),
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
    Sends a PDF chunk to the PaddleOCR API and returns the parsed result.
    """
    print(f"[*] uploading chunk: {os.path.basename(chunk_path)}")

    with open(chunk_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    payload = {
        "file": file_data,
        "fileType": 0,  # 0 for PDF
        "useDocOrientationClassify": True,
        "useDocUnwarping": True,
        "useChartRecognition": False,
        "layoutThreshold": 0.3,
        "textRecScoreThresh": 0.3,
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=API_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result = response.json()

            # Handle API responses safely
            # The API seems to return 'result' directly in some cases, or an 'error' field
            if "error" in result:
                print(
                    f"[!] API Error for chunk {os.path.basename(chunk_path)}: {result['error']}"
                )
                return None

            return result

        except (requests.exceptions.RequestException, ConnectionError) as e:
            wait_time = (
                2**attempt
            ) * 5  # Exponential backoff: 5, 10, 20, 40, 80 seconds
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

