"""CLI entry point for the Paddle OCR PDF-to-EPUB pipeline."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

from .config import (
    tqdm,
    API_TOKEN,
    MINERU_API_TOKEN,
    DEFAULT_COVER_JPEG_QUALITY,
    DEFAULT_COVER_MAX_EDGE,
    DEFAULT_EPUB_LANGUAGE,
)
from .epub_builder import create_epub
from .metadata import (
    download_image,
    extract_candidate_headings,
    extract_cover_image,
    extract_metadata_interactive,
    filter_heading_candidates,
    review_toc_interactive,
)
from .page_image_fallback import apply_page_image_fallbacks
from .page_order_repair import repair_page_order_by_printed_numbers
from .paddle_api import (
    check_dependencies,
    parse_pdf_chunk,
    split_pdf,
)
from . import mineru_api


def main():
    if not check_dependencies():
        return

    def _parse_bool_string(value: str) -> bool:
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError("expected a boolean value")

    parser = argparse.ArgumentParser(description="Scanned PDF to Epub Converter")
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument(
        "--output", "-o", help="Path to output EPUB file (default: input_name.epub)"
    )
    parser.add_argument("--title", help="Book title (skips interactive prompt)")
    parser.add_argument("--author", help="Author name (skips interactive prompt)")
    parser.add_argument("--language", default=DEFAULT_EPUB_LANGUAGE,
                        help=f"EPUB language tag (default: {DEFAULT_EPUB_LANGUAGE})")
    parser.add_argument("--cover-max-edge", type=int, default=DEFAULT_COVER_MAX_EDGE,
                        help=f"Max cover image edge in pixels (default: {DEFAULT_COVER_MAX_EDGE})")
    parser.add_argument("--cover-quality", type=int, default=DEFAULT_COVER_JPEG_QUALITY,
                        help=f"JPEG cover quality 1-100 (default: {DEFAULT_COVER_JPEG_QUALITY})")
    parser.add_argument("--strict-ocr-noise", action="store_true",
                        help="Fail EPUB generation when suspicious OCR artifacts remain")
    parser.add_argument("--auto-toc", action="store_true",
                        help="Skip interactive TOC review; use auto-detected headings")
    parser.add_argument("--no-toc", action="store_true",
                        help="Skip heading detection; produce single-chapter EPUB")
    parser.add_argument("--api", choices=["paddle", "mineru"], default="paddle",
                        help="OCR API backend to use (default: paddle)")
    parser.add_argument(
        "--ocr-preset",
        choices=["default", "vertical-zh-hant"],
        default="default",
        help="OCR tuning preset (default: default)",
    )
    parser.add_argument(
        "--paddle-use-layout-detection",
        type=_parse_bool_string,
        default=None,
        help="Paddle optionalPayload.useLayoutDetection (true/false)",
    )
    parser.add_argument(
        "--paddle-layout-shape-mode",
        choices=["rect", "quad", "poly", "auto"],
        default=None,
        help="Paddle optionalPayload.layoutShapeMode",
    )
    parser.add_argument(
        "--paddle-layout-merge-bboxes-mode",
        choices=["large", "small", "union"],
        default=None,
        help="Paddle optionalPayload.layoutMergeBboxesMode",
    )
    parser.add_argument(
        "--paddle-temperature",
        type=float,
        default=None,
        help="Paddle optionalPayload.temperature",
    )
    parser.add_argument(
        "--paddle-top-p",
        type=float,
        default=None,
        help="Paddle optionalPayload.topP",
    )
    parser.add_argument(
        "--paddle-repetition-penalty",
        type=float,
        default=None,
        help="Paddle optionalPayload.repetitionPenalty",
    )
    parser.add_argument(
        "--paddle-min-pixels",
        type=int,
        default=None,
        help="Paddle optionalPayload.minPixels",
    )
    parser.add_argument(
        "--paddle-max-pixels",
        type=int,
        default=None,
        help="Paddle optionalPayload.maxPixels",
    )
    parser.add_argument(
        "--paddle-prettify-markdown",
        type=_parse_bool_string,
        default=None,
        help="Paddle optionalPayload.prettifyMarkdown (true/false)",
    )
    parser.add_argument(
        "--paddle-visualize",
        type=_parse_bool_string,
        default=None,
        help="Paddle optionalPayload.visualize (true/false)",
    )
    parser.add_argument(
        "--paddle-force-rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=None,
        help="Rotate pages before OCR (0/90/180/270)",
    )
    parser.add_argument(
        "--paddle-padding-x",
        type=float,
        default=None,
        help="Horizontal page padding ratio (applied to both left and right)",
    )
    parser.add_argument(
        "--paddle-padding-y",
        type=float,
        default=None,
        help="Vertical page padding ratio (applied to both top and bottom)",
    )
    parser.add_argument(
        "--mineru-language",
        default=None,
        help="MinerU OCR language (e.g. chinese_cht, ch_server)",
    )
    args = parser.parse_args()

    input_path = args.input_pdf
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    if args.auto_toc and args.no_toc:
        print("[!] Error: --auto-toc and --no-toc are mutually exclusive.")
        return

    if args.api == "mineru":
        if not MINERU_API_TOKEN:
            print("[!] Error: MINERU_API_TOKEN environment variable is not set.")
            print("    Please set it using: export MINERU_API_TOKEN='your_token_here'")
            return
    elif not API_TOKEN:
        print("[!] Error: PADDLE_API_TOKEN environment variable is not set.")
        print("    Please set it using: export PADDLE_API_TOKEN='your_token_here'")
        return

    if not args.output:
        args.output = os.path.splitext(input_path)[0] + ".epub"

    # Create a unique work directory keyed on both the resolved path and the
    # file's current content (mtime + size as a fast proxy). This means:
    #  - Two PDFs with the same name in different directories get separate dirs.
    #  - Replacing book.pdf in-place invalidates the existing checkpoints so
    #    stale OCR output is never silently reused for a changed document.
    _stat = os.stat(input_path)
    _content_key = f"{os.path.realpath(input_path)}:{_stat.st_mtime}:{_stat.st_size}"
    file_hash = hashlib.md5(_content_key.encode("utf-8")).hexdigest()[:8]
    work_dir = f"paddle_epub_work_{file_hash}"

    print(f"[*] Work directory: {work_dir}")

    image_dir = os.path.join(work_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    chunk_paths = []  # populated by split_pdf(); kept here so finally block can clean up
    chunk_temp_dir = None

    try:
        paddle_optional_payload_overrides = {}
        paddle_split_options = {}
        mineru_options = {}

        if args.ocr_preset == "vertical-zh-hant":
            paddle_optional_payload_overrides.update(
                {"temperature": 0.1, "topP": 0.75, "useLayoutDetection": True}
            )
            mineru_options["language"] = "chinese_cht"

        if args.paddle_use_layout_detection is not None:
            paddle_optional_payload_overrides["useLayoutDetection"] = args.paddle_use_layout_detection
        if args.paddle_layout_shape_mode is not None:
            paddle_optional_payload_overrides["layoutShapeMode"] = args.paddle_layout_shape_mode
        if args.paddle_layout_merge_bboxes_mode is not None:
            paddle_optional_payload_overrides["layoutMergeBboxesMode"] = args.paddle_layout_merge_bboxes_mode
        if args.paddle_temperature is not None:
            paddle_optional_payload_overrides["temperature"] = args.paddle_temperature
        if args.paddle_top_p is not None:
            paddle_optional_payload_overrides["topP"] = args.paddle_top_p
        if args.paddle_repetition_penalty is not None:
            paddle_optional_payload_overrides["repetitionPenalty"] = args.paddle_repetition_penalty
        if args.paddle_min_pixels is not None:
            paddle_optional_payload_overrides["minPixels"] = args.paddle_min_pixels
        if args.paddle_max_pixels is not None:
            paddle_optional_payload_overrides["maxPixels"] = args.paddle_max_pixels
        if args.paddle_prettify_markdown is not None:
            paddle_optional_payload_overrides["prettifyMarkdown"] = args.paddle_prettify_markdown
        if args.paddle_visualize is not None:
            paddle_optional_payload_overrides["visualize"] = args.paddle_visualize

        if args.paddle_force_rotate is not None:
            paddle_split_options["force_rotate"] = args.paddle_force_rotate
        if args.paddle_padding_x is not None:
            paddle_split_options["padding_x"] = args.paddle_padding_x
        if args.paddle_padding_y is not None:
            paddle_split_options["padding_top"] = args.paddle_padding_y
            paddle_split_options["padding_bottom"] = args.paddle_padding_y

        if args.mineru_language is not None and args.mineru_language.strip():
            mineru_options["language"] = args.mineru_language.strip()

        # Step 1: Chunking
        print("[-] Step 1: Splitting PDF...")
        if args.api == "mineru":
            chunk_paths = mineru_api.split_pdf(input_path)
        else:
            chunk_paths = split_pdf(input_path, options=paddle_split_options)
        # split_pdf stores chunks in a fresh mkdtemp; capture it for cleanup
        chunk_temp_dir = os.path.dirname(chunk_paths[0]) if chunk_paths else None

        # Step 1.5: Extract cover image
        cover_path = extract_cover_image(
            input_path,
            os.path.join(work_dir, "cover.jpg"),
            max_edge=args.cover_max_edge,
            jpg_quality=args.cover_quality,
        )

        # Step 2: API Processing
        results = []
        total = len(chunk_paths)
        api_label = "MinerU" if args.api == "mineru" else "PaddleOCR"
        api_token = MINERU_API_TOKEN if args.api == "mineru" else API_TOKEN
        print(f"[-] Step 2: Processing {total} chunks via {api_label} API...")
        if tqdm is not None:
            pbar = tqdm(total=total, unit="chunk", ncols=80,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            _write = pbar.write
        else:
            pbar = None
            _write = print
        for i, chunk in enumerate(chunk_paths):
            chunk_name = os.path.basename(chunk)
            json_checkpoint = os.path.join(work_dir, chunk_name + ".json")

            if os.path.exists(json_checkpoint):
                with open(json_checkpoint, "r") as f:
                    res = json.load(f)
                # MinerU checkpoints stash the result ZIP URL so they
                # can be re-parsed with the latest _parse_zip logic.
                if args.api == "mineru" and "_mineru_zip_url" in res:
                    reparsed = mineru_api.reparse_checkpoint(res, api_token)
                    if reparsed is not res:
                        res = reparsed
                        with open(json_checkpoint, "w") as f:
                            json.dump(res, f)
            else:
                # Rate limiting: Sleep before new request
                if i > 0:
                    _write("    ...waiting 5s to respect API rate limits...")
                    time.sleep(5)

                _MAX_RETRIES = 3
                for attempt in range(1, _MAX_RETRIES + 1):
                    if args.api == "mineru":
                        res = mineru_api.parse_pdf_chunk(chunk, api_token, mineru_options)
                    else:
                        res = parse_pdf_chunk(chunk, api_token, paddle_optional_payload_overrides)
                    if res:
                        break
                    if attempt < _MAX_RETRIES:
                        wait = attempt * 10
                        _write(
                            f"    ...chunk {i + 1} attempt {attempt}/{_MAX_RETRIES} "
                            f"failed ({api_label} parsing error), retrying in {wait}s..."
                        )
                        time.sleep(wait)

                if res:
                    # Save checkpoint
                    os.makedirs(os.path.dirname(json_checkpoint), exist_ok=True)
                    with open(json_checkpoint, "w") as f:
                        json.dump(res, f)

            if res:
                results.append(res)

                # Download images immediately to save locally
                layout_results = res.get("result", {}).get("layoutParsingResults", [])
                for page_res in layout_results:
                    images_map = page_res["markdown"].get("images", {})
                    for rel_path, img_url in images_map.items():
                        local_path = os.path.join(image_dir, rel_path)
                        if not os.path.exists(
                            local_path
                        ):  # Don't re-download if exists
                            download_image(img_url, local_path)
            else:
                _write(
                    f"[!] CRITICAL: Failed to process chunk {i + 1} "
                    f"after {_MAX_RETRIES} attempts. Aborting to prevent "
                    f"incomplete book."
                )
                _write(
                    "    Please resolve connectivity issues and re-run the script to resume."
                )
                if pbar is not None:
                    pbar.close()
                sys.exit(1)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Step 2.25: Render whole-page fallbacks for visual pages Paddle did not
        # expose as image assets (for example rotated genealogy diagrams).
        # NOTE: This runs before page order repair. Visual pages rarely carry
        # printed page numbers, but if they do, the fallback image may end up
        # at the wrong position after repair_page_order_by_printed_numbers swaps pages.
        apply_page_image_fallbacks(input_path, results, image_dir)

        # Step 2.35: Repair scanner page-order inversions before TOC detection
        # and chapter splitting. Some scans alternate adjacent printed pages
        # (e.g. 257, 256), while OCR results preserve PDF physical order.
        repair_page_order_by_printed_numbers(input_path, results)

        # Step 2.5: Metadata extraction
        default_title = os.path.splitext(os.path.basename(input_path))[0]
        if args.title:
            # Both title and author (if any) supplied via CLI — skip interactive prompt
            metadata = {"title": args.title, "author": args.author}
        elif args.author:
            # Author supplied but not title — infer/prompt for title only
            inferred = extract_metadata_interactive(results, default_title)
            metadata = {"title": inferred["title"], "author": args.author}
        else:
            metadata = extract_metadata_interactive(results, default_title)

        # Step 2.75: TOC Review
        if args.no_toc:
            confirmed_headings = []
        elif args.auto_toc:
            print("[-] Step 2.75: Auto-detecting chapter headings...")
            candidates = extract_candidate_headings(results)
            confirmed_headings = filter_heading_candidates(candidates)
            print(f"  ({len(confirmed_headings)} chapter headings detected)")
        else:
            print("[-] Step 2.75: Detecting chapter headings...")
            candidates = extract_candidate_headings(results)
            filtered = filter_heading_candidates(candidates)
            if len(candidates) > len(filtered):
                print(f"  ({len(filtered)} chapter headings found, {len(candidates) - len(filtered)} sub-headings hidden)")
            confirmed_headings = review_toc_interactive(filtered, all_candidates=candidates)

        # Step 3: Generation
        print("[-] Step 3: Generating EPUB...")
        create_epub(metadata["title"], results, args.output, image_dir,
                    cover_image_path=cover_path, author=metadata["author"],
                    confirmed_headings=confirmed_headings, language=args.language,
                    strict_ocr_validation=args.strict_ocr_noise)

    finally:
        # Clean up the temporary split-PDF directory created by split_pdf().
        # work_dir is intentionally kept so the user can inspect OCR checkpoints.
        if chunk_temp_dir and os.path.isdir(chunk_temp_dir):
            try:
                shutil.rmtree(chunk_temp_dir)
            except OSError as _e:
                print(f"[!] Could not remove temp chunk dir '{chunk_temp_dir}': {_e}")
        print(
            f"[*] Done. Intermediate files are in '{work_dir}'. You can delete this folder if verified."
        )


if __name__ == "__main__":
    main()
