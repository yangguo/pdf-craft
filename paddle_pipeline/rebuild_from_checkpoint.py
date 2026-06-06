"""Rebuild an EPUB from OCR checkpoints and an optional manual TOC file."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from .config import DEFAULT_EPUB_LANGUAGE
from .epub_builder import create_epub


_CHECKPOINT_RE = re.compile(r"^chunk_(\d+)_(\d+)\.pdf\.json$")


def _checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    match = _CHECKPOINT_RE.match(os.path.basename(path))
    if not match:
        return (10**9, 10**9, path)
    return (int(match.group(1)), int(match.group(2)), path)


def load_checkpoint_results(work_dir: str) -> list[dict[str, Any]]:
    """Load chunk checkpoint JSON files in numeric page order."""
    paths = [
        os.path.join(work_dir, name)
        for name in os.listdir(work_dir)
        if _CHECKPOINT_RE.match(name)
    ]
    results: list[dict[str, Any]] = []
    for path in sorted(paths, key=_checkpoint_sort_key):
        with open(path, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def load_manual_toc(path: str | None) -> list[dict[str, Any]] | None:
    """Load a manual TOC JSON list or an object with a ``headings`` list."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("headings"), list):
        return data["headings"]
    raise ValueError("Manual TOC must be a JSON list or an object with headings[].")


def rebuild_epub_from_checkpoint(
    work_dir: str,
    output: str,
    *,
    title: str,
    author: str | None = None,
    toc_path: str | None = None,
    cover: str | None = None,
    language: str = DEFAULT_EPUB_LANGUAGE,
    strict_ocr_validation: bool = False,
) -> None:
    """Rebuild an EPUB using existing OCR checkpoint data."""
    results = load_checkpoint_results(work_dir)
    if not results:
        raise ValueError(f"No checkpoint files found in {work_dir}")

    image_dir = os.path.join(work_dir, "images")
    cover_path = cover
    if cover_path is None:
        candidate = os.path.join(work_dir, "cover.jpg")
        if os.path.exists(candidate):
            cover_path = candidate

    create_epub(
        title,
        results,
        output,
        image_dir,
        cover_image_path=cover_path,
        author=author,
        confirmed_headings=load_manual_toc(toc_path),
        language=language,
        strict_ocr_validation=strict_ocr_validation,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild an EPUB from existing OCR checkpoints."
    )
    parser.add_argument("work_dir", help="Directory containing chunk_*.pdf.json files")
    parser.add_argument("--output", "-o", required=True, help="Output EPUB path")
    parser.add_argument("--title", required=True, help="EPUB title")
    parser.add_argument("--author", help="EPUB author")
    parser.add_argument("--toc", help="Manual TOC JSON file")
    parser.add_argument("--cover", help="Cover image path; defaults to work_dir/cover.jpg")
    parser.add_argument(
        "--language",
        default=DEFAULT_EPUB_LANGUAGE,
        help=f"EPUB language tag (default: {DEFAULT_EPUB_LANGUAGE})",
    )
    parser.add_argument(
        "--strict-ocr-noise",
        action="store_true",
        help="Fail EPUB generation when suspicious OCR artifacts remain",
    )
    args = parser.parse_args(argv)

    rebuild_epub_from_checkpoint(
        args.work_dir,
        args.output,
        title=args.title,
        author=args.author,
        toc_path=args.toc,
        cover=args.cover,
        language=args.language,
        strict_ocr_validation=args.strict_ocr_noise,
    )


if __name__ == "__main__":
    main()
