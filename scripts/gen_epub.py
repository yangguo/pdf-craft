import argparse
import hashlib
from pathlib import Path

from pdf_craft import LaTeXRender, OCREventKind, TableRender, transform_epub


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF to EPUB using DeepSeek OCR.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to the output EPUB file (default: <pdf_stem>.epub in analysing dir).")
    parser.add_argument("--analysing-dir", type=Path, default=None, help="Directory for intermediate OCR/analysis files (default: analysing_<pdf_stem>_<hash>/).")
    parser.add_argument("--models-cache", type=Path, default=None, help="Directory for cached OCR models (default: models-cache/).")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pdf_path: Path = args.pdf.resolve()
    stem = pdf_path.stem

    # Include both the resolved path and a lightweight content signature so the
    # default OCR cache is invalidated when the same file path is replaced.
    pdf_stat = pdf_path.stat()
    _path_hash = hashlib.md5(
        f"{pdf_path}:{pdf_stat.st_mtime_ns}:{pdf_stat.st_size}".encode()
    ).hexdigest()[:8]
    analysing_dir_path = (args.analysing_dir or project_root / f"analysing_{stem}_{_path_hash}").resolve()
    epub_path = (args.output or analysing_dir_path / f"{stem}.epub").resolve()
    models_cache_path = (args.models_cache or project_root / "models-cache").resolve()

    print(f"Input:  {pdf_path}")
    print(f"Output: {epub_path}")

    transform_epub(
        pdf_path=pdf_path,
        epub_path=epub_path,
        analysing_path=analysing_dir_path,
        models_cache_path=models_cache_path,
        includes_footnotes=True,
        generate_plot=True,
        table_render=TableRender.HTML,
        latex_render=LaTeXRender.MATHML,
        on_ocr_event=lambda e: print(
            f"OCR {OCREventKind(e.kind).name} - Page {e.page_index}/{e.total_pages} - {_format_duration(e.cost_time_ms)}"
        ),
    )

    print(f"\nDone! EPUB saved to: {epub_path}")


def _format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        seconds = ms / 1000
        return f"{seconds:.2f}s"
    else:
        minutes = ms // 60000
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.2f}s"


if __name__ == "__main__":
    main()
