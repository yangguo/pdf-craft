from __future__ import annotations

import importlib.util
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "converting-pdfs-to-epub"
    / "scripts"
    / "pdf_to_epub_hybrid.py"
)
SPEC = importlib.util.spec_from_file_location("pdf_to_epub_hybrid", SCRIPT)
assert SPEC and SPEC.loader
hybrid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hybrid
SPEC.loader.exec_module(hybrid)


def _draw_repeated_furniture(
    page: fitz.Page,
    count: int = 10,
    start_y: int = 20,
    start_x: int = 10,
) -> None:
    for offset in range(count):
        y = start_y + offset * 3
        page.draw_line((start_x, y), (start_x + 20, y))


def test_candidate_visual_pages_calibrates_recurring_drawings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "visuals.pdf"
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    for _ in range(3):
        page = document.new_page()
        _draw_repeated_furniture(page)
        page.insert_text((50, 100), "Long prose page content that represents ordinary body text. " * 2)
    outlier = document.new_page()
    _draw_repeated_furniture(outlier, count=40)
    outlier.insert_text((50, 160), "Long prose page content that represents ordinary body text. " * 2)
    image_page = document.new_page()
    _draw_repeated_furniture(image_page)
    image_page.insert_text((50, 220), "Long prose page content that represents ordinary body text. " * 2)
    image_page.insert_image(fitz.Rect(100, 100, 180, 180), stream=image_buffer.getvalue())
    caption_page = document.new_page()
    caption_page.draw_line((50, 100), (250, 100))
    caption_page.insert_text((50, 80), "圖 1.1 測試圖", fontname="china-t")
    reference_page = document.new_page()
    reference_page.draw_line((50, 100), (250, 100))
    reference_page.insert_text((50, 80), "正文中請見圖 1.1。", fontname="china-t")
    document.save(pdf_path)
    document.close()

    with fitz.open(pdf_path) as parsed:
        assert hybrid.candidate_visual_pages(parsed) == {4, 5, 6}


def test_candidate_visual_pages_keeps_uniform_low_text_diagrams(tmp_path: Path) -> None:
    pdf_path = tmp_path / "diagrams.pdf"
    document = fitz.open()
    for _ in range(2):
        page = document.new_page()
        _draw_repeated_furniture(page, count=12, start_y=200, start_x=100)
        page.insert_text((50, 100), "Diagram")
    document.save(pdf_path)
    document.close()

    with fitz.open(pdf_path) as parsed:
        assert hybrid.candidate_visual_pages(parsed) == {2}


def test_reflow_text_blocks_joins_cjk_lines_at_paragraph_indentation() -> None:
    blocks = [
        (78.0, 100.0, 350.0, 112.0, "文化交流從古到今都是歷史家關心的課題，", 0, 0),
        (56.0, 116.0, 350.0, 128.0, "進入全球化時代，", 1, 0),
        (56.0, 132.0, 350.0, 144.0, "文化之間的相遇更受重視。", 2, 0),
        (78.0, 164.0, 350.0, 176.0, "第二段從縮排開始，", 3, 0),
        (56.0, 180.0, 350.0, 192.0, "接續文字不應被拆成另一段。", 4, 0),
    ]

    assert hybrid.reflow_text_blocks(blocks) == [
        "文化交流從古到今都是歷史家關心的課題，進入全球化時代，文化之間的相遇更受重視。",
        "第二段從縮排開始，接續文字不應被拆成另一段。",
    ]


def test_reflow_text_blocks_preserves_non_cjk_block_boundaries() -> None:
    blocks = [
        (78.0, 100.0, 350.0, 112.0, "First English block", 0, 0),
        (56.0, 116.0, 350.0, 128.0, "Second English block", 1, 0),
    ]

    assert hybrid.reflow_text_blocks(blocks) == [
        "First English block",
        "Second English block",
    ]


def test_reflow_text_blocks_removes_spaces_inside_multiline_cjk_block() -> None:
    blocks = [
        (78.0, 100.0, 350.0, 128.0, "文化交流從古到今\n都是歷史家關心的課題。", 0, 0),
    ]

    assert hybrid.reflow_text_blocks(blocks) == [
        "文化交流從古到今都是歷史家關心的課題。",
    ]


def _write_single_page_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "A short test document.")
    document.save(path)
    document.close()


def test_convert_writes_rich_metadata_options(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    _write_single_page_pdf(pdf_path)

    result = hybrid.main(
        [
            "convert",
            str(pdf_path),
            str(epub_path),
            "--title",
            "Test Book",
            "--author",
            "Test Editor",
            "--isbn",
            "978-1-2345-6789-0",
            "--publisher",
            "Test Press",
            "--published",
            "2026-07-18",
            "--description",
            "A conversion fixture.",
        ]
    )

    assert result == 0
    with zipfile.ZipFile(epub_path) as archive:
        opf_name = next(name for name in archive.namelist() if name.endswith(".opf"))
        opf = archive.read(opf_name).decode("utf-8")
    assert "urn:isbn:978-1-2345-6789-0" in opf
    assert "Test Press" in opf
    assert "2026-07-18" in opf
    assert "A conversion fixture." in opf


def test_convert_uses_selected_language_in_xhtml_root(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    _write_single_page_pdf(pdf_path)

    result = hybrid.main(
        [
            "convert",
            str(pdf_path),
            str(epub_path),
            "--language",
            "zh-Hant",
        ]
    )

    assert result == 0
    with zipfile.ZipFile(epub_path) as archive:
        chapter_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".xhtml") and not name.endswith(("nav.xhtml", "cover.xhtml"))
        )
        chapter = archive.read(chapter_name).decode("utf-8")
    assert 'lang="zh-Hant"' in chapter
    assert 'xml:lang="zh-Hant"' in chapter


def _write_test_epub(
    path: Path,
    *,
    chapter: str = '<html xmlns="http://www.w3.org/1999/xhtml"><body><img src="images/cover.jpg"/></body></html>',
    mimetype_first: bool = True,
    mimetype_stored: bool = True,
) -> None:
    members = [
        (
            "mimetype",
            b"application/epub+zip",
            zipfile.ZIP_STORED if mimetype_stored else zipfile.ZIP_DEFLATED,
        ),
        (
            "META-INF/container.xml",
            b'<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            zipfile.ZIP_DEFLATED,
        ),
        (
            "EPUB/content.opf",
            b'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"><metadata><meta name="cover" content="cover-image"/></metadata><manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/><item id="cover-image" href="images/cover.jpg" media-type="image/jpeg"/></manifest><spine><itemref idref="chapter"/></spine></package>',
            zipfile.ZIP_DEFLATED,
        ),
        ("EPUB/chapter.xhtml", chapter.encode("utf-8"), zipfile.ZIP_DEFLATED),
        ("EPUB/images/cover.jpg", b"jpeg", zipfile.ZIP_DEFLATED),
    ]
    if not mimetype_first:
        members[0], members[1] = members[1], members[0]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data, compression in members:
            archive.writestr(name, data, compress_type=compression)


def _verify_args(path: Path):
    return hybrid.build_parser().parse_args(["verify", str(path)])


def test_verify_rejects_missing_local_resource(tmp_path: Path, capsys) -> None:
    epub_path = tmp_path / "missing-resource.epub"
    _write_test_epub(
        epub_path,
        chapter='<html xmlns="http://www.w3.org/1999/xhtml"><body><img src="images/missing.jpg"/></body></html>',
    )

    assert hybrid.verify_epub(_verify_args(epub_path)) == 1
    assert "missing local references" in capsys.readouterr().err


def test_verify_rejects_malformed_xml(tmp_path: Path, capsys) -> None:
    epub_path = tmp_path / "malformed.epub"
    _write_test_epub(epub_path, chapter="<html><body></html>")

    assert hybrid.verify_epub(_verify_args(epub_path)) == 1
    assert "invalid XML" in capsys.readouterr().err


def test_verify_requires_first_uncompressed_mimetype(tmp_path: Path, capsys) -> None:
    epub_path = tmp_path / "bad-mimetype.epub"
    _write_test_epub(epub_path, mimetype_first=False, mimetype_stored=False)

    assert hybrid.verify_epub(_verify_args(epub_path)) == 1
    error = capsys.readouterr().err
    assert "mimetype must be the first ZIP member" in error
    assert "mimetype must be stored without compression" in error
