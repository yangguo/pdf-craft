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


def test_candidate_visual_pages_skips_captionless_page_raster(tmp_path: Path) -> None:
    pdf_path = tmp_path / "page-raster.pdf"
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(50, 50, 545, 790), stream=image_buffer.getvalue())
    document.save(pdf_path)
    document.close()

    with fitz.open(pdf_path) as parsed:
        assert hybrid.candidate_visual_pages(parsed, skip_page_rasters=True) == set()


def test_visual_crop_rect_uses_embedded_image_bounds() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(100, 200, 300, 300), stream=image_buffer.getvalue())

    crop = hybrid.visual_crop_rect(page)

    assert crop is not None
    assert round(crop.x0, 1) == 92.0
    assert round(crop.y0, 1) == 192.0
    assert round(crop.x1, 1) == 308.0
    assert round(crop.y1, 1) == 308.0
    document.close()


def test_visual_crop_rect_uses_vector_drawing_bounds() -> None:
    document = fitz.open()
    page = document.new_page()
    page.draw_rect(fitz.Rect(120, 300, 420, 500), fill=(0.8, 0.8, 0.8))

    crop = hybrid.visual_crop_rect(page)

    assert crop is not None
    assert round(crop.x0, 1) == 112.0
    assert round(crop.y0, 1) == 292.0
    assert round(crop.x1, 1) == 428.0
    assert round(crop.y1, 1) == 508.0
    document.close()


def test_visual_crop_rect_ignores_unrelated_drawings_alongside_images() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(100, 200, 300, 300), stream=image_buffer.getvalue())
    page.draw_rect(fitz.Rect(320, 400, 500, 550), fill=(0.8, 0.8, 0.8))

    crop = hybrid.visual_crop_rect(page)

    assert crop is not None
    assert round(crop.x0, 1) == 92.0
    assert round(crop.y0, 1) == 192.0
    assert round(crop.x1, 1) == 308.0
    assert round(crop.y1, 1) == 308.0
    document.close()


def test_visual_crop_rect_uses_multiple_interior_vector_lines() -> None:
    document = fitz.open()
    page = document.new_page()
    page.draw_line((120, 300), (420, 300))
    page.draw_line((120, 350), (420, 350))
    page.draw_line((120, 500), (420, 500))

    crop = hybrid.visual_crop_rect(page)

    assert crop is not None
    assert round(crop.x0, 1) == 112.0
    assert round(crop.y0, 1) == 291.0
    assert round(crop.x1, 1) == 428.0
    assert round(crop.y1, 1) == 509.0
    document.close()


def test_visual_crop_rect_ignores_line_only_page_furniture() -> None:
    document = fitz.open()
    page = document.new_page()
    page.draw_line((100, 200), (500, 200))
    page.draw_line((100, 400), (500, 400))

    assert hybrid.visual_crop_rect(page) is None
    document.close()


def test_convert_skips_visual_page_without_visual_region(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    page.insert_text((72, 100), "Body text")
    page.draw_line((100, 200), (500, 200))
    document.save(pdf_path)
    document.close()

    result = hybrid.main(
        [
            "convert",
            str(pdf_path),
            str(epub_path),
            "--visual-pages",
            "2",
        ]
    )

    assert result == 0
    with zipfile.ZipFile(epub_path) as archive:
        assert "EPUB/images/visual_p0002.jpg" not in archive.namelist()


def test_convert_does_not_add_internal_visual_snapshot_caption(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    page.insert_text((72, 100), "Figure 1. A test figure.")
    page.insert_image(fitz.Rect(100, 200, 300, 300), stream=image_buffer.getvalue())
    document.save(pdf_path)
    document.close()

    result = hybrid.main(
        [
            "convert",
            str(pdf_path),
            str(epub_path),
            "--visual-pages",
            "2",
        ]
    )

    assert result == 0
    with zipfile.ZipFile(epub_path) as archive:
        chapter = archive.read("EPUB/01_Pages_1_2.xhtml").decode("utf-8")
    assert "PDF page 2 figure/table visual snapshot" not in chapter


def test_caption_pattern_matches_simplified_chinese_figure_titles() -> None:
    match = hybrid.CAPTION_PATTERN.match("图 3-5 第一张图")

    assert match is not None
    assert match.group("kind") == "图"
    assert match.group("chapter") == "3"
    assert match.group("index") == "5"


def test_captioned_visual_regions_pair_figure_titles() -> None:
    document = fitz.open()
    page = document.new_page()
    page.draw_rect(fitz.Rect(100, 100, 300, 180), fill=(0.8, 0.8, 0.8))
    page.insert_text((120, 200), "Figure 3-5 First figure")
    page.draw_rect(fitz.Rect(100, 260, 300, 340), fill=(0.8, 0.8, 0.8))
    page.insert_text((120, 360), "Figure 3-6 Second figure")

    visuals = hybrid.captioned_visual_regions(page)

    assert [visual.label for visual in visuals] == ["Figure 3-5", "Figure 3-6"]
    assert [visual.caption for visual in visuals] == ["Figure 3-5 First figure", "Figure 3-6 Second figure"]
    assert visuals[0].rect.y1 < visuals[0].caption_rect.y0
    assert visuals[1].rect.y1 < visuals[1].caption_rect.y0
    document.close()


def test_convert_splits_captioned_figures_and_writes_figcaptions(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    page.insert_image(fitz.Rect(100, 100, 300, 180), stream=image_buffer.getvalue())
    page.insert_text((120, 200), "Figure 1-1 First figure")
    page.insert_image(fitz.Rect(100, 260, 300, 340), stream=image_buffer.getvalue())
    page.insert_text((120, 360), "Figure 1-2 Second figure")
    document.save(pdf_path)
    document.close()

    assert hybrid.main(["convert", str(pdf_path), str(epub_path), "--visual-pages", "2"]) == 0

    with zipfile.ZipFile(epub_path) as archive:
        names = archive.namelist()
        chapter = archive.read("EPUB/01_Pages_1_2.xhtml").decode("utf-8")
    assert "EPUB/images/visual_p0002_f01.jpg" in names
    assert "EPUB/images/visual_p0002_f02.jpg" in names
    assert "<figcaption>Figure 1-1 First figure</figcaption>" in chapter
    assert "<figcaption>Figure 1-2 Second figure</figcaption>" in chapter


def test_page_body_and_footnotes_moves_smaller_bottom_text_out_of_body() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 130), "Main body paragraph.", fontsize=10)
    page.insert_text((72, 640), "A source footnote.", fontsize=7)

    body, footnotes = hybrid.page_body_and_footnotes(page, 1, drop_headers=True)

    assert body == ["Main body paragraph."]
    assert footnotes == ["A source footnote."]
    document.close()


def test_is_readable_footnote_rejects_garbled_hidden_text() -> None:
    assert hybrid.is_readable_footnote("LH, Notes on WNJH 5, p. 7.")
    assert not hybrid.is_readable_footnote("\ufffd\ufffd如何行动，\ufffd实\ufffd际\ufffd")
    assert hybrid.is_readable_footnote("4 4 " + "可读正文" * 30)


def test_page_text_blocks_ignores_kenten_emphasis_glyphs() -> None:
    class FakePage:
        rect = fitz.Rect(0, 0, 400, 600)

        def get_text(self, kind: str) -> dict:
            assert kind == "dict"
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (50, 100, 350, 112),
                        "lines": [
                            {
                                "spans": [
                                    {
                                        "text": "这是应该保留的正文。",
                                        "size": 10,
                                        "font": "SimSun",
                                    },
                                    {
                                        "text": "444444444444",
                                        "size": 4,
                                        "font": "KentenGeneric",
                                    },
                                ]
                            }
                        ],
                    }
                ]
            }

    assert hybrid.page_text_blocks(FakePage(), 1, drop_headers=True) == ["这是应该保留的正文。"]


def test_convert_removes_text_inside_visual_region(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    image_buffer = BytesIO()
    Image.new("RGB", (20, 20), "navy").save(image_buffer, "PNG")

    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    page.insert_image(fitz.Rect(100, 100, 300, 180), stream=image_buffer.getvalue())
    page.insert_text((120, 140), "Chart label")
    page.insert_text((120, 200), "Figure 1-1 A chart")
    page.insert_text((72, 280), "Body prose remains selectable.")
    document.save(pdf_path)
    document.close()

    assert hybrid.main(["convert", str(pdf_path), str(epub_path), "--visual-pages", "2"]) == 0

    with zipfile.ZipFile(epub_path) as archive:
        chapter = archive.read("EPUB/01_Pages_1_2.xhtml").decode("utf-8")
    assert "Chart label" not in chapter
    assert "Body prose remains selectable." in chapter


def test_convert_writes_footnotes_at_the_end_of_a_chapter(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    epub_path = tmp_path / "book.epub"
    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    page.insert_text((72, 130), "Body prose.", fontsize=10)
    page.insert_text((72, 640), "A chapter endnote.", fontsize=7)
    document.save(pdf_path)
    document.close()

    assert hybrid.main(
        [
            "convert",
            str(pdf_path),
            str(epub_path),
            "--section",
            "Chapter:1-2",
        ]
    ) == 0

    with zipfile.ZipFile(epub_path) as archive:
        chapter = archive.read("EPUB/01_Chapter.xhtml").decode("utf-8")
    assert '<section class="endnotes"><h2>注释</h2><ol>' in chapter
    assert "<li>A chapter endnote.</li>" in chapter
    assert chapter.index("Body prose.") < chapter.index("A chapter endnote.")


def test_remove_duplicate_section_heading_only_at_section_start() -> None:
    blocks = ["Cyber operations", "AI-augmented cyber operations"]

    assert hybrid.remove_duplicate_section_heading(
        blocks, "Cyber operations", is_section_start=True
    ) == ["AI-augmented cyber operations"]
    assert hybrid.remove_duplicate_section_heading(
        blocks, "Cyber operations", is_section_start=False
    ) == blocks


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


def test_reflow_text_blocks_does_not_split_same_line_inset_and_honors_explicit_indent() -> None:
    blocks = [
        (56.0, 100.0, 190.0, 114.0, "「他不過是他媽的一個攝像，我的", 0, 0),
        (169.0, 100.0, 236.0, 120.0, "攝像！」", 1, 0),
        (56.0, 116.0, 330.0, 130.0, "\u3000\u3000 然而，照現在的情形來看，一切不都扯平了嗎？", 2, 0),
    ]

    assert hybrid.reflow_text_blocks(blocks) == [
        "「他不過是他媽的一個攝像，我的攝像！」",
        "然而，照現在的情形來看，一切不都扯平了嗎？",
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


def test_header_filter_preserves_body_blocks_near_bottom_margin() -> None:
    body_block = (72.0, 660.0, 540.0, 719.0, "The final paragraph on a dense page.", 0, 0)
    footer_block = (72.0, 744.0, 540.0, 754.0, "Report title 22", 0, 0)

    assert not hybrid.is_header_or_footer(body_block, drop_headers=True, page_height=792.0)
    assert hybrid.is_header_or_footer(footer_block, drop_headers=True, page_height=792.0)


def test_reflow_text_blocks_removes_spaces_inside_multiline_cjk_block() -> None:
    blocks = [
        (78.0, 100.0, 350.0, 128.0, "文化交流從古到今\n都是歷史家關心的課題。", 0, 0),
    ]

    assert hybrid.reflow_text_blocks(blocks) == [
        "文化交流從古到今都是歷史家關心的課題。",
    ]


def test_normalize_text_removes_embedded_control_characters() -> None:
    assert hybrid.normalize_text("图3-4\x07标题") == "图3-4标题"


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


def test_late_page_reading_order_does_not_assume_columns() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((250, 100), "First in vertical order.")
    page.insert_text((72, 200), "Second in vertical order.")
    assert hybrid.page_text_blocks(page, 301, False) == [
        "First in vertical order.", "Second in vertical order."
    ]
    assert hybrid.page_text_blocks(page, 301, False, column_split=200) == [
        "Second in vertical order.", "First in vertical order."
    ]
    document.close()


def test_readable_footnote_preserves_legitimate_repeated_digits() -> None:
    assert hybrid.is_readable_footnote("Burbank and Cooper 2010: 444")
    assert hybrid.is_readable_footnote("44 次觀察，數值 4 4 是原文內容。")


def test_candidate_visual_pages_retains_full_page_artwork() -> None:
    document = fitz.open()
    document.new_page().insert_text((72, 100), "Cover")
    page = document.new_page()
    buffer = BytesIO()
    Image.new("RGB", (30, 30), "navy").save(buffer, "PNG")
    page.insert_image(fitz.Rect(50, 50, 545, 790), stream=buffer.getvalue())
    assert hybrid.candidate_visual_pages(document) == {2}
    assert hybrid.candidate_visual_pages(document, skip_page_rasters=True) == set()
    document.close()


def test_reflow_continuous_inset_lines_stay_in_one_paragraph() -> None:
    blocks = [
        (56, 100, 350, 112, "前一段。", 0, 0),
        (78, 116, 350, 128, "引文第一行，", 1, 0),
        (78, 132, 350, 144, "引文第二行，", 2, 0),
        (78, 148, 350, 160, "引文第三行。", 3, 0),
    ]
    assert hybrid.reflow_text_blocks(blocks) == ["前一段。", "引文第一行，引文第二行，引文第三行。"]


def test_flow_page_sections_preserves_inline_text_links_and_page_anchors() -> None:
    from xml.etree import ElementTree as ET
    ns = 'http://www.w3.org/1999/xhtml'
    root = ET.fromstring(f'''<html xmlns="{ns}"><body><h1>Chapter</h1>
    <section class="pdf-page" id="p0001"><p>開頭<em>文字</em>未完</p></section>
    <section class="pdf-page" id="p0002"><p id="source-note">續文<a href="#note">注</a>結尾。</p><p>新段。</p></section>
    </body></html>''')
    hybrid.flow_page_sections(root, continuation_pages={2})
    body = root.find(f'{{{ns}}}body')
    assert body is not None
    paragraphs = body.findall(f'{{{ns}}}p')
    assert [''.join(p.itertext()) for p in paragraphs] == ['開頭文字未完續文注結尾。', '新段。']
    assert paragraphs[0].find(f'{{{ns}}}em').text == '文字'
    assert paragraphs[0].find(f'{{{ns}}}a').get('href') == '#note'
    assert {e.get('id') for e in body.iter() if e.get('id')} == {'p0001', 'p0002', 'source-note'}
    assert not body.findall(f'{{{ns}}}section')


def test_flow_page_sections_respects_figures_and_unmarked_boundaries() -> None:
    from xml.etree import ElementTree as ET
    ns = 'http://www.w3.org/1999/xhtml'
    root = ET.fromstring(f'''<html xmlns="{ns}"><body>
    <section class="pdf-page" id="p0001"><p>未完</p><figure><img src="local.jpg"/></figure></section>
    <section class="pdf-page" id="p0002"><p>圖片後文字</p></section>
    <section class="pdf-page" id="p0003"><p>新段落</p></section>
    <section class="endnotes"><ol><li>注釋</li></ol></section>
    </body></html>''')
    hybrid.flow_page_sections(root, continuation_pages={2})
    body = root.find(f'{{{ns}}}body')
    assert [''.join(p.itertext()) for p in body.findall(f'{{{ns}}}p')] == ['未完', '圖片後文字', '新段落']
    assert body.find(f'{{{ns}}}figure') is not None
    assert body.find(f'{{{ns}}}section').get('class') == 'endnotes'


def test_page_continuation_requires_cjk_geometry_and_no_new_indent() -> None:
    document = fitz.open()
    left = document.new_page()
    left.insert_text((72, 100), '中文正文前行，中文正文前行，中文正文前行，', fontname='china-t')
    left.insert_text((72, 720), '中文正文未完，中文正文未完，中文正文未完，', fontname='china-t')
    right = document.new_page()
    # Refresh pages after adding a page (PyMuPDF invalidates previous handles).
    left = document[0]
    right.insert_text((72, 90), '接續中文內容。', fontname='china-t')
    assert hybrid.page_continues_paragraph(left, right, drop_headers=True)
    indented = document.new_page()
    indented.insert_text((94, 90), '新段落中文內容。', fontname='china-t')
    indented.insert_text((72, 120), '後續中文內容。', fontname='china-t')
    assert not hybrid.page_continues_paragraph(document[0], indented, drop_headers=True)
    top_only = document.new_page()
    top_only.insert_text((72, 100), '未到頁尾的短段，', fontname='china-t')
    assert not hybrid.page_continues_paragraph(top_only, document[1], drop_headers=True)
    document.close()


def test_convert_flow_default_and_pages_layout_keep_source_ids(tmp_path: Path) -> None:
    from xml.etree import ElementTree as ET
    document = fitz.open()
    for text in ['First paragraph.', 'Second paragraph.']:
        document.new_page().insert_text((72, 100), text)
    pdf = tmp_path / 'flow.pdf'
    document.save(pdf)
    document.close()
    for layout in ['flow', 'pages']:
        out = tmp_path / f'{layout}.epub'
        assert hybrid.main(['convert', str(pdf), str(out), '--layout', layout, '--translator', 'Test Translator']) == 0
        with zipfile.ZipFile(out) as archive:
            name = next(n for n in archive.namelist() if n.endswith('.xhtml') and '01_' in n)
            root = ET.fromstring(archive.read(name))
            assert {'p0001', 'p0002'} <= {e.get('id') for e in root.iter()}
            sections = [e for e in root.iter() if e.get('class') == 'pdf-page']
            assert bool(sections) == (layout == 'pages')
            css = archive.read('EPUB/style/book.css').decode()
            assert ('page-break-after: always' in css) == (layout == 'pages')
            opf = ET.fromstring(archive.read('EPUB/content.opf'))
            contributors = opf.findall('.//{http://purl.org/dc/elements/1.1/}contributor')
            assert [c.text for c in contributors] == ['Test Translator']
            roles = [e for e in opf.iter() if e.get('property') == 'role']
            assert any(e.text == 'trl' and e.get('refines') == '#' + contributors[0].get('id') for e in roles)


def test_default_sections_preserves_front_matter_and_deduplicates_starts() -> None:
    document = fitz.open()
    for _ in range(5):
        document.new_page()
    document.set_toc([[1, 'Chapter One', 3], [1, 'Same start', 3], [1, 'Chapter Two', 5]])
    sections = hybrid.default_sections(document)
    assert [(s.title, s.start, s.end) for s in sections] == [
        ('Front Matter', 1, 2), ('Chapter One', 3, 4), ('Chapter Two', 5, 5)
    ]
    assert [n for s in sections for n in range(s.start, s.end + 1)] == list(range(1, 6))
    document.close()


def test_convert_cross_page_join_is_opt_in_and_does_not_cross_sections(tmp_path: Path) -> None:
    from xml.etree import ElementTree as ET
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), '中文正文前行，中文正文前行，中文正文前行，', fontname='china-t')
    page.insert_text((72, 720), '中文正文未完，中文正文未完，中文正文未完，', fontname='china-t')
    document.new_page().insert_text((72, 90), '接續中文內容。', fontname='china-t')
    pdf = tmp_path / 'continuation.pdf'
    document.save(pdf)
    document.close()
    ns = {'x': 'http://www.w3.org/1999/xhtml'}
    for join in [False, True]:
        out = tmp_path / f'continuation-{join}.epub'
        options = ['--join-page-continuations'] if join else []
        assert hybrid.main(['convert', str(pdf), str(out), '--language', 'zh-Hant', *options]) == 0
        with zipfile.ZipFile(out) as archive:
            name = next(n for n in archive.namelist() if n.endswith('.xhtml') and '01_' in n)
            root = ET.fromstring(archive.read(name))
            paragraphs = root.findall('.//x:body/x:p', ns)
            assert len(paragraphs) == (2 if join else 3)
            text = ''.join(''.join(p.itertext()) for p in paragraphs)
            assert text == '中文正文前行，中文正文前行，中文正文前行，中文正文未完，中文正文未完，中文正文未完，接續中文內容。'
    out = tmp_path / 'separate.epub'
    assert hybrid.main(['convert', str(pdf), str(out), '--join-page-continuations', '--section', 'First:1-1', '--section', 'Second:2-2']) == 0
    with zipfile.ZipFile(out) as archive:
        second = next(n for n in archive.namelist() if n.endswith('.xhtml') and '02_' in n)
        root = ET.fromstring(archive.read(second))
        assert [''.join(p.itertext()) for p in root.findall('.//x:body/x:p', ns)] == ['接續中文內容。']
