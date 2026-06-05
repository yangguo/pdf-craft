import tempfile
import unittest
from unittest import mock
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _write_checkpoint(path: Path, page_texts: list[str]) -> None:
    import json

    path.write_text(
        json.dumps(
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": text}} for text in page_texts
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class TestAutoFixGarbledVerification(unittest.TestCase):
    def test_verify_epub_package_accepts_valid_ebooklib_epub(self):
        from ebooklib import epub

        from auto_fix_garbled import _verify_epub_package

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            book = epub.EpubBook()
            book.set_identifier("test-book")
            book.set_title("Test Book")
            book.set_language("zh-Hant")
            book.add_author("Tester")
            chapter = epub.EpubHtml(
                title="Chapter 1",
                file_name="chapter.xhtml",
                content="<html><body><p>正常中文正文。</p></body></html>",
            )
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.spine = ["nav", chapter]
            epub.write_epub(str(epub_path), book, {})

            report = _verify_epub_package(str(epub_path))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["mimetype_first"], report)
        self.assertTrue(report["mimetype_stored"], report)
        self.assertTrue(report["has_opf"], report)
        self.assertTrue(report["has_nav"], report)
        self.assertTrue(report["has_ncx"], report)
        self.assertIsInstance(report["ebooklib_items"], int)

    def test_verify_epub_package_reports_missing_mimetype_and_nav(self):
        from auto_fix_garbled import _verify_epub_package

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "broken.epub"
            with ZipFile(epub_path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("EPUB/chapter.xhtml", "<html><body>bad</body></html>")

            report = _verify_epub_package(str(epub_path))

        self.assertFalse(report["ok"], report)
        self.assertIn("mimetype is not the first zip entry", report["errors"])
        self.assertIn("content.opf missing", report["errors"])
        self.assertIn("EPUB nav document missing", report["errors"])

    def test_verify_epub_package_uses_container_rootfile_path(self):
        from auto_fix_garbled import _verify_epub_package

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "rootfile.epub"
            with ZipFile(epub_path, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip", compress_type=0)
                archive.writestr(
                    "META-INF/container.xml",
                    (
                        '<?xml version="1.0"?>'
                        '<container version="1.0" '
                        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                        "<rootfiles>"
                        '<rootfile full-path="OEBPS/book.opf" '
                        'media-type="application/oebps-package+xml"/>'
                        "</rootfiles>"
                        "</container>"
                    ),
                )
                archive.writestr("OEBPS/book.opf", "<package></package>")
                archive.writestr("OEBPS/nav.xhtml", "<html><body>nav</body></html>")
                archive.writestr("toc.ncx", "<ncx></ncx>")

            with mock.patch("auto_fix_garbled.ebooklib_epub", None):
                report = _verify_epub_package(str(epub_path))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["has_opf"], report)


class TestAutoFixGarbledMainMetadata(unittest.TestCase):
    def test_main_allows_scan_without_metadata_when_epub_exists_and_clean(self):
        from auto_fix_garbled import main

        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "book.pdf"
            output_epub = Path(td) / "book.epub"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            output_epub.write_bytes(b"placeholder")
            with (
                mock.patch("auto_fix_garbled._find_garbled_spans", return_value={}),
                mock.patch(
                    "auto_fix_garbled._verify_epub_package",
                    return_value={
                        "path": str(output_epub),
                        "ok": True,
                        "errors": [],
                        "warnings": [],
                        "entry_count": 1,
                        "mimetype_first": True,
                        "mimetype_stored": True,
                        "has_opf": True,
                        "has_nav": True,
                        "has_ncx": True,
                        "ebooklib_items": 1,
                    },
                ),
            ):
                main([str(pdf_path), "--output", str(output_epub)])


class TestAutoFixGarbledMapping(unittest.TestCase):
    def test_map_spans_to_pages_skips_ambiguous_duplicate_matches(self):
        from auto_fix_garbled import _map_spans_to_pages

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint(
                work_dir / "chunk_0_2.pdf.json",
                [
                    "這一頁包含重複片段甲乙丙丁戊己庚辛。",
                    "下一頁也包含重複片段甲乙丙丁戊己庚辛。",
                ],
            )
            pages = _map_spans_to_pages(
                str(work_dir),
                {"chapter_1.xhtml": ["疑似亂碼：甲乙丙丁戊己庚辛"]},
            )

        self.assertEqual(set(), pages)

    def test_map_spans_to_pages_matches_extension_a_characters(self):
        from auto_fix_garbled import _map_spans_to_pages

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint(
                work_dir / "chunk_0_1.pdf.json",
                ["文本含有㐀㐁㐂㐃㐄㐅這段罕見字串。"],
            )
            pages = _map_spans_to_pages(
                str(work_dir),
                {"chapter_1.xhtml": ["疑似亂碼：㐀㐁㐂㐃㐄㐅"]},
            )

        self.assertEqual({1}, pages)


class TestSplitPdfCompatibility(unittest.TestCase):
    def test_split_pdf_applies_legacy_bottom_padding_percent(self):
        from paddle_pipeline import paddle_api

        fitz = paddle_api.fitz
        if fitz is None:
            self.skipTest("PyMuPDF is not installed")

        with tempfile.TemporaryDirectory() as td:
            source_pdf = Path(td) / "source.pdf"
            doc = fitz.open()
            doc.new_page(width=100, height=200)
            doc.save(source_pdf)
            doc.close()

            with (
                mock.patch.object(paddle_api, "PADDLE_PAGE_MARGIN_PT", 36),
                mock.patch.object(paddle_api, "PADDLE_BOTTOM_PADDING_PERCENT", 10, create=True),
            ):
                chunk_paths = paddle_api.split_pdf(str(source_pdf), chunk_size=1)

            try:
                chunk_doc = fitz.open(chunk_paths[0])
                mediabox = chunk_doc[0].mediabox
                chunk_doc.close()
            finally:
                if chunk_paths:
                    import shutil

                    shutil.rmtree(Path(chunk_paths[0]).parent, ignore_errors=True)

        self.assertAlmostEqual(0, mediabox.x0)
        self.assertAlmostEqual(0, mediabox.y0)
        self.assertAlmostEqual(172, mediabox.x1)
        self.assertAlmostEqual(292, mediabox.y1)


if __name__ == "__main__":
    unittest.main()
