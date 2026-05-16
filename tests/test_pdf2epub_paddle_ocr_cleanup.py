import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile
from unittest import mock


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "pdf2epub_paddle.py"
    spec = importlib.util.spec_from_file_location("pdf2epub_paddle_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class TestPdf2EpubPaddleOcrCleanup(unittest.TestCase):
    def test_clean_ocr_noise_preserves_display_array_and_keeps_prose(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "# 中国要变什么？\n\n"
            r"$$ \begin{array}{ccc} 2 & 3 & 4 \\ 5 & 6 & 7 \\ 8 & 9 & 10 \end{array} $$"
            "\n\n过去学术界习惯把鸦片战争作为近代中国之起点。"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("# 中国要变什么？", cleaned)
        self.assertIn("过去学术界习惯", cleaned)
        self.assertIn(r"\begin{array}", cleaned)
        self.assertIn("2 & 3 & 4", cleaned)
        self.assertIn(r"\end{array}", cleaned)

    def test_clean_ocr_noise_preserves_valid_aligned_equations(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "Before equation.\n\n"
            r"$$\begin{aligned} a &= b + c \\ d &= e + f \end{aligned}$$"
            "\n\nAfter equation."
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before equation.", cleaned)
        self.assertIn(r"\begin{aligned}", cleaned)
        self.assertIn("a &= b + c", cleaned)
        self.assertIn(r"\end{aligned}", cleaned)
        self.assertIn("After equation.", cleaned)

    def test_clean_ocr_noise_preserves_prose_after_unclosed_array_marker(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "Intro paragraph.\n"
            r"\[\begin{array}{c} noisy ornament"
            "\n\nThis prose should remain in the chapter."
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Intro paragraph.", cleaned)
        self.assertIn("This prose should remain in the chapter.", cleaned)

    def test_clean_ocr_noise_normalizes_inline_false_latex(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            r"20 $ \frac{1}{2} $7年，香港三联书店出版。"
            r"海关内部衍生了邮政、海巡 $ \downarrow $气象、天文等多个部门。"
            r"不惜使用相当夸张的说法 $ \underset{\cdot}{颂} $圣。"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("2017年，香港三联书店出版", cleaned)
        self.assertIn("海巡、气象、天文", cleaned)
        self.assertIn("颂圣", cleaned)
        self.assertNotIn("\\frac", cleaned)
        self.assertNotIn("\\downarrow", cleaned)
        self.assertNotIn("\\underset", cleaned)

    def test_clean_ocr_noise_preserves_isolated_downarrow_math(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = r"The diagram uses the symbol $ \downarrow $ between two nodes."

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn(r"$ \downarrow $", cleaned)
        self.assertNotIn("、", cleaned)

    def test_clean_ocr_noise_removes_numeric_only_year_table_from_blank_page(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        rows = []
        for year in range(2021, 2045):
            rows.append(
                "<tr>"
                f"<td>{year}</td><td>{year + 1}</td><td>{year + 2}</td>"
                "</tr>"
            )
        raw = "Before.\n\n<table>" + "".join(rows) + "</table>\n\nAfter."

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        self.assertNotIn("<table", cleaned)
        self.assertNotIn("2044", cleaned)

    def test_clean_ocr_noise_removes_dotted_numeric_table_from_blank_page(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        cells = "".join(
            f"<td>1.1.1.{index}</td>"
            for index in range(377, 397)
        )
        raw = (
            "<p>Before.</p>"
            "<table><tr><td>序：章之三</td>"
            + cells
            + "</tr></table>"
            "<p>After.</p>"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        self.assertNotIn("<table", cleaned)
        self.assertNotIn("1.1.1.382", cleaned)

    def test_scan_epub_for_ocr_noise_reports_specific_false_latex_artifacts(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(any(item["file"] == "EPUB/chapter.xhtml" for item in findings))
        self.assertTrue(any(item["token"] == "\\underset{\\cdot}" for item in findings))

    def test_scan_epub_for_ocr_noise_reports_numeric_only_year_table(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        rows = []
        for year in range(2021, 2045):
            rows.append(
                "<tr>"
                f"<td>{year}</td><td>{year + 1}</td><td>{year + 2}</td>"
                "</tr>"
            )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><table>"
                    + "".join(rows)
                    + "</table></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(
            any(item["token"] == "numeric-only OCR table" for item in findings)
        )

    def test_scan_epub_for_ocr_noise_reports_dotted_numeric_table(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        cells = "".join(
            f"<td>1.1.1.{index}</td>"
            for index in range(377, 397)
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><table><tr><td>序：章之三</td>"
                    + cells
                    + "</tr></table></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(
            any(item["token"] == "dotted numeric OCR table" for item in findings)
        )

    def test_scan_epub_for_ocr_noise_allows_valid_math_latex(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>Area is $a \times b$ and half is $\frac{1}{2}$.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_scan_epub_for_ocr_noise_allows_valid_downarrow_math(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>The sequence satisfies $a_n \downarrow 0$.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_validate_epub_no_ocr_noise_allows_isolated_downarrow_in_strict_mode(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>The symbol is $ \downarrow $.</p></body></html>",
                )

            findings = mod.validate_epub_no_ocr_noise(epub_path, strict=True)

        self.assertEqual([], findings)

    def test_scan_epub_for_ocr_noise_ignores_structural_metadata_files(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "metadata.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    r"<metadata><dc:title>Arrow $ \downarrow $ Theory</dc:title></metadata>",
                )
                zf.writestr(
                    "EPUB/toc.ncx",
                    r"<navLabel><text>Arrow $ \downarrow $ Theory</text></navLabel>",
                )
                zf.writestr(
                    "EPUB/nav.xhtml",
                    r"<html><body><nav>Arrow $ \downarrow $ Theory</nav></body></html>",
                )
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><p>Clean OCR chapter content.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_validate_epub_no_ocr_noise_reports_truncated_finding_count(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                for index in range(9):
                    zf.writestr(
                        f"EPUB/chapter-{index}.xhtml",
                        r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                    )

            with self.assertRaises(RuntimeError) as cm:
                mod.validate_epub_no_ocr_noise(epub_path, strict=True)

        self.assertIn("showing 8 of 9", str(cm.exception))

    def test_validate_epub_no_ocr_noise_warns_without_failing_by_default(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                )

            with mock.patch("builtins.print") as mock_print:
                findings = mod.validate_epub_no_ocr_noise(epub_path)

        self.assertTrue(findings)
        self.assertTrue(any("[!]" in str(call.args[0]) for call in mock_print.call_args_list))

    def test_resolve_epub_fragment_href_keeps_same_file_fragment_in_source_file(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "_resolve_epub_fragment_href"))

        resolved = mod._resolve_epub_fragment_href("EPUB/nav.xhtml", "#toc-header")

        self.assertEqual(("EPUB/nav.xhtml", "toc-header"), resolved)

    def test_ensure_toc_targets_start_pages_marks_only_toc_heading_targets(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol><li>"
                        "<a href=\"chapter.xhtml#section-1\">Section</a>"
                        "</li></ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><head><link href=\"style/nav.css\" rel=\"stylesheet\" "
                        "type=\"text/css\"/></head><body>"
                        "<h2 id=\"intro\">Intro</h2>"
                        "<p>Before.</p>"
                        "<h2 id=\"section-1\">Section</h2>"
                        "<p>After.</p>"
                        "</body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                infos = zf.infolist()
                css = zf.read("EPUB/style/nav.css").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertEqual("mimetype", infos[0].filename)
        self.assertEqual(ZIP_STORED, infos[0].compress_type)
        self.assertIn(".toc-page-start", css)
        self.assertIn("break-before: page", css)
        self.assertIn('<h2 id="section-1" class="toc-page-start">Section</h2>', chapter)
        self.assertIn('<h2 id="intro">Intro</h2>', chapter)

    def test_ensure_toc_targets_start_pages_preserves_existing_heading_class(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/toc.ncx",
                    '<ncx><navMap><navPoint><content src="chapter.xhtml#section-1"/>'
                    '</navPoint></navMap></ncx>',
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h2 class=\"chapter-title\" id=\"section-1\">"
                        "Section</h2></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertIn(
            '<h2 class="chapter-title toc-page-start" id="section-1">Section</h2>',
            chapter,
        )

    def test_ensure_toc_targets_start_pages_moves_chapter_link_to_number_heading(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol><li>"
                        "<a href=\"chapter.xhtml#title\">第五章 對外關係</a>"
                        "</li></ol></nav></body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/toc.ncx",
                    (
                        "<ncx><navMap><navPoint><navLabel><text>第五章 對外關係"
                        "</text></navLabel><content src=\"chapter.xhtml#title\"/>"
                        "</navPoint></navMap></ncx>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>Before.</p>"
                        "<h2 id=\"chapter-number\">第五章</h2>"
                        "<h2 id=\"title\" class=\"toc-page-start\">對外關係</h2>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                ncx = zf.read("EPUB/toc.ncx").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertIn('href="chapter.xhtml#chapter-number"', nav)
        self.assertIn('src="chapter.xhtml#chapter-number"', ncx)
        self.assertIn('<h2 id="chapter-number" class="toc-page-start">第五章</h2>', chapter)
        self.assertIn('<h2 id="title">對外關係</h2>', chapter)

    def test_ensure_toc_targets_start_pages_moves_part_link_to_part_line(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#title\">第一編 傳統制度的延續，1600-1800年</a></li>"
                        "<li><a href=\"chapter.xhtml#title\">第二章 清帝國的興盛</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>第一編</p><p>傳統制度的延續</p>"
                        "<p>1600-1800年</p><h2 id=\"title\">清帝國的興盛</h2>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        part_href = re.search(r'href="chapter.xhtml#([^"]+)">第一編', nav)
        chapter_href = re.search(r'href="chapter.xhtml#([^"]+)">第二章', nav)
        self.assertIsNotNone(part_href)
        self.assertIsNotNone(chapter_href)
        self.assertNotEqual("title", part_href.group(1))
        self.assertNotEqual("title", chapter_href.group(1))
        self.assertIn(
            f'<p id="{part_href.group(1)}" class="toc-page-start">第一編</p>',
            chapter,
        )
        self.assertIn(
            f'<h2 id="{chapter_href.group(1)}" class="toc-page-start">第二章</h2>',
            chapter,
        )
        self.assertIn('<h2 id="title">清帝國的興盛</h2>', chapter)

    def test_ensure_toc_targets_start_pages_inserts_missing_part_heading(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#title\">第三編 外國帝國主義加劇時期的自強運動 1861-1895年</a></li>"
                        "<li><a href=\"chapter.xhtml#title\">第十一章 清朝中興與自強運動</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h1 id=\"title\">清朝中興與自強運動</h1>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        part_href = re.search(r'href="chapter.xhtml#([^"]+)">第三編', nav)
        chapter_href = re.search(r'href="chapter.xhtml#([^"]+)">第十一章', nav)
        self.assertIsNotNone(part_href)
        self.assertIsNotNone(chapter_href)
        self.assertNotEqual("title", part_href.group(1))
        self.assertNotEqual("title", chapter_href.group(1))
        self.assertLess(chapter.index("第三編"), chapter.index("第十一章"))
        self.assertIn(
            f'<p id="{part_href.group(1)}" class="toc-page-start">第三編</p>',
            chapter,
        )
        self.assertIn("<p>外國帝國主義加劇時期的自強運動 1861-1895年</p>", chapter)
        self.assertIn(
            f'<h1 id="{chapter_href.group(1)}" class="toc-page-start">第十一章</h1>',
            chapter,
        )

    def test_ensure_toc_targets_start_pages_moves_part_heading_before_chapter_number(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#part\">第三編 外國帝國主義加劇時期的自強運動 1861-1895年</a></li>"
                        "<li><a href=\"chapter.xhtml#chapter\">第十一章 清朝中興與自強運動</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h1 id=\"chapter\" class=\"toc-page-start\">第十一章</h1>"
                        "<p id=\"part\" class=\"toc-page-start\">第三編</p>"
                        "<p>外國帝國主義加劇時期的自強運動 1861-1895年</p>"
                        "<h1 id=\"title\">清朝中興與自強運動</h1></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertLess(chapter.index("第三編"), chapter.index("第十一章"))
        self.assertLess(chapter.index("第十一章"), chapter.index("清朝中興與自強運動"))

    def test_write_validated_epub_validates_once_after_toc_patching(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "write_validated_epub"))

        events = []

        def fake_write_epub(output_file, _book, _options):
            events.append("write")
            with ZipFile(output_file, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><p>Clean OCR chapter content.</p></body></html>",
                )

        def fake_ensure_toc_targets_start_pages(_output_file):
            events.append("toc")
            return {"targets": 0, "xhtml_files": [], "css_files": []}

        def fake_validate_epub_no_ocr_noise(_output_file, strict=False):
            events.append(f"validate:{strict}")
            return []

        fake_epub = SimpleNamespace(write_epub=fake_write_epub)

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "book.epub"
            with mock.patch.object(mod, "epub", fake_epub, create=True), \
                    mock.patch.object(
                        mod,
                        "ensure_toc_targets_start_pages",
                        fake_ensure_toc_targets_start_pages,
                    ), \
                    mock.patch.object(
                        mod,
                        "validate_epub_no_ocr_noise",
                        fake_validate_epub_no_ocr_noise,
                    ):
                mod.write_validated_epub(
                    object(),
                    str(output_path),
                    strict_ocr_validation=False,
                )

            self.assertTrue(output_path.exists())

        self.assertEqual(["write", "toc", "validate:False"], events)

    def test_create_epub_does_not_leave_output_file_when_validation_fails(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        class FakeBook:
            def set_identifier(self, _identifier):
                pass

            def set_title(self, _title):
                pass

            def set_language(self, _language):
                pass

            def add_author(self, _author):
                pass

            def add_item(self, _item):
                pass

            def set_cover(self, _filename, _data):
                pass

        class FakeItem:
            def __init__(self, *args, **kwargs):
                pass

            def add_item(self, _item):
                pass

        def fake_write_epub(output_file, _book, _options):
            Path(output_file).write_text("noisy epub", encoding="utf-8")

        fake_epub = SimpleNamespace(
            EpubBook=FakeBook,
            EpubItem=FakeItem,
            EpubImage=FakeItem,
            EpubHtml=FakeItem,
            EpubNcx=FakeItem,
            EpubNav=FakeItem,
            write_epub=fake_write_epub,
        )

        def fail_validation(_output_file, *args, **kwargs):
            raise RuntimeError("validation failed")

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "book.epub"
            with mock.patch.object(mod, "epub", fake_epub, create=True), \
                    mock.patch.object(mod, "validate_epub_no_ocr_noise", fail_validation):
                with self.assertRaises(RuntimeError):
                    mod.create_epub(
                        "Title",
                        [],
                        str(output_path),
                        str(Path(td) / "images"),
                        strict_ocr_validation=True,
                    )

            self.assertFalse(output_path.exists())

    def test_invalid_numeric_env_values_fall_back_to_defaults(self):
        env_overrides = {
            "PADDLE_CHUNK_SIZE": "abc",
            "PADDLE_API_TIMEOUT_SECONDS": "def",
            "EPUB_COVER_MAX_EDGE": "ghi",
            "EPUB_COVER_JPEG_QUALITY": "jkl",
        }
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in env_overrides
        }
        clean_env.update(env_overrides)

        with mock.patch.dict(os.environ, clean_env, clear=True):
            mod = _load_script_module()

        self.assertEqual(5, mod.CHUNK_SIZE)
        self.assertEqual(600, mod.API_TIMEOUT_SECONDS)
        self.assertEqual(2000, mod.DEFAULT_COVER_MAX_EDGE)
        self.assertEqual(82, mod.DEFAULT_COVER_JPEG_QUALITY)

    def test_non_positive_numeric_env_values_fall_back_to_defaults(self):
        env_overrides = {
            "PADDLE_CHUNK_SIZE": "0",
            "PADDLE_API_TIMEOUT_SECONDS": "-1",
            "EPUB_COVER_MAX_EDGE": "0",
            "EPUB_COVER_JPEG_QUALITY": "-10",
        }
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in env_overrides
        }
        clean_env.update(env_overrides)

        with mock.patch.dict(os.environ, clean_env, clear=True):
            mod = _load_script_module()

        self.assertEqual(5, mod.CHUNK_SIZE)
        self.assertEqual(600, mod.API_TIMEOUT_SECONDS)
        self.assertEqual(2000, mod.DEFAULT_COVER_MAX_EDGE)
        self.assertEqual(82, mod.DEFAULT_COVER_JPEG_QUALITY)
