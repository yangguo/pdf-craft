import importlib.util
import os
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
