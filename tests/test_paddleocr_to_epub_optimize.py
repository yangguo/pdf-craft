import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "contrib" / "paddleocr_to_epub.py"
    spec = importlib.util.spec_from_file_location("paddleocr_to_epub_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class TestChapterMerge(unittest.TestCase):
    def test_merge_pages_into_chapters_dedups_running_header(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "merge_pages_into_chapters"))

        pages = [
            {"page": 1, "file": "p1.png", "text": "はじめに\n\n本文1"},
            {"page": 2, "file": "p2.png", "text": "はじめに\n\n本文2"},
            {"page": 3, "file": "p3.png", "text": "# 第1章\n\n本文3"},
            {"page": 4, "file": "p4.png", "text": "第1章\n\n本文4"},
        ]

        chapters = mod.merge_pages_into_chapters(pages)
        self.assertEqual([c["title"] for c in chapters], ["はじめに", "第1章"])
        self.assertEqual([p["page"] for p in chapters[0]["pages"]], [1, 2])
        self.assertEqual([p["page"] for p in chapters[1]["pages"]], [3, 4])

    def test_merge_pages_strips_title_line_from_body(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "merge_pages_into_chapters"))

        pages = [
            {"page": 1, "file": "p1.png", "text": "# 第1章\n\n本文"},
            {"page": 2, "file": "p2.png", "text": "第1章\n\n続き"},
        ]
        chapters = mod.merge_pages_into_chapters(pages)
        self.assertEqual([c["title"] for c in chapters], ["第1章"])
        self.assertEqual(chapters[0]["pages"][0]["text"], "本文")
        self.assertEqual(chapters[0]["pages"][1]["text"], "続き")


class TestTextCleaning(unittest.TestCase):
    def test_clean_text_strips_noise_and_html(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_text"))

        raw = (
            "❌ No document content detected\n"
            "<div style=\"text-align: center;\">![](foo.png)</div>\n"
            "  # Title  \n"
            "\n"
            "Hello\n"
            "World\n"
        )
        cleaned = mod.clean_text(raw)
        self.assertNotIn("No document content detected", cleaned)
        self.assertNotIn("<div", cleaned)
        self.assertNotIn("![](", cleaned)
        self.assertIn("# Title", cleaned)
        self.assertIn("Hello", cleaned)
        self.assertIn("World", cleaned)

    def test_clean_text_strips_html_img_tags(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_text"))

        raw = (
            '<div style="text-align: center;"><img src="imgs/foo.jpg" alt="Image" width="34%" /></div>\n'
            "\n"
            "カマドウマ\n"
            "\n"
            '<img src="imgs/bar.jpg" alt="Image" />\n'
            "本文\n"
        )
        cleaned = mod.clean_text(raw)
        self.assertNotIn("<img", cleaned)
        self.assertNotIn("imgs/", cleaned)
        self.assertIn("カマドウマ", cleaned)
        self.assertIn("本文", cleaned)

    def test_clean_text_drops_empty_tables(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_text"))

        raw = "<table border=1><tr><td></td></tr></table>"
        self.assertEqual(mod.clean_text(raw), "")

    def test_clean_text_drops_numeric_only_lines(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_text"))

        self.assertEqual(mod.clean_text("11 333"), "")


class TestImagePolicy(unittest.TestCase):
    def test_should_include_page_image_auto(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "should_include_page_image"))

        should = mod.should_include_page_image
        self.assertTrue(should({"page": 1, "text": ""}, policy="auto", short_text_threshold=120))
        self.assertTrue(
            should(
                {"page": 2, "text": "❌ No document content detected"},
                policy="auto",
                short_text_threshold=120,
            )
        )
        self.assertTrue(
            should({"page": 3, "text": "短い"}, policy="auto", short_text_threshold=120)
        )
        self.assertFalse(
            should({"page": 4, "text": "長い" * 200}, policy="auto", short_text_threshold=120)
        )

    def test_should_include_page_image_force_image(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "should_include_page_image"))

        should = mod.should_include_page_image
        page = {"page": 1, "text": "長い" * 200, "force_image": True}
        self.assertTrue(should(page, policy="auto", short_text_threshold=0))

    def test_should_include_page_image_auto_includes_html_image_pages_even_when_threshold_zero(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "should_include_page_image"))

        should = mod.should_include_page_image
        page = {
            "page": 1,
            "text": '<div style="text-align: center;"><img src="imgs/x.jpg" alt="Image" /></div>\nCaption',
        }
        self.assertTrue(should(page, policy="auto", short_text_threshold=0))


class TestBlankImageDetection(unittest.TestCase):
    def test_is_blank_image(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "is_blank_image"))

        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            blank = td_path / "blank.png"
            not_blank = td_path / "not_blank.png"

            Image.new("RGB", (200, 200), (255, 255, 255)).save(blank)
            img = Image.new("RGB", (200, 200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle((60, 80, 140, 120), fill=(0, 0, 0))
            img.save(not_blank)

            self.assertTrue(mod.is_blank_image(blank))
            self.assertFalse(mod.is_blank_image(not_blank))


class TestMarkdownBuild(unittest.TestCase):
    def test_build_markdown_skips_blank_images(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "build_markdown"))

        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            blank = td_path / "blank.png"
            Image.new("RGB", (200, 200), (255, 255, 255)).save(blank)

            chapters = [{"title": "T", "pages": [{"page": 1, "file": str(blank), "text": ""}]}]
            md = mod.build_markdown(chapters, image_policy="auto", short_text_threshold=0)
            self.assertNotIn(str(blank), md)

    def test_build_markdown_uses_page_image_and_drops_noisy_img_page_text(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "build_markdown"))

        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            page_img = td_path / "page.png"
            img = Image.new("RGB", (200, 200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle((50, 50, 150, 150), fill=(0, 0, 0))
            img.save(page_img)

            chapters = [
                {
                    "title": "T",
                    "pages": [
                        {
                            "page": 1,
                            "file": str(page_img),
                            "text": "カマドウマ\n本文",
                            "raw_text": '<div style="text-align: center;"><img src="imgs/x.jpg" /></div>\nカマドウマ\n本文',
                        }
                    ],
                }
            ]
            md = mod.build_markdown(chapters, image_policy="auto", short_text_threshold=0)
            self.assertIn(f"![]({page_img})", md)
            self.assertIn("カマドウマ", md)
            self.assertIn("本文", md)


class TestImageCropping(unittest.TestCase):
    def test_crop_image_whitespace_reduces_dimensions(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "crop_image_whitespace"))

        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "src.png"
            dst = td_path / "dst.png"

            img = Image.new("RGB", (200, 200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle((60, 80, 140, 120), fill=(0, 0, 0))
            img.save(src)

            mod.crop_image_whitespace(src, dst, threshold=250, margin=0)

            cropped = Image.open(dst)
            self.assertLess(cropped.size[0], 200)
            self.assertLess(cropped.size[1], 200)


class TestEpubCssPatch(unittest.TestCase):
    def test_patch_epub_css_appends_and_preserves_mimetype_first(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "patch_epub_css"))

        from zipfile import ZIP_STORED, ZipFile

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr("OEBPS/styles/style.css", "p { margin: 1em; }")
                zf.writestr("OEBPS/nav.xhtml", "<html></html>")

            mod.patch_epub_css(epub_path, "p { margin: 0; }")

            with ZipFile(epub_path, "r") as zf:
                infos = zf.infolist()
                self.assertEqual(infos[0].filename, "mimetype")
                self.assertEqual(infos[0].compress_type, ZIP_STORED)
                css = zf.read("OEBPS/styles/style.css").decode("utf-8")
                self.assertIn("p { margin: 0; }", css)


class TestEpubBuild(unittest.TestCase):
    def test_build_epub_skips_missing_page_images(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "build_epub"))

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "book.epub"
            missing_image = Path(td) / "missing.png"
            chapters = [
                {
                    "title": "T",
                    "pages": [
                        {
                            "page": 1,
                            "file": str(missing_image),
                            "text": "本文",
                            "raw_text": "本文",
                        }
                    ],
                }
            ]

            mod.build_epub(
                chapters,
                output_path,
                title="Book",
                author="Author",
                cover_image=None,
            )

            self.assertTrue(output_path.exists())
