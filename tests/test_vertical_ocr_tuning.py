import sys
import tempfile
import unittest

from pathlib import Path
from unittest import mock

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPaddleVerticalTuning(unittest.TestCase):
    def test_build_optional_payload_uses_vertical_defaults(self):
        from paddle_pipeline import paddle_api

        payload = paddle_api._build_optional_payload()

        self.assertEqual(0.35, payload["layoutThreshold"])
        self.assertEqual(0.1, payload["temperature"])
        self.assertEqual(1.05, payload["repetitionPenalty"])
        self.assertEqual(0.75, payload["topP"])

    def test_build_optional_payload_allows_module_overrides(self):
        from paddle_pipeline import paddle_api

        with (
            mock.patch.object(paddle_api, "PADDLE_LAYOUT_THRESHOLD", 0.4),
            mock.patch.object(paddle_api, "PADDLE_TEMPERATURE", 0.2),
            mock.patch.object(paddle_api, "PADDLE_REPETITION_PENALTY", 1.1),
            mock.patch.object(paddle_api, "PADDLE_TOP_P", 0.8),
        ):
            payload = paddle_api._build_optional_payload()

        self.assertEqual(0.4, payload["layoutThreshold"])
        self.assertEqual(0.2, payload["temperature"])
        self.assertEqual(1.1, payload["repetitionPenalty"])
        self.assertEqual(0.8, payload["topP"])

    def test_split_pdf_adds_top_and_bottom_padding(self):
        from paddle_pipeline import paddle_api

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source.pdf"
            doc = fitz.open()
            doc.new_page(width=200, height=300)
            doc.save(src)
            doc.close()

            chunk_path = paddle_api.split_pdf(str(src), chunk_size=1)[0]

            chunk_doc = fitz.open(chunk_path)
            try:
                page = chunk_doc[0]
                self.assertEqual(fitz.Rect(0, -15, 200, 315), page.mediabox)
                self.assertEqual(330.0, page.rect.height)
            finally:
                chunk_doc.close()


class TestMineruVerticalTuning(unittest.TestCase):
    def test_build_upload_request_uses_traditional_defaults(self):
        from paddle_pipeline import mineru_api

        payload = mineru_api._build_upload_request("chunk.pdf")

        self.assertEqual("ch_tra", payload["language"])
        self.assertFalse(payload["enable_table"])
        self.assertTrue(payload["is_ocr"])
        self.assertFalse(payload["enable_formula"])

    def test_build_upload_request_allows_module_overrides(self):
        from paddle_pipeline import mineru_api

        with (
            mock.patch.object(mineru_api, "MINERU_LANGUAGE", "ch_server"),
            mock.patch.object(mineru_api, "MINERU_ENABLE_TABLE", True),
        ):
            payload = mineru_api._build_upload_request("chunk.pdf")

        self.assertEqual("ch_server", payload["language"])
        self.assertTrue(payload["enable_table"])

    def test_split_pdf_adds_left_top_and_bottom_padding(self):
        from paddle_pipeline import mineru_api

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source.pdf"
            doc = fitz.open()
            doc.new_page(width=200, height=300)
            doc.save(src)
            doc.close()

            chunk_path = mineru_api.split_pdf(str(src), chunk_size=1)[0]

            chunk_doc = fitz.open(chunk_path)
            try:
                page = chunk_doc[0]
                self.assertEqual(fitz.Rect(-8, -15, 200, 315), page.mediabox)
                self.assertEqual(208.0, page.rect.width)
                self.assertEqual(330.0, page.rect.height)
            finally:
                chunk_doc.close()


class TestTraditionalSentenceStarters(unittest.TestCase):
    def test_preserves_paragraph_break_for_traditional_sentence_starters(self):
        from paddle_pipeline.mineru_api import _clean_mineru_markdown

        for starter in ("於", "其", "吾", "余", "餘", "蓋", "嗟", "夫", "凡"):
            with self.subTest(starter=starter):
                md = (
                    "前面一段結尾沒有標點\n\n"
                    f"{starter}是新的段落開始了\n"
                )

                result = _clean_mineru_markdown(md)

                self.assertIn(f"沒有標點\n\n{starter}是新的段落開始了", result)


if __name__ == "__main__":
    unittest.main()
