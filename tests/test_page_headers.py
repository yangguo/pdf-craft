import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paddle_pipeline.epub_builder import (
    _build_header_fingerprints,
    _strip_page_headers,
)


class TestBuildHeaderFingerprints(unittest.TestCase):
    """Tests for _build_header_fingerprints — chapter-title and book-title fingerprinting."""

    def test_includes_chapter_title(self):
        headings = [{"title": "第三章 中英談判內外", "page": 1}]
        fps = _build_header_fingerprints(headings)
        self.assertIn("第三章中英談判內外", fps)

    def test_splits_chapter_number_and_suffix(self):
        headings = [{"title": "第十四章 「六四」風雲", "page": 1}]
        fps = _build_header_fingerprints(headings)
        self.assertIn("第十四章", fps)
        self.assertIn("「六四」風雲", fps)

    def test_includes_book_title(self):
        fps = _build_header_fingerprints(None, book_title="許家屯香港回憶錄")
        self.assertIn("許家屯香港回憶錄", fps)

    def test_book_title_whitespace_stripped(self):
        fps = _build_header_fingerprints([], book_title="  許家屯  香港  回憶錄  ")
        self.assertIn("許家屯香港回憶錄", fps)

    def test_no_empty_fingerprint(self):
        fps = _build_header_fingerprints(None, book_title="")
        self.assertNotIn("", fps)

    def test_no_headings_produces_empty_set_without_book_title(self):
        fps = _build_header_fingerprints(None)
        self.assertEqual(fps, set())


class TestStripPageHeaders(unittest.TestCase):
    """Tests for _strip_page_headers — page header/footer removal."""

    def test_strips_book_title_at_top(self):
        fps = _build_header_fingerprints([], book_title="許家屯香港回憶錄")
        lines = [
            "許家屯香港回憶錄",
            "這是正文的第一段內容，應該被保留。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("許家屯香港回憶錄", result)
        self.assertIn("這是正文的第一段內容，應該被保留。", result)

    def test_strips_chapter_title_at_bottom(self):
        headings = [{"title": "第三章 中英談判內外", "page": 1}]
        fps = _build_header_fingerprints(headings)
        lines = [
            "這是一段正文內容。",
            "第三章中英談判內外",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("第三章中英談判內外", result)
        self.assertIn("這是一段正文內容。", result)

    def test_strips_chapter_header_mid_page(self):
        headings = [{"title": "第三章 中英談判內外", "page": 1}]
        fps = _build_header_fingerprints(headings)
        lines = [
            "第一段正文。",
            "第三章中英談判內外",
            "第二段正文繼續。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("第三章中英談判內外", result)

    def test_strips_page_number(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "正文內容在這裡。",
            "42",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("42", result)
        self.assertIn("正文內容在這裡。", result)

    def test_strips_chinese_page_number(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "正文內容。",
            "二七",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("二七", result)

    def test_strips_title_bracket_short_line(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "「六四風雲」",
            "正文內容。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("「六四風雲」", result)

    def test_preserves_mid_page_quoted_body_line(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "他停了一下，然後回答。",
            "「我不同意。」",
            "會議室裡一時安靜下來。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertIn("「我不同意。」", result)

    def test_preserves_mismatched_quoted_line_at_edge(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "「錯配\"",
            "正文內容。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertIn("「錯配\"", result)

    def test_preserves_long_chapterish_body_text(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "第三章講述了中英談判的漫長過程，這是一段非常重要的歷史時期",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertIn(lines[0], result)

    def test_strips_short_chapterish_header(self):
        fps = _build_header_fingerprints(None)
        lines = [
            "第三章中英談判內外",
            "正文。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("第三章中英談判內外", result)

    def test_preserves_normal_body_text(self):
        headings = [{"title": "第三章 中英談判內外", "page": 1}]
        fps = _build_header_fingerprints(headings)
        lines = [
            "這是一段正常的正文內容，不應該被過濾掉。",
            "第二段文字繼續。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertEqual(len(result), 2)

    def test_strips_book_title_mid_page(self):
        fps = _build_header_fingerprints([], book_title="許家屯香港回憶錄")
        lines = [
            "正文段落一。",
            "許家屯香港回憶錄",
            "正文段落二。",
        ]
        result = _strip_page_headers(lines, fps)
        self.assertNotIn("許家屯香港回憶錄", result)
        self.assertIn("正文段落一。", result)
        self.assertIn("正文段落二。", result)


class TestMineruPageHeaderDetection(unittest.TestCase):
    """Tests for MinerU-level _detect_page_headers — plain-text header detection."""

    def _load_detect(self):
        from paddle_pipeline.mineru_api import _detect_page_headers
        return _detect_page_headers

    def test_detects_repeated_book_title(self):
        detect = self._load_detect()
        lines = [
            "# 許家屯香港回憶錄",
            "正文第一頁。",
            "# 許家屯香港回憶錄",
            "正文第二頁。",
            "# 許家屯香港回憶錄",
            "正文第三頁。",
        ]
        md_headers, plain_headers = detect(lines)
        self.assertIn("許家屯香港回憶錄", plain_headers)

    def test_detects_chapterish_plain_text_2plus(self):
        detect = self._load_detect()
        lines = [
            "正文行一。",
            "第三章中英談判內外",
            "正文行二。",
            "第三章中英談判內外",
        ]
        md_headers, plain_headers = detect(lines)
        self.assertIn("第三章中英談判內外", plain_headers)

    def test_detects_bracket_title_2plus(self):
        detect = self._load_detect()
        lines = [
            "正文行一。",
            "「六四」風雲",
            "正文行二。",
            "「六四」風雲",
        ]
        md_headers, plain_headers = detect(lines)
        self.assertIn("「六四」風雲", plain_headers)


if __name__ == "__main__":
    unittest.main()
