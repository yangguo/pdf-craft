import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_clean():
    from paddle_pipeline.mineru_api import _clean_mineru_markdown
    return _clean_mineru_markdown


class TestMineruCleaning(unittest.TestCase):
    """Tests for _clean_mineru_markdown — MinerU VLM image artifact removal."""

    # --- <details> block removal (step 0) ---

    def test_removes_details_block_with_label(self):
        clean = _load_clean()
        md = (
            "![](images/img.jpg)\n"
            "<details>\n<summary>natural_image</summary>\n\n</details>\n"
            "text after\n"
        )
        result = clean(md)
        self.assertNotIn("<details>", result)
        self.assertNotIn("natural_image", result)
        self.assertIn("text after", result)
        self.assertIn("![](images/img.jpg)", result)

    def test_removes_details_block_with_description(self):
        clean = _load_clean()
        md = (
            "![](images/img.jpg)\n"
            "<details>\n<summary>text_image</summary>\n\n"
            "Black and white photo with no visible text\n"
            "</details>\n"
            "text after\n"
        )
        result = clean(md)
        self.assertNotIn("<details>", result)
        self.assertNotIn("Black and white photo", result)
        self.assertIn("text after", result)

    # --- standalone label removal (step 1) ---

    def test_removes_standalone_label_line(self):
        clean = _load_clean()
        md = "before\nnatural_image\nafter\n"
        result = clean(md)
        self.assertNotIn("natural_image", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    def test_removes_label_with_colon_description(self):
        clean = _load_clean()
        md = (
            "before\n"
            "natural_image: Full-body photo of a person wearing a jacket\n"
            "after\n"
        )
        result = clean(md)
        self.assertNotIn("Full-body photo", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    # --- image alt text cleaning (step 2) ---

    def test_strips_label_alt_text_from_image(self):
        clean = _load_clean()
        md = "![natural_image](images/img.jpg)"
        result = clean(md)
        self.assertIn("[](images/img.jpg)", result)
        self.assertNotIn("natural_image", result)

    def test_strips_long_english_alt_text(self):
        clean = _load_clean()
        md = (
            "![Full-body photo of a person wearing a beige "
            "trousers and light-colored jacket, standing with hands "
            "on hips (no visible text or symbols)](images/img.jpg)"
        )
        result = clean(md)
        self.assertIn("[](images/img.jpg)", result)
        self.assertNotIn("Full-body photo", result)

    def test_preserves_short_legitimate_alt_text(self):
        clean = _load_clean()
        md = "![Figure 1: System Architecture](images/arch.png)"
        result = clean(md)
        self.assertIn("![Figure 1: System Architecture](images/arch.png)", result)

    def test_preserves_empty_alt_text(self):
        clean = _load_clean()
        md = "![](images/img.jpg)"
        result = clean(md)
        self.assertIn("![](images/img.jpg)", result)

    # --- standalone description removal (step 3) ---

    def test_removes_ai_description_with_note(self):
        clean = _load_clean()
        md = (
            "before\n"
            "Full-body photo of a person wearing a beige trousers and "
            "light-colored jacket, standing with hands on hips (no visible text or symbols)\n"
            "after\n"
        )
        result = clean(md)
        self.assertNotIn("Full-body photo", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    # --- regression guards (false-positive prevention) ---

    def test_preserves_figure_caption(self):
        clean = _load_clean()
        md = "Figure 1. The relationship between X and Y over time"
        result = clean(md)
        self.assertIn("Figure 1.", result)

    def test_preserves_photography_word(self):
        clean = _load_clean()
        md = "photography techniques for landscape composition"
        result = clean(md)
        self.assertIn("photography", result)

    def test_preserves_photographic_word(self):
        clean = _load_clean()
        md = "photographic evidence suggests otherwise for consideration"
        result = clean(md)
        self.assertIn("photographic", result)

    def test_preserves_chinese_text(self):
        clean = _load_clean()
        md = (
            "第一章 引言\n\n"
            "這是一段正常的中文文本內容，不應該被過濾掉。\n\n"
            "第二段文字繼續，包含一些標點符號。\n"
        )
        result = clean(md)
        self.assertIn("第一章 引言", result)
        self.assertIn("中文文本內容", result)
        self.assertIn("第二段文字繼續", result)

    def test_removes_text_image_label(self):
        clean = _load_clean()
        md = "before\ntext_image\nafter\n"
        result = clean(md)
        self.assertNotIn("text_image", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    # --- OCR mid-line truncation repair (step 3.6) ---

    def test_merges_truncated_line_with_continuation(self):
        clean = _load_clean()
        md = (
            "第二，中國不是阿根廷，英國無力用福島的方式，賴在香港不走。"
            "第三，英方已知中國收回香港的「底牌」：採用「一國兩制」方針\n\n"
            "國在香港既有的利益，還可利用這一方針，争取更多的利益。\n"
        )
        result = clean(md)
        self.assertIn("方針國在香港既有的利益", result)
        # Should not have the broken split
        self.assertNotIn("方針\n\n國", result)

    def test_preserves_real_paragraph_break(self):
        clean = _load_clean()
        md = (
            "這是一段正文結束。\n\n"
            "第二段從這裡開始。\n"
        )
        result = clean(md)
        self.assertIn("結束。\n\n第二段", result)

    def test_preserves_sentence_start_with_第(self):
        clean = _load_clean()
        md = (
            "前面一段結尾文字沒有標點\n\n"
            "第二節新的開始\n"
        )
        result = clean(md)
        # "第" is a sentence starter, should NOT merge
        self.assertIn("文字沒有標點\n\n第二節", result)

    def test_preserves_sentence_start_with_common_function_word(self):
        clean = _load_clean()
        md = (
            "前文描述到此仍未收束而且還在延伸\n\n"
            "在這個背景下，新的段落需要保留。\n"
        )
        result = clean(md)
        self.assertIn("還在延伸\n\n在這個背景下", result)


if __name__ == "__main__":
    unittest.main()
