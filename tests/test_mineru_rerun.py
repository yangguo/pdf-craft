import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _make_pdf(path: Path, page_count: int) -> None:
    from paddle_pipeline.config import fitz

    if fitz is None:
        raise unittest.SkipTest("PyMuPDF dependency not installed")

    doc = fitz.open()
    for page_number in range(1, page_count + 1):
        page = doc.new_page(width=180, height=240)
        page.insert_text((36, 72), f"page {page_number}")
    doc.save(path)
    doc.close()


def _layout_texts(result):
    return [
        item["markdown"]["text"]
        for item in result["result"]["layoutParsingResults"]
    ]


class TestMineruRerun(unittest.TestCase):
    def test_parse_page_ranges_accepts_commas_and_ranges(self):
        from paddle_pipeline.mineru_rerun import parse_page_ranges

        self.assertEqual([5, 71, 75, 76, 77, 78, 79, 82], parse_page_ranges("5,71,75-79,82"))

    def test_rerun_pages_replaces_only_target_checkpoint_page(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=8)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_4_8.pdf.json"
            old_result = {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "old page 5 bad OCR", "images": {}}},
                        {"markdown": {"text": "old page 6", "images": {}}},
                        {"markdown": {"text": "old page 7", "images": {}}},
                        {"markdown": {"text": "old page 8", "images": {}}},
                    ]
                },
                "_mineru_zip_url": "https://example.test/old.zip",
            }
            checkpoint.write_text(json.dumps(old_result), encoding="utf-8")

            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": (
                                    "第二，自治的主體是港人。基本法草案授予香港特區"
                                    "高度的自治權限，並貫徹「港人治港」。"
                                ),
                                "images": {},
                            }
                        }
                    ]
                },
                "_mineru_zip_url": "https://example.test/page-5.zip",
            }
            calls = []

            def fake_parse(chunk_path, token=None):
                calls.append(Path(chunk_path).name)
                return new_page_result

            with mock.patch.object(mineru_api, "parse_pdf_chunk", fake_parse):
                summaries = rerun_mineru_pages(
                    str(pdf_path),
                    str(work_dir),
                    pages=[5],
                    token="token",
                    chunk_size=4,
                    replace_page=True,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(1, len(calls))
        self.assertEqual([5], [item.page_number for item in summaries])
        self.assertEqual(str(checkpoint), summaries[0].checkpoint_path)
        self.assertEqual(
            [
                "第二，自治的主體是港人。基本法草案授予香港特區高度的自治權限，並貫徹「港人治港」。",
                "old page 6",
                "old page 7",
                "old page 8",
            ],
            _layout_texts(saved),
        )
        self.assertNotIn("_mineru_zip_url", saved)
        self.assertEqual("https://example.test/old.zip", saved["_mineru_original_zip_url"])
        self.assertEqual(
            [{
                "page": 5,
                "mode": "replace_page",
                "zip_url": "https://example.test/page-5.zip",
            }],
            saved["_mineru_partial_reruns"],
        )

    def test_rerun_strips_mineru_single_page_running_header(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=4)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_0_4.pdf.json"
            checkpoint.write_text(
                json.dumps({
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "old page 1", "images": {}}},
                            {"markdown": {"text": "old page 2 body", "images": {}}},
                            {"markdown": {"text": "old page 3", "images": {}}},
                            {"markdown": {"text": "old page 4", "images": {}}},
                        ]
                    }
                }),
                encoding="utf-8",
            )

            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "# 許家屯香港回憶錄\n\nnew page 2 corrected body",
                                "images": {},
                            }
                        }
                    ]
                }
            }

            with mock.patch.object(mineru_api, "parse_pdf_chunk", return_value=new_page_result):
                rerun_mineru_pages(
                    str(pdf_path),
                    str(work_dir),
                    pages=[2],
                    token="token",
                    chunk_size=4,
                    replace_page=True,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(
            ["old page 1", "new page 2 corrected body", "old page 3", "old page 4"],
            _layout_texts(saved),
        )

    def test_rerun_last_chunk_uses_actual_pdf_page_count_in_checkpoint_name(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=7)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_4_7.pdf.json"
            checkpoint.write_text(
                json.dumps({
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "old page 5", "images": {}}},
                            {"markdown": {"text": "old page 6", "images": {}}},
                            {"markdown": {"text": "old page 7 bad OCR", "images": {}}},
                        ]
                    }
                }),
                encoding="utf-8",
            )

            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "new page 7 corrected", "images": {}}}
                    ]
                }
            }

            with mock.patch.object(mineru_api, "parse_pdf_chunk", return_value=new_page_result):
                summaries = rerun_mineru_pages(
                    str(pdf_path),
                    str(work_dir),
                    pages=[7],
                    token="token",
                    chunk_size=4,
                    replace_page=True,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(str(checkpoint), summaries[0].checkpoint_path)
        self.assertEqual(
            ["old page 5", "old page 6", "new page 7 corrected"],
            _layout_texts(saved),
        )

    def test_rerun_refuses_checkpoint_with_missing_physical_pages(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=8)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_4_8.pdf.json"
            checkpoint.write_text(
                json.dumps({
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "old page 5", "images": {}}},
                            {"markdown": {"text": "old page 7", "images": {}}},
                            {"markdown": {"text": "old page 8", "images": {}}},
                        ]
                    }
                }),
                encoding="utf-8",
            )

            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "new page 7 corrected", "images": {}}}
                    ]
                }
            }

            with mock.patch.object(mineru_api, "parse_pdf_chunk", return_value=new_page_result):
                with self.assertRaises(RuntimeError) as cm:
                    rerun_mineru_pages(
                        str(pdf_path),
                        str(work_dir),
                        pages=[7],
                        token="token",
                        chunk_size=4,
                        replace_page=True,
                    )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertIn("does not match physical chunk length", str(cm.exception))
        self.assertEqual(["old page 5", "old page 7", "old page 8"], _layout_texts(saved))

    def test_rerun_patches_only_suspicious_ocr_span_by_default(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=4)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_0_4.pdf.json"
            repeated_context = (
                "香港問題是中國近代史上的重要問題。中英雙方經過多輪談判，"
                "逐步形成了有關香港前途和制度安排的基本方針。"
            ) * 20
            original_page = (
                "談心會剛結束，李先念的祕書打電話告訴我，"
                "先念同志最近身體不好。我表示要去看望他。當天，我就去李家。\n\n"
                "李先念見了我以後，誤有炎言尺會果宜一書一堂遺事"
                "「下茂故二八辛戰巴。"
            )
            checkpoint.write_text(
                json.dumps({
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": repeated_context, "images": {}}},
                            {"markdown": {"text": original_page, "images": {}}},
                            {"markdown": {"text": repeated_context, "images": {}}},
                            {"markdown": {"text": repeated_context, "images": {}}},
                        ]
                    }
                }),
                encoding="utf-8",
            )

            mineru_page = (
                "談心會剛結束，李先念的祕書打電話告訴我，"
                "先念同志最近身體不好。我表示要去看望他。当天，我就去李家。\n\n"
                "李先念見了我以後，沒有談這次會，寒暄一番之後就講："
                "「你歲數不小了，辭職吧。」"
            )
            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": mineru_page, "images": {}}}
                    ]
                }
            }

            with mock.patch.object(mineru_api, "parse_pdf_chunk", return_value=new_page_result):
                rerun_mineru_pages(
                    str(pdf_path),
                    str(work_dir),
                    pages=[2],
                    token="token",
                    chunk_size=4,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            patched_text = _layout_texts(saved)[1]

        self.assertIn("當天，我就去李家", patched_text)
        self.assertNotIn("当天，我就去李家", patched_text)
        self.assertNotIn("誤有炎言尺會果宜", patched_text)
        self.assertIn(
            "李先念見了我以後，沒有談這次會，寒暄一番之後就講",
            patched_text,
        )
        self.assertEqual(
            [{"page": 2, "mode": "patch_suspicious_spans", "patched_spans": 1}],
            saved["_mineru_partial_reruns"],
        )

    def test_patch_missing_page_start_replaces_partial_overlap(self):
        from paddle_pipeline.mineru_rerun import _patch_missing_page_start

        original = {
            "markdown": {
                "text": (
                    "之。就像同時代的一些人（包括共產黨在内），"
                    "他有很長一段時間和祕密的政治、犯罪團體關係密切。"
                )
            }
        }
        replacement = {
            "markdown": {
                "text": (
                    "行動和祕密警察的鎮壓，令數千人喪失性命。"
                    "他和其後不民主的強人領導一樣僞善，"
                    "但他不是個犬儒之人。就像同時代的一些人（包括共產黨在內），"
                    "他有很長一段時間和祕密的政治犯罪團體關係密切。"
                )
            }
        }

        patched, count = _patch_missing_page_start(original, replacement)
        text = patched["markdown"]["text"]

        self.assertEqual(1, count)
        self.assertTrue(text.startswith("行動和祕密警察的鎮壓"))
        self.assertIn("犬儒之人。就像同時代", text)
        self.assertNotIn("犬儒之人。之。就像", text)

    def test_rerun_can_patch_missing_page_start(self):
        from paddle_pipeline import mineru_api
        from paddle_pipeline.mineru_rerun import rerun_mineru_pages

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "book.pdf"
            _make_pdf(pdf_path, page_count=2)

            work_dir = root / "work"
            work_dir.mkdir()
            checkpoint = work_dir / "chunk_0_2.pdf.json"
            checkpoint.write_text(
                json.dumps({
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "上一頁以無情的軍事", "images": {}}},
                            {
                                "markdown": {
                                    "text": (
                                        "之。就像同時代的一些人（包括共產黨在内），"
                                        "他有很長一段時間和祕密的政治、犯罪團體關係密切。"
                                    ),
                                    "images": {},
                                }
                            },
                        ]
                    }
                }),
                encoding="utf-8",
            )
            new_page_result = {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": (
                                    "行動和祕密警察的鎮壓，令數千人喪失性命。"
                                    "他和其後不民主的強人領導一樣僞善，"
                                    "但他不是個犬儒之人。就像同時代的一些人（包括共產黨在內），"
                                    "他有很長一段時間和祕密的政治犯罪團體關係密切。"
                                ),
                                "images": {},
                            }
                        }
                    ]
                }
            }

            with mock.patch.object(mineru_api, "parse_pdf_chunk", return_value=new_page_result):
                rerun_mineru_pages(
                    str(pdf_path),
                    str(work_dir),
                    pages=[2],
                    token="token",
                    chunk_size=2,
                    patch_page_start=True,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            patched_text = _layout_texts(saved)[1]

        self.assertTrue(patched_text.startswith("行動和祕密警察的鎮壓"))
        self.assertIn("犬儒之人。就像同時代", patched_text)
        self.assertEqual(
            [{"page": 2, "mode": "patch_missing_page_start", "patched_spans": 1}],
            saved["_mineru_partial_reruns"],
        )

    def test_should_patch_hunk_rejects_garbled_to_garbled_replacement(self):
        """_should_patch_hunk must return False when the replacement span is
        also garbled — garbled→garbled patches are disallowed.

        Both spans use characters that appear exactly once in the corpus, so
        every bigram is a singleton and both score near 1.0 (garbled)."""
        from collections import Counter
        from paddle_pipeline.mineru_rerun import _should_patch_hunk
        from paddle_pipeline.ocr_review import _cjk_chars

        base_context = "香港問題是中國近代史上的重要問題中英雙方經過多輪談判" * 50
        # Both spans use chars that appear exactly once in the corpus — their
        # bigrams are singletons (count=1), so both score near 1.0 (garbled).
        original_span = "誤齶焱巆墀壯耷縹嗶欐垯艿"    # 12 unique CJK chars
        replacement_span = "鑲燼覦罹褻蓿灃痂蝻蚼鏤殛"  # 12 unique CJK chars
        corpus = base_context + original_span + replacement_span
        chars_list = _cjk_chars(corpus)
        char_freq = Counter(chars_list)
        bigram_freq = Counter(
            chars_list[i] + chars_list[i + 1] for i in range(len(chars_list) - 1)
        )

        result = _should_patch_hunk(original_span, replacement_span, char_freq, bigram_freq)
        self.assertFalse(result, "Should not patch garbled span with another garbled replacement")

    def test_should_patch_hunk_rejects_replacement_with_unseen_garbled_bigrams(self):
        """Unseen replacement bigrams must not be treated as safe."""
        from collections import Counter
        from paddle_pipeline.mineru_rerun import _should_patch_hunk
        from paddle_pipeline.ocr_review import _cjk_chars

        base_context = "香港問題是中國近代史上的重要問題中英雙方經過多輪談判" * 50
        original_span = "誤齶焱巆墀壯耷縹嗶欐垯艿"
        replacement_span = "鑲燼覦罹褻蓿灃痂蝻蚼鏤殛"

        # Model is built from checkpoint-only text: replacement chars/bigrams are unseen.
        chars_list = _cjk_chars(base_context + original_span)
        char_freq = Counter(chars_list)
        bigram_freq = Counter(
            chars_list[i] + chars_list[i + 1] for i in range(len(chars_list) - 1)
        )

        result = _should_patch_hunk(original_span, replacement_span, char_freq, bigram_freq)
        self.assertFalse(result, "Should reject replacement that introduces unseen garbled bigrams")

    def test_strip_single_page_running_header_preserves_chapter_headings(self):
        """_strip_single_page_running_header must preserve chapter-pattern
        headings (e.g. '第三章') unconditionally, even when the original OCR
        text was too garbled to contain the heading text for confirmation."""
        from paddle_pipeline.mineru_rerun import _strip_single_page_running_header

        original = {"markdown": {"text": "誤齶焱巆炎墀壯亱耷縹稂嗶欐垯艿誤齶焱巆炎"}}
        replacement = {
            "markdown": {
                "text": "# 第三章 香港的自治權\n\n本章討論香港回歸後的自治架構與制度安排。",
            }
        }

        result = _strip_single_page_running_header(replacement, original)
        text = result["markdown"]["text"]
        self.assertTrue(
            text.startswith("# 第三章"),
            f"Chapter heading should be preserved but got: {text[:80]}",
        )


if __name__ == "__main__":
    unittest.main()
