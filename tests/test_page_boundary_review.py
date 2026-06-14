import json
import tempfile
import unittest

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _write_checkpoint(path: Path, page_texts: list[str]) -> None:
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


def _write_checkpoint_pages(path: Path, pages: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"result": {"layoutParsingResults": pages}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_epub(path: Path, chapter_text: str) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=0)
        archive.writestr(
            "OEBPS/chapter.xhtml",
            (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<body><p>{chapter_text}</p></body></html>"
            ),
        )


class TestPageBoundaryReview(unittest.TestCase):
    def test_flags_known_cross_page_missing_sentence_boundary(self):
        from paddle_pipeline.page_boundary_review import find_page_boundary_candidates

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint(
                work_dir / "chunk_52_54.pdf.json",
                [
                    (
                        "他甚至想要娶她，但是根據她寫給他的一封信所述，"
                        "即令她想和他長相"
                    ),
                    "蔣和許多女子發生性關係的一個結果，就是他自認感染性病。",
                ],
            )

            candidates = find_page_boundary_candidates(
                str(work_dir), min_score=0.65, limit=10,
            )

        self.assertEqual(1, len(candidates), candidates)
        candidate = candidates[0]
        self.assertEqual(53, candidate["previous_page"])
        self.assertEqual(54, candidate["next_page"])
        self.assertGreaterEqual(candidate["score"], 0.65)
        self.assertIn("open_tail_suffix", candidate["reasons"])
        self.assertIn("即令她想和他長相", candidate["tail"])
        self.assertIn("蔣和許多女子", candidate["head"])

    def test_flags_image_first_page_head_garble_after_open_tail(self):
        from paddle_pipeline.page_boundary_review import find_page_boundary_candidates

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint_pages(
                work_dir / "chunk_32_34.pdf.json",
                [
                    {
                        "markdown": {
                            "text": (
                                "中共自己也找到好幾座紅"
                            ),
                            "images": {},
                        }
                    },
                    {
                        "markdown": {
                            "text": (
                                '<div><img src="imgs/photo.jpg" alt="Image" /></div>\n\n'
                                "軍源拉白坩一室之一\n\n"
                                "可能是根據岡村提供的報告，蔣知道蘇聯拿到多少日本武器。"
                            ),
                            "images": {"imgs/photo.jpg": ""},
                        }
                    },
                    {
                        "markdown": {
                            "text": "後續正常頁面已經完整結束。",
                            "images": {},
                        }
                    },
                ],
            )

            candidates = find_page_boundary_candidates(
                str(work_dir), min_score=0.70, limit=10,
            )

        self.assertEqual(1, len(candidates), candidates)
        candidate = candidates[0]
        self.assertEqual(33, candidate["previous_page"])
        self.assertEqual(34, candidate["next_page"])
        self.assertGreaterEqual(candidate["score"], 0.70)
        self.assertIn("next_page_starts_with_image", candidate["reasons"])
        self.assertIn("garbled_page_head", candidate["reasons"])
        self.assertTrue(candidate["next_starts_with_image"])
        self.assertIn("軍源拉白坩一室之一", candidate["head"])

    def test_flags_markdown_image_with_cjk_alt_text_before_page_head(self):
        from paddle_pipeline.page_boundary_review import find_page_boundary_candidates

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint_pages(
                work_dir / "chunk_32_34.pdf.json",
                [
                    {"markdown": {"text": "中共自己也找到好幾座紅"}},
                    {
                        "markdown": {
                            "text": (
                                "![圖片說明](imgs/photo.jpg)\n\n"
                                "軍源拉白坩一室之一\n\n"
                                "可能是根據岡村提供的報告，蔣知道蘇聯拿到多少日本武器。"
                            ),
                        }
                    },
                ],
            )

            candidates = find_page_boundary_candidates(
                str(work_dir), min_score=0.70, limit=10,
            )

        self.assertEqual(1, len(candidates), candidates)
        candidate = candidates[0]
        self.assertEqual(33, candidate["previous_page"])
        self.assertEqual(34, candidate["next_page"])
        self.assertIn("next_page_starts_with_image", candidate["reasons"])
        self.assertIn("garbled_page_head", candidate["reasons"])
        self.assertTrue(candidate["next_starts_with_image"])
        self.assertGreaterEqual(candidate["head_singleton_bigram_ratio"], 0.75)
        self.assertIn("軍源拉白坩一室之一", candidate["head"])

    def test_ignores_terminal_page_boundary(self):
        from paddle_pipeline.page_boundary_review import find_page_boundary_candidates

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint(
                work_dir / "chunk_0_2.pdf.json",
                [
                    "這一頁的句子已經完整結束。",
                    "下一頁從新的句子開始，並沒有頁首漏字。",
                ],
            )

            candidates = find_page_boundary_candidates(str(work_dir))

        self.assertEqual([], candidates)

    def test_detects_page_start_omission_from_second_ocr_alignment(self):
        from paddle_pipeline.page_boundary_review import compare_page_start_ocr

        checkpoint_head = "蔣和許多女子發生性關係的一個結果就是他自認感染性病"
        second_ocr = (
            "廝守她並不想被一紙正式婚約所綁這段時期儘管感情不睦"
            "蔣仍不時返回寧波見姚冶誠蔣和許多女子發生性關係"
        )

        result = compare_page_start_ocr(checkpoint_head, second_ocr, window=16)

        self.assertTrue(result["possible_omission"], result)
        self.assertGreaterEqual(result["second_ocr_offset"], 30)
        self.assertGreaterEqual(result["similarity"], 0.80)

    def test_second_ocr_alignment_accepts_matching_page_start(self):
        from paddle_pipeline.page_boundary_review import compare_page_start_ocr

        checkpoint_head = "蔣和許多女子發生性關係的一個結果就是他自認感染性病"
        second_ocr = "蔣和許多女子發生性關係的一個結果就是他自認感染性病"

        result = compare_page_start_ocr(checkpoint_head, second_ocr, window=16)

        self.assertFalse(result["possible_omission"], result)
        self.assertLessEqual(result["second_ocr_offset"], 2)
        self.assertGreaterEqual(result["similarity"], 0.95)

    def test_annotates_boundary_when_checkpoint_join_exists_in_epub(self):
        from paddle_pipeline.page_boundary_review import (
            annotate_candidates_with_epub_continuity,
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            _write_epub(
                epub_path,
                (
                    "他甚至想要娶她，但是根據她寫給他的一封信所述，"
                    "即令她想和他長相廝守，她並不想被一紙正式婚約所綁。"
                ),
            )
            candidates = [
                {
                    "previous_page": 53,
                    "next_page": 54,
                    "tail": "即令她想和他長相",
                    "head": "廝守，她並不想被一紙正式婚約所綁。",
                }
            ]

            annotated = annotate_candidates_with_epub_continuity(
                candidates, str(epub_path), window=10,
            )

        self.assertEqual("checkpoint_boundary_present", annotated[0]["epub_status"])
        self.assertTrue(annotated[0]["epub_boundary_present"])
        self.assertGreaterEqual(annotated[0]["epub_match_index"], 0)
        self.assertIn(
            "即令她想和他長相廝守她並不想被一紙",
            annotated[0]["epub_boundary_window"],
        )

    def test_annotates_boundary_when_checkpoint_join_is_missing_from_epub(self):
        from paddle_pipeline.page_boundary_review import (
            annotate_candidates_with_epub_continuity,
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            _write_epub(
                epub_path,
                (
                    "他甚至想要娶她，但是根據她寫給他的一封信所述，"
                    "即令她想和他長相蔣和許多女子發生性關係。"
                ),
            )
            candidates = [
                {
                    "previous_page": 53,
                    "next_page": 54,
                    "tail": "即令她想和他長相",
                    "head": "廝守，她並不想被一紙正式婚約所綁。",
                }
            ]

            annotated = annotate_candidates_with_epub_continuity(
                candidates, str(epub_path), window=10,
            )

        self.assertEqual("checkpoint_boundary_missing_from_epub", annotated[0]["epub_status"])
        self.assertFalse(annotated[0]["epub_boundary_present"])
        self.assertEqual(-1, annotated[0]["epub_match_index"])


if __name__ == "__main__":
    unittest.main()
