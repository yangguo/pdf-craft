import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


class TestOcrReview(unittest.TestCase):
    def test_review_scanner_reports_short_mixed_cjk_noise_candidates(self):
        from paddle_pipeline.ocr_review import find_suspicious_cjk_spans_in_epub

        normal_context = (
            "香港問題是中國近代史上的重要問題。中英雙方經過多輪談判，"
            "逐步形成了有關香港前途和制度安排的基本方針。回顧當時情況，"
            "很多人既有期待，也有疑慮，需要耐心解釋和溝通。"
        ) * 80
        bad_one = (
            "李先念見了我以後，誤有炎言尺會果宜一書一堂遺事"
            "「下茂故二八辛戰巴。"
        )
        bad_two = (
            "第二，自治的主體是港人。基本去享家受子香巷寺區高度勺自台蘿最，"
            "並貫散「巷人治港」。"
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "mimetype", "application/epub+zip", compress_type=ZIP_STORED
                )
                archive.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body>"
                        f"<p>{normal_context}</p>"
                        f"<p>{bad_one}</p>"
                        f"<p>{bad_two}</p>"
                        "</body></html>"
                    ),
                )

            candidates = find_suspicious_cjk_spans_in_epub(
                str(epub_path), limit=20, min_score=0.68
            )

        excerpts = [item["excerpt"] for item in candidates]
        self.assertTrue(
            any("誤有炎言尺會果宜" in excerpt for excerpt in excerpts),
            excerpts,
        )
        self.assertTrue(
            any("基本去享家受子香巷" in excerpt for excerpt in excerpts),
            excerpts,
        )
        self.assertTrue(all("score" in item for item in candidates))
        self.assertTrue(all("reasons" in item for item in candidates))

    def test_review_scanner_does_not_report_domain_repeated_normal_sentence(self):
        from paddle_pipeline.ocr_review import find_suspicious_cjk_spans_in_epub

        normal_context = (
            "中英聯合聲明正式簽署後，香港前途逐漸明朗。"
            "中國政府多次說明基本方針政策，強調保持原有制度和生活方式。"
            "談判過程仍有分歧，但各方都需要用清楚文字固定承諾。"
        ) * 40
        one_off_sentence = (
            "公元一九八四年中英聯合聲明正式簽署後香港前途逐漸明朗"
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "mimetype", "application/epub+zip", compress_type=ZIP_STORED
                )
                archive.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{normal_context}</p><p>{one_off_sentence}</p></body></html>",
                )

            candidates = find_suspicious_cjk_spans_in_epub(
                str(epub_path), limit=20, min_score=0.68
            )

        self.assertFalse(
            any(one_off_sentence in item["excerpt"] for item in candidates),
            candidates,
        )

    def test_review_scanner_ignores_short_normal_question(self):
        from paddle_pipeline.ocr_review import find_suspicious_cjk_spans_in_epub

        normal_context = (
            "香港問題是中國近代史上的重要問題。中英雙方經過多輪談判，"
            "逐步形成了有關香港前途和制度安排的基本方針。"
        ) * 80
        # This sentence is composed entirely of bigrams that recur throughout
        # normal_context (≥80 repetitions), so its anchor_bigram_ratio is high
        # and it scores well below the 0.68 threshold.
        normal_sentence = "中英雙方經過多輪談判逐步形成基本方針"

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "mimetype", "application/epub+zip", compress_type=ZIP_STORED
                )
                archive.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{normal_context}</p><p>{normal_sentence}</p></body></html>",
                )

            candidates = find_suspicious_cjk_spans_in_epub(
                str(epub_path), limit=20, min_score=0.68, min_cjk=10
            )

        self.assertFalse(
            any(normal_sentence in item["excerpt"] for item in candidates),
            candidates,
        )


if __name__ == "__main__":
    unittest.main()
