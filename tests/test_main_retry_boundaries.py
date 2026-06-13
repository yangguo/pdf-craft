import importlib
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


class TestMainRetryBoundaries(unittest.TestCase):
    def test_paddle_parse_none_is_not_retried_by_main(self):
        main_module = importlib.import_module("paddle_pipeline.main")

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                input_pdf = Path(temp_dir, "book.pdf")
                input_pdf.write_bytes(b"%PDF-1.4\n")

                chunk_dir = Path(temp_dir, "chunks")
                chunk_dir.mkdir()
                chunk_path = chunk_dir / "chunk_0_1.pdf"
                chunk_path.write_bytes(b"%PDF-1.4\n")

                parse_mock = mock.Mock(return_value=None)

                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(sys, "argv", ["pdf2epub", str(input_pdf)])
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "check_dependencies", return_value=True)
                    )
                    stack.enter_context(mock.patch.object(main_module, "API_TOKEN", "token"))
                    stack.enter_context(mock.patch.object(main_module, "tqdm", None))
                    stack.enter_context(
                        mock.patch.object(main_module, "split_pdf", return_value=[str(chunk_path)])
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "extract_cover_image", return_value=None)
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "parse_pdf_chunk", parse_mock)
                    )
                    stack.enter_context(
                        mock.patch.object(main_module.time, "sleep", return_value=None)
                    )

                    with self.assertRaises(SystemExit) as exit_ctx:
                        main_module.main()

                self.assertEqual(exit_ctx.exception.code, 1)
                self.assertEqual(parse_mock.call_count, 1)
            finally:
                os.chdir(old_cwd)

    def test_weread_chapterize_runs_after_epub_generation(self):
        main_module = importlib.import_module("paddle_pipeline.main")

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                input_pdf = Path(temp_dir, "book.pdf")
                input_pdf.write_bytes(b"%PDF-1.4\n")
                output_epub = Path(temp_dir, "out.epub")

                chunk_dir = Path(temp_dir, "chunks")
                chunk_dir.mkdir()
                chunk_path = chunk_dir / "chunk_0_1.pdf"
                chunk_path.write_bytes(b"%PDF-1.4\n")

                parse_result = {"result": {"layoutParsingResults": []}}
                events = []
                chapterize_result = mock.Mock(segment_count=0)

                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(
                            sys,
                            "argv",
                            [
                                "pdf2epub",
                                str(input_pdf),
                                "--output",
                                str(output_epub),
                                "--title",
                                "Book",
                                "--author",
                                "Author",
                                "--auto-toc",
                                "--weread-chapterize",
                            ],
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "check_dependencies", return_value=True)
                    )
                    stack.enter_context(mock.patch.object(main_module, "API_TOKEN", "token"))
                    stack.enter_context(mock.patch.object(main_module, "tqdm", None))
                    stack.enter_context(
                        mock.patch.object(main_module, "split_pdf", return_value=[str(chunk_path)])
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "extract_cover_image", return_value=None)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "parse_pdf_chunk", return_value=parse_result
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "apply_page_image_fallbacks", return_value=None
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "repair_page_order_by_printed_numbers",
                            return_value=None,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "extract_candidate_headings", return_value=[]
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "filter_heading_candidates", return_value=[]
                        )
                    )
                    create_epub_mock = stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "create_epub",
                            side_effect=lambda *args, **kwargs: events.append("create_epub"),
                        )
                    )
                    chapterize_mock = stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "chapterize_epub_for_weread",
                            create=True,
                            side_effect=lambda *args, **kwargs: (
                                events.append("chapterize") or chapterize_result
                            ),
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(main_module.time, "sleep", return_value=None)
                    )

                    main_module.main()

                create_epub_mock.assert_called_once()
                self.assertEqual(events, ["create_epub", "chapterize"])
                chapterize_mock.assert_called_once_with(
                    str(output_epub),
                    backup_path=str(
                        output_epub.with_name("out.before_weread_chapterize.epub")
                    ),
                )
            finally:
                os.chdir(old_cwd)

    def test_weread_chapterize_is_opt_in(self):
        main_module = importlib.import_module("paddle_pipeline.main")

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                input_pdf = Path(temp_dir, "book.pdf")
                input_pdf.write_bytes(b"%PDF-1.4\n")
                output_epub = Path(temp_dir, "out.epub")

                chunk_dir = Path(temp_dir, "chunks")
                chunk_dir.mkdir()
                chunk_path = chunk_dir / "chunk_0_1.pdf"
                chunk_path.write_bytes(b"%PDF-1.4\n")

                parse_result = {"result": {"layoutParsingResults": []}}

                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(
                            sys,
                            "argv",
                            [
                                "pdf2epub",
                                str(input_pdf),
                                "--output",
                                str(output_epub),
                                "--title",
                                "Book",
                                "--author",
                                "Author",
                                "--auto-toc",
                            ],
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "check_dependencies", return_value=True)
                    )
                    stack.enter_context(mock.patch.object(main_module, "API_TOKEN", "token"))
                    stack.enter_context(mock.patch.object(main_module, "tqdm", None))
                    stack.enter_context(
                        mock.patch.object(main_module, "split_pdf", return_value=[str(chunk_path)])
                    )
                    stack.enter_context(
                        mock.patch.object(main_module, "extract_cover_image", return_value=None)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "parse_pdf_chunk", return_value=parse_result
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "apply_page_image_fallbacks", return_value=None
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "repair_page_order_by_printed_numbers",
                            return_value=None,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "extract_candidate_headings", return_value=[]
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module, "filter_heading_candidates", return_value=[]
                        )
                    )
                    create_epub_mock = stack.enter_context(
                        mock.patch.object(main_module, "create_epub")
                    )
                    chapterize_mock = stack.enter_context(
                        mock.patch.object(
                            main_module, "chapterize_epub_for_weread", create=True
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(main_module.time, "sleep", return_value=None)
                    )

                    main_module.main()

                create_epub_mock.assert_called_once()
                chapterize_mock.assert_not_called()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
