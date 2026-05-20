import sys
import unittest
from pathlib import Path
from unittest import mock


class TestPdf2EpubPaddleDependencyCheck(unittest.TestCase):
    def test_check_dependencies_reports_missing_requests(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import paddle_pipeline.paddle_api as api

        with mock.patch.object(api, "requests", None):
            self.assertFalse(api.check_dependencies())

    def test_check_dependencies_all_present(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import paddle_pipeline.paddle_api as api

        if api.requests is None or api.fitz is None or api.epub is None:
            self.skipTest("Dependencies not installed")
        self.assertTrue(api.check_dependencies())
