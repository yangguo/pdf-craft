import sys
import subprocess
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

    def test_missing_tqdm_does_not_hide_core_dependencies(self):
        try:
            import requests  # noqa: F401
            import fitz  # noqa: F401
            from ebooklib import epub  # noqa: F401
        except ImportError:
            self.skipTest("Core dependencies not installed")

        repo_root = Path(__file__).resolve().parents[1]
        script = """
import builtins

original_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tqdm" or name.startswith("tqdm."):
        raise ImportError("forced missing tqdm")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

from paddle_pipeline import config

assert config.requests is not None, "requests hidden by missing tqdm"
assert config.fitz is not None, "fitz hidden by missing tqdm"
assert config.epub is not None, "ebooklib.epub hidden by missing tqdm"
assert config.tqdm is None, "tqdm should be optional"
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
