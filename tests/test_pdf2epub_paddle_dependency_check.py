import builtins
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_script_module_without_requests():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "tmp"
        / "pdf2epub-paddle"
        / "pdf2epub_paddle.py"
    )
    spec = importlib.util.spec_from_file_location("pdf2epub_paddle_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    fitz_mod = types.ModuleType("fitz")
    ebooklib_mod = types.ModuleType("ebooklib")
    setattr(ebooklib_mod, "epub", object())
    dotenv_mod = types.ModuleType("dotenv")
    setattr(dotenv_mod, "load_dotenv", lambda: None)

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "requests":
            raise ImportError("requests missing")
        return orig_import(name, globals, locals, fromlist, level)

    module = importlib.util.module_from_spec(spec)
    patch_modules = patch.dict(
        sys.modules,
        {"fitz": fitz_mod, "ebooklib": ebooklib_mod, "dotenv": dotenv_mod},
    )
    patch_import = patch("builtins.__import__", new=fake_import)

    patch_modules.start()
    patch_import.start()
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        patch_import.stop()
        patch_modules.stop()
        raise

    return module, patch_import, patch_modules


class TestPdf2EpubPaddleDependencyCheck(unittest.TestCase):
    def test_check_dependencies_reports_missing_requests(self):
        mod, patch_import, patch_modules = _load_script_module_without_requests()
        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                ok = mod.check_dependencies()

            self.assertFalse(ok)
            self.assertIn("requests", stdout.getvalue())
        finally:
            patch_import.stop()
            patch_modules.stop()
