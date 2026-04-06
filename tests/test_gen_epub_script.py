import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_script_module(fake_transform_epub):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gen_epub.py"
    spec = importlib.util.spec_from_file_location("gen_epub_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    fake_pdf_craft = types.SimpleNamespace(
        LaTeXRender=types.SimpleNamespace(MATHML="MATHML"),
        OCREventKind=lambda kind: kind,
        TableRender=types.SimpleNamespace(HTML="HTML"),
        transform_epub=fake_transform_epub,
    )

    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pdf_craft": fake_pdf_craft}):
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class TestGenEpubScript(unittest.TestCase):
    def test_default_analysing_dir_changes_when_pdf_is_replaced_in_place(self):
        calls = []

        def fake_transform_epub(**kwargs):
            calls.append(kwargs)

        mod = _load_script_module(fake_transform_epub)

        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "book.pdf"
            pdf_path.write_bytes(b"first version")
            os.utime(pdf_path, (1_700_000_000, 1_700_000_000))

            with patch.object(sys, "argv", ["gen_epub.py", str(pdf_path)]):
                mod.main()

            first_analysing_path = Path(calls[-1]["analysing_path"])

            pdf_path.write_bytes(b"second version with different size")
            os.utime(pdf_path, (1_700_000_100, 1_700_000_100))

            with patch.object(sys, "argv", ["gen_epub.py", str(pdf_path)]):
                mod.main()

            second_analysing_path = Path(calls[-1]["analysing_path"])

        self.assertNotEqual(first_analysing_path, second_analysing_path)
