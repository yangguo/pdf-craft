import json
import tempfile
import unittest
from pathlib import Path


def _write_checkpoint(path: Path, marker: str) -> None:
    path.write_text(
        json.dumps(
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": marker, "images": {}}}
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class TestRebuildFromCheckpoint(unittest.TestCase):
    def test_load_checkpoint_results_sorts_chunks_by_numeric_page_range(self):
        from paddle_pipeline.rebuild_from_checkpoint import load_checkpoint_results

        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            _write_checkpoint(work_dir / "chunk_10_15.pdf.json", "page 11")
            _write_checkpoint(work_dir / "chunk_0_5.pdf.json", "page 1")
            _write_checkpoint(work_dir / "chunk_5_10.pdf.json", "page 6")
            (work_dir / "notes.json").write_text("{}", encoding="utf-8")

            results = load_checkpoint_results(str(work_dir))

        first_page_texts = [
            result["result"]["layoutParsingResults"][0]["markdown"]["text"]
            for result in results
        ]
        self.assertEqual(["page 1", "page 6", "page 11"], first_page_texts)


if __name__ == "__main__":
    unittest.main()
