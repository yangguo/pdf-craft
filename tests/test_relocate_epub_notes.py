from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "converting-pdfs-to-epub"
    / "scripts"
    / "relocate_epub_notes.py"
)

XHTML = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Chapter</title></head>
  <body>
    <h1>Chapter</h1>
    <p id="p1">First prose <a epub:type="noteref" href="#fn1">1</a>.</p>
    <aside class="footnotes" epub:type="footnotes">
      <p id="fn1">1. First note. <a epub:type="backlink" href="#p1">back</a></p>
    </aside>
    <p id="p2">Second prose.</p>
    <aside class="footnotes extra" epub:type="footnotes">
      <p id="fn2">2. Second note.</p>
    </aside>
    <p id="p3">Third prose.</p>
  </body>
</html>
"""


def _load_relocator():
    assert SCRIPT.exists(), "relocator script is not implemented"
    spec = importlib.util.spec_from_file_location("relocate_epub_notes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_relocate_notes_to_chapter_end_preserves_prose_and_links() -> None:
    relocator = _load_relocator()

    rewritten, moved = relocator.relocate_notes_in_xhtml(XHTML)

    assert moved == 2
    root = ElementTree.fromstring(rewritten)
    ns = {
        "x": "http://www.w3.org/1999/xhtml",
        "epub": "http://www.idpf.org/2007/ops",
    }
    body = root.find("x:body", ns)
    assert body is not None
    assert [child.attrib.get("id") for child in body if child.tag.endswith("p")] == [
        "p1",
        "p2",
        "p3",
    ]
    assert not body.findall("x:aside", ns)

    endnotes = body[-1]
    assert endnotes.tag.endswith("section")
    assert endnotes.attrib["class"] == "chapter-notes"
    assert endnotes.attrib["{http://www.idpf.org/2007/ops}type"] == "endnotes"
    heading = endnotes.find("x:h2", ns)
    assert heading is not None
    assert heading.text == "注釋"
    notes = endnotes.findall("x:aside", ns)
    note_paragraphs = [note.find("x:p", ns) for note in notes]
    assert all(paragraph is not None for paragraph in note_paragraphs)
    assert [
        paragraph.attrib["id"]
        for paragraph in note_paragraphs
        if paragraph is not None
    ] == ["fn1", "fn2"]
    assert root.find(".//x:a[@href='#fn1']", ns) is not None
    assert root.find(".//x:a[@href='#p1']", ns) is not None


def test_relocate_notes_leaves_chapter_without_notes_byte_identical() -> None:
    relocator = _load_relocator()
    chapter = XHTML.replace(
        b"""    <aside class="footnotes" epub:type="footnotes">
      <p id="fn1">1. First note. <a epub:type="backlink" href="#p1">back</a></p>
    </aside>
""",
        b"",
    ).replace(
        b"""    <aside class="footnotes extra" epub:type="footnotes">
      <p id="fn2">2. Second note.</p>
    </aside>
""",
        b"",
    )

    rewritten, moved = relocator.relocate_notes_in_xhtml(chapter)

    assert moved == 0
    assert rewritten == chapter


def test_relocate_notes_is_idempotent() -> None:
    relocator = _load_relocator()

    rewritten, moved = relocator.relocate_notes_in_xhtml(XHTML)
    second_pass, moved_again = relocator.relocate_notes_in_xhtml(rewritten)

    assert moved == 2
    assert moved_again == 0
    assert second_pass == rewritten


def _write_fixture_epub(path: Path) -> bytes:
    plain_chapter = XHTML.replace(
        b"""    <aside class="footnotes" epub:type="footnotes">
      <p id="fn1">1. First note. <a epub:type="backlink" href="#p1">back</a></p>
    </aside>
""",
        b"",
    ).replace(
        b"""    <aside class="footnotes extra" epub:type="footnotes">
      <p id="fn2">2. Second note.</p>
    </aside>
""",
        b"",
    )
    image = b"\xff\xd8fixture-image\xff\xd9"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            b"application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr(
            "META-INF/container.xml",
            b'<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "EPUB/content.opf",
            b'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"><metadata><dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Fixture</dc:title></metadata><manifest/></package>',
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "EPUB/chapter.xhtml",
            XHTML,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "EPUB/plain.xhtml",
            plain_chapter,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "EPUB/style/book.css",
            b"body { line-height: 1.5; }\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "EPUB/images/cover.jpg",
            image,
            compress_type=zipfile.ZIP_DEFLATED,
        )
    return image


def test_relocate_epub_notes_repackages_safely(tmp_path: Path) -> None:
    relocator = _load_relocator()
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    image = _write_fixture_epub(source)

    stats = relocator.relocate_epub_notes(source, output)

    assert stats == {"chapters_changed": 1, "notes_moved": 2}
    assert output.stat().st_mode & 0o777 == source.stat().st_mode & 0o777
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(output) as after:
        infos = after.infolist()
        assert infos[0].filename == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert after.read("mimetype") == b"application/epub+zip"
        assert after.read("META-INF/container.xml") == before.read(
            "META-INF/container.xml"
        )
        assert after.read("EPUB/content.opf") == before.read("EPUB/content.opf")
        assert after.read("EPUB/plain.xhtml") == before.read("EPUB/plain.xhtml")
        assert after.read("EPUB/images/cover.jpg") == image
        assert b".chapter-notes" in after.read("EPUB/style/book.css")

        chapter = ElementTree.fromstring(after.read("EPUB/chapter.xhtml"))
        body = chapter.find(f"{{http://www.w3.org/1999/xhtml}}body")
        assert body is not None
        assert body[-1].attrib["class"] == "chapter-notes"
        assert not body.findall("{http://www.w3.org/1999/xhtml}aside")


def test_relocate_epub_notes_rejects_in_place_rewrite(tmp_path: Path) -> None:
    relocator = _load_relocator()
    source = tmp_path / "source.epub"
    _write_fixture_epub(source)

    with pytest.raises(ValueError, match="must be different"):
        relocator.relocate_epub_notes(source, source)


def test_main_writes_output_and_reports_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    relocator = _load_relocator()
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_fixture_epub(source)

    assert relocator.main([str(source), str(output)]) == 0
    report = capsys.readouterr().out
    assert '"chapters_changed": 1' in report
    assert '"notes_moved": 2' in report
    assert output.exists()
