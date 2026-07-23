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

INLINE_MARKER_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter</title></head>
  <body>
    <p id="before">具有繼往開來的意義。</p>
    <blockquote><p>[35] 可見即</p></blockquote>
    <p id="after">使國家開始走下坡的時候，和親仍能發揮作用。</p>
  </body>
</html>
""".encode("utf-8")

INTERLEAVED_MARKER_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Chapter</title></head>
  <body>
    <p>前頁正文。</p>
    <blockquote><p>[9] 跨頁未完</p></blockquote>
    <aside class="footnotes" epub:type="footnotes"><p>9. 注釋。</p></aside>
    <span class="page-anchor" id="page-2"/>
    <p>正文續句。</p>
  </body>
</html>
""".encode("utf-8")


def _load_relocator():
    assert SCRIPT.exists(), "relocator script is not implemented"
    spec = importlib.util.spec_from_file_location("relocate_epub_notes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repair_inline_note_marker_joins_cjk_paragraph_fragments() -> None:
    relocator = _load_relocator()

    rewritten, repaired, joined = relocator.repair_inline_note_markers_in_xhtml(
        INLINE_MARKER_XHTML
    )

    assert (repaired, joined) == (1, 1)
    root = ElementTree.fromstring(rewritten)
    body = root.find("{http://www.w3.org/1999/xhtml}body")
    assert body is not None
    assert [child.tag.rsplit("}", 1)[-1] for child in body] == ["p"]
    paragraph = body[0]
    assert "".join(paragraph.itertext()) == (
        "具有繼往開來的意義。[35]可見即使國家開始走下坡的時候，"
        "和親仍能發揮作用。"
    )
    marker = paragraph.find("{http://www.w3.org/1999/xhtml}sup")
    assert marker is not None
    assert marker.attrib["class"] == "inline-note-marker"
    assert marker.text == "[35]"


def _repair_body(body_html: str):
    relocator = _load_relocator()
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title></head>'
        f"<body>{body_html}</body></html>"
    ).encode("utf-8")
    rewritten, repaired, joined = relocator.repair_inline_note_markers_in_xhtml(
        chapter
    )
    root = ElementTree.fromstring(rewritten)
    body = root.find("{http://www.w3.org/1999/xhtml}body")
    assert body is not None
    return body, repaired, joined


def test_repair_inline_note_marker_preserves_genuine_quotation() -> None:
    body, repaired, joined = _repair_body(
        "<p>正文。</p><blockquote><p>真正的長篇引文。</p></blockquote><p>後文。</p>"
    )

    assert (repaired, joined) == (0, 0)
    assert [child.tag.rsplit("}", 1)[-1] for child in body] == [
        "p",
        "blockquote",
        "p",
    ]


def test_repair_marker_only_keeps_next_paragraph_separate() -> None:
    body, repaired, joined = _repair_body(
        "<p>前段需要注釋。</p><blockquote><p>[4]</p></blockquote>"
        "<p>鴉片戰爭後是新段落。</p>"
    )

    assert (repaired, joined) == (1, 0)
    assert [child.tag.rsplit("}", 1)[-1] for child in body] == ["p", "p"]
    assert "".join(body[0].itertext()) == "前段需要注釋。[4]"
    assert "".join(body[1].itertext()) == "鴉片戰爭後是新段落。"


def test_repair_sentence_ending_marker_keeps_next_paragraph_separate() -> None:
    body, repaired, joined = _repair_body(
        "<p>前文，</p><blockquote><p>[15] 同時價值也獲得重檢。</p></blockquote>"
        "<p>下一段另起。</p>"
    )

    assert (repaired, joined) == (1, 0)
    assert [child.tag.rsplit("}", 1)[-1] for child in body] == ["p", "p"]
    assert "".join(body[0].itertext()) == "前文，[15]同時價值也獲得重檢。"


def test_repair_marker_joins_across_following_page_anchor() -> None:
    body, repaired, joined = _repair_body(
        '<p>前文。</p><blockquote><p>[2] 進步觀念成為</p></blockquote>'
        '<span class="page-anchor" id="page-2"/><p>啟蒙時代的共同信仰。</p>'
    )

    assert (repaired, joined) == (1, 1)
    assert len(body) == 1
    assert "".join(body[0].itertext()) == "前文。[2]進步觀念成為啟蒙時代的共同信仰。"
    anchor = body[0].find("{http://www.w3.org/1999/xhtml}span")
    assert anchor is not None
    assert anchor.attrib["id"] == "page-2"


def test_repair_marker_joins_after_preceding_page_anchor() -> None:
    body, repaired, joined = _repair_body(
        '<p>跨頁前文。</p><span class="page-anchor" id="page-3"/>'
        "<blockquote><p>[50] 自可加</p></blockquote><p>深認識。</p>"
    )

    assert (repaired, joined) == (1, 1)
    assert len(body) == 1
    assert "".join(body[0].itertext()) == "跨頁前文。[50]自可加深認識。"
    anchor = body[0].find("{http://www.w3.org/1999/xhtml}span")
    assert anchor is not None
    assert anchor.attrib["id"] == "page-3"


def test_repair_orphan_marker_becomes_paragraph_without_touching_quote() -> None:
    body, repaired, joined = _repair_body(
        "<blockquote><p>真正引文。</p></blockquote>"
        "<blockquote><p>[51] 於晚清讀書界</p></blockquote>"
        "<p>一直備受推崇。</p>"
    )

    assert (repaired, joined) == (1, 1)
    assert [child.tag.rsplit("}", 1)[-1] for child in body] == [
        "blockquote",
        "p",
    ]
    assert "".join(body[1].itertext()) == "[51]於晚清讀書界一直備受推崇。"


@pytest.mark.parametrize(
    "fragment",
    [
        "<h3>41. 頁底注釋續文。</h3>",
        "<blockquote><p>係資料—德島</p></blockquote>",
    ],
)
def test_relocate_interposed_note_fragment_restores_cross_page_prose(
    fragment: str,
) -> None:
    relocator = _load_relocator()
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>T</title></head>'
        "<body><p>前文。</p><blockquote><p>[42] 各婦女界</p></blockquote>"
        f'{fragment}<span class="page-anchor" id="page-2"/><p>精英續文。</p>'
        '<section class="chapter-notes" epub:type="endnotes"><h2>注釋</h2>'
        '<aside class="footnotes" epub:type="footnotes"><p>既有注釋。</p></aside>'
        "</section></body></html>"
    ).encode("utf-8")

    relocated, fragments_moved = (
        relocator.relocate_interposed_note_fragments_in_xhtml(chapter)
    )
    rewritten, repaired, joined = relocator.repair_inline_note_markers_in_xhtml(
        relocated
    )

    assert fragments_moved == 1
    assert (repaired, joined) == (1, 1)
    root = ElementTree.fromstring(rewritten)
    ns = {
        "x": "http://www.w3.org/1999/xhtml",
        "epub": "http://www.idpf.org/2007/ops",
    }
    body = root.find("x:body", ns)
    assert body is not None
    assert "".join(body[0].itertext()) == "前文。[42]各婦女界精英續文。"
    endnotes = body.find("x:section[@class='chapter-notes']", ns)
    assert endnotes is not None
    moved = endnotes.findall("x:aside[@class='footnotes relocated-fragment']", ns)
    assert len(moved) == 1
    assert moved[0].find("x:p/x:p", ns) is None
    assert "".join(moved[0].itertext()) in {
        "41. 頁底注釋續文。",
        "係資料—德島",
    }


def test_reorder_displaced_prefix_moves_text_before_inline_marker() -> None:
    relocator = _load_relocator()
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title></head>'
        '<body><p>借款仍有可<sup class="inline-note-marker">[46]</sup>'
        "蔣介石於16望，順聞。」日收到電文。</p></body></html>"
    ).encode("utf-8")

    rewritten, applied = relocator.reorder_displaced_prefixes_in_xhtml(
        chapter,
        [
            {
                "marker": "46",
                "remainder": "蔣介石於16",
                "continuation_starts": "日收到",
            }
        ],
    )

    assert applied == 1
    root = ElementTree.fromstring(rewritten)
    paragraph = root.find(
        "{http://www.w3.org/1999/xhtml}body/"
        "{http://www.w3.org/1999/xhtml}p"
    )
    assert paragraph is not None
    assert "".join(paragraph.itertext()) == (
        "借款仍有可望，順聞。」[46]蔣介石於16日收到電文。"
    )


def test_repair_leading_paragraph_marker_as_superscript() -> None:
    relocator = _load_relocator()
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title></head>'
        "<body><p>[48] 在中國需要援助時，仍繼續正文。</p></body></html>"
    ).encode("utf-8")

    rewritten, repaired = (
        relocator.repair_leading_paragraph_markers_in_xhtml(chapter)
    )

    assert repaired == 1
    root = ElementTree.fromstring(rewritten)
    paragraph = root.find(
        "{http://www.w3.org/1999/xhtml}body/"
        "{http://www.w3.org/1999/xhtml}p"
    )
    assert paragraph is not None
    assert "".join(paragraph.itertext()) == "[48]在中國需要援助時，仍繼續正文。"
    marker = paragraph[0]
    assert marker.tag == "{http://www.w3.org/1999/xhtml}sup"
    assert marker.attrib["class"] == "inline-note-marker"


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
            "EPUB/marker.xhtml",
            INTERLEAVED_MARKER_XHTML,
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

    assert stats == {
        "chapters_changed": 2,
        "notes_moved": 3,
        "note_fragments_relocated": 0,
        "inline_markers_repaired": 1,
        "leading_markers_repaired": 0,
        "continuations_joined": 1,
        "reading_order_corrections_applied": 0,
    }
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
        assert b".inline-note-marker" in after.read("EPUB/style/book.css")

        chapter = ElementTree.fromstring(after.read("EPUB/chapter.xhtml"))
        body = chapter.find(f"{{http://www.w3.org/1999/xhtml}}body")
        assert body is not None
        assert body[-1].attrib["class"] == "chapter-notes"
        assert not body.findall("{http://www.w3.org/1999/xhtml}aside")

        marker_chapter = ElementTree.fromstring(after.read("EPUB/marker.xhtml"))
        marker_body = marker_chapter.find("{http://www.w3.org/1999/xhtml}body")
        assert marker_body is not None
        assert len(marker_body) == 2
        assert marker_body[0].find(
            "{http://www.w3.org/1999/xhtml}sup"
        ) is not None
        assert "".join(marker_body[0].itertext()) == (
            "前頁正文。[9]跨頁未完正文續句。"
        )
        assert marker_body[-1].attrib["class"] == "chapter-notes"


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
    assert '"chapters_changed": 2' in report
    assert '"notes_moved": 3' in report
    assert '"inline_markers_repaired": 1' in report
    assert '"continuations_joined": 1' in report
    assert output.exists()
