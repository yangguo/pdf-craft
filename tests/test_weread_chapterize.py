import re
import zipfile
from pathlib import Path

from paddle_pipeline.weread_chapterize import (
    chapterize_epub_for_weread,
    verify_weread_chapterized_epub,
)


def _write_sample_epub(path: Path) -> None:
    nav = """<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body><nav epub:type="toc" id="toc"><ol>
<li><a href="front.xhtml#maps">地圖目錄</a></li>
<li><a href="chapter.xhtml#ch1">第一章 測試章</a><ol>
  <li><a href="chapter.xhtml#sec1">阿美士德使團，1816年</a></li>
  <li><a href="chapter.xhtml#sec2">戰爭爆發</a></li>
</ol></li>
<li><a href="chapter.xhtml#part2">第二編 測試編，1900-1910年</a></li>
<li><a href="chapter.xhtml#ch2">第二章 後續章</a><ol>
  <li><a href="chapter.xhtml#sec3">新文化運動的展開</a></li>
</ol></li>
</ol></nav></body></html>
"""
    ncx = """<?xml version='1.0' encoding='utf-8'?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="sample"/></head>
<docTitle><text>Sample</text></docTitle><navMap>
<navPoint id="n1" playOrder="1"><navLabel><text>地圖目錄</text></navLabel><content src="front.xhtml#maps"/></navPoint>
</navMap></ncx>
"""
    opf = """<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Sample</dc:title><dc:identifier id="id">sample-id</dc:identifier><dc:language>zh</dc:language>
</metadata>
<manifest>
<item href="nav.xhtml" id="nav" media-type="application/xhtml+xml" properties="nav"/>
<item href="front.xhtml" id="front" media-type="application/xhtml+xml"/>
<item href="chapter.xhtml" id="chapter" media-type="application/xhtml+xml"/>
<item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
<item href="style/nav.css" id="css" media-type="text/css"/>
</manifest>
<spine toc="ncx"><itemref idref="nav"/><itemref idref="front"/><itemref idref="chapter"/></spine>
</package>
"""
    front = """<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="maps">地圖目錄</h1><p>front matter text</p>
</body></html>
"""
    chapter = """<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="ch1">第一章 測試章</h1><p id="sec1">阿美士德使團，1816年 remains in body.</p>
<p id="sec2">戰爭爆發 remains in body.</p>
<h1 id="part2">第二編 測試編，1900-1910年</h1><p>part intro</p>
<h1 id="ch2">第二章 後續章</h1><p id="sec3">新文化運動的展開 remains in body.</p>
</body></html>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("EPUB/content.opf", opf)
        archive.writestr("EPUB/nav.xhtml", nav)
        archive.writestr("EPUB/toc.ncx", ncx)
        archive.writestr("EPUB/front.xhtml", front)
        archive.writestr("EPUB/chapter.xhtml", chapter)
        archive.writestr("EPUB/style/nav.css", "h1 { font-size: 1.2em; }")


def _write_wrapper_boundary_epub(path: Path) -> None:
    nav = """<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body><nav epub:type="toc" id="toc"><ol>
<li><a href="chapter.xhtml#ch1">第一章 包裝章</a></li>
</ol></nav></body></html>
"""
    ncx = """<?xml version='1.0' encoding='utf-8'?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="sample"/></head>
<docTitle><text>Sample</text></docTitle><navMap>
<navPoint id="n1" playOrder="1"><navLabel><text>第一章 包裝章</text></navLabel><content src="chapter.xhtml#ch1"/></navPoint>
</navMap></ncx>
"""
    opf = """<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Sample</dc:title><dc:identifier id="id">sample-id</dc:identifier><dc:language>zh</dc:language>
</metadata>
<manifest>
<item href="nav.xhtml" id="nav" media-type="application/xhtml+xml" properties="nav"/>
<item href="chapter.xhtml" id="chapter" media-type="application/xhtml+xml"/>
<item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
</manifest>
<spine toc="ncx"><itemref idref="nav"/><itemref idref="chapter"/></spine>
</package>
"""
    chapter = """<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<section id="ch1"><h1>第一章 包裝章</h1><p>wrapped paragraph must survive.</p></section>
<p>following paragraph must survive.</p>
</body></html>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("EPUB/content.opf", opf)
        archive.writestr("EPUB/nav.xhtml", nav)
        archive.writestr("EPUB/toc.ncx", ncx)
        archive.writestr("EPUB/chapter.xhtml", chapter)


def _write_nonstandard_opf_id_epub(path: Path) -> None:
    base = path.with_suffix(".base.epub")
    _write_sample_epub(base)
    with zipfile.ZipFile(base) as archive:
        infos = archive.infolist()
        payloads = {info.filename: archive.read(info.filename) for info in infos}
    opf = payloads["EPUB/content.opf"].decode("utf-8")
    opf = opf.replace('id="nav"', 'id="navdoc"')
    opf = opf.replace('id="ncx"', 'id="tocfile"')
    opf = opf.replace('toc="ncx"', 'toc="tocfile"')
    opf = opf.replace('idref="nav"', 'idref="navdoc"')
    payloads["EPUB/content.opf"] = opf.encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for info in infos:
            archive.writestr(info, payloads[info.filename])
    base.unlink()


def _zip_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member).decode("utf-8")


def test_chapterize_epub_flattens_toc_and_splits_spine(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_sample_epub(source)

    before_issues = verify_weread_chapterized_epub(source)
    assert before_issues

    result = chapterize_epub_for_weread(source, output)

    assert result.segment_count == 4
    assert result.backup_path is None
    assert verify_weread_chapterized_epub(output) == []

    nav = _zip_text(output, "EPUB/nav.xhtml")
    assert nav.count("<li>") == 4
    assert "阿美士德使團" not in nav
    assert "戰爭爆發" not in nav
    assert "新文化運動的展開" not in nav

    opf = _zip_text(output, "EPUB/content.opf")
    assert "front.xhtml" not in opf
    assert "chapter.xhtml" not in opf
    assert opf.count('idref="weread_') == 4

    with zipfile.ZipFile(output) as archive:
        content_files = [
            name for name in archive.namelist()
            if name.startswith("EPUB/") and name.endswith(".xhtml")
            and name not in {"EPUB/nav.xhtml", "EPUB/cover.xhtml"}
        ]
        assert content_files == [
            "EPUB/weread_001.xhtml",
            "EPUB/weread_002.xhtml",
            "EPUB/weread_003.xhtml",
            "EPUB/weread_004.xhtml",
        ]
        combined = "".join(archive.read(name).decode("utf-8") for name in content_files)

    assert "阿美士德使團，1816年 remains in body." in combined
    assert "戰爭爆發 remains in body." in combined
    assert "新文化運動的展開 remains in body." in combined
    assert len(re.findall(r"<h1\b", combined)) == 4


def test_chapterize_epub_can_patch_in_place_with_backup(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    backup = tmp_path / "source.before.epub"
    _write_sample_epub(source)

    result = chapterize_epub_for_weread(source, backup_path=backup)

    assert result.output_path == source
    assert result.backup_path == backup
    assert backup.exists()
    assert verify_weread_chapterized_epub(source) == []
    assert verify_weread_chapterized_epub(backup)


def test_chapterize_preserves_body_when_boundary_id_is_on_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_wrapper_boundary_epub(source)

    chapterize_epub_for_weread(source, output)

    body = _zip_text(output, "EPUB/weread_001.xhtml")
    assert "wrapped paragraph must survive." in body
    assert "following paragraph must survive." in body
    assert len(re.findall(r"<h1\b", body)) == 1
    assert body.count('id="ch1"') == 1


def test_chapterize_preserves_existing_nav_and_ncx_manifest_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_nonstandard_opf_id_epub(source)

    chapterize_epub_for_weread(source, output)

    opf = _zip_text(output, "EPUB/content.opf")
    assert 'toc="tocfile"' in opf
    assert 'idref="navdoc"' in opf
    assert 'toc="ncx"' not in opf
    assert 'idref="nav"' not in opf
    assert verify_weread_chapterized_epub(output) == []


def test_verify_reports_missing_required_members_without_raising(tmp_path: Path) -> None:
    source = tmp_path / "broken.epub"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr(
            "mimetype",
            "application/epub+zip",
            compress_type=zipfile.ZIP_DEFLATED,
        )

    issues = verify_weread_chapterized_epub(source)

    assert "mimetype is not the first ZIP member" in issues
    assert "mimetype must be stored without compression" in issues
    assert "missing required member: EPUB/content.opf" in issues
    assert "missing required member: EPUB/nav.xhtml" in issues
    assert "missing required member: EPUB/toc.ncx" in issues
