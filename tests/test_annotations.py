from __future__ import annotations

from pathlib import Path

from PIL import Image

from echolabel.data.annotations import (
    build_annotation_tables,
    pair_images_and_xml,
    reconcile_with_source,
    split_key,
    strip_extension,
)

XML_WITH_SIZE = """<annotation><size><width>{w}</width><height>{h}</height><depth>3</depth></size>
<object><name>{label}</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>11</xmax><ymax>6</ymax></bndbox></object></annotation>"""
XML_NO_SIZE = """<annotation><object><name>{label}</name>
<bndbox><xmin>2</xmin><ymin>2</ymin><xmax>12</xmax><ymax>7</ymax></bndbox></object></annotation>"""


def _make_image(path: Path, w: int = 40, h: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (10, 20, 30)).save(path)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build_tree(root: Path) -> tuple[Path, Path]:
    images, xml = root / "images", root / "xml"
    _make_image(images / "2018/ST1/Left/a.jpg")
    _write(xml / "2018/ST1/Left/a.xml", XML_WITH_SIZE.format(w=40, h=30, label="herring"))
    _make_image(images / "2018/ST1/Left/b.png", 50, 20)
    _write(xml / "2018/ST1/Left/b.xml", XML_NO_SIZE.format(label="krill"))
    _make_image(images / "2024/T1/Left/c.jpg")
    _write(xml / "2024/T1/Left/d.xml", XML_WITH_SIZE.format(w=40, h=30, label="krill"))
    # Same basename as a.jpg under another station: must not pair with the 2018/ST1 file.
    _make_image(images / "2018/ST2/Left/a.jpg")
    # XML one directory level shallower than its image: paired by stem within the tow.
    _make_image(images / "2023/20230518T1608Z/Left/e_left.jpg")
    _write(xml / "2023/20230518T1608Z/e_left.xml", XML_WITH_SIZE.format(w=40, h=30, label="krill"))
    # XML under a station whose image lives under a transect directory of the same year.
    _make_image(images / "2023/20230520T1632Z/Left/f.jpg")
    _write(xml / "2023/ST95/Left/f.xml", XML_WITH_SIZE.format(w=40, h=30, label="krill"))
    # Nested duplicate trees that must be skipped when excluded.
    _make_image(images / "images/2018/ST1/Left/a.jpg")
    _write(xml / "xml/2018/ST1/Left/a.xml", XML_WITH_SIZE.format(w=40, h=30, label="herring"))
    return images, xml


def test_helpers() -> None:
    assert strip_extension("2018/ST1/Left/a.jpg") == "2018/ST1/Left/a"
    parts = split_key("2018/ST1/Left/a")
    assert parts == {
        "year": "2018",
        "station": "ST1",
        "camera": "Left",
        "rel_dir": "2018/ST1/Left",
        "basename": "a",
    }
    assert split_key("a")["year"] == ""
    deep = split_key("2020/rect/20200525T0445Z/x")
    assert deep["station"] == "20200525T0445Z" and deep["camera"] == ""
    shallow = split_key("2023/20230518T1608Z/x_left")
    assert shallow["station"] == "20230518T1608Z" and shallow["camera"] == ""


def test_pairing(tmp_path: Path) -> None:
    images, xml = _build_tree(tmp_path)
    pairs_all, _ = pair_images_and_xml(images, xml)
    assert "images/2018/ST1/Left/a" in set(pairs_all["key"])
    pairs, dup = pair_images_and_xml(images, xml, exclude_top=("images", "xml"))
    assert "images/2018/ST1/Left/a" not in set(pairs["key"])
    status = dict(zip(pairs["key"], pairs["status"], strict=True))
    assert status["2018/ST1/Left/a"] == "paired"
    assert status["2018/ST1/Left/b"] == "paired"
    assert status["2024/T1/Left/c"] == "image_only"
    assert status["2024/T1/Left/d"] == "xml_only"
    assert status["2018/ST2/Left/a"] == "image_only"
    assert status["2023/20230518T1608Z/e_left"] == "paired_by_stem"
    assert "2023/20230518T1608Z/Left/e_left" not in status
    row = pairs.set_index("key").loc["2023/20230518T1608Z/e_left"]
    assert row["image_rel"] == "2023/20230518T1608Z/Left/e_left.jpg"
    assert status["2023/ST95/Left/f"] == "xml_only"
    assert (
        pairs.set_index("key").loc["2023/ST95/Left/f", "stem_match_elsewhere"] == "2023/20230520T1632Z/Left/f"
    )
    assert dup == []


def test_tables_fill_size_from_image(tmp_path: Path) -> None:
    images, xml = _build_tree(tmp_path)
    tables = build_annotation_tables(images, xml, n_workers=2, exclude_top=("images", "xml"))
    img = tables["images"].set_index("key")
    assert img.loc["2018/ST1/Left/a", "size_source"] == "xml"
    assert img.loc["2018/ST1/Left/b", "size_source"] == "image"
    assert img.loc["2018/ST1/Left/b", "width_used"] == 50
    assert img.loc["2018/ST1/Left/b", "height_used"] == 20
    assert "missing_size" in img.loc["2018/ST1/Left/b", "anomalies"]
    assert img.loc["2023/20230518T1608Z/e_left", "image_readable"]
    boxes = tables["boxes"]
    assert len(boxes) == 5
    b = boxes.set_index("key").loc["2018/ST1/Left/b"]
    assert abs(b["area_frac"] - (10 * 5) / (50 * 20)) < 1e-9


def test_reconcile(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    src = tmp_path / "src"
    for root in (dest, src):
        _write(root / "xml/2018/a.xml", "<annotation/>")
    _write(dest / "xml/xml/2018/a.xml", "<annotation/>")
    _write(dest / "xml/2018/only_dest.xml", "<annotation><x/></annotation>")
    table, summary = reconcile_with_source(dest, src, n_workers=2)
    assert summary["n_extra_in_dest"] == 2
    assert summary["n_missing_in_dest"] == 0
    row = table.set_index("rel").loc["xml/xml/2018/a.xml"]
    assert row["identical_copy_exists"]
    assert row["duplicate_of"] == "xml/2018/a.xml"
    assert not table.set_index("rel").loc["xml/2018/only_dest.xml", "identical_copy_exists"]
