from __future__ import annotations

from pathlib import Path

from echolabel.data.voc import parse_voc_xml

FULL = """<annotation>
  <folder>mackerel</folder>
  <filename>ST4_539-20180506041404783.jpg</filename>
  <size><width>1228</width><height>1027</height><depth>3</depth></size>
  <object><name>mackerel</name><difficult>0</difficult>
    <bndbox><xmin>316</xmin><ymin>486</ymin><xmax>755</xmax><ymax>692</ymax></bndbox></object>
  <object><name>krill</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>1300</xmax><ymax>20</ymax></bndbox></object>
</annotation>"""

NO_SIZE = """<annotation>
  <filename>x.jpg</filename>
  <object><name> Herring </name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>9</ymax></bndbox></object>
  <object><name>krill</name></object>
</annotation>"""


def test_full_annotation(tmp_path: Path) -> None:
    p = tmp_path / "a.xml"
    p.write_text(FULL, encoding="utf-8")
    ann = parse_voc_xml(p)
    assert ann.width == 1228 and ann.height == 1027 and ann.depth == 3
    assert ann.folder == "mackerel"
    assert ann.filename == "ST4_539-20180506041404783.jpg"
    assert [b.label for b in ann.boxes] == ["mackerel", "krill"]
    assert ann.boxes[0].width == 439 and ann.boxes[0].height == 206
    assert "box_outside_image" in ann.anomalies
    assert "missing_size" not in ann.anomalies


def test_missing_size_and_degenerate(tmp_path: Path) -> None:
    p = tmp_path / "b.xml"
    p.write_text(NO_SIZE, encoding="utf-8")
    ann = parse_voc_xml(p)
    assert not ann.has_size
    assert "missing_size" in ann.anomalies
    assert "degenerate_box" in ann.anomalies
    assert "missing_bndbox" in ann.anomalies
    assert len(ann.boxes) == 1
    assert ann.boxes[0].label == "Herring"


def test_unparseable(tmp_path: Path) -> None:
    p = tmp_path / "c.xml"
    p.write_bytes(b"")
    ann = parse_voc_xml(p)
    assert not ann.boxes
    assert ann.anomalies and ann.anomalies[0].startswith(("unparseable", "empty_document"))
