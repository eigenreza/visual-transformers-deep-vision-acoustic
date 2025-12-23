"""Parsing of Pascal VOC annotation files.

The parser is tolerant. Missing or malformed fields are recorded as anomaly
codes on the returned object instead of raising, so that a whole collection can
be inventoried in one pass. Box coordinates are kept in pixels as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

BOX_TAGS = ("xmin", "ymin", "xmax", "ymax")


@dataclass
class VocBox:
    """One bounding box in pixel coordinates."""

    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: int = 0
    truncated: int = 0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class VocAnnotation:
    """Content of one VOC XML file."""

    path: Path
    folder: str | None
    filename: str | None
    width: int | None
    height: int | None
    depth: int | None
    boxes: list[VocBox] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    @property
    def has_size(self) -> bool:
        return self.width is not None and self.height is not None


def _text(node: etree._Element | None, tag: str) -> str | None:
    if node is None:
        return None
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _to_int(value: str | None, anomalies: list[str], what: str) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        anomalies.append(f"non_numeric_{what}")
        return None


def _to_float(value: str | None, anomalies: list[str], what: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        anomalies.append(f"non_numeric_{what}")
        return None


def parse_voc_xml(path: str | Path, outside_tolerance_px: float = 1.0) -> VocAnnotation:
    """Parse one VOC file.

    Boxes that extend more than ``outside_tolerance_px`` beyond the recorded
    image size are flagged as ``box_outside_image``. A tolerance of one pixel
    accepts the common one-based convention for the far corner.
    """
    path = Path(path)
    anomalies: list[str] = []
    parser = etree.XMLParser(recover=True, remove_blank_text=True)
    try:
        root = etree.parse(str(path), parser).getroot()
    except (etree.XMLSyntaxError, OSError) as exc:
        return VocAnnotation(path, None, None, None, None, None, [], [f"unparseable:{type(exc).__name__}"])
    if root is None:
        return VocAnnotation(path, None, None, None, None, None, [], ["empty_document"])
    if root.tag != "annotation":
        anomalies.append(f"root_tag_{root.tag}")

    size = root.find("size")
    width = _to_int(_text(size, "width"), anomalies, "width")
    height = _to_int(_text(size, "height"), anomalies, "height")
    depth = _to_int(_text(size, "depth"), anomalies, "depth")
    if width is None or height is None:
        anomalies.append("missing_size")
        width = height = None
    elif width <= 0 or height <= 0:
        anomalies.append("nonpositive_size")
        width = height = None

    boxes: list[VocBox] = []
    for obj in root.findall("object"):
        label = _text(obj, "name")
        if label is None:
            anomalies.append("missing_label")
            label = ""
        bnd = obj.find("bndbox")
        if bnd is None:
            anomalies.append("missing_bndbox")
            continue
        coords = [_to_float(_text(bnd, tag), anomalies, tag) for tag in BOX_TAGS]
        if any(c is None for c in coords):
            anomalies.append("incomplete_bndbox")
            continue
        xmin, ymin, xmax, ymax = (float(c) for c in coords)  # type: ignore[arg-type]
        if xmax <= xmin or ymax <= ymin:
            anomalies.append("degenerate_box")
        if width is not None and height is not None:
            tol = outside_tolerance_px
            if xmin < -tol or ymin < -tol or xmax > width + tol or ymax > height + tol:
                anomalies.append("box_outside_image")
        difficult = _to_int(_text(obj, "difficult"), anomalies, "difficult") or 0
        truncated = _to_int(_text(obj, "truncated"), anomalies, "truncated") or 0
        boxes.append(VocBox(label, xmin, ymin, xmax, ymax, difficult, truncated))
    if not boxes:
        anomalies.append("no_boxes")

    return VocAnnotation(
        path=path,
        folder=_text(root, "folder"),
        filename=_text(root, "filename"),
        width=width,
        height=height,
        depth=depth,
        boxes=boxes,
        anomalies=anomalies,
    )
