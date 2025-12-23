"""Stage manifests.

A manifest records what a stage consumed and produced: the configuration and
its hash, a description of every input (size and modification time for files,
modification time for directories), the package version, a SHA-256 hash of the
package source and a SHA-256 hash of every output file. A stage whose manifest
matches the current configuration, inputs and source can be skipped; a change
to any source file of the package invalidates every manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import config_hash

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def package_version() -> str:
    try:
        return version("echolabel")
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def source_hash() -> str:
    """SHA-256 hex digest over the Python source files of the package, in path order.

    Paths enter the digest relative to the package with forward slashes and
    line endings are normalised, so that the digest is the same on every
    machine that holds the same source.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def describe_input(path: str | Path) -> dict[str, Any]:
    """Cheap description of an input path used to detect changes between runs."""
    p = Path(path)
    if p.is_file():
        st = p.stat()
        return {"path": str(p), "kind": "file", "size": st.st_size, "mtime": st.st_mtime}
    if p.is_dir():
        st = p.stat()
        return {"path": str(p), "kind": "dir", "mtime": st.st_mtime}
    return {"path": str(p), "kind": "missing"}


def write_manifest(
    out_dir: str | Path,
    stage: str,
    cfg: dict[str, Any],
    inputs: dict[str, str | Path],
    outputs: list[str | Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": package_version(),
        "source_hash": source_hash(),
        "hostname": platform.node(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "inputs": {name: describe_input(p) for name, p in inputs.items()},
        "outputs": {},
    }
    for out in outputs:
        p = Path(out)
        if p.is_file():
            rel = os.path.relpath(p, out_dir)
            manifest["outputs"][rel] = {"sha256": file_sha256(p), "size": p.stat().st_size}
    if extra:
        manifest["extra"] = extra
    target = out_dir / MANIFEST_NAME
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info("wrote manifest %s", target)
    return target


def read_manifest(out_dir: str | Path) -> dict[str, Any] | None:
    target = Path(out_dir) / MANIFEST_NAME
    if not target.is_file():
        return None
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_is_current(out_dir: str | Path, cfg: dict[str, Any], inputs: dict[str, str | Path]) -> bool:
    """True when a manifest exists with the same configuration hash, source hash and inputs."""
    manifest = read_manifest(out_dir)
    if manifest is None:
        return False
    if manifest.get("source_hash") != source_hash():
        return False
    if manifest.get("config_hash") != config_hash(cfg):
        return False
    recorded = manifest.get("inputs", {})
    for name, p in inputs.items():
        if recorded.get(name) != describe_input(p):
            return False
    for rel in manifest.get("outputs", {}):
        if not (Path(out_dir) / rel).is_file():
            return False
    return True


def verify_outputs(out_dir: str | Path, roots: dict[str, str] | None = None) -> dict[str, str]:
    """Compare the output files of a stage directory with the hashes recorded in its manifest.

    Returns one status per recorded output: ``ok``, ``missing``, ``changed``
    or ``placeholder``. A text file whose bytes differ from the recorded ones
    in their line endings alone counts as ``ok``, since a version control
    system may store it with other line endings than the stage wrote. A copy
    in which machine-specific roots are replaced by placeholders such as
    ``<project_dir>`` is compared after the placeholders are replaced by the
    entries of ``roots`` (the paths file of the machine that wrote it);
    without ``roots`` it cannot be compared and its status is
    ``placeholder``. Raises FileNotFoundError when the directory has no
    manifest.
    """
    manifest = read_manifest(out_dir)
    if manifest is None:
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {out_dir}")
    status: dict[str, str] = {}
    for rel, record in manifest.get("outputs", {}).items():
        path = Path(out_dir) / rel
        if not path.is_file():
            status[rel] = "missing"
        else:
            recorded = record.get("sha256")
            same = file_sha256(path) == recorded or _matches_up_to_line_endings(path, recorded)
            if same:
                status[rel] = "ok"
            elif _has_placeholder(path):
                restored = roots is not None and _matches_up_to_line_endings(path, recorded, roots=roots)
                status[rel] = "ok" if restored else ("changed" if roots is not None else "placeholder")
            else:
                status[rel] = "changed"
    return status


PLACEHOLDER_KEYS = ("source_data", "annotated_data", "annotated_data_source", "project_dir", "scratch_dir")


def replace_roots(text: str, roots: dict[str, str]) -> str:
    """Replace the machine-specific roots of a paths file in a text by placeholders such as ``<project_dir>``.

    Longer roots are replaced first, so that a root that contains another one is not cut.
    """
    for key, root in sorted(roots.items(), key=lambda item: -len(str(item[1] or ""))):
        if key in PLACEHOLDER_KEYS and root:
            text = text.replace(str(root), f"<{key}>")
    return text


def _has_placeholder(path: Path, max_bytes: int = 50_000_000) -> bool:
    if path.stat().st_size > max_bytes:
        return False
    data = path.read_bytes()
    return b"\0" not in data and any(f"<{key}>".encode() in data for key in PLACEHOLDER_KEYS)


def _matches_up_to_line_endings(
    path: Path, recorded: str | None, max_bytes: int = 50_000_000, roots: dict[str, str] | None = None
) -> bool:
    """Whether the file has the recorded hash once its line endings are written as LF or as CRLF.

    With ``roots`` the placeholders of the file are first replaced by the roots that they stand for.
    """
    if recorded is None or path.stat().st_size > max_bytes:
        return False
    data = path.read_bytes()
    if b"\0" in data:
        return False
    for key, root in (roots or {}).items():
        if key in PLACEHOLDER_KEYS and root:
            data = data.replace(f"<{key}>".encode(), str(root).encode())
    unix = data.replace(b"\r\n", b"\n")
    return any(
        hashlib.sha256(variant).hexdigest() == recorded for variant in (unix, unix.replace(b"\n", b"\r\n"))
    )
