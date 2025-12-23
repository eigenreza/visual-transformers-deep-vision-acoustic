from __future__ import annotations

from pathlib import Path

from echolabel.config import config_hash
from echolabel.manifest import manifest_is_current, read_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    inp = tmp_path / "input.txt"
    inp.write_text("data")
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.csv").write_text("a,b\n1,2\n")
    cfg = {"stage": "x", "value": 1}
    write_manifest(out, "x", cfg, {"inp": inp}, [out / "result.csv"])
    m = read_manifest(out)
    assert m is not None and m["config_hash"] == config_hash(cfg)
    assert "result.csv" in m["outputs"]
    assert manifest_is_current(out, cfg, {"inp": inp})
    assert not manifest_is_current(out, {"stage": "x", "value": 2}, {"inp": inp})
    inp.write_text("changed data with different length")
    assert not manifest_is_current(out, cfg, {"inp": inp})


def test_manifest_is_stale_after_a_source_change(tmp_path: Path, monkeypatch) -> None:
    import echolabel.manifest as manifest

    out = tmp_path / "out"
    out.mkdir()
    (out / "result.csv").write_text("a\n1\n")
    cfg = {"stage": "x"}
    write_manifest(out, "x", cfg, {}, [out / "result.csv"])
    recorded = read_manifest(out)["source_hash"]
    assert len(recorded) == 64 and recorded == manifest.source_hash()
    assert manifest_is_current(out, cfg, {})
    # Any change to the source of the package changes the hash and invalidates the manifest.
    monkeypatch.setattr(manifest, "source_hash", lambda: "0" * 64)
    assert not manifest.manifest_is_current(out, cfg, {})


def test_source_hash_follows_the_source(tmp_path: Path, monkeypatch) -> None:
    import echolabel.manifest as manifest

    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    (package / "sub" / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(manifest, "__file__", str(package / "manifest.py"))
    manifest.source_hash.cache_clear()
    first = manifest.source_hash()
    manifest.source_hash.cache_clear()
    assert manifest.source_hash() == first
    (package / "sub" / "b.py").write_text("y = 3\n")
    manifest.source_hash.cache_clear()
    assert manifest.source_hash() != first
    manifest.source_hash.cache_clear()


def test_verify_outputs_reports_changed_and_missing_files(tmp_path: Path) -> None:
    from echolabel.manifest import verify_outputs

    out = tmp_path / "out"
    out.mkdir()
    for name in ("a.csv", "b.csv", "c.csv"):
        (out / name).write_text(name)
    write_manifest(out, "x", {"stage": "x"}, {}, [out / "a.csv", out / "b.csv", out / "c.csv"])
    assert set(verify_outputs(out).values()) == {"ok"}
    (out / "b.csv").write_text("edited")
    (out / "c.csv").unlink()
    assert verify_outputs(out) == {"a.csv": "ok", "b.csv": "changed", "c.csv": "missing"}


def test_verify_outputs_ignores_line_endings_of_text_files(tmp_path: Path) -> None:
    from echolabel.manifest import verify_outputs

    out = tmp_path / "out"
    out.mkdir()
    (out / "table.csv").write_bytes(b"a,b\r\n1,2\r\n")
    (out / "blob.bin").write_bytes(b"\x00\x01\r\n\x02")
    write_manifest(out, "x", {"stage": "x"}, {}, [out / "table.csv", out / "blob.bin"])
    (out / "table.csv").write_bytes(b"a,b\n1,2\n")
    (out / "blob.bin").write_bytes(b"\x00\x01\n\x02")
    assert verify_outputs(out) == {"table.csv": "ok", "blob.bin": "changed"}
    (out / "table.csv").write_bytes(b"a,b\n1,3\n")
    assert verify_outputs(out)["table.csv"] == "changed"


def test_verify_outputs_compares_copies_with_placeholders_through_the_roots(tmp_path: Path) -> None:
    from echolabel.manifest import replace_roots, verify_outputs

    roots = {"project_dir": "/srv/project", "scratch_dir": "/srv/project_scratch", "source_data": "/srv/data"}
    out = tmp_path / "out"
    out.mkdir()
    original = '{"weights": "/srv/project_scratch/w.pt", "table": "/srv/project/runs/t.csv", "n": 3}\n'
    (out / "summary.json").write_text(original, newline="\n")
    (out / "plain.csv").write_text("a,b\n1,2\n", newline="\n")
    write_manifest(out, "x", {"stage": "x"}, {}, [out / "summary.json", out / "plain.csv"])
    redacted = replace_roots(original, roots)
    assert redacted == '{"weights": "<scratch_dir>/w.pt", "table": "<project_dir>/runs/t.csv", "n": 3}\n'
    (out / "summary.json").write_text(redacted, newline="\n")
    assert verify_outputs(out) == {"summary.json": "placeholder", "plain.csv": "ok"}
    assert verify_outputs(out, roots) == {"summary.json": "ok", "plain.csv": "ok"}
    # Other roots than those of the machine that wrote the file do not restore it.
    assert verify_outputs(out, {**roots, "project_dir": "/elsewhere"})["summary.json"] == "changed"
    (out / "summary.json").write_text(redacted.replace("3", "4"), newline="\n")
    assert verify_outputs(out, roots)["summary.json"] == "changed"
