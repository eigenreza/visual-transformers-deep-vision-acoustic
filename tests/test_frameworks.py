from __future__ import annotations

from pathlib import Path

import yaml

from echolabel.detection.frameworks import (
    build_framework_config,
    multiscale_sizes,
    train_command,
    train_transforms,
    write_framework_config,
)


def test_multiscale_sizes() -> None:
    s640 = multiscale_sizes(640)
    assert s640 == [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    s1024 = multiscale_sizes(1024)
    assert min(s1024) == 768 and max(s1024) == 1280 and s1024.count(1024) == 3


def test_transforms_follow_augmentation_flags() -> None:
    ops, strong = train_transforms(
        800, {"photometric_p": 0.5, "iou_crop_p": 0.0, "zoom_out": False, "hflip": True}
    )
    types = [o["type"] for o in ops]
    assert "RandomIoUCrop" not in types and "RandomZoomOut" not in types
    assert strong == ["RandomPhotometricDistort"]
    assert {"type": "Resize", "size": [800, 800]} in ops
    assert types[-1] == "ConvertBoxes"


def test_config_and_command(tmp_path: Path) -> None:
    cfg = build_framework_config(
        framework="dfine",
        base_config=Path("/repo/configs/dfine/custom/dfine_hgnetv2_l_custom.yml"),
        output_dir=tmp_path / "out",
        images_root=Path("/data/images"),
        train_json=Path("/data/train.json"),
        val_json=Path("/data/val.json"),
        num_classes=15,
        input_size=1024,
        batch_size=8,
        epochs=60,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        warmup_iterations=500,
        checkpoint_freq=10,
        num_workers=4,
        aug={"multiscale": True},
    )
    assert cfg["num_classes"] == 15 and cfg["remap_mscoco_category"] is False
    assert cfg["eval_spatial_size"] == [1024, 1024]
    assert cfg["epochs"] == 60 and "epoches" not in cfg
    assert cfg["train_dataloader"]["collate_fn"]["base_size"] == 1024
    assert cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 52
    assert cfg["HGNetv2"] == {"pretrained": False}
    path = write_framework_config(cfg, tmp_path / "cfg.yml")
    loaded = yaml.safe_load(path.read_text())
    assert loaded["__include__"] == ["/repo/configs/dfine/custom/dfine_hgnetv2_l_custom.yml"]
    cmd = train_command("dfine", path, tmp_path / "out", 3, pretrained=Path("/w/p.pth"))
    assert cmd[1] == "train.py" and "-t" in cmd and "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "3"
    cmd_r = train_command(
        "rtdetrv2", path, tmp_path / "out", 0, pretrained=Path("/w/p.pth"), resume=Path("/o/last.pth")
    )
    assert cmd_r[1] == "tools/train.py" and "-r" in cmd_r and "-t" not in cmd_r
    rt = build_framework_config(
        framework="rtdetrv2",
        base_config=Path("/repo/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"),
        output_dir=tmp_path / "out2",
        images_root=Path("/data/images"),
        train_json=Path("/data/train.json"),
        val_json=Path("/data/val.json"),
        num_classes=15,
        input_size=640,
        batch_size=16,
        epochs=60,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        warmup_iterations=500,
        checkpoint_freq=10,
        num_workers=4,
        aug={"multiscale": True},
    )
    assert rt["epoches"] == 60 and rt["train_dataloader"]["collate_fn"]["scales"] == multiscale_sizes(640)
    assert rt["PResNet"] == {"pretrained": False}
