from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "build-brush-static-dataset.py"
SPEC = importlib.util.spec_from_file_location("build_brush_static_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selects_one_dynamic_frame_plus_global_static(tmp_path: Path) -> None:
    path = tmp_path / "points.npz"
    np.savez(
        path,
        positions=np.arange(18, dtype=np.float32).reshape(6, 3),
        colors=np.ones((6, 3), dtype=np.float32),
        source_frames=np.array([1, 1, 4, 4, 7, 7]),
        is_static=np.array([0, 1, 0, 1, 0, 1]),
    )
    with np.load(path, allow_pickle=False) as npz:
        keep, details = MODULE._npz_rows_for_static_frame(npz, 4, allow_nearest=False)

    assert keep.tolist() == [False, True, True, True, False, True]
    assert details["selected_dynamic_rows"] == 1
    assert details["selected_static_rows"] == 3
    assert details["selected_instant"] == 4


def test_missing_keyframe_is_an_error_unless_nearest_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "points.npz"
    np.savez(
        path,
        positions=np.ones((2, 3), dtype=np.float32),
        colors=np.ones((2, 3), dtype=np.float32),
        source_frames=np.array([1, 4]),
    )
    with np.load(path, allow_pickle=False) as npz:
        with pytest.raises(ValueError, match="no dynamic rows for source frame 2"):
            MODULE._npz_rows_for_static_frame(npz, 2, allow_nearest=False)
    with np.load(path, allow_pickle=False) as npz:
        keep, details = MODULE._npz_rows_for_static_frame(npz, 2, allow_nearest=True)

    assert keep.tolist() == [True, False]
    assert details["selected_instant"] == 1


def test_binary_ply_contains_brush_rgb_fields(tmp_path: Path) -> None:
    path = tmp_path / "init.ply"
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32)
    MODULE._write_brush_init_ply(path, positions, colors)

    payload = path.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"element vertex 2" in header
    assert b"property float red" in header
    assert len(body) == 2 * 6 * 4


def test_rejects_unsafe_colmap_names() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE._safe_colmap_name("../outside.jpg")


def test_builds_complete_nested_colmap_dataset(tmp_path: Path) -> None:
    data = tmp_path / "capture"
    model = data / "sparse/reference"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 PINHOLE 8 6 5 5 4 3\n")
    (model / "images.txt").write_text("1 1 0 0 0 0 0 1 1 0001/000000.jpg\n\n")
    (model / "points3D.txt").write_text("1 0 0 2 255 0 0 0\n")

    image_dir = data / "images/0001"
    mask_dir = data / "masks/0001"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), (20, 40, 60)).save(image_dir / "000001.jpg")
    Image.new("L", (8, 6), 255).save(mask_dir / "000001.png")

    output = tmp_path / "brush-static"
    args = MODULE.parse_args(
        [
            "--data-dir", str(data),
            "--reference-colmap", str(model),
            "--output-dir", str(output),
            "--frame", "1",
        ]
    )
    manifest = MODULE.build_dataset(args)

    assert (output / "sparse/0/cameras.txt").is_file()
    assert (output / "images/0001/000000.jpg").is_file()
    assert (output / "masks/0001/000000.png").is_file()
    assert manifest["registered_views"] == 1
    assert manifest["masks"]["copied"] == 1
    assert manifest["initialization"]["kind"] == "colmap_points3D"


def test_transparent_mode_embeds_soft_mask_alpha(tmp_path: Path) -> None:
    data = tmp_path / "capture"
    model = data / "sparse/reference"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 PINHOLE 8 6 5 5 4 3\n")
    (model / "images.txt").write_text("1 1 0 0 0 0 0 1 1 0001/000000.jpg\n\n")
    (model / "points3D.txt").write_text("1 0 0 2 255 0 0 0\n")
    (data / "images/0001").mkdir(parents=True)
    (data / "masks/0001").mkdir(parents=True)
    Image.new("RGB", (8, 6), (20, 40, 60)).save(data / "images/0001/000001.jpg")
    mask = Image.new("L", (8, 6), 0)
    mask.putpixel((3, 2), 127)
    mask.save(data / "masks/0001/000001.png")

    output = tmp_path / "brush-alpha"
    args = MODULE.parse_args(
        [
            "--data-dir", str(data),
            "--reference-colmap", str(model),
            "--output-dir", str(output),
            "--frame", "1",
            "--mask-mode", "transparent",
        ]
    )
    manifest = MODULE.build_dataset(args)

    with Image.open(output / "images/0001/000000.jpg") as rgba:
        assert rgba.format == "PNG"
        assert rgba.mode == "RGBA"
        assert rgba.getpixel((3, 2))[3] == 127
        assert rgba.getpixel((0, 0))[3] == 0
    assert not (output / "masks").exists()
    assert manifest["masks"]["mode"] == "transparent"
    assert manifest["masks"]["embedded_alpha"] == 1
