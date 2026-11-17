from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BUILD_PATH = ROOT / "prep/easyvolcap/build-l3mhet-camera-dataset.py"
EXPORT_PATH = ROOT / "prep/easyvolcap/export-l3mhet-cameras-to-colmap.py"
UNDISTORT_PATH = ROOT / "prep/easyvolcap/build-l3mhet-undistorted-colmap.py"
RAW_DEPTH_PATH = ROOT / "prep/easyvolcap/export-l3mhet-raw-depth.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load("build_l3mhet_camera_dataset", BUILD_PATH)
EXPORT = _load("export_l3mhet_cameras", EXPORT_PATH)
UNDISTORT = _load("build_l3mhet_undistorted_colmap", UNDISTORT_PATH)
RAW_DEPTH = _load("export_l3mhet_raw_depth", RAW_DEPTH_PATH)


def test_raw_depth_export_preserves_float_values_and_restores_crop():
    from types import SimpleNamespace

    class Meta(SimpleNamespace):
        def __contains__(self, key):
            return hasattr(self, key)

    flat = np.arange(12, dtype=np.float32).reshape(1, 12, 1) / 7.0
    plane = RAW_DEPTH._image_plane(flat, 3, 4)
    meta = {
        "orig_h": np.asarray([5]),
        "orig_w": np.asarray([7]),
        "crop_x": np.asarray([2]),
        "crop_y": np.asarray([1]),
    }
    restored = RAW_DEPTH._restore_full_image(plane, Meta(**meta), fill_value=np.nan)

    assert restored.dtype == np.float32
    assert restored.shape == (5, 7)
    assert np.allclose(restored[1:4, 2:6], plane)
    assert np.isnan(restored[0, 0])
    K = np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 12.0], [0.0, 0.0, 1.0]])
    restored_K = RAW_DEPTH._restore_full_intrinsics(K, Meta(**meta))
    assert np.allclose(restored_K[:2, 2], [12.0, 13.0])


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "sparse"
    model.mkdir()
    (model / "cameras.txt").write_text(
        "1 OPENCV 8 6 5 5.5 4 3 0.01 -0.02 0.001 -0.002\n"
        "2 OPENCV 8 6 5 5.5 4 3 0.02 -0.01 0.002 -0.001\n"
    )
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 0000/000001.jpg\n\n2 1 0 0 0 1 0 0 2 0001/000001.jpg\n\n"
    )
    (model / "points3D.txt").write_text("1 -1 -2 3 255 0 0 0\n2 2 4 9 0 255 0 0\n")
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    for camera in ("0000", "0001"):
        (images / camera).mkdir(parents=True)
        (masks / camera).mkdir(parents=True)
        Image.new("RGB", (8, 6), (20, 40, 60)).save(images / camera / "000001.jpg")
        Image.new("RGB", (8, 6), (30, 50, 70)).save(images / camera / "000002.jpg")
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 1:6] = 255
        mask[2, 3] = 0  # internal false-negative hole
        mask[0, 7] = 255  # disconnected foreground; must not be removed
        Image.fromarray(mask).save(masks / camera / "000001.png")
    return model, images, masks


def test_builds_official_static_l3mhet_layout_and_fills_only_holes(
    tmp_path: Path,
) -> None:
    model, images, masks = _fixture(tmp_path)
    output = tmp_path / "easyvolcap"
    args = BUILD.parse_args(
        [
            "--reference-colmap",
            str(model),
            "--images-root",
            str(images),
            "--masks-root",
            str(masks),
            "--output-dir",
            str(output),
            "--frame",
            "1",
            "--fill-mask-holes",
            "--images-mode",
            "copy",
            "--repo-root",
            str(ROOT),
        ]
    )
    manifest = BUILD.build(args)

    assert manifest["views"] == 2
    assert manifest["fill_mask_holes"] is True
    assert (output / "images/0000/000000.jpg").is_file()
    assert (output / "intri.yml").is_file()
    assert (output / "extri.yml").is_file()
    config = (output / "l3mhet_camera_refine.yaml").read_text()
    assert "configs/models/l3mhet.yaml" in config
    assert "configs/specs/optcam.yaml" in config
    assert "configs/specs/mask.yaml" in config
    assert "use_aligned_cameras: false" in config

    with Image.open(output / "masks/0000/000000.jpg") as decoded:
        prepared = np.asarray(decoded.convert("L"))
    assert prepared[2, 3] == 255
    assert prepared[0, 7] == 255
    assert prepared[5, 0] == 0


def test_export_preserves_intrinsics_points_and_zero_residual_poses(
    tmp_path: Path,
) -> None:
    model, images, masks = _fixture(tmp_path)
    dataset = tmp_path / "easyvolcap"
    build_args = BUILD.parse_args(
        [
            "--reference-colmap",
            str(model),
            "--images-root",
            str(images),
            "--masks-root",
            str(masks),
            "--output-dir",
            str(dataset),
            "--frame",
            "1",
            "--images-mode",
            "copy",
            "--repo-root",
            str(ROOT),
        ]
    )
    BUILD.build(build_args)

    optimized = dataset / "optimized"
    optimized.mkdir()
    (optimized / "extri.yml").write_bytes((dataset / "extri.yml").read_bytes())
    output = tmp_path / "optimized-colmap"
    export_args = EXPORT.parse_args(
        [
            "--reference-colmap",
            str(model),
            "--optimized-dir",
            str(optimized),
            "--output-sparse",
            str(output),
            "--repo-root",
            str(ROOT),
        ]
    )
    report = EXPORT.export(export_args)

    assert report["cameras"] == 2
    assert report["rotation_change_deg"]["max"] < 1e-6
    assert report["center_change_world"]["max"] < 1e-10
    colmap_io = BUILD._load_colmap_io(ROOT)
    source_cameras, source_images, source_points = colmap_io.read_model(
        str(model), ".txt"
    )
    cameras, exported_images, points = colmap_io.read_model(str(output), ".bin")
    assert set(cameras) == set(source_cameras)
    assert set(points) == set(source_points)
    for image_id in source_images:
        assert np.allclose(exported_images[image_id].qvec, source_images[image_id].qvec)
        assert np.allclose(exported_images[image_id].tvec, source_images[image_id].tvec)


def test_builds_easyvolcap_equivalent_undistorted_pinhole_model(
    tmp_path: Path,
) -> None:
    cv2 = __import__("pytest").importorskip("cv2")
    model, source_images_root, masks = _fixture(tmp_path)
    dataset = tmp_path / "easyvolcap"
    BUILD.build(
        BUILD.parse_args(
            [
                "--reference-colmap",
                str(model),
                "--images-root",
                str(source_images_root),
                "--masks-root",
                str(masks),
                "--output-dir",
                str(dataset),
                "--frame",
                "1",
                "--images-mode",
                "copy",
                "--repo-root",
                str(ROOT),
            ]
        )
    )
    optimized = dataset / "optimized"
    optimized.mkdir()
    (optimized / "intri.yml").write_bytes((dataset / "intri.yml").read_bytes())
    (optimized / "extri.yml").write_bytes((dataset / "extri.yml").read_bytes())

    output = tmp_path / "undistorted"
    report = UNDISTORT.build(
        UNDISTORT.parse_args(
            [
                "--reference-colmap",
                str(model),
                "--optimized-dir",
                str(optimized),
                "--images-root",
                str(source_images_root),
                "--output-dir",
                str(output),
                "--frame",
                "1",
                "--repo-root",
                str(ROOT),
            ]
        )
    )

    assert report["views"] == 2
    assert report["feature_observations"] == 0
    assert report["points3D"] == 0
    assert (output / "images/0000/000001.png").is_file()
    colmap_io = BUILD._load_colmap_io(ROOT)
    cameras, images, points = colmap_io.read_model(str(output / "sparse/0"), ".bin")
    assert not points
    assert all(camera.model == "PINHOLE" for camera in cameras.values())
    assert all(len(image.xys) == 0 for image in images.values())
    assert {image.name for image in images.values()} == {
        "0000/000001.png",
        "0001/000001.png",
    }

    K = np.array([[5.0, 0.0, 4.0], [0.0, 5.5, 3.0], [0.0, 0.0, 1.0]])
    D = np.array([0.01, -0.02, 0.001, -0.002])
    expected_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (8, 6), 0, (8, 6))
    camera = cameras[images[1].camera_id]
    assert np.allclose(
        camera.params,
        [expected_K[0, 0], expected_K[1, 1], expected_K[0, 2], expected_K[1, 2]],
    )


def test_undistortion_can_prepare_adjacent_unregistered_frames(
    tmp_path: Path,
) -> None:
    __import__("pytest").importorskip("cv2")
    model, source_images_root, masks = _fixture(tmp_path)
    dataset = tmp_path / "easyvolcap"
    BUILD.build(
        BUILD.parse_args(
            [
                "--reference-colmap",
                str(model),
                "--images-root",
                str(source_images_root),
                "--masks-root",
                str(masks),
                "--output-dir",
                str(dataset),
                "--frame",
                "1",
                "--images-mode",
                "copy",
                "--repo-root",
                str(ROOT),
            ]
        )
    )
    optimized = dataset / "optimized"
    optimized.mkdir()
    (optimized / "intri.yml").write_bytes((dataset / "intri.yml").read_bytes())
    (optimized / "extri.yml").write_bytes((dataset / "extri.yml").read_bytes())

    output = tmp_path / "undistorted-sequence"
    report = UNDISTORT.build(
        UNDISTORT.parse_args(
            [
                "--reference-colmap",
                str(model),
                "--optimized-dir",
                str(optimized),
                "--images-root",
                str(source_images_root),
                "--output-dir",
                str(output),
                "--frame",
                "1",
                "--frame-end",
                "2",
                "--repo-root",
                str(ROOT),
            ]
        )
    )

    assert report["prepared_frames"] == [1, 2]
    assert (output / "images/0000/000001.png").is_file()
    assert (output / "images/0000/000002.png").is_file()
    colmap_io = BUILD._load_colmap_io(ROOT)
    _, images, _ = colmap_io.read_model(str(output / "sparse/0"), ".bin")
    assert {image.name for image in images.values()} == {
        "0000/000001.png",
        "0001/000001.png",
    }
