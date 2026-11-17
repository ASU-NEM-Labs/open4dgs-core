#!/usr/bin/env python3
"""Build the distortion-free COLMAP dataset seen by an EasyVolcap run.

EasyVolcap's default ``dist_opt_K=True`` image loader calls
``cv2.getOptimalNewCameraMatrix(..., alpha=0)`` and ``cv2.undistort`` before
training.  Its optimized ``intri.yml`` still stores the pre-undistortion K and
distortion coefficients.  This converter applies that same operation to the
source images, writes lossless PNGs, and emits matching PINHOLE cameras with
the optimized world-to-camera poses from ``extri.yml``.

The output intentionally contains no feature observations or sparse points:
their pixel coordinates belong to the distorted images.  ``--frame-end`` can
prepare a short contiguous image sequence while registering only ``--frame``
in COLMAP.  This is useful when a static reconstruction still needs adjacent
frames for per-camera synchronization compensation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file

from typing import Any

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_colmap_io(repo_root: Path):
    module_path = trigs_file("io/colmap.py", repo_root)
    spec = importlib.util.spec_from_file_location("trigs_colmap_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_extension(path: Path) -> str:
    for extension in (".bin", ".txt"):
        if all(
            (path / f"{stem}{extension}").is_file()
            for stem in ("cameras", "images", "points3D")
        ):
            return extension
    raise FileNotFoundError(f"No complete COLMAP model found under {path}")


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe or camera-less COLMAP image name: {name!r}")
    return path


def _read_names(storage: Any) -> list[str]:
    node = storage.getNode("names")
    if node.empty() or not node.isSeq():
        raise ValueError("EasyVolcap camera file has no names sequence")
    names: list[str] = []
    for index in range(node.size()):
        item = node.at(index)
        value = item.string()
        if not value:
            value = str(int(item.real()))
        names.append(value)
    return names


def _required_matrix(storage: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = storage.getNode(key).mat()
    if value is None:
        raise ValueError(f"EasyVolcap camera file is missing {key}")
    array = np.asarray(value, dtype=np.float64)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
    return array.reshape(shape)


def _optional_matrix(storage: Any, key: str) -> np.ndarray:
    value = storage.getNode(key).mat()
    if value is None:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _required_integer(storage: Any, key: str) -> int:
    node = storage.getNode(key)
    if node.empty():
        raise ValueError(f"EasyVolcap camera file is missing {key}")
    value = node.real()
    if not np.isfinite(value) or value <= 0 or float(value) != float(int(value)):
        raise ValueError(f"Invalid integer value for {key}: {value}")
    return int(value)


def _read_intrinsics(path: Path) -> dict[str, dict[str, Any]]:
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Cannot open EasyVolcap intrinsics: {path}")
    try:
        output: dict[str, dict[str, Any]] = {}
        for name in _read_names(storage):
            output[name] = {
                "K": _required_matrix(storage, f"K_{name}", (3, 3)),
                "D": _optional_matrix(storage, f"D_{name}"),
                "height": _required_integer(storage, f"H_{name}"),
                "width": _required_integer(storage, f"W_{name}"),
            }
        return output
    finally:
        storage.release()


def _read_extrinsics(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Cannot open EasyVolcap extrinsics: {path}")
    try:
        output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in _read_names(storage):
            rotation = storage.getNode(f"Rot_{name}").mat()
            if rotation is None:
                rotation_vector = storage.getNode(f"R_{name}").mat()
                if rotation_vector is None:
                    raise ValueError(f"No rotation stored for camera {name}")
                rotation = cv2.Rodrigues(rotation_vector)[0]
            translation = _required_matrix(storage, f"T_{name}", (3,))
            output[name] = (
                np.asarray(rotation, dtype=np.float64).reshape(3, 3),
                translation,
            )
        return output
    finally:
        storage.release()


def _find_source_image(
    images_root: Path, camera_name: str, frame: int, digits: int
) -> Path:
    directory = images_root / camera_name
    stem = f"{frame:0{digits}d}"
    found = sorted(
        path
        for path in directory.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source image for camera {camera_name}, frame "
            f"{frame} under {directory}; found {[str(path) for path in found]}"
        )
    return found[0]


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    reference = args.reference_colmap.resolve()
    optimized = args.optimized_dir.resolve()
    images_root = args.images_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    colmap_io = _load_colmap_io(args.repo_root.resolve())
    extension = _model_extension(reference)
    _, source_images, _ = colmap_io.read_model(str(reference), extension)
    intrinsics = _read_intrinsics(optimized / "intri.yml")
    extrinsics = _read_extrinsics(optimized / "extri.yml")

    camera_to_image: dict[str, Any] = {}
    for image in source_images.values():
        camera_name = _safe_name(image.name).parts[0]
        if camera_name in camera_to_image:
            raise ValueError(
                f"Reference model contains multiple images for camera {camera_name}"
            )
        camera_to_image[camera_name] = image
    expected = set(camera_to_image)
    if expected != set(intrinsics) or expected != set(extrinsics):
        raise ValueError(
            "Camera-name mismatch: "
            f"missing intrinsics={sorted(expected - set(intrinsics))}, "
            f"extra intrinsics={sorted(set(intrinsics) - expected)}, "
            f"missing extrinsics={sorted(expected - set(extrinsics))}, "
            f"extra extrinsics={sorted(set(extrinsics) - expected)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=str(output.parent))
    )
    rows: list[dict[str, Any]] = []
    focal_changes: list[float] = []
    try:
        sparse = temporary / "sparse/0"
        sparse.mkdir(parents=True)
        cameras: dict[int, Any] = {}
        images: dict[int, Any] = {}

        frame_end = int(args.frame if args.frame_end is None else args.frame_end)
        if frame_end < int(args.frame):
            raise ValueError("--frame-end must be greater than or equal to --frame")
        prepared_frames = list(range(int(args.frame), frame_end + 1))

        for camera_name, source_record in sorted(camera_to_image.items()):
            camera = intrinsics[camera_name]
            width = int(camera["width"])
            height = int(camera["height"])
            K = np.asarray(camera["K"], dtype=np.float64)
            D = np.asarray(camera["D"], dtype=np.float64)
            has_distortion = bool(np.sum(np.abs(D)) != 0.0)
            if has_distortion:
                new_K, roi = cv2.getOptimalNewCameraMatrix(
                    K, D, (width, height), 0, (width, height)
                )
            else:
                new_K = K.copy()
                roi = (0, 0, width, height)

            source_paths: list[str] = []
            output_names: list[str] = []
            for frame in prepared_frames:
                source_path = _find_source_image(
                    images_root, camera_name, frame, args.frame_digits
                )
                decoded = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"OpenCV could not decode {source_path}")
                if decoded.shape[:2] != (height, width):
                    raise ValueError(
                        f"Image size mismatch for {camera_name}, frame {frame}: "
                        f"decoded {decoded.shape[1]}x{decoded.shape[0]}, "
                        f"EasyVolcap {width}x{height}"
                    )
                prepared = (
                    cv2.undistort(decoded, K, D, newCameraMatrix=new_K)
                    if has_distortion
                    else decoded
                )
                output_name = PurePosixPath(camera_name) / (
                    f"{frame:0{args.frame_digits}d}.png"
                )
                output_path = temporary / "images" / Path(*output_name.parts)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                success, encoded = cv2.imencode(
                    ".png",
                    prepared,
                    [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
                )
                if not success:
                    raise RuntimeError(f"OpenCV could not encode {output_path}")
                output_path.write_bytes(encoded.tobytes())
                source_paths.append(str(source_path))
                output_names.append(output_name.as_posix())

            registered_output_name = PurePosixPath(camera_name) / (
                f"{args.frame:0{args.frame_digits}d}.png"
            )

            # Use a camera per registered image.  This remains correct even if
            # the reference model shared an intrinsics record between images.
            camera_id = int(source_record.id)
            cameras[camera_id] = colmap_io.Camera(
                id=camera_id,
                model="PINHOLE",
                width=width,
                height=height,
                params=np.array(
                    [new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]],
                    dtype=np.float64,
                ),
            )
            rotation, translation = extrinsics[camera_name]
            images[int(source_record.id)] = colmap_io.Image(
                id=int(source_record.id),
                qvec=colmap_io.rotmat2qvec(rotation),
                tvec=translation,
                camera_id=camera_id,
                name=registered_output_name.as_posix(),
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
            focal_change = float(
                np.mean([new_K[0, 0] / K[0, 0], new_K[1, 1] / K[1, 1]])
            )
            focal_changes.append(focal_change)
            rows.append(
                {
                    "camera": camera_name,
                    "source_images": source_paths,
                    "output_images": output_names,
                    "width": width,
                    "height": height,
                    "had_distortion": has_distortion,
                    "source_K": K.tolist(),
                    "pinhole_K": np.asarray(new_K).tolist(),
                    "optimal_roi_not_applied": [int(value) for value in roi],
                    "mean_focal_scale": focal_change,
                }
            )

        # Stale points and observations are invalid after nonlinear pixel
        # remapping.  The downstream Brush builder supplies init.ply from NPZ.
        colmap_io.write_model(cameras, images, {}, str(sparse), ".bin")
        report = {
            "format": "easyvolcap-undistorted-pinhole-colmap-v1",
            "method": "cv2.getOptimalNewCameraMatrix(alpha=0) + cv2.undistort",
            "reference_colmap": str(reference),
            "optimized_dir": str(optimized),
            "images_root": str(images_root),
            "output_dir": str(output),
            "frame": int(args.frame),
            "frame_end": frame_end,
            "prepared_frames": prepared_frames,
            "views": len(rows),
            "image_encoding": "lossless PNG",
            "feature_observations": 0,
            "points3D": 0,
            "mean_focal_scale": _summary(focal_changes),
            "per_camera": rows,
        }
        (temporary / "easyvolcap-undistorted-colmap.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        temporary.replace(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-colmap", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Also prepare every frame through this inclusive endpoint.",
    )
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--png-compression", type=int, choices=range(10), default=6)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args(argv)


def main() -> None:
    report = build(parse_args())
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_camera"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
