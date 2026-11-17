#!/usr/bin/env python3
"""Export EasyVolcap L3MHET optimized extrinsics to a COLMAP model.

The reference COLMAP intrinsics, image names, and points are preserved exactly.
Only the world-to-camera rotation and translation are replaced from the
``optimized/extri.yml`` produced by 4K4D/EasyVolcap.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file

from typing import Any

import numpy as np


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


def _read_name_list(storage: Any, key: str = "names") -> list[str]:
    node = storage.getNode(key)
    names = []
    for index in range(node.size()):
        value = node.at(index).string()
        if not value:
            value = str(int(node.at(index).real()))
        names.append(value)
    return names


def _read_optimized_extrinsics(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Cannot open EasyVolcap extrinsics: {path}")
    try:
        names = _read_name_list(storage)
        output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            rotation = storage.getNode(f"Rot_{name}").mat()
            if rotation is None:
                rotation_vector = storage.getNode(f"R_{name}").mat()
                if rotation_vector is None:
                    raise ValueError(f"No rotation stored for camera {name}")
                rotation = cv2.Rodrigues(rotation_vector)[0]
            translation = storage.getNode(f"T_{name}").mat()
            if translation is None:
                raise ValueError(f"No translation stored for camera {name}")
            output[name] = (
                np.asarray(rotation, dtype=np.float64).reshape(3, 3),
                np.asarray(translation, dtype=np.float64).reshape(3),
            )
        return output
    finally:
        storage.release()


def _rotation_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    reference = args.reference_colmap.resolve()
    optimized = args.optimized_dir.resolve()
    output = args.output_sparse.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    colmap_io = _load_colmap_io(args.repo_root.resolve())
    extension = _model_extension(reference)
    cameras, images, points = colmap_io.read_model(str(reference), extension)
    optimized_extrinsics = _read_optimized_extrinsics(optimized / "extri.yml")

    expected_names = {Path(image.name).parts[0] for image in images.values()}
    if expected_names != set(optimized_extrinsics):
        missing = sorted(expected_names - set(optimized_extrinsics))
        extra = sorted(set(optimized_extrinsics) - expected_names)
        raise ValueError(f"Camera-name mismatch; missing={missing}, extra={extra}")

    updated = {}
    rows = []
    angles: list[float] = []
    centers: list[float] = []
    for image_id, image in images.items():
        camera_name = Path(image.name).parts[0]
        rotation, translation = optimized_extrinsics[camera_name]
        source_rotation = colmap_io.qvec2rotmat(image.qvec)
        source_center = -(source_rotation.T @ np.asarray(image.tvec))
        optimized_center = -(rotation.T @ translation)
        angle = _rotation_degrees(rotation @ source_rotation.T)
        center_change = float(np.linalg.norm(optimized_center - source_center))
        angles.append(angle)
        centers.append(center_change)
        rows.append(
            {
                "camera": camera_name,
                "image": image.name,
                "rotation_change_deg": angle,
                "center_change_world": center_change,
            }
        )
        updated[image_id] = colmap_io.Image(
            id=image.id,
            qvec=colmap_io.rotmat2qvec(rotation),
            tvec=translation,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    output.mkdir(parents=True)
    colmap_io.write_model(cameras, updated, points, str(output), ".bin")
    report = {
        "method": "4K4D/EasyVolcap L3MHET OptimizableCamera",
        "reference_colmap": str(reference),
        "optimized_dir": str(optimized),
        "output_sparse": str(output),
        "cameras": len(rows),
        "rotation_change_deg": _summary(angles),
        "center_change_world": _summary(centers),
        "per_camera": rows,
    }
    report_path = output.parent / f"{output.name}-camera-export-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-colmap", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--output-sparse", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args(argv)


def main() -> None:
    report = export(parse_args())
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_camera"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
