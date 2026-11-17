#!/usr/bin/env python3
"""Export gsplat ``simple_trainer`` pose optimization to a COLMAP model.

The current gsplat example trainer optimizes a camera-to-world residual in its
normalized scene coordinates.  This script uses gsplat's own COLMAP parser to
recreate that normalization and image ordering, applies the saved residuals,
then maps the cameras back to the source COLMAP coordinate system.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file


import numpy as np


def _load_colmap_io(repo_root: Path):
    module_path = trigs_file("io/colmap.py", repo_root)
    spec = importlib.util.spec_from_file_location("trigs_colmap_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gsplat_parser(examples_dir: Path):
    examples_dir = examples_dir.resolve()
    if not (examples_dir / "datasets/colmap.py").is_file():
        raise FileNotFoundError(
            f"Expected gsplat examples/datasets/colmap.py under {examples_dir}"
        )
    sys.path.insert(0, str(examples_dir))
    try:
        from datasets.colmap import Parser  # type: ignore
        from datasets.normalize import transform_cameras  # type: ignore
    finally:
        sys.path.pop(0)
    return Parser, transform_cameras


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Match gsplat examples/utils.py::rotation_6d_to_matrix exactly."""
    d6 = np.asarray(d6, dtype=np.float64)
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6) rotations, got {d6.shape}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(norm1, np.finfo(np.float64).eps)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(b2, axis=-1, keepdims=True)
    b2 = b2 / np.maximum(norm2, np.finfo(np.float64).eps)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _pose_delta_matrices(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 9:
        raise ValueError(f"Expected pose weights shaped (N, 9), got {weights.shape}")
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    matrices = np.broadcast_to(np.eye(4), (len(weights), 4, 4)).copy()
    matrices[:, :3, :3] = _rotation_6d_to_matrix(weights[:, 3:] + identity_6d)
    matrices[:, :3, 3] = weights[:, :3]
    return matrices


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def export(
    dataset: Path,
    checkpoint: Path,
    output_sparse: Path,
    gsplat_examples_dir: Path,
    repo_root: Path,
    *,
    data_factor: int,
    test_every: int,
    normalize_world_space: bool,
) -> dict:
    import torch

    dataset = dataset.resolve()
    checkpoint = checkpoint.resolve()
    output_sparse = output_sparse.resolve()
    if output_sparse.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_sparse}")
    if test_every <= 1:
        raise ValueError(
            "test_every must exceed 1. Use a value larger than the image count "
            "to keep one fixed gauge-anchor camera and optimize all remaining cameras."
        )

    Parser, transform_cameras = _load_gsplat_parser(gsplat_examples_dir)
    parser = Parser(
        data_dir=str(dataset),
        factor=data_factor,
        normalize=normalize_world_space,
        test_every=test_every,
    )
    all_indices = np.arange(len(parser.image_names), dtype=np.int64)
    train_indices = all_indices[all_indices % test_every != 0]
    anchor_indices = all_indices[all_indices % test_every == 0]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pose_state = saved.get("pose_adjust")
    if not isinstance(pose_state, dict) or "embeds.weight" not in pose_state:
        raise ValueError(
            f"{checkpoint} has no pose_adjust/embeds.weight; train with --pose_opt"
        )
    weights = pose_state["embeds.weight"].detach().cpu().numpy()
    if weights.shape != (len(train_indices), 9):
        raise ValueError(
            "Pose table does not match this dataset/split: "
            f"checkpoint={weights.shape}, expected={(len(train_indices), 9)}. "
            "Use the same --data_factor, --test_every and normalization settings "
            "for training and export."
        )

    normalized_c2w = np.asarray(parser.camtoworlds, dtype=np.float64).copy()
    deltas = _pose_delta_matrices(weights)
    normalized_c2w[train_indices] = normalized_c2w[train_indices] @ deltas
    if normalize_world_space:
        optimized_c2w = transform_cameras(
            np.linalg.inv(np.asarray(parser.transform, dtype=np.float64)),
            normalized_c2w,
        )
    else:
        optimized_c2w = normalized_c2w

    colmap_io = _load_colmap_io(repo_root.resolve())
    source_sparse = dataset / "sparse/0"
    if not source_sparse.is_dir():
        source_sparse = dataset / "sparse"
    cameras = colmap_io.read_cameras_binary(str(source_sparse / "cameras.bin"))
    images = colmap_io.read_images_binary(str(source_sparse / "images.bin"))
    image_by_name = {image.name: image for image in images.values()}
    if set(parser.image_names) != set(image_by_name):
        missing = sorted(set(parser.image_names) - set(image_by_name))
        extra = sorted(set(image_by_name) - set(parser.image_names))
        raise ValueError(
            f"Parser/COLMAP image mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    optimized_images = dict(images)
    rows = []
    train_lookup = {int(index): row for row, index in enumerate(train_indices)}
    rotation_changes: list[float] = []
    center_changes: list[float] = []
    for dataset_index, name in enumerate(parser.image_names):
        source_image = image_by_name[name]
        source_w2c = np.eye(4, dtype=np.float64)
        source_w2c[:3, :3] = colmap_io.qvec2rotmat(source_image.qvec)
        source_w2c[:3, 3] = source_image.tvec
        source_c2w = np.linalg.inv(source_w2c)
        optimized_w2c = np.linalg.inv(optimized_c2w[dataset_index])
        optimized_images[source_image.id] = colmap_io.Image(
            id=source_image.id,
            qvec=colmap_io.rotmat2qvec(optimized_w2c[:3, :3]),
            tvec=optimized_w2c[:3, 3],
            camera_id=source_image.camera_id,
            name=source_image.name,
            xys=source_image.xys,
            point3D_ids=source_image.point3D_ids,
        )

        relative_rotation = optimized_c2w[dataset_index, :3, :3] @ source_c2w[
            :3, :3
        ].T
        angle = _rotation_angle_degrees(relative_rotation)
        center_shift = float(
            np.linalg.norm(
                optimized_c2w[dataset_index, :3, 3] - source_c2w[:3, 3]
            )
        )
        rotation_changes.append(angle)
        center_changes.append(center_shift)
        pose_row = train_lookup.get(dataset_index)
        rows.append(
            {
                "image": name,
                "dataset_index": dataset_index,
                "optimized": pose_row is not None,
                "pose_table_row": pose_row,
                "rotation_change_deg": angle,
                "center_change_world": center_shift,
            }
        )

    output_sparse.mkdir(parents=True)
    colmap_io.write_cameras_binary(cameras, str(output_sparse / "cameras.bin"))
    colmap_io.write_images_binary(
        optimized_images, str(output_sparse / "images.bin")
    )
    points_path = source_sparse / "points3D.bin"
    if points_path.is_file():
        shutil.copyfile(points_path, output_sparse / "points3D.bin")

    report = {
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(saved.get("step", -1)),
        "output_sparse": str(output_sparse),
        "data_factor": data_factor,
        "test_every": test_every,
        "normalize_world_space": normalize_world_space,
        "images": len(parser.image_names),
        "optimized_images": len(train_indices),
        "anchor_images": [parser.image_names[int(i)] for i in anchor_indices],
        "rotation_change_deg": _summary(rotation_changes),
        "center_change_world": _summary(center_changes),
        "per_image": rows,
    }
    report_path = output_sparse.parent / "gsplat-pose-export-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-sparse", type=Path, required=True)
    parser.add_argument(
        "--gsplat-examples-dir",
        type=Path,
        required=True,
        help="Path to the checkout containing simple_trainer.py and datasets/",
    )
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=1_000_000)
    parser.add_argument(
        "--no-normalize-world-space",
        action="store_true",
        help="Use only if training also used --no-normalize_world_space",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    report = export(
        args.dataset,
        args.checkpoint,
        args.output_sparse,
        args.gsplat_examples_dir,
        args.repo_root,
        data_factor=args.data_factor,
        test_every=args.test_every,
        normalize_world_space=not args.no_normalize_world_space,
    )
    summary = {key: value for key, value in report.items() if key != "per_image"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
