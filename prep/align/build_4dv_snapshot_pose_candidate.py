#!/usr/bin/env python3
"""Build a reference-export candidate using selected 4DV snapshot poses.

The 4DV capture manifest and its bound reference snapshot can carry different
extrinsics for repointed PTZ cameras.  This tool aligns the snapshot camera
ring to an existing COLMAP reference with a proper 3D similarity transform,
then replaces explicitly selected camera poses.  It can also replace those
cameras' complete projection models with the snapshot intrinsics.  Image names,
scene metadata, and every unselected camera remain untouched.

The output is a candidate for epipolar A/B validation; it does not modify a
COLMAP sparse model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -np.asarray(rotation, dtype=np.float64).T @ np.asarray(
        translation, dtype=np.float64
    )


def proper_similarity_row(
    source: np.ndarray, target: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * source @ rotation + translation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape [N,3]")
    if len(source) < 3:
        raise ValueError("at least three camera centers are required")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    x = source - source_mean
    y = target - target_mean
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3, dtype=np.float64)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    correction = np.diag(sign)
    rotation = u @ correction @ vt
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        raise ValueError("snapshot camera centers are degenerate")
    scale = float(np.sum(singular * sign) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid fitted similarity scale {scale}")
    translation = target_mean - scale * source_mean @ rotation
    return scale, rotation, translation


def parse_camera_list(text: str) -> Tuple[str, ...]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower().startswith("jcam"):
            item = item[4:]
        values.append(f"{int(item):04d}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one camera folder")
    return tuple(dict.fromkeys(values))


def snapshot_extrinsics(record: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rows = record.get("extri", {}).get("data", [])
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        camera_id = str(row["camera_id"])
        matrix = np.asarray(row["extrinsic"], dtype=np.float64)
        if matrix.shape != (3, 4):
            raise ValueError(f"{camera_id}: expected a 3x4 extrinsic")
        folder = f"{int(camera_id.removeprefix('JCam')):04d}"
        result[folder] = (matrix[:, :3], matrix[:, 3])
    return result


def snapshot_intrinsics(record: dict) -> Dict[str, np.ndarray]:
    rows = record.get("intri", {}).get("data", [])
    result: Dict[str, np.ndarray] = {}
    for row in rows:
        camera_id = str(row["camera_id"])
        matrix = np.asarray(row["intrinsic"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"{camera_id}: expected a 3x3 intrinsic matrix")
        if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"{camera_id}: invalid intrinsic matrix")
        folder = f"{int(camera_id.removeprefix('JCam')):04d}"
        result[folder] = matrix
    return result


def build_candidate(
    base: dict,
    snapshot_record: dict,
    replace_folders: Iterable[str],
    replace_projection: bool = False,
) -> Tuple[dict, dict]:
    candidate = copy.deepcopy(base)
    base_by_folder = {str(camera["folder"]): camera for camera in candidate["cameras"]}
    snapshot_by_folder = snapshot_extrinsics(snapshot_record)
    snapshot_k_by_folder = snapshot_intrinsics(snapshot_record)
    common = sorted(set(base_by_folder) & set(snapshot_by_folder))
    if len(common) < 3:
        raise ValueError("base and snapshot do not share enough camera folders")

    source_centers = []
    target_centers = []
    for folder in common:
        snapshot_r, snapshot_t = snapshot_by_folder[folder]
        base_camera = base_by_folder[folder]
        source_centers.append(camera_center(snapshot_r, snapshot_t))
        target_centers.append(
            camera_center(
                np.asarray(base_camera["R"], dtype=np.float64),
                np.asarray(base_camera["t"], dtype=np.float64),
            )
        )
    source_centers_arr = np.stack(source_centers)
    target_centers_arr = np.stack(target_centers)
    scale, world_rotation_row, world_translation = proper_similarity_row(
        source_centers_arr, target_centers_arr
    )
    aligned_centers = (
        scale * source_centers_arr @ world_rotation_row + world_translation
    )
    center_errors = np.linalg.norm(aligned_centers - target_centers_arr, axis=1)

    replacements = {}
    for folder in replace_folders:
        if folder not in base_by_folder:
            raise ValueError(f"base reference has no camera folder {folder}")
        if folder not in snapshot_by_folder:
            raise ValueError(f"snapshot record has no camera folder {folder}")
        camera = base_by_folder[folder]
        old_r = np.asarray(camera["R"], dtype=np.float64)
        old_t = np.asarray(camera["t"], dtype=np.float64)
        snapshot_r, snapshot_t = snapshot_by_folder[folder]
        snapshot_center = camera_center(snapshot_r, snapshot_t)
        mapped_center = (
            scale * snapshot_center @ world_rotation_row + world_translation
        )
        # If X_base = scale * X_snapshot @ W + translation in row-vector
        # notation, the corresponding world-to-camera rotation is R_snapshot W.
        mapped_r = snapshot_r @ world_rotation_row
        if np.linalg.det(mapped_r) < 0.999 or np.linalg.det(mapped_r) > 1.001:
            raise ValueError(f"{folder}: mapped rotation is not proper")
        mapped_t = -mapped_r @ mapped_center
        delta = old_r @ mapped_r.T
        angle = float(
            np.degrees(
                np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
            )
        )
        center_delta = float(
            np.linalg.norm(camera_center(old_r, old_t) - mapped_center)
        )
        camera["R"] = mapped_r.tolist()
        camera["t"] = mapped_t.tolist()
        projection = None
        if replace_projection:
            if folder not in snapshot_k_by_folder:
                raise ValueError(f"snapshot record has no intrinsics for camera folder {folder}")
            intrinsic = snapshot_k_by_folder[folder]
            old_projection = {
                "camera_model_name": camera.get("camera_model_name"),
                "params": copy.deepcopy(camera.get("params")),
            }
            # The 4DV record supplies K but no lens-distortion coefficients.
            # Treating it as OPENCV while retaining COLMAP's weakly estimated
            # distortion would create a projection model that never existed.
            camera["camera_model_name"] = "PINHOLE"
            camera["params"] = [
                float(intrinsic[0, 0]), float(intrinsic[1, 1]),
                float(intrinsic[0, 2]), float(intrinsic[1, 2]),
            ]
            projection = {
                "old": old_projection,
                "new": {
                    "camera_model_name": "PINHOLE",
                    "params": copy.deepcopy(camera["params"]),
                },
            }
        replacements[folder] = {
            "rotation_delta_deg": angle,
            "center_delta": center_delta,
            "projection_replaced": bool(replace_projection),
            "projection": projection,
        }

    diagnostics = {
        "common_cameras": len(common),
        "similarity_scale": scale,
        "center_alignment_median": float(np.median(center_errors)),
        "center_alignment_p90": float(np.quantile(center_errors, 0.90)),
        "center_alignment_max": float(np.max(center_errors)),
        "replacements": replacements,
    }
    candidate["pose_candidate"] = {
        "source": "4dv_bound_snapshot_record",
        "projection_source": (
            "4dv_bound_snapshot_record" if replace_projection else "base_reference"
        ),
        **diagnostics,
    }
    return candidate, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reference", default="reference_export.json")
    parser.add_argument("--snapshot-record", required=True)
    parser.add_argument("--replace-cameras", type=parse_camera_list, required=True)
    parser.add_argument(
        "--replace-projection",
        action="store_true",
        help=(
            "Also replace selected cameras' intrinsics/distortion with the snapshot K "
            "as a PINHOLE model. Use this to test the complete 4DV calibration rather "
            "than a snapshot-pose/COLMAP-intrinsics hybrid."
        ),
    )
    parser.add_argument("--output-reference", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = json.loads(Path(args.base_reference).read_text(encoding="utf-8"))
    record = json.loads(Path(args.snapshot_record).read_text(encoding="utf-8"))
    candidate, diagnostics = build_candidate(
        base, record, args.replace_cameras, replace_projection=args.replace_projection
    )
    output = Path(args.output_reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    print(
        f"Aligned {diagnostics['common_cameras']} snapshot cameras: "
        f"center error median={diagnostics['center_alignment_median']:.4f}, "
        f"p90={diagnostics['center_alignment_p90']:.4f}, "
        f"max={diagnostics['center_alignment_max']:.4f}"
    )
    for folder, row in diagnostics["replacements"].items():
        print(
            f"  {folder}: pose rotation delta={row['rotation_delta_deg']:.3f} deg, "
            f"center delta={row['center_delta']:.4f}"
        )
    print(f"Wrote pose candidate: {output}")


if __name__ == "__main__":
    main()
