#!/usr/bin/env python3
"""
Generate a reference_export.json and scene_meta.json from an existing COLMAP sparse folder.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pycolmap


def _get_image_pose_world_to_cam(image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return COLMAP image world->camera R, t robustly across pycolmap versions.
    """
    if hasattr(image, "cam_from_world"):
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()

        rot = cam_from_world.rotation
        if hasattr(rot, "matrix"):
            R = rot.matrix()
            if callable(R):
                R = R()
        else:
            R = np.asarray(rot)

        t = cam_from_world.translation
        return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)

    if hasattr(image, "rotation_matrix"):
        R = image.rotation_matrix()
        if callable(R):
            R = R()
        t = image.tvec
        return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)

    raise RuntimeError(
        "Could not read image pose from pycolmap Image. "
        "Update _get_image_pose_world_to_cam for this pycolmap version."
    )


def _camera_model_name(camera) -> str:
    model = camera.model
    if hasattr(model, "name"):
        return str(model.name)
    return str(model)


def get_scene_bounds(rec: pycolmap.Reconstruction) -> Dict[str, Any]:
    """
    Calculate the bounding sphere and center of the 3D points.
    """
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)

    if len(xyz) == 0:
        return {
            "scene_center": [0.0, 0.0, 0.0],
            "scene_radius_p90": 1.0,
            "scene_radius_p95": 1.0,
            "scene_radius_max": 1.0,
            "n_points": 0,
        }
        
    center = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - center[None, :], axis=1)
    
    return {
        "scene_center": center.tolist(),
        "scene_radius_p90": float(np.percentile(dists, 90)),
        "scene_radius_p95": float(np.percentile(dists, 95)),
        "scene_radius_max": float(dists.max()),
        "n_points": int(len(xyz)),
    }


def export_reference_json(
    sparse_dir: Path,
    out_json: Path,
    meta_json: Path
) -> None:
    """
    Export camera data and scene metadata for RoMa/TriGS prep.
    """
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Could not find sparse directory: {sparse_dir}")

    print(f"[INFO] Loading reconstruction from {sparse_dir}...")
    rec = pycolmap.Reconstruction(str(sparse_dir))
    
    print(f"[INFO] Loaded {len(rec.cameras)} cameras, {len(rec.images)} images, and {len(rec.points3D)} points.")

    # Generate and save scene metadata
    scene_meta = get_scene_bounds(rec)
    
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_json, "w") as f:
        json.dump(scene_meta, f, indent=2)
    print(f"[INFO] Wrote scene metadata: {meta_json}")

    # Process cameras
    cameras = []

    # Sort by folder/name for deterministic camera order.
    images_sorted = sorted(
        rec.images.values(),
        key=lambda img: str(img.name),
    )

    for img in images_sorted:
        cam = rec.cameras[img.camera_id]
        R, t = _get_image_pose_world_to_cam(img)

        img_name = str(img.name)
        p = Path(img_name)

        if p.parent.as_posix() == ".":
            # Fallback if names are flat
            folder = p.stem
        else:
            folder = p.parent.as_posix()

        cameras.append(
            {
                "name": img_name,
                "folder": folder,
                "image_name": p.name,
                "R": R.tolist(),
                "t": t.reshape(3).tolist(),
                "camera_model_name": _camera_model_name(cam),
                "width": int(cam.width),
                "height": int(cam.height),
                "params": np.asarray(cam.params, dtype=np.float64).tolist(),
            }
        )

    payload = {
        "cameras": cameras,
        "scene_meta": scene_meta,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Wrote reference export: {out_json}")
    print(f"[INFO] Exported {len(cameras)} cameras successfully.")


def main():
    parser = argparse.ArgumentParser(description="Export reference_export.json from a COLMAP sparse directory.")
    parser.add_argument("--sparse-dir", type=str, required=True, help="Path to the COLMAP sparse directory (e.g., path/to/sparse/0)")
    parser.add_argument("--out-json", type=str, required=True, help="Path to the output reference_export.json")
    parser.add_argument("--meta-json", type=str, default=None, help="Path to output scene_meta.json. Defaults to inside sparse-dir.")

    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    
    if args.meta_json:
        meta_json = Path(args.meta_json).expanduser().resolve()
    else:
        meta_json = sparse_dir / "scene_meta.json"

    export_reference_json(sparse_dir, out_json, meta_json)


if __name__ == "__main__":
    main()