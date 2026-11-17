#!/usr/bin/env python3
"""
Build a clean reference COLMAP model for a static multi-camera rig,
using a TriGS-compatible image-name layout.

Expected source dataset layout:

    data_dir/
        images/
            01/000001.jpg
            02/000001.jpg
            ...

This script creates a temporary COLMAP reference set like:

    data_dir/_colmap_ref/ref_images/
        01/000001.jpg
        02/000001.jpg
        ...

so COLMAP image names become:

    01/000001.jpg
    02/000001.jpg

instead of flat names like:

    01.jpg
    02.jpg

That makes sparse/0 compatible with TriGSVanilla's COLMAP parser.
"""

from __future__ import annotations

import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pycolmap


def run(cmd):
    print("\n[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def select_largest_sparse_model(sparse_dir: Path) -> Tuple[Path, Any]:
    """Return the retained COLMAP model with the most registered images.

    COLMAP can emit several numbered reconstructions even when the mapper is
    asked for a single model.  Directory ``0`` is not guaranteed to be the
    largest (or the last model seen in the mapper log), so treating
    ``sparse/0`` as the result can silently publish a tiny two-view component.
    """
    candidates = []
    for model_path in sorted(
        (p for p in sparse_dir.iterdir() if p.is_dir()),
        key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
    ):
        try:
            rec = pycolmap.Reconstruction(str(model_path))
        except Exception as exc:
            print(f"[WARN] Could not load retained model {model_path}: {exc}")
            continue
        candidates.append((len(rec.images), len(rec.points3D), model_path, rec))

    if not candidates:
        raise RuntimeError(f"Reference SfM failed: no readable models under {sparse_dir}")

    print("[INFO] Retained COLMAP models:")
    for n_images, n_points, model_path, _rec in candidates:
        print(
            f"  {model_path.name}: {n_images} registered images, "
            f"{n_points} points"
        )

    _n_images, _n_points, selected_path, selected_rec = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    print(
        f"[INFO] Selected largest retained model: {selected_path} "
        f"({_n_images} registered images, {_n_points} points)"
    )
    return selected_path, selected_rec


def collect_frame_images(images_dir: Path, frame_idx: int) -> Dict[str, Path]:
    exts = [".jpg", ".jpeg", ".png"]
    frame_stem = f"{frame_idx:06d}"

    paths: Dict[str, Path] = {}
    for cam in sorted(images_dir.iterdir()):
        if not cam.is_dir():
            continue

        found = None
        for ext in exts:
            p = cam / f"{frame_stem}{ext}"
            if p.exists():
                found = p
                break

        if found is not None:
            paths[cam.name] = found

    return paths


def copy_reference_images_trigs_layout(
    frame_paths: Dict[str, Path],
    dst_dir: Path,
    ref_frame: int,
    force_name_frame: int | None = None,
) -> Dict[str, str]:
    """
    Copy reference images into nested TriGS-compatible layout.

    Args:
        frame_paths:
            cam_folder -> source image path.
        dst_dir:
            _colmap_ref/ref_images.
        ref_frame:
            Actual source reference frame.
        force_name_frame:
            Optional image filename frame index to use inside COLMAP.
            If None, preserves ref_frame.

    Returns:
        Mapping cam_folder -> relative COLMAP image name, e.g. "01/000001.jpg".
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_name_frame = ref_frame if force_name_frame is None else force_name_frame
    frame_stem = f"{image_name_frame:06d}"

    rel_names: Dict[str, str] = {}

    for cam, src in frame_paths.items():
        cam_dir = dst_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)

        suffix = src.suffix.lower()
        dst_name = f"{frame_stem}{suffix}"
        dst = cam_dir / dst_name

        shutil.copy2(src, dst)

        rel_names[cam] = f"{cam}/{dst_name}"

    return rel_names


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


def export_scene_bounds(model_path: Path, out_json: Path) -> Dict[str, Any]:
    rec = pycolmap.Reconstruction(str(model_path))
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)

    if len(xyz) == 0:
        meta = {
            "scene_center": [0.0, 0.0, 0.0],
            "scene_radius_p90": 1.0,
            "scene_radius_p95": 1.0,
            "scene_radius_max": 1.0,
            "n_points": 0,
        }
    else:
        center = np.median(xyz, axis=0)
        dists = np.linalg.norm(xyz - center[None, :], axis=1)
        meta = {
            "scene_center": center.tolist(),
            "scene_radius_p90": float(np.percentile(dists, 90)),
            "scene_radius_p95": float(np.percentile(dists, 95)),
            "scene_radius_max": float(dists.max()),
            "n_points": int(len(xyz)),
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Wrote scene metadata: {out_json}")
    print(json.dumps(meta, indent=2))

    return meta


def export_reference_json(
    model_path: Path,
    out_json: Path,
    scene_meta: Dict[str, Any] | None,
) -> None:
    """
    Export camera data for RoMa/TriGS prep.

    Output format matches the reference JSON expected by the RoMa prep script:
        cameras: [
          {
            name,
            folder,
            R,
            t,
            camera_model_name,
            width,
            height,
            params
          }
        ],
        scene_meta: {...}
    """
    rec = pycolmap.Reconstruction(str(model_path))

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
            # Should not happen in TriGS layout, but keep fallback.
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
    print(f"[INFO] Exported {len(cameras)} cameras")


def verify_trigs_sparse_model(model_path: Path, source_images_dir: Path) -> None:
    rec = pycolmap.Reconstruction(str(model_path))
    names = [str(img.name) for img in rec.images.values()]

    flat = [n for n in names if "/" not in n]
    if flat:
        raise RuntimeError(
            "TriGS-incompatible flat COLMAP image names remain. Examples: "
            + ", ".join(flat[:10])
        )

    missing = []
    for name in names:
        p = source_images_dir / name
        if not p.exists():
            missing.append(name)

    if missing:
        raise RuntimeError(
            "COLMAP image names do not exist under source images/. Examples: "
            + ", ".join(missing[:10])
        )

    print("[INFO] TriGS sparse model verification OK")
    print("[INFO] First 20 COLMAP image names:")
    for n in sorted(names)[:20]:
        print(" ", n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ref-frame", type=int, required=True)
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help=(
            "Optional isolated COLMAP work directory. Defaults to "
            "data-dir/_colmap_ref. Use a separate path when comparing static "
            "and dynamic reference solves."
        ),
    )
    parser.add_argument(
        "--reference-images-dir",
        type=str,
        default=None,
        help=(
            "Optional static reference image tree with the same camera-folder "
            "layout as data-dir/images. Defaults to data-dir/images. This is "
            "useful for 4DV snapshot/reference images."
        ),
    )
    parser.add_argument("--camera-model", type=str, default="OPENCV")
    parser.add_argument(
        "--matcher",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "sequential"],
    )
    parser.add_argument("--use-gpu", type=int, default=1)
    parser.add_argument("--min-num-matches", type=int, default=15)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-all-cameras",
        action="store_true",
        help=(
            "Fail before publishing sparse/0 and reference_export.json unless "
            "the selected COLMAP model registers every reference image."
        ),
    )

    parser.add_argument(
        "--single-camera-per-folder",
        type=int,
        default=1,
        help=(
            "Use COLMAP ImageReader.single_camera_per_folder. "
            "For TriGS nested camera folders, 1 is usually the right setting."
        ),
    )

    parser.add_argument(
        "--force-colmap-name-frame",
        type=int,
        default=-1,
        help=(
            "Optional filename frame index to use in _colmap_ref/ref_images. "
            "Default -1 preserves --ref-frame. Usually leave this alone."
        ),
    )

    parser.add_argument(
        "--reference-json-name",
        type=str,
        default="reference_export.json",
        help="Output reference JSON name under data-dir.",
    )

    parser.add_argument(
        "--skip-model-analyzer",
        action="store_true",
        help="Skip colmap model_analyzer.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    images_dir = data_dir / "images"
    reference_source_dir = (
        Path(args.reference_images_dir).expanduser().resolve()
        if args.reference_images_dir
        else images_dir
    )

    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else data_dir / "_colmap_ref"
    )
    ref_images_dir = work_dir / "ref_images"
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dst_sparse0 = data_dir / "sparse" / "0"
    meta_json = data_dir / "sparse" / "scene_meta.json"
    reference_json = data_dir / args.reference_json_name

    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not reference_source_dir.exists():
        raise FileNotFoundError(
            f"Missing reference images directory: {reference_source_dir}"
        )

    frame_paths = collect_frame_images(reference_source_dir, args.ref_frame)
    if not frame_paths:
        raise RuntimeError(
            f"No images found for reference frame {args.ref_frame} under "
            f"{reference_source_dir}"
        )

    print(
        f"[INFO] Found {len(frame_paths)} camera images for frame "
        f"{args.ref_frame} under {reference_source_dir}"
    )

    force_name_frame = None
    if args.force_colmap_name_frame >= 0:
        force_name_frame = args.force_colmap_name_frame

    rel_names = copy_reference_images_trigs_layout(
        frame_paths=frame_paths,
        dst_dir=ref_images_dir,
        ref_frame=args.ref_frame,
        force_name_frame=force_name_frame,
    )

    print("[INFO] Copied reference images in TriGS nested layout:")
    for cam in sorted(rel_names)[:20]:
        print(f"  {rel_names[cam]}")
    if len(rel_names) > 20:
        print(f"  ... {len(rel_names) - 20} more")

    work_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(ref_images_dir),
        "--ImageReader.camera_model", args.camera_model,
        "--ImageReader.single_camera_per_folder", str(args.single_camera_per_folder),
        "--SiftExtraction.use_gpu", str(args.use_gpu),
        "--SiftExtraction.max_num_features", "12000",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ])

    if args.matcher == "exhaustive":
        run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", str(args.use_gpu),
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_ratio", "0.75",
            "--SiftMatching.max_distance", "0.6",
            "--SiftMatching.cross_check", "1",
        ])
    else:
        run([
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", str(args.use_gpu),
            "--SiftMatching.guided_matching", "1",
        ])

    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(ref_images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.multiple_models", "0",
        "--Mapper.abs_pose_min_num_inliers", str(args.min_num_matches),
        "--Mapper.abs_pose_min_inlier_ratio", str(args.min_inlier_ratio),
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.filter_max_reproj_error", "2.0",
        "--Mapper.filter_min_tri_angle", "2.5",
    ])

    model_path, rec = select_largest_sparse_model(sparse_dir)
    print(
        f"[INFO] Reference model: {len(rec.cameras)} cameras, "
        f"{len(rec.images)} images, {len(rec.points3D)} points"
    )
    if args.require_all_cameras and len(rec.images) != len(frame_paths):
        missing = len(frame_paths) - len(rec.images)
        raise RuntimeError(
            "Largest reference model is incomplete: "
            f"registered {len(rec.images)}/{len(frame_paths)} images "
            f"({missing} missing). Refusing to replace sparse/0 or "
            "reference_export.json."
        )

    # Do not touch the published model until SfM has completed and any
    # completeness requirement has passed. This makes --require-all-cameras a
    # real publication gate rather than deleting the prior model up front.
    if dst_sparse0.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Published sparse model already exists at {dst_sparse0}; "
                "use --overwrite after preserving it"
            )
        shutil.rmtree(dst_sparse0)
    dst_sparse0.mkdir(parents=True, exist_ok=True)
    for f in model_path.iterdir():
        shutil.copy2(f, dst_sparse0 / f.name)

    print(f"[INFO] Copied TriGS-compatible reference model to: {dst_sparse0}")

    scene_meta = export_scene_bounds(model_path, meta_json)
    export_reference_json(model_path, reference_json, scene_meta)

    verify_trigs_sparse_model(dst_sparse0, images_dir)

    if not args.skip_model_analyzer:
        print("\n" + "=" * 40)
        print("      COLMAP MODEL ANALYSIS")
        print("=" * 40)
        run([
            "colmap", "model_analyzer",
            "--path", str(model_path),
        ])


if __name__ == "__main__":
    main()
