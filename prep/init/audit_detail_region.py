#!/usr/bin/env python3
"""Audit why a small scene region fails multiview reconstruction.

This is deliberately a diagnostic, not another point-generation path.  It
answers four separate questions for a projected world-space sphere:

1. Does RoMa return confident correspondences in the region?
2. Do those correspondences agree with the supplied camera calibration?
3. Do independently matched pairs join into genuine 3+-camera tracks?
4. Is temporal motion large enough for the recorded camera skew to matter?

The audit writes a JSON report plus overlays for the geometrically worst
camera pairs.  It never writes or modifies a TriGS NPZ.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_prep_module():
    path = Path(__file__).with_name("prep-roma-trigs-memory.py")
    name = "_trigs_prep_roma_for_detail_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_vec3(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in str(text).split(",")]
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return np.asarray(values, dtype=np.float64)


def parse_camera_pairs(text: str) -> List[Tuple[str, str]]:
    """Parse ``folder_a:folder_b`` pairs without assuming numeric folders."""
    result: List[Tuple[str, str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values = [value.strip() for value in item.split(":")]
        if len(values) != 2 or not all(values) or values[0] == values[1]:
            raise argparse.ArgumentTypeError(
                "expected comma-separated camera pairs such as 0026:0079"
            )
        result.append((values[0], values[1]))
    if not result:
        raise argparse.ArgumentTypeError("at least one camera pair is required")
    return result


def select_explicit_camera_pairs(
    cameras, visible: Sequence[int], requested: Sequence[Tuple[str, str]]
) -> List[Tuple[int, int]]:
    by_folder = {str(camera.folder): idx for idx, camera in enumerate(cameras)}
    visible_set = set(int(idx) for idx in visible)
    selected = set()
    for folder_a, folder_b in requested:
        missing = [folder for folder in (folder_a, folder_b) if folder not in by_folder]
        if missing:
            raise ValueError("Unknown camera folder(s): " + ", ".join(missing))
        cam_a, cam_b = by_folder[folder_a], by_folder[folder_b]
        invisible = [
            str(cameras[idx].folder) for idx in (cam_a, cam_b) if idx not in visible_set
        ]
        if invisible:
            raise ValueError(
                "Requested camera(s) do not see the detail center: "
                + ", ".join(invisible)
            )
        selected.add(tuple(sorted((cam_a, cam_b))))
    return sorted(selected)


def skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def calibrated_fundamental(camera_a, camera_b) -> np.ndarray:
    """Fundamental matrix on the cameras' ideal, undistorted pixel planes."""
    r_rel = camera_b.R @ camera_a.R.T
    t_rel = camera_b.t - r_rel @ camera_a.t
    essential = skew(t_rel) @ r_rel
    fundamental = np.linalg.inv(camera_b._K).T @ essential @ np.linalg.inv(camera_a._K)
    norm = float(np.linalg.norm(fundamental))
    return fundamental / max(norm, 1e-12)


def signed_camera_axis_transforms() -> List[Tuple[str, np.ndarray]]:
    """Return every signed permutation of the camera-coordinate axes.

    This small finite family covers OpenCV/OpenGL axis changes, image-axis
    mirrors, and swapped sensor axes.  It is diagnostic only: finding a good
    transform identifies a convention mismatch but does not silently mutate a
    calibration.
    """
    result: List[Tuple[str, np.ndarray]] = []
    axes = ("x", "y", "z")
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, source_axis in enumerate(permutation):
                matrix[row, source_axis] = signs[row]
            label = ",".join(
                ("+" if signs[row] > 0 else "-") + axes[source_axis]
                for row, source_axis in enumerate(permutation)
            )
            result.append((label, matrix))
    return result


def calibrated_fundamental_with_axis_transform(
    camera_a, camera_b, camera_axis_transform: np.ndarray
) -> np.ndarray:
    transform = np.asarray(camera_axis_transform, dtype=np.float64).reshape(3, 3)
    rotation_a = transform @ camera_a.R
    translation_a = transform @ camera_a.t
    rotation_b = transform @ camera_b.R
    translation_b = transform @ camera_b.t
    r_rel = rotation_b @ rotation_a.T
    t_rel = translation_b - r_rel @ translation_a
    essential = skew(t_rel) @ r_rel
    fundamental = np.linalg.inv(camera_b._K).T @ essential @ np.linalg.inv(camera_a._K)
    norm = float(np.linalg.norm(fundamental))
    return fundamental / max(norm, 1e-12)


def align_cameras_by_folder(primary_cameras, comparison_cameras):
    """Return comparison cameras in primary order, rejecting ambiguous mappings."""
    by_folder = {}
    for camera in comparison_cameras:
        folder = str(camera.folder)
        if folder in by_folder:
            raise ValueError(f"Comparison calibration repeats camera folder {folder!r}")
        by_folder[folder] = camera
    missing = [str(camera.folder) for camera in primary_cameras if str(camera.folder) not in by_folder]
    if missing:
        raise ValueError(
            "Comparison calibration is missing primary camera folders: "
            + ", ".join(missing[:12])
        )
    aligned = [by_folder[str(camera.folder)] for camera in primary_cameras]
    for primary, comparison in zip(primary_cameras, aligned):
        if (int(primary.width), int(primary.height)) != (
            int(comparison.width), int(comparison.height)
        ):
            raise ValueError(
                f"Camera {primary.folder} has incompatible image sizes: "
                f"primary={(primary.width, primary.height)} "
                f"comparison={(comparison.width, comparison.height)}"
            )
    return aligned


def quantiles(values: Sequence[float]) -> Dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"p10": None, "p50": None, "p90": None, "p99": None, "max": None}
    q = np.quantile(array, [0.10, 0.50, 0.90, 0.99])
    return {
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "p99": float(q[3]),
        "max": float(np.max(array)),
    }


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(max(int(denominator), 1))


def classify_report(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = report["summary"]
    pair_success = float(summary["pair_success_ratio"])
    match_yield = float(summary["match_yield_ratio"])
    calibrated = float(summary["calibrated_inlier_ratio_2_5px"])
    fitted = float(summary["fitted_inlier_ratio"])
    three_plus = float(summary["detail_three_plus_track_ratio"])
    temporal_p90 = summary.get("temporal_flow_p90_px_per_frame")
    sync_p90 = summary.get("predicted_sync_shift_p90_px")
    calibrated_camera_outliers = [
        row for row in report.get("per_camera", [])
        if int(row.get("match_observations", 0)) >= 100
        and float(row.get("calibrated_inlier_ratio_2_5px", 0.0)) < 0.05
    ]

    if pair_success < 0.50 or match_yield < 0.05:
        code = "roma_correspondence_failure"
        explanation = (
            "RoMa does not produce enough confident regional correspondences. "
            "This is a matcher/resolution/appearance problem before calibration or tracking."
        )
    elif sync_p90 is not None and float(sync_p90) > 2.0:
        code = "timing_or_motion_failure"
        explanation = (
            "Measured regional motion combined with sub-frame camera skew exceeds two pixels. "
            "Timing compensation is large enough to corrupt cross-camera geometry."
        )
    elif calibrated < 0.50 and fitted >= max(0.65, calibrated + 0.20):
        code = "camera_calibration_mismatch"
        explanation = (
            "RoMa finds a coherent epipolar model, but the supplied camera poses/intrinsics "
            "do not explain those same matches. The camera calibration is the primary failure."
        )
    elif calibrated < 0.50:
        code = "wrong_matches_or_occlusion"
        explanation = (
            "Regional matches do not agree with either the supplied calibration or a strong "
            "pairwise epipolar model. Inspect the overlays for wrong-surface/occlusion matches."
        )
    elif calibrated_camera_outliers:
        folders = ", ".join(str(row["folder"]) for row in calibrated_camera_outliers[:8])
        code = "camera_calibration_outlier_subset"
        explanation = (
            "The aggregate calibration passes, but one or more well-observed cameras have "
            "fewer than 5% calibrated inliers. Those camera poses/intrinsics (or their "
            f"regional matches) must be resolved before track assembly: {folders}."
        )
    elif three_plus < 0.25:
        code = "multiview_track_assembly_failure"
        explanation = (
            "The calibrated pair matches are valid, but they remain separate two-camera "
            "components instead of joining into 3+-camera tracks. Track assembly/sampling is "
            "the primary failure; adding pair-only points cannot fix it."
        )
    else:
        code = "no_single_failure_identified"
        explanation = (
            "Matcher, calibration, track support, and timing all pass the conservative gates. "
            "Use the per-camera residual table and overlays to find a smaller outlier subset."
        )

    return {
        "primary_cause": code,
        "explanation": explanation,
        "thresholds": {
            "pair_success_min": 0.50,
            "match_yield_min": 0.05,
            "calibrated_inlier_min": 0.50,
            "fitted_inlier_strong": 0.65,
            "detail_three_plus_track_min": 0.25,
            "predicted_sync_shift_max_px": 2.0,
            "per_camera_calibrated_inlier_min": 0.05,
            "per_camera_min_match_observations": 100,
        },
        "temporal_flow_p90_px_per_frame": temporal_p90,
    }


def _write_pair_overlay(
    destination: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    crop_a: Tuple[int, int, int, int],
    crop_b: Tuple[int, int, int, int],
    points_a: np.ndarray,
    points_b: np.ndarray,
    residuals: np.ndarray,
    fitted_inliers: np.ndarray,
    label: str,
    threshold_px: float,
    seed: int,
) -> None:
    x0a, y0a, x1a, y1a = crop_a
    x0b, y0b, x1b, y1b = crop_b
    left = np.ascontiguousarray(image_a[y0a:y1a, x0a:x1a])
    right = np.ascontiguousarray(image_b[y0b:y1b, x0b:x1b])
    target_h = 600

    def resize(image: np.ndarray) -> Tuple[np.ndarray, float]:
        scale = target_h / max(float(image.shape[0]), 1.0)
        width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (width, target_h), interpolation=cv2.INTER_AREA), scale

    left_r, sa = resize(left)
    right_r, sb = resize(right)
    canvas = np.concatenate([left_r, right_r], axis=1)
    rng = np.random.default_rng(seed)
    count = min(len(points_a), 180)
    indices = (
        np.arange(len(points_a), dtype=np.int64)
        if count == len(points_a)
        else np.sort(rng.choice(len(points_a), size=count, replace=False))
    )
    for idx in indices:
        pa = ((points_a[idx] - np.asarray([x0a, y0a])) * sa).round().astype(int)
        pb = ((points_b[idx] - np.asarray([x0b, y0b])) * sb).round().astype(int)
        pb[0] += left_r.shape[1]
        calibrated = float(residuals[idx]) <= float(threshold_px)
        pair_fitted = bool(fitted_inliers[idx])
        # RGB: green agrees with the supplied calibration, amber agrees only
        # with the independently fitted pair model, red agrees with neither.
        color = (
            (40, 220, 40) if calibrated
            else (245, 180, 30) if pair_fitted
            else (235, 55, 55)
        )
        cv2.line(canvas, tuple(pa), tuple(pb), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pa), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pb), 2, color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas, label + "  green=calibrated amber=pair-fit red=neither",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(destination, quality=92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", default="reference_export.json")
    parser.add_argument(
        "--compare-reference-json",
        default=None,
        help=(
            "Optional second calibration scored on the exact same RoMa matches. "
            "Camera folders and image dimensions must match --reference-json."
        ),
    )
    parser.add_argument(
        "--compare-axis-search",
        action="store_true",
        help=(
            "Score all 48 signed camera-axis permutations for the comparison "
            "calibration. This diagnoses OpenCV/OpenGL, sensor-axis, or mirror "
            "convention mismatches without modifying either reference file."
        ),
    )
    parser.add_argument("--images-root", default="images")
    parser.add_argument("--image-pattern", default="{frame:06d}.jpg")
    parser.add_argument("--sync-json", default="sync.json")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help=(
            "Skip the t->t+1 motion control and require only --frame. Use this "
            "for a one-frame static snapshot calibration control."
        ),
    )
    parser.add_argument("--detail-sphere-center", type=parse_vec3, required=True)
    parser.add_argument("--detail-sphere-radius", type=float, required=True)
    parser.add_argument(
        "--max-view-angle-deg",
        type=float,
        default=180.0,
        help=(
            "Only audit camera pairs whose center-to-detail viewing rays differ by "
            "at most this angle. Use 45 for a convex head region."
        ),
    )
    parser.add_argument(
        "--camera-pairs",
        type=parse_camera_pairs,
        default=None,
        help=(
            "Audit only explicit folder pairs, e.g. 0026:0079 or "
            "0026:0079,0008:0026. This overrides --max-view-angle-deg for pair selection."
        ),
    )
    parser.add_argument("--crop-radius", type=float, default=0.0)
    parser.add_argument("--crop-scale", type=float, default=1.25)
    parser.add_argument("--crop-min-size-px", type=int, default=256)
    parser.add_argument(
        "--full-image-pairs",
        action="store_true",
        help=(
            "Match complete images instead of projected detail crops. Intended "
            "for explicit-pair static-scene calibration controls."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roma-backend", choices=("v1", "v2"), default="v2")
    parser.add_argument("--roma-model", default="outdoor")
    parser.add_argument("--roma-v2-setting", default="precise")
    parser.add_argument("--roma-v2-mask-oversample", type=float, default=2.0)
    parser.add_argument("--roma-v2-spatial-grid", type=int, default=8)
    parser.add_argument("--roma-v2-spatial-fraction", type=float, default=0.5)
    parser.add_argument("--roma-num-matches", type=int, default=6000)
    parser.add_argument("--roma-min-certainty", type=float, default=0.05)
    parser.add_argument("--pair-ransac-px", type=float, default=0.85)
    parser.add_argument("--min-pair-inliers", type=int, default=60)
    parser.add_argument("--same-camera-merge-px", type=float, default=2.0)
    parser.add_argument("--reproj-thresh-px", type=float, default=2.5)
    parser.add_argument("--calibrated-threshold-px", type=float, default=2.5)
    parser.add_argument("--output-dir", default="detail_audit")
    parser.add_argument("--output-json", default="detail_audit/report.json")
    parser.add_argument("--overlay-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (args.frame_start <= args.frame <= args.frame_end):
        raise ValueError("--frame must lie inside --frame-start/--frame-end")
    if not args.skip_temporal and args.frame >= args.frame_end:
        raise ValueError("the audit needs --frame < --frame-end for its temporal check")
    if args.detail_sphere_radius <= 0.0:
        raise ValueError("--detail-sphere-radius must be positive")
    if not (0.0 < args.max_view_angle_deg <= 180.0):
        raise ValueError("--max-view-angle-deg must be in (0, 180]")
    if args.compare_axis_search and not args.compare_reference_json:
        raise ValueError("--compare-axis-search requires --compare-reference-json")
    if args.full_image_pairs and args.camera_pairs is None:
        raise ValueError("--full-image-pairs requires explicit --camera-pairs")

    prep = _load_prep_module()
    cameras, _scene_meta = prep.load_reference_export(Path(args.reference_json))
    comparison_cameras = None
    if args.compare_reference_json:
        comparison_raw, _comparison_scene_meta = prep.load_reference_export(
            Path(args.compare_reference_json)
        )
        comparison_cameras = align_cameras_by_folder(cameras, comparison_raw)
    center = np.asarray(args.detail_sphere_center, dtype=np.float64)
    crop_radius = float(args.crop_radius or args.detail_sphere_radius)
    visible, pairs = prep.choose_detail_sphere_camera_pairs(
        cameras,
        center,
        min_image_margin_px=0.0,
        max_view_angle_deg=args.max_view_angle_deg,
    )
    if len(visible) < 2:
        raise RuntimeError(f"Only {len(visible)} cameras see the requested detail center")
    if args.camera_pairs is not None:
        pairs = select_explicit_camera_pairs(cameras, visible, args.camera_pairs)

    offsets_sec, sync_keys = prep.load_sync_offsets_for_cameras(args.sync_json, cameras)
    frames_now = prep.sync_aware_camera_frames(
        args.frame,
        fps=args.fps,
        offsets_sec=offsets_sec,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    frames_next = (
        frames_now.copy()
        if args.skip_temporal else
        prep.sync_aware_camera_frames(
            args.frame + 1,
            fps=args.fps,
            offsets_sec=offsets_sec,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    )
    residual_frames = prep.camera_capture_residual_frames(
        args.frame, frames_now, fps=args.fps, offsets_sec=offsets_sec
    )
    paths_now, images_now = prep.load_frame_images_for_camera_frames(
        Path(args.images_root), cameras, frames_now, args.image_pattern
    )
    if args.skip_temporal:
        paths_next, images_next = paths_now, images_now
    else:
        paths_next, images_next = prep.load_frame_images_for_camera_frames(
            Path(args.images_root), cameras, frames_next, args.image_pattern
        )

    model = prep.create_roma_backend(
        backend=args.roma_backend,
        device=args.device,
        model=args.roma_model,
        v2_setting=args.roma_v2_setting,
        symmetric=True,
        upsample_preds=True,
        v2_mask_oversample=args.roma_v2_mask_oversample,
        v2_spatial_grid=args.roma_v2_spatial_grid,
        v2_spatial_fraction=args.roma_v2_spatial_fraction,
    )
    print(
        f"Detail audit: frame={args.frame}, cameras={len(visible)}, pairs={len(pairs)}, "
        f"backend={model.name}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: List[Dict[str, Any]] = []
    overlay_rows: List[Dict[str, Any]] = []
    pair_matches = []
    camera_residuals: Dict[int, List[np.ndarray]] = {idx: [] for idx in visible}
    camera_inliers: Dict[int, List[np.ndarray]] = {idx: [] for idx in visible}
    comparison_camera_residuals: Dict[int, List[np.ndarray]] = {
        idx: [] for idx in visible
    }
    comparison_camera_inliers: Dict[int, List[np.ndarray]] = {
        idx: [] for idx in visible
    }
    temporal_rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="trigs-detail-audit-") as temp_name:
        temp_dir = Path(temp_name)
        crops_now: Dict[int, Tuple[Path, np.ndarray, Tuple[int, int, int, int], np.ndarray]] = {}
        crops_next: Dict[int, Tuple[Path, np.ndarray, Tuple[int, int, int, int], np.ndarray]] = {}
        required_cameras = (
            sorted({idx for pair in pairs for idx in pair})
            if args.full_image_pairs else visible
        )
        for cam_idx in required_cameras:
            bounds = (
                (0, 0, int(cameras[cam_idx].width), int(cameras[cam_idx].height))
                if args.full_image_pairs else
                prep.project_detail_sphere_crop_bounds(
                    cameras[cam_idx], center, crop_radius,
                    scale=args.crop_scale, min_size_px=args.crop_min_size_px,
                )
            )
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            offset = np.asarray([x0, y0], dtype=np.float32)
            crop_targets = [
                ("now", images_now[cam_idx], int(frames_now[cam_idx]), crops_now)
            ]
            if not args.skip_temporal:
                crop_targets.append(
                    ("next", images_next[cam_idx], int(frames_next[cam_idx]), crops_next)
                )
            for label, image, frame_no, target in crop_targets:
                crop = np.ascontiguousarray(image[y0:y1, x0:x1])
                path = temp_dir / f"cam-{cam_idx:04d}-{label}-{frame_no:06d}.png"
                Image.fromarray(crop).save(path)
                target[cam_idx] = (path, crop, bounds, offset)

        usable = sorted(crops_now) if args.skip_temporal else sorted(set(crops_now) & set(crops_next))
        pairs = [(a, b) for a, b in pairs if a in usable and b in usable]

        # Same-camera motion is measured in precisely the crop used below.
        temporal_cameras = [] if args.skip_temporal else usable
        for cam_idx in tqdm(temporal_cameras, desc="temporal detail crops"):
            pa, ia, _ba, _oa = crops_now[cam_idx]
            pb, ib, _bb, _ob = crops_next[cam_idx]
            ka, kb, cert = prep.run_roma_pair(
                model, pa, pb, ia, ib,
                args.roma_num_matches, args.roma_min_certainty,
                args.pair_ransac_px, 8, args.device,
                use_fundamental_ransac=False,
            )
            flow = np.linalg.norm(kb - ka, axis=1) if len(ka) else np.zeros(0)
            predicted = flow * abs(float(residual_frames[cam_idx]))
            temporal_rows.append({
                "camera_index": int(cam_idx),
                "folder": cameras[cam_idx].folder,
                "source_frame": int(frames_now[cam_idx]),
                "next_source_frame": int(frames_next[cam_idx]),
                "sync_key": sync_keys[cam_idx],
                "sync_offset_ms": float(offsets_sec[cam_idx] * 1000.0),
                "capture_residual_frames": float(residual_frames[cam_idx]),
                "matches": int(len(flow)),
                "flow_px_per_frame": quantiles(flow),
                "predicted_sync_shift_px": quantiles(predicted),
                "certainty": quantiles(cert),
            })

        for cam_a, cam_b in tqdm(pairs, desc="cross-camera detail pairs"):
            path_a, image_a, bounds_a, offset_a = crops_now[cam_a]
            path_b, image_b, bounds_b, offset_b = crops_now[cam_b]
            local_a, local_b, cert = prep.run_roma_pair(
                model, path_a, path_b, image_a, image_b,
                args.roma_num_matches, args.roma_min_certainty,
                args.pair_ransac_px, args.min_pair_inliers, args.device,
                use_fundamental_ransac=False,
            )
            global_a = np.ascontiguousarray(local_a + offset_a[None, :]) if len(local_a) else local_a
            global_b = np.ascontiguousarray(local_b + offset_b[None, :]) if len(local_b) else local_b
            row: Dict[str, Any] = {
                "camera_a": int(cam_a), "camera_b": int(cam_b),
                "folder_a": cameras[cam_a].folder, "folder_b": cameras[cam_b].folder,
                "source_frame_a": int(frames_now[cam_a]),
                "source_frame_b": int(frames_now[cam_b]),
                "capture_residual_frames_a": float(residual_frames[cam_a]),
                "capture_residual_frames_b": float(residual_frames[cam_b]),
                "matches_after_certainty": int(len(global_a)),
                "match_yield_ratio": ratio(len(global_a), args.roma_num_matches),
                "certainty": quantiles(cert),
            }
            if len(global_a) < args.min_pair_inliers:
                row.update({
                    "fitted_inliers": 0, "fitted_inlier_ratio": 0.0,
                    "calibrated_inliers_0_85px": 0,
                    "calibrated_inliers_2_5px": 0,
                    "calibrated_inliers_5px": 0,
                    "calibrated_inlier_ratio_2_5px": 0.0,
                    "calibrated_epipolar_px": quantiles([]),
                })
                if comparison_cameras is not None:
                    row.update({
                        "comparison_calibrated_inliers_0_85px": 0,
                        "comparison_calibrated_inliers_2_5px": 0,
                        "comparison_calibrated_inliers_5px": 0,
                        "comparison_calibrated_inlier_ratio_2_5px": 0.0,
                        "comparison_calibrated_epipolar_px": quantiles([]),
                    })
                pair_rows.append(row)
                continue

            ideal_a = cameras[cam_a].undistort_image_points_pixel(global_a)
            ideal_b = cameras[cam_b].undistort_image_points_pixel(global_b)
            fundamental = calibrated_fundamental(cameras[cam_a], cameras[cam_b])
            residual_px = prep.symmetric_epipolar_distance_px(fundamental, ideal_a, ideal_b)
            calibrated_keep = residual_px <= float(args.calibrated_threshold_px)
            fitted = prep.pair_ransac_with_motion_slack(
                global_a, global_b, None,
                args.pair_ransac_px, args.min_pair_inliers,
                0.0, 0.0,
                pair_label=f"{cameras[cam_a].folder}<->{cameras[cam_b].folder}",
                camera_a=cameras[cam_a], camera_b=cameras[cam_b],
            )
            fitted_keep = np.zeros(len(global_a), dtype=bool) if fitted is None else fitted[0]
            row.update({
                "fitted_inliers": int(fitted_keep.sum()),
                "fitted_inlier_ratio": ratio(int(fitted_keep.sum()), len(global_a)),
                "calibrated_inliers_0_85px": int((residual_px <= 0.85).sum()),
                "calibrated_inliers_2_5px": int((residual_px <= 2.5).sum()),
                "calibrated_inliers_5px": int((residual_px <= 5.0).sum()),
                "calibrated_inlier_ratio_2_5px": ratio(int((residual_px <= 2.5).sum()), len(global_a)),
                "calibrated_epipolar_px": quantiles(residual_px),
            })
            if comparison_cameras is not None:
                comparison_a = comparison_cameras[cam_a]
                comparison_b = comparison_cameras[cam_b]
                comparison_ideal_a = comparison_a.undistort_image_points_pixel(global_a)
                comparison_ideal_b = comparison_b.undistort_image_points_pixel(global_b)
                comparison_fundamental = calibrated_fundamental(comparison_a, comparison_b)
                comparison_residual_px = prep.symmetric_epipolar_distance_px(
                    comparison_fundamental, comparison_ideal_a, comparison_ideal_b
                )
                comparison_keep = (
                    comparison_residual_px <= float(args.calibrated_threshold_px)
                )
                row.update({
                    "comparison_calibrated_inliers_0_85px": int(
                        (comparison_residual_px <= 0.85).sum()
                    ),
                    "comparison_calibrated_inliers_2_5px": int(
                        (comparison_residual_px <= 2.5).sum()
                    ),
                    "comparison_calibrated_inliers_5px": int(
                        (comparison_residual_px <= 5.0).sum()
                    ),
                    "comparison_calibrated_inlier_ratio_2_5px": ratio(
                        int((comparison_residual_px <= 2.5).sum()), len(global_a)
                    ),
                    "comparison_calibrated_epipolar_px": quantiles(
                        comparison_residual_px
                    ),
                })
                if args.compare_axis_search:
                    axis_rows = []
                    for axis_label, axis_transform in signed_camera_axis_transforms():
                        axis_fundamental = calibrated_fundamental_with_axis_transform(
                            comparison_a, comparison_b, axis_transform
                        )
                        axis_residual_px = prep.symmetric_epipolar_distance_px(
                            axis_fundamental, comparison_ideal_a, comparison_ideal_b
                        )
                        axis_rows.append({
                            "label": axis_label,
                            "matrix": axis_transform.tolist(),
                            "determinant": float(np.linalg.det(axis_transform)),
                            "calibrated_inliers_2_5px": int(
                                (axis_residual_px <= 2.5).sum()
                            ),
                            "calibrated_inlier_ratio_2_5px": ratio(
                                int((axis_residual_px <= 2.5).sum()), len(global_a)
                            ),
                            "calibrated_epipolar_px": quantiles(axis_residual_px),
                        })
                    axis_rows.sort(
                        key=lambda item: (
                            -item["calibrated_inlier_ratio_2_5px"],
                            item["calibrated_epipolar_px"]["p50"]
                            if item["calibrated_epipolar_px"]["p50"] is not None
                            else float("inf"),
                        )
                    )
                    row["comparison_axis_search"] = axis_rows
                comparison_camera_residuals[cam_a].append(comparison_residual_px)
                comparison_camera_residuals[cam_b].append(comparison_residual_px)
                comparison_camera_inliers[cam_a].append(comparison_keep)
                comparison_camera_inliers[cam_b].append(comparison_keep)
            pair_rows.append(row)
            camera_residuals[cam_a].append(residual_px)
            camera_residuals[cam_b].append(residual_px)
            camera_inliers[cam_a].append(calibrated_keep)
            camera_inliers[cam_b].append(calibrated_keep)
            # Track assembly must be tested with observations that the supplied
            # calibration itself explains.  A separately fitted F can be
            # coherent even when it represents the wrong surface or exposes a
            # bad camera pose; feeding those rows into the component builder
            # would make the assembly stage look healthier than it is.
            if calibrated_keep.any():
                pair_matches.append((
                    cam_a, cam_b,
                    global_a[calibrated_keep], global_b[calibrated_keep],
                    None, None, None, None,
                ))
            overlay_rows.append({
                "score": float(row["calibrated_inlier_ratio_2_5px"]),
                "cam_a": cam_a, "cam_b": cam_b,
                "bounds_a": bounds_a, "bounds_b": bounds_b,
                "points_a": global_a, "points_b": global_b,
                "residuals": residual_px,
                "fitted_inliers": fitted_keep,
                "row": row,
            })

        overlay_rows.sort(key=lambda item: item["score"])
        for rank, item in enumerate(overlay_rows[: max(int(args.overlay_count), 0)]):
            a, b = item["cam_a"], item["cam_b"]
            row = item["row"]
            label = (
                f"{cameras[a].folder}<->{cameras[b].folder}  "
                f"known-calib inliers={100.0 * row['calibrated_inlier_ratio_2_5px']:.1f}%  "
                f"fit inliers={100.0 * row['fitted_inlier_ratio']:.1f}%"
            )
            _write_pair_overlay(
                output_dir / "overlays" / f"worst-{rank + 1:02d}-{cameras[a].folder}-{cameras[b].folder}.jpg",
                images_now[a], images_now[b], item["bounds_a"], item["bounds_b"],
                item["points_a"], item["points_b"], item["residuals"],
                item["fitted_inliers"],
                label, args.calibrated_threshold_px, args.seed + rank,
            )

    components = prep.build_multiview_components(pair_matches, args.same_camera_merge_px)
    support_hist: Dict[str, int] = {}
    detail_support_hist: Dict[str, int] = {}
    detail_accepted = 0
    detail_three_plus = 0
    detail_reprojection_values: List[float] = []
    for component in components:
        observations = prep.average_observations_per_camera(component)
        support = len(observations)
        support_hist[str(support)] = support_hist.get(str(support), 0) + 1
        if support < 2:
            continue
        point = prep.triangulate_multiview_dlt(observations, cameras)
        if point is None:
            continue
        errors, depths = prep.compute_reprojection_errors(point, observations, cameras)
        if np.any(depths <= 0) or not np.all(np.isfinite(errors)):
            continue
        if float(np.mean(errors)) > args.reproj_thresh_px:
            continue
        if float(np.linalg.norm(point - center)) > crop_radius:
            continue
        detail_accepted += 1
        detail_reprojection_values.extend(errors.tolist())
        detail_support_hist[str(support)] = detail_support_hist.get(str(support), 0) + 1
        if support >= 3:
            detail_three_plus += 1

    total_matches = int(sum(row["matches_after_certainty"] for row in pair_rows))
    total_fitted = int(sum(row["fitted_inliers"] for row in pair_rows))
    total_calibrated = int(sum(row["calibrated_inliers_2_5px"] for row in pair_rows))
    total_comparison_calibrated = (
        int(sum(row["comparison_calibrated_inliers_2_5px"] for row in pair_rows))
        if comparison_cameras is not None else None
    )
    successful_pairs = int(sum(row["matches_after_certainty"] >= args.min_pair_inliers for row in pair_rows))
    temporal_flow_p90 = [
        row["flow_px_per_frame"]["p90"] for row in temporal_rows
        if row["flow_px_per_frame"]["p90"] is not None
    ]
    predicted_sync_p90 = [
        row["predicted_sync_shift_px"]["p90"] for row in temporal_rows
        if row["predicted_sync_shift_px"]["p90"] is not None
    ]

    per_camera = []
    for cam_idx in visible:
        residual = np.concatenate(camera_residuals[cam_idx]) if camera_residuals[cam_idx] else np.zeros(0)
        inlier = np.concatenate(camera_inliers[cam_idx]) if camera_inliers[cam_idx] else np.zeros(0, dtype=bool)
        camera_row = {
            "camera_index": int(cam_idx),
            "folder": cameras[cam_idx].folder,
            "match_observations": int(len(residual)),
            "calibrated_inlier_ratio_2_5px": float(np.mean(inlier)) if len(inlier) else 0.0,
            "calibrated_epipolar_px": quantiles(residual),
        }
        if comparison_cameras is not None:
            comparison_residual = (
                np.concatenate(comparison_camera_residuals[cam_idx])
                if comparison_camera_residuals[cam_idx] else np.zeros(0)
            )
            comparison_inlier = (
                np.concatenate(comparison_camera_inliers[cam_idx])
                if comparison_camera_inliers[cam_idx]
                else np.zeros(0, dtype=bool)
            )
            camera_row.update({
                "comparison_calibrated_inlier_ratio_2_5px": (
                    float(np.mean(comparison_inlier)) if len(comparison_inlier) else 0.0
                ),
                "comparison_calibrated_epipolar_px": quantiles(comparison_residual),
            })
        per_camera.append(camera_row)
    per_camera.sort(key=lambda item: item["calibrated_inlier_ratio_2_5px"])

    summary = {
        "visible_cameras": int(len(visible)),
        "attempted_pairs": int(len(pair_rows)),
        "successful_pairs": successful_pairs,
        "pair_success_ratio": ratio(successful_pairs, len(pair_rows)),
        "matches_after_certainty": total_matches,
        "match_yield_ratio": ratio(total_matches, len(pair_rows) * args.roma_num_matches),
        "fitted_inliers": total_fitted,
        "fitted_inlier_ratio": ratio(total_fitted, total_matches),
        "calibrated_inliers_2_5px": total_calibrated,
        "calibrated_inlier_ratio_2_5px": ratio(total_calibrated, total_matches),
        "components": int(len(components)),
        "detail_accepted_components": int(detail_accepted),
        "detail_three_plus_components": int(detail_three_plus),
        "detail_three_plus_track_ratio": ratio(detail_three_plus, detail_accepted),
        "detail_reprojection_px": quantiles(detail_reprojection_values),
        # Use the 90th percentile across cameras rather than the median.  A
        # small subset of close-up cameras can be exactly what breaks the head
        # while leaving the body reconstruction apparently healthy.
        "temporal_flow_p90_px_per_frame": (
            float(np.quantile(temporal_flow_p90, 0.90)) if temporal_flow_p90 else None
        ),
        "predicted_sync_shift_p90_px": (
            float(np.quantile(predicted_sync_p90, 0.90)) if predicted_sync_p90 else None
        ),
    }
    if comparison_cameras is not None:
        comparison_ratio = ratio(total_comparison_calibrated, total_matches)
        primary_ratio = ratio(total_calibrated, total_matches)
        summary["comparison_calibration"] = {
            "reference_json": str(args.compare_reference_json),
            "calibrated_inliers_2_5px": int(total_comparison_calibrated),
            "calibrated_inlier_ratio_2_5px": comparison_ratio,
            "inlier_ratio_delta_vs_primary": comparison_ratio - primary_ratio,
        }
    report: Dict[str, Any] = {
        "schema_version": 2,
        "inputs": {
            "reference_json": str(args.reference_json),
            "compare_reference_json": (
                str(args.compare_reference_json) if args.compare_reference_json else None
            ),
            "compare_axis_search": bool(args.compare_axis_search),
            "images_root": str(args.images_root),
            "sync_json": str(args.sync_json),
            "frame": int(args.frame),
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "fps": float(args.fps),
            "detail_sphere_center": center.tolist(),
            "detail_sphere_radius": float(args.detail_sphere_radius),
            "max_view_angle_deg": float(args.max_view_angle_deg),
            "skip_temporal": bool(args.skip_temporal),
            "full_image_pairs": bool(args.full_image_pairs),
            "camera_pairs": (
                [[folder_a, folder_b] for folder_a, folder_b in args.camera_pairs]
                if args.camera_pairs is not None else None
            ),
            "analysis_crop_radius": crop_radius,
            "roma_backend": str(model.name),
            "roma_num_matches": int(args.roma_num_matches),
            "roma_min_certainty": float(args.roma_min_certainty),
        },
        "camera_frames": [int(value) for value in frames_now.tolist()],
        "visible_camera_indices": [int(value) for value in visible],
        "summary": summary,
        "classification": {},
        "support_hist": support_hist,
        "detail_support_hist": detail_support_hist,
        "per_camera": per_camera,
        "temporal": temporal_rows,
        "pairs": pair_rows,
    }
    report["classification"] = classify_report(report)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== DETAIL AUDIT RESULT ===")
    print(f"Primary cause: {report['classification']['primary_cause']}")
    print(report["classification"]["explanation"])
    print(
        f"RoMa pairs: {successful_pairs}/{len(pair_rows)}; "
        f"match yield={100.0 * summary['match_yield_ratio']:.1f}%"
    )
    print(
        f"Known-calibration inliers @2.5px: "
        f"{100.0 * summary['calibrated_inlier_ratio_2_5px']:.1f}% | "
        f"pair-fitted inliers: {100.0 * summary['fitted_inlier_ratio']:.1f}%"
    )
    if comparison_cameras is not None:
        comparison_summary = summary["comparison_calibration"]
        print(
            f"Comparison-calibration inliers @2.5px: "
            f"{100.0 * comparison_summary['calibrated_inlier_ratio_2_5px']:.1f}% "
            f"(delta={100.0 * comparison_summary['inlier_ratio_delta_vs_primary']:+.1f} points)"
        )
        if args.compare_axis_search:
            best_axis_rows = [
                row["comparison_axis_search"][0]
                for row in pair_rows
                if row.get("comparison_axis_search")
            ]
            if best_axis_rows:
                best_axis = max(
                    best_axis_rows,
                    key=lambda item: item["calibrated_inlier_ratio_2_5px"],
                )
                print(
                    "Best comparison camera-axis convention: "
                    f"{best_axis['label']} | inliers @2.5px="
                    f"{100.0 * best_axis['calibrated_inlier_ratio_2_5px']:.1f}% | "
                    f"p90={best_axis['calibrated_epipolar_px']['p90']}px"
                )
    print(
        f"Head/detail 3+-view tracks: {detail_three_plus}/{detail_accepted} "
        f"({100.0 * summary['detail_three_plus_track_ratio']:.1f}%)"
    )
    print(
        f"Temporal flow p90: {summary['temporal_flow_p90_px_per_frame']} px/frame | "
        f"predicted sync shift p90: {summary['predicted_sync_shift_p90_px']} px"
    )
    print("Worst cameras by known-calibration inlier ratio:")
    for row in per_camera[: min(8, len(per_camera))]:
        print(
            f"  {row['folder']}: {100.0 * row['calibrated_inlier_ratio_2_5px']:.1f}% "
            f"(epipolar p90={row['calibrated_epipolar_px']['p90']})"
        )
    print(f"Report: {output_json}")
    print(f"Overlays: {output_dir / 'overlays'}")


if __name__ == "__main__":
    main()
