#!/usr/bin/env python3
"""Plan automatic RoMa crop passes from L3MHET depth coverage holes.

Depth is used only to choose image regions and camera pairs. The resulting
plan contains no 3D points and does not relax any RoMa geometry gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass
class GuideCamera:
    index: int
    name: str
    folder: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    depth: np.ndarray
    opacity: np.ndarray
    depth_factor: int = 1

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera = (self.R @ points.T).T + self.t
        valid = camera[:, 2] > 1e-8
        uv = np.full((len(points), 2), np.nan, dtype=np.float64)
        uv[valid, 0] = self.K[0, 0] * camera[valid, 0] / camera[valid, 2] + self.K[0, 2]
        uv[valid, 1] = self.K[1, 1] * camera[valid, 1] / camera[valid, 2] + self.K[1, 2]
        valid &= np.isfinite(uv).all(axis=1)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < self.width)
        valid &= (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        distance = np.linalg.norm(points - self.center[None], axis=1)
        return uv, distance, valid

    def backproject(self, uv: np.ndarray, distance: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        distance = np.asarray(distance, dtype=np.float64).reshape(-1)
        pixel = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
        ray_camera = pixel @ np.linalg.inv(self.K).T
        ray_world = ray_camera @ self.R
        ray_world /= np.linalg.norm(ray_world, axis=1, keepdims=True).clip(1e-12)
        return self.center[None] + distance[:, None] * ray_world


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(array: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(uv).all(axis=1)
    xy = np.zeros((len(uv), 2), dtype=np.int64)
    xy[finite] = np.rint(uv[finite]).astype(np.int64)
    valid = (
        finite
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < array.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < array.shape[0])
    )
    values = np.full(len(uv), np.nan, dtype=np.float64)
    values[valid] = array[xy[valid, 1], xy[valid, 0]]
    return values, valid


def _sample_camera(
    camera: GuideCamera, array: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _sample(array, np.asarray(uv, dtype=np.float64) / camera.depth_factor)


def _depth_consistent(
    camera: GuideCamera,
    points: np.ndarray,
    *,
    opacity_min: float,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv, predicted, valid = camera.project(points)
    observed, sampled = _sample_camera(camera, camera.depth, uv)
    opacity, sampled_opacity = _sample_camera(camera, camera.opacity, uv)
    tolerance = abs_tolerance + rel_tolerance * np.maximum(predicted, observed)
    valid &= sampled & sampled_opacity
    valid &= np.isfinite(observed) & (observed > 0)
    valid &= opacity >= opacity_min
    valid &= np.abs(predicted - observed) <= tolerance
    return uv, valid


def _square_crop(
    uv: np.ndarray,
    width: int,
    height: int,
    *,
    padding: int,
    minimum: int,
    maximum: int,
) -> list[int] | None:
    uv = np.asarray(uv, dtype=np.float64)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0:
        return None
    lo = np.floor(np.min(uv, axis=0) - padding)
    hi = np.ceil(np.max(uv, axis=0) + padding + 1)
    size = math.ceil(max(hi[0] - lo[0], hi[1] - lo[1], minimum))
    size = min(size, maximum, width, height)
    center = 0.5 * (lo + hi)
    x0 = round(center[0] - size / 2)
    y0 = round(center[1] - size / 2)
    x0 = min(max(x0, 0), width - size)
    y0 = min(max(y0, 0), height - size)
    return [x0, y0, x0 + size, y0 + size]


def _load_cameras(
    reference_json: Path,
    depth_root: Path,
    depth_frame_index: int,
    depth_sample_factor: int,
) -> tuple[list[GuideCamera], dict[str, Any]]:
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    manifest_path = depth_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "l3mhet-raw-depth-v1":
        raise ValueError(f"Unsupported raw-depth manifest: {manifest.get('format')}")
    if manifest.get("depth_semantics") != "distance_along_normalized_world_ray":
        raise ValueError("Depth must be distance along a normalized world ray")
    entries = {
        (str(row["camera"]), int(row["frame_index"])): row
        for row in manifest["entries"]
    }
    cameras: list[GuideCamera] = []
    validation: list[dict[str, float]] = []
    for index, item in enumerate(reference["cameras"]):
        key = (str(item["folder"]), depth_frame_index)
        if key not in entries:
            key = (str(item["name"]), depth_frame_index)
        if key not in entries:
            raise KeyError(
                f"No depth entry for camera {item['folder']!r}, frame {depth_frame_index}"
            )
        with np.load(depth_root / entries[key]["path"], allow_pickle=False) as archive:
            full_depth = np.asarray(archive["depth"], dtype=np.float32)
            full_opacity = np.asarray(archive["opacity"], dtype=np.float32)
            K = np.asarray(archive["K"], dtype=np.float64)
            R = np.asarray(archive["R"], dtype=np.float64)
            t = np.asarray(archive["T"], dtype=np.float64).reshape(3)
        params = np.asarray(item["params"], dtype=np.float64)
        if str(item["camera_model_name"]).upper() != "PINHOLE":
            raise ValueError(
                "Depth-guided plans currently require PINHOLE input images"
            )
        ref_K = np.asarray(
            [[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]],
            dtype=np.float64,
        )
        validation.append(
            {
                "K_max_abs": float(np.max(np.abs(K - ref_K))),
                "R_max_abs": float(np.max(np.abs(R - np.asarray(item["R"])))),
                "t_max_abs": float(np.max(np.abs(t - np.asarray(item["t"])))),
            }
        )
        if full_depth.shape != full_opacity.shape or full_depth.shape != (
            int(item["height"]),
            int(item["width"]),
        ):
            raise ValueError(
                f"Depth shape {full_depth.shape} does not match camera {item['folder']} "
                f"shape {(item['height'], item['width'])}"
            )
        depth = np.ascontiguousarray(
            full_depth[::depth_sample_factor, ::depth_sample_factor]
        )
        opacity = np.ascontiguousarray(
            full_opacity[::depth_sample_factor, ::depth_sample_factor]
        )
        cameras.append(
            GuideCamera(
                index=index,
                name=str(item["name"]),
                folder=str(item["folder"]),
                width=int(item["width"]),
                height=int(item["height"]),
                K=K,
                R=R,
                t=t,
                depth=depth,
                opacity=opacity,
                depth_factor=depth_sample_factor,
            )
        )
    worst = {
        key: max(row[key] for row in validation)
        for key in ("K_max_abs", "R_max_abs", "t_max_abs")
    }
    # Extrinsics must agree closely: otherwise a depth-selected crop would not
    # refer to the same geometry used by the RoMa triangulator.
    if (
        worst["K_max_abs"] > 2.0
        or worst["R_max_abs"] > 2e-4
        or worst["t_max_abs"] > 2e-4
    ):
        raise ValueError(f"Depth/reference camera mismatch: {worst}")
    return cameras, {"manifest": manifest_path.as_posix(), "camera_validation": worst}


def _candidate_tiles(
    hole: np.ndarray,
    *,
    cell_px: int,
    tile_px: int,
    tile_stride_px: int,
    min_cells: int,
    max_tiles: int,
) -> list[tuple[list[int], np.ndarray]]:
    height_cells, width_cells = hole.shape
    tile_cells = max(1, math.ceil(tile_px / cell_px))
    stride_cells = max(1, round(tile_stride_px / cell_px))
    candidates: list[tuple[int, int, int, int, int]] = []
    for y0 in range(0, height_cells, stride_cells):
        for x0 in range(0, width_cells, stride_cells):
            y1 = min(y0 + tile_cells, height_cells)
            x1 = min(x0 + tile_cells, width_cells)
            count = int(np.count_nonzero(hole[y0:y1, x0:x1]))
            if count >= min_cells:
                candidates.append((count, x0, y0, x1, y1))
    candidates.sort(reverse=True)
    claimed = np.zeros_like(hole, dtype=bool)
    selected: list[tuple[list[int], np.ndarray]] = []
    for _count, x0, y0, x1, y1 in candidates:
        local = hole[y0:y1, x0:x1] & ~claimed[y0:y1, x0:x1]
        if int(np.count_nonzero(local)) < min_cells:
            continue
        ys, xs = np.where(hole[y0:y1, x0:x1])
        cells = np.stack([xs + x0, ys + y0], axis=1)
        selected.append(
            (
                [
                    x0 * cell_px,
                    y0 * cell_px,
                    min(x1 * cell_px, width_cells * cell_px),
                    min(y1 * cell_px, height_cells * cell_px),
                ],
                cells,
            )
        )
        claimed[y0:y1, x0:x1] |= hole[y0:y1, x0:x1]
        if len(selected) >= max_tiles:
            break
    return selected


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    reference_json = args.reference_json.resolve()
    depth_root = args.depth_root.resolve()
    cameras, depth_info = _load_cameras(
        reference_json,
        depth_root,
        int(args.depth_frame_index),
        int(args.depth_sample_factor),
    )
    with np.load(args.npz_path.resolve(), allow_pickle=False) as archive:
        existing_points = np.asarray(archive["positions"], dtype=np.float64)
    if existing_points.ndim == 3:
        existing_points = existing_points.reshape(-1, 3)
    if existing_points.ndim != 2 or existing_points.shape[1] != 3:
        raise ValueError(f"Unsupported positions shape: {existing_points.shape}")

    sampled_valid_depth = [
        camera.depth[
            np.isfinite(camera.depth)
            & (camera.depth > 0)
            & (camera.opacity >= args.opacity_min)
        ][::100]
        for camera in cameras
    ]
    sampled_valid_depth = [values for values in sampled_valid_depth if len(values)]
    if not sampled_valid_depth:
        raise ValueError("Raw-depth export has no finite high-opacity depth samples")
    valid_depth_values = np.concatenate(sampled_valid_depth)
    median_depth = float(np.median(valid_depth_values))
    abs_tolerance = (
        float(args.depth_abs_tolerance)
        if args.depth_abs_tolerance > 0
        else 0.0025 * median_depth
    )
    rel_tolerance = float(args.depth_rel_tolerance)

    plan_entries: list[dict[str, Any]] = []
    camera_reports: list[dict[str, Any]] = []
    overlay_dir = args.overlay_dir.resolve() if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        cell = int(args.coverage_cell_px)
        xs = np.arange(cell // 2, camera.width, cell, dtype=np.int64)
        ys = np.arange(cell // 2, camera.height, cell, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_uv = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        grid_depth, _ = _sample_camera(camera, camera.depth, grid_uv)
        grid_opacity, _ = _sample_camera(camera, camera.opacity, grid_uv)
        grid_depth = grid_depth.reshape(grid_x.shape)
        grid_opacity = grid_opacity.reshape(grid_x.shape)
        valid_grid = (
            np.isfinite(grid_depth)
            & (grid_depth > 0)
            & (grid_opacity >= float(args.opacity_min))
        )
        valid_grid = valid_grid.reshape(len(ys), len(xs))

        uv_existing, consistent = _depth_consistent(
            camera,
            existing_points,
            opacity_min=float(args.opacity_min),
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        coverage = np.zeros_like(valid_grid, dtype=bool)
        covered_uv = uv_existing[consistent]
        if len(covered_uv):
            cx = np.clip((covered_uv[:, 0] / cell).astype(int), 0, len(xs) - 1)
            cy = np.clip((covered_uv[:, 1] / cell).astype(int), 0, len(ys) - 1)
            coverage[cy, cx] = True
        if int(args.coverage_dilate_cells) > 0:
            coverage = binary_dilation(
                coverage,
                iterations=int(args.coverage_dilate_cells),
            )
        near_coverage = distance_transform_edt(~coverage) * cell <= float(
            args.max_hole_distance_px
        )
        hole = valid_grid & ~coverage & near_coverage

        planned_for_camera = 0
        for crop_a, hole_cells in _candidate_tiles(
            hole,
            cell_px=cell,
            tile_px=int(args.tile_size_px),
            tile_stride_px=int(args.tile_stride_px),
            min_cells=int(args.min_hole_cells),
            max_tiles=int(args.max_tiles_per_camera),
        ):
            source_uv = (hole_cells.astype(np.float64) + 0.5) * cell
            source_uv[:, 0] = np.clip(source_uv[:, 0], 0, camera.width - 1)
            source_uv[:, 1] = np.clip(source_uv[:, 1], 0, camera.height - 1)
            if len(source_uv) > int(args.max_depth_samples_per_tile):
                indices = np.linspace(
                    0, len(source_uv) - 1, int(args.max_depth_samples_per_tile)
                ).astype(int)
                source_uv = source_uv[indices]
            source_depth, sampled = _sample_camera(camera, camera.depth, source_uv)
            source_uv = source_uv[sampled]
            source_depth = source_depth[sampled]
            world = camera.backproject(source_uv, source_depth)

            candidates: list[
                tuple[float, GuideCamera, np.ndarray, np.ndarray, float]
            ] = []
            for other in cameras:
                if other.index == camera.index:
                    continue
                uv_other, consistent_other = _depth_consistent(
                    other,
                    world,
                    opacity_min=float(args.opacity_min),
                    abs_tolerance=abs_tolerance,
                    rel_tolerance=rel_tolerance,
                )
                support = int(np.count_nonzero(consistent_other))
                support_ratio = support / max(len(world), 1)
                if support < int(args.min_pair_depth_support) or support_ratio < float(
                    args.min_pair_depth_support_ratio
                ):
                    continue
                src_view = camera.center[None] - world[consistent_other]
                dst_view = other.center[None] - world[consistent_other]
                src_view /= np.linalg.norm(src_view, axis=1, keepdims=True).clip(1e-12)
                dst_view /= np.linalg.norm(dst_view, axis=1, keepdims=True).clip(1e-12)
                angle = np.degrees(
                    np.arccos(np.clip(np.sum(src_view * dst_view, axis=1), -1, 1))
                )
                median_angle = float(np.median(angle))
                if not (
                    float(args.min_pair_view_angle_deg)
                    <= median_angle
                    <= float(args.max_pair_view_angle_deg)
                ):
                    continue
                score = support * support_ratio * math.sqrt(max(median_angle, 1.0))
                candidates.append(
                    (score, other, uv_other, consistent_other, median_angle)
                )
            candidates.sort(key=lambda row: row[0], reverse=True)
            for score, other, uv_other, consistent_other, median_angle in candidates[
                : int(args.neighbors_per_tile)
            ]:
                crop_b = _square_crop(
                    uv_other[consistent_other],
                    other.width,
                    other.height,
                    padding=int(args.crop_padding_px),
                    minimum=int(args.min_crop_size_px),
                    maximum=int(args.max_crop_size_px),
                )
                if crop_b is None:
                    continue
                crop_a[2] = min(crop_a[2], camera.width)
                crop_a[3] = min(crop_a[3], camera.height)
                plan_entries.append(
                    {
                        "id": f"{camera.index:04d}-{planned_for_camera:03d}-{other.index:04d}",
                        "cam_a": camera.index,
                        "cam_b": other.index,
                        "camera_a": camera.folder,
                        "camera_b": other.folder,
                        "crop_a": [int(v) for v in crop_a],
                        "crop_b": [int(v) for v in crop_b],
                        "hole_cells": len(hole_cells),
                        "depth_samples": len(world),
                        "depth_consistent_samples": int(
                            np.count_nonzero(consistent_other)
                        ),
                        "depth_consistent_ratio": float(np.mean(consistent_other)),
                        "median_view_angle_deg": median_angle,
                        "score": float(score),
                    }
                )
            planned_for_camera += 1

        camera_reports.append(
            {
                "camera_index": camera.index,
                "camera": camera.folder,
                "valid_cells": int(np.count_nonzero(valid_grid)),
                "covered_cells": int(np.count_nonzero(valid_grid & coverage)),
                "hole_cells": int(np.count_nonzero(hole)),
                "planned_tiles": planned_for_camera,
            }
        )

        if overlay_dir is not None:
            depth = camera.depth.copy()
            finite = np.isfinite(depth) & (depth > 0)
            lo, hi = np.percentile(depth[finite], [5, 95]) if finite.any() else (0, 1)
            gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            gray[~finite] = 0
            low_res = np.repeat((gray[..., None] * 180).astype(np.uint8), 3, axis=2)
            base = np.asarray(
                Image.fromarray(low_res).resize(
                    (camera.width, camera.height), Image.Resampling.BILINEAR
                )
            )
            overlay = Image.fromarray(base).convert("RGB")
            cell_overlay = np.zeros((*valid_grid.shape, 4), dtype=np.uint8)
            cell_overlay[valid_grid & coverage] = (0, 255, 0, 80)
            cell_overlay[hole] = (255, 0, 0, 100)
            coverage_overlay = Image.fromarray(cell_overlay, mode="RGBA").resize(
                (camera.width, camera.height), Image.Resampling.NEAREST
            )
            overlay = Image.alpha_composite(overlay.convert("RGBA"), coverage_overlay)
            draw = ImageDraw.Draw(overlay, "RGBA")
            for entry in plan_entries:
                if entry["cam_a"] == camera.index:
                    draw.rectangle(entry["crop_a"], outline=(255, 255, 0, 255), width=5)
            overlay.save(overlay_dir / f"camera-{camera.index:04d}-{camera.folder}.png")

    # The same physical pair/crop can be proposed from several overlapping
    # source tiles. Keep the strongest exact proposal without changing crops.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in plan_entries:
        key = (
            entry["cam_a"],
            entry["cam_b"],
            *entry["crop_a"],
            *entry["crop_b"],
        )
        if key not in unique or entry["score"] > unique[key]["score"]:
            unique[key] = entry
    plan_entries = sorted(unique.values(), key=lambda row: row["score"], reverse=True)
    if int(args.max_plan_entries) > 0:
        plan_entries = plan_entries[: int(args.max_plan_entries)]

    return {
        "format": "trigs-depth-guided-roma-plan-v1",
        "purpose": "crop_and_pair_selection_only",
        "target_frame": int(args.target_frame),
        "depth_frame_index": int(args.depth_frame_index),
        "depth_sample_factor": int(args.depth_sample_factor),
        "reference_json": str(reference_json),
        "reference_sha256": _sha256(reference_json),
        "source_npz": str(args.npz_path.resolve()),
        "source_npz_sha256": _sha256(args.npz_path.resolve()),
        "depth_root": str(depth_root),
        "depth_info": depth_info,
        "thresholds": {
            "opacity_min": float(args.opacity_min),
            "depth_abs_tolerance": abs_tolerance,
            "depth_rel_tolerance": rel_tolerance,
            "median_valid_depth": median_depth,
            "coverage_cell_px": int(args.coverage_cell_px),
            "max_hole_distance_px": float(args.max_hole_distance_px),
            "min_pair_view_angle_deg": float(args.min_pair_view_angle_deg),
            "max_pair_view_angle_deg": float(args.max_pair_view_angle_deg),
        },
        "camera_reports": camera_reports,
        "entries": plan_entries,
        "summary": {
            "cameras": len(cameras),
            "entries": len(plan_entries),
            "cameras_with_holes": sum(row["hole_cells"] > 0 for row in camera_reports),
            "total_hole_cells": sum(row["hole_cells"] for row in camera_reports),
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-frame-index", type=int, default=0)
    parser.add_argument(
        "--depth-sample-factor",
        type=int,
        default=2,
        help=(
            "Subsample raw depth/opacity in memory while preserving full-image "
            "crop coordinates (default: 2)."
        ),
    )
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--target-frame", type=int, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--opacity-min", type=float, default=0.5)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.0)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--coverage-cell-px", type=int, default=24)
    parser.add_argument("--coverage-dilate-cells", type=int, default=0)
    parser.add_argument("--max-hole-distance-px", type=float, default=512.0)
    parser.add_argument("--tile-size-px", type=int, default=512)
    parser.add_argument("--tile-stride-px", type=int, default=384)
    parser.add_argument("--min-hole-cells", type=int, default=12)
    parser.add_argument("--max-tiles-per-camera", type=int, default=8)
    parser.add_argument("--max-depth-samples-per-tile", type=int, default=512)
    parser.add_argument("--min-pair-depth-support", type=int, default=12)
    parser.add_argument("--min-pair-depth-support-ratio", type=float, default=0.15)
    parser.add_argument("--min-pair-view-angle-deg", type=float, default=2.0)
    parser.add_argument("--max-pair-view-angle-deg", type=float, default=75.0)
    parser.add_argument("--neighbors-per-tile", type=int, default=2)
    parser.add_argument("--crop-padding-px", type=int, default=64)
    parser.add_argument("--min-crop-size-px", type=int, default=256)
    parser.add_argument("--max-crop-size-px", type=int, default=768)
    parser.add_argument("--max-plan-entries", type=int, default=768)
    args = parser.parse_args(argv)
    for name in (
        "coverage_cell_px",
        "depth_sample_factor",
        "tile_size_px",
        "tile_stride_px",
        "min_hole_cells",
        "max_tiles_per_camera",
        "max_depth_samples_per_tile",
        "min_pair_depth_support",
        "neighbors_per_tile",
        "min_crop_size_px",
        "max_crop_size_px",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.opacity_min <= 1:
        parser.error("--opacity-min must be in [0,1]")
    if args.min_pair_view_angle_deg >= args.max_pair_view_angle_deg:
        parser.error("minimum pair angle must be below maximum pair angle")
    return args


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
