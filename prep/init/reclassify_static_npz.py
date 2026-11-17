#!/usr/bin/env python3
"""Turn low-motion TriGS rows into one persistent, voxel-fused static pool.

This is primarily a recovery path for a known-static capture that was prepared
with dynamic protection enabled. It avoids re-running RoMa: measured velocities
already stored in the coverage NPZ are used to identify static rows, including
protected rows only when explicitly requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def _merge_groups(
    positions: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable sort order, group starts, and one representative per voxel."""
    if len(positions) == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty
    if radius <= 0.0:
        order = np.arange(len(positions), dtype=np.int64)
        return order, order, order

    coords = np.floor(positions / float(radius)).astype(np.int64)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order]
    changes = np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)
    starts = np.r_[0, np.flatnonzero(changes) + 1].astype(np.int64)
    return order, starts, order[starts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--static-motion-floor",
        required=True,
        type=float,
        help="Maximum speed in world units per video frame.",
    )
    parser.add_argument(
        "--allow-protected",
        action="store_true",
        help=(
            "Also reclassify low-motion dynamic_protect rows. Use only when the "
            "capture interval is known to be static."
        ),
    )
    parser.add_argument(
        "--static-spatial-dedup-radius",
        type=float,
        default=0.0,
        help="Voxel size used to fuse static rows across keyframes. 0 keeps all rows.",
    )
    parser.add_argument(
        "--static-max-points",
        type=int,
        default=0,
        help="Optional deterministic cap after voxel fusion. 0 disables.",
    )
    parser.add_argument(
        "--drop-nonstatic",
        action="store_true",
        help=(
            "Drop rows above the motion floor instead of preserving them as "
            "dynamic. Appropriate only when the entire interval is known static."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.static_motion_floor <= 0.0:
        raise SystemExit("--static-motion-floor must be greater than zero")
    if args.static_spatial_dedup_radius < 0.0:
        raise SystemExit("--static-spatial-dedup-radius cannot be negative")
    if args.static_max_points < 0:
        raise SystemExit("--static-max-points cannot be negative")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Refusing to overwrite --input; choose a different --output")

    data = dict(np.load(args.input, allow_pickle=True))
    required = {"positions", "colors", "velocities", "has_velocity", "is_static"}
    missing = sorted(required.difference(data))
    if missing:
        raise SystemExit(f"Input NPZ is missing required arrays: {missing}")

    positions = np.asarray(data["positions"])
    n_rows = len(positions)
    if _scalar_text(data.get("velocity_units", np.asarray(""))) != "world_per_video_frame":
        raise SystemExit(
            "This recovery tool requires velocity_units=world_per_video_frame; "
            "re-run prep with per-video-frame velocities first."
        )

    velocities = np.asarray(data["velocities"])
    has_velocity = np.asarray(data["has_velocity"]).reshape(-1).astype(bool)
    existing_static = np.asarray(data["is_static"]).reshape(-1).astype(bool)
    protected = np.asarray(
        data.get("dynamic_protect", np.zeros((n_rows,), dtype=bool))
    ).reshape(-1).astype(bool)
    if any(len(x) != n_rows for x in (velocities, has_velocity, existing_static, protected)):
        raise SystemExit("Input NPZ row arrays do not all match positions")

    speed = np.linalg.norm(velocities, axis=1)
    eligible = has_velocity & (speed <= float(args.static_motion_floor))
    if not args.allow_protected:
        eligible &= ~protected
    static_mask = existing_static | eligible
    dynamic_mask = np.zeros((n_rows,), dtype=bool) if args.drop_nonstatic else ~static_mask

    static_positions = positions[static_mask]
    order, starts, representatives_local = _merge_groups(
        static_positions,
        float(args.static_spatial_dedup_radius),
    )
    if len(starts):
        counts = np.diff(np.r_[starts, len(order)]).astype(np.float64)
        merged_positions = (
            np.add.reduceat(static_positions[order].astype(np.float64), starts, axis=0)
            / counts[:, None]
        ).astype(positions.dtype)
        static_colors = np.asarray(data["colors"])[static_mask]
        merged_colors = (
            np.add.reduceat(static_colors[order].astype(np.float64), starts, axis=0)
            / counts[:, None]
        ).astype(static_colors.dtype)
    else:
        merged_positions = static_positions.copy()
        merged_colors = np.asarray(data["colors"])[static_mask].copy()

    if args.static_max_points and len(merged_positions) > args.static_max_points:
        rng = np.random.default_rng(args.seed)
        keep_groups = np.sort(
            rng.choice(len(merged_positions), size=args.static_max_points, replace=False)
        )
        merged_positions = merged_positions[keep_groups]
        merged_colors = merged_colors[keep_groups]
        representatives_local = representatives_local[keep_groups]

    static_indices = np.flatnonzero(static_mask)
    representatives = static_indices[representatives_local]
    n_dynamic = int(dynamic_mask.sum())
    n_static = len(merged_positions)

    output: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] != n_rows:
            output[key] = arr
            continue

        dynamic_block = arr[dynamic_mask]
        if key == "positions":
            static_block = merged_positions
        elif key == "colors":
            static_block = merged_colors
        elif key == "velocities":
            static_block = np.zeros((n_static,) + arr.shape[1:], dtype=arr.dtype)
        elif key == "has_velocity":
            static_block = np.zeros((n_static,) + arr.shape[1:], dtype=arr.dtype)
        elif key in ("is_static", "stage_keep"):
            static_block = np.ones((n_static,) + arr.shape[1:], dtype=arr.dtype)
        elif key in ("dynamic_protect", "dynamic_allowed"):
            static_block = np.zeros((n_static,) + arr.shape[1:], dtype=arr.dtype)
        else:
            static_block = arr[representatives]
        output[key] = np.concatenate([dynamic_block, static_block], axis=0)

    output["static_reclassify_motion_floor"] = np.asarray(
        args.static_motion_floor, dtype=np.float32
    )
    output["static_reclassify_allow_protected"] = np.asarray(
        args.allow_protected, dtype=bool
    )
    output["static_reclassify_dedup_radius"] = np.asarray(
        args.static_spatial_dedup_radius, dtype=np.float32
    )
    output["static_reclassify_dropped_nonstatic"] = np.asarray(
        args.drop_nonstatic, dtype=bool
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"input:                 {n_rows:,}")
    print(f"new low-motion static: {int((eligible & ~existing_static).sum()):,}")
    if args.drop_nonstatic:
        print(f"nonstatic dropped:      {int((~static_mask).sum()):,}")
    print(f"dynamic kept:          {n_dynamic:,}")
    print(f"static fused:          {int(static_mask.sum()):,} -> {n_static:,}")
    print(f"output:                {n_dynamic + n_static:,} ({args.output})")


if __name__ == "__main__":
    main()
