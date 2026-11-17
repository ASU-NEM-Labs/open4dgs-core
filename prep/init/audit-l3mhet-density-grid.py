#!/usr/bin/env python3
"""Audit an L3MHET density grid against the RoMa point cloud.

The question this answers is whether the field's geometry can be trusted where
RoMa has no points, and the only honest way to decide that is to check it where
RoMa *does* have points. Surface vertices are extracted at a sweep of density
thresholds and compared against the RoMa cloud two ways:

``agreement``  vertices that have a RoMa point nearby -- how far off are they?
               If the field's geometry is right this lands near the voxel size.
``novel``      vertices with no RoMa point nearby. These are the candidate
               seeds. Their location is reported so it can be confirmed to be
               the region RoMa actually failed on, rather than floaters.

Reads only; writes nothing unless ``--write-points`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def load_grid(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        grid = {
            "density": np.asarray(archive["density"], dtype=np.float32),
            "occupancy": np.asarray(archive["occupancy"], dtype=np.float32),
            "xs": np.asarray(archive["xs"], dtype=np.float64),
            "ys": np.asarray(archive["ys"], dtype=np.float64),
            "zs": np.asarray(archive["zs"], dtype=np.float64),
            "voxel_size": float(archive["voxel_size"]),
            "metadata": json.loads(str(archive["metadata"])),
        }
    if grid["metadata"].get("format") != "l3mhet-density-grid-v1":
        raise ValueError(f"Unsupported grid format: {grid['metadata'].get('format')}")
    return grid


def isosurface(grid: dict[str, Any], field: np.ndarray, level: float) -> np.ndarray:
    """Level-set points in world coordinates, or an empty array.

    Every grid edge whose endpoints straddle ``level`` contributes one point,
    placed by linear interpolation along that edge. That is the same sub-voxel
    placement marching cubes uses for its vertices, without the triangulation
    we have no use for -- these become Gaussian seeds, not a mesh.
    """
    axes = (grid["xs"], grid["ys"], grid["zs"])
    chunks: list[np.ndarray] = []
    for axis, coords in enumerate(axes):
        if coords.size < 2:
            continue
        low_slice: list[Any] = [slice(None)] * 3
        high_slice: list[Any] = [slice(None)] * 3
        low_slice[axis] = slice(0, -1)
        high_slice[axis] = slice(1, None)
        low = field[tuple(low_slice)]
        high = field[tuple(high_slice)]
        crossing = (low < level) != (high < level)
        crossing &= np.isfinite(low) & np.isfinite(high)
        if not crossing.any():
            continue
        index = np.nonzero(crossing)
        start = low[index].astype(np.float64)
        end = high[index].astype(np.float64)
        span = end - start
        # A zero span cannot straddle the level, but guard the divide anyway.
        alpha = np.where(np.abs(span) > 1e-12, (level - start) / np.where(span == 0, 1.0, span), 0.5)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        points = np.stack([axes[a][index[a]] for a in range(3)], axis=1).astype(np.float64)
        step = coords[index[axis] + 1] - coords[index[axis]]
        points[:, axis] += alpha * step
        chunks.append(points)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    grid = load_grid(args.grid_path.resolve())
    field = grid["occupancy"] if args.field == "occupancy" else grid["density"]

    with np.load(args.npz_path.resolve(), allow_pickle=False) as archive:
        roma = np.asarray(archive["positions"], dtype=np.float64).reshape(-1, 3)
    lo = np.array([grid["xs"][0], grid["ys"][0], grid["zs"][0]])
    hi = np.array([grid["xs"][-1], grid["ys"][-1], grid["zs"][-1]])
    inside = np.all((roma >= lo[None]) & (roma <= hi[None]), axis=1)
    roma_box = roma[inside]
    if len(roma_box) == 0:
        raise ValueError(
            "No RoMa points fall inside the grid box, so there is nothing to "
            "validate against. Extend the box to cover a region RoMa solved."
        )
    tree = cKDTree(roma_box)

    finite = np.isfinite(field)
    report: dict[str, Any] = {
        "grid_shape": list(field.shape),
        "voxel_size": grid["voxel_size"],
        "field": args.field,
        "field_percentiles": {
            str(p): float(np.percentile(field[finite], p))
            for p in (50, 90, 99, 99.9, 100)
        },
        "roma_points_in_box": int(len(roma_box)),
        "near_radius": args.near_radius,
        "thresholds": [],
    }

    for level in args.thresholds:
        verts = isosurface(grid, field, float(level))
        entry: dict[str, Any] = {"threshold": float(level), "vertices": len(verts)}
        if len(verts):
            distance, _ = tree.query(verts, k=1)
            near = distance <= args.near_radius
            entry["agreement_fraction"] = float(np.mean(near))
            if near.any():
                entry["agreement_dist_p50"] = float(np.percentile(distance[near], 50))
                entry["agreement_dist_p90"] = float(np.percentile(distance[near], 90))
            novel = verts[~near]
            entry["novel_vertices"] = int(len(novel))
            if len(novel):
                entry["novel_centroid"] = [round(float(v), 4) for v in novel.mean(0)]
                entry["novel_bbox_min"] = [round(float(v), 4) for v in novel.min(0)]
                entry["novel_bbox_max"] = [round(float(v), 4) for v in novel.max(0)]
            # Reverse direction catches geometry RoMa has that the field lost.
            reverse, _ = cKDTree(verts).query(roma_box, k=1)
            entry["roma_covered_fraction"] = float(
                np.mean(reverse <= args.near_radius)
            )
        report["thresholds"].append(entry)

    if args.write_points is not None:
        if args.threshold is None:
            raise ValueError("--write-points requires --threshold")
        verts = isosurface(grid, field, float(args.threshold))
        distance, _ = tree.query(verts, k=1) if len(verts) else (np.zeros(0), None)
        keep = verts[distance > args.near_radius] if len(verts) else verts
        output = args.write_points.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            positions=keep.astype(np.float32),
            threshold=np.asarray(float(args.threshold)),
            field=np.asarray(args.field),
            near_radius=np.asarray(float(args.near_radius)),
            source_grid=np.asarray(str(args.grid_path)),
        )
        report["written"] = {"path": str(output), "points": int(len(keep))}

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-path", type=Path, required=True)
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=("density", "occupancy"),
        default="occupancy",
        help="Occupancy is bounded in [0, 1] so its thresholds transfer "
        "between captures; density does not.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument(
        "--near-radius",
        type=float,
        default=0.01,
        help="A vertex within this distance of a RoMa point counts as "
        "agreeing with it; beyond it, as new geometry.",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--write-points", type=Path)
    return parser


if __name__ == "__main__":
    print(json.dumps(audit(build_parser().parse_args()), indent=2))
