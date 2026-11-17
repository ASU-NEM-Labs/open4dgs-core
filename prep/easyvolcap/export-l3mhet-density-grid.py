#!/usr/bin/env python3
"""Export raw L3MHET density/occupancy on a world-space voxel grid.

This queries the radiance field directly at known world coordinates. Unlike
rendered depth it involves no ray parameterization, no per-view scale, and no
camera alignment, so the exported values are unambiguously located in the same
world frame as the dataset cameras.

The whole grid is written out rather than a thresholded point set: choosing a
density threshold is a judgement call that should be made and revised offline,
without paying for another GPU pass.

Arguments before ``--`` belong to this script. Arguments after ``--`` are
passed to EasyVolcap's normal test configuration loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def parse_local_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Inclusive lower corner of the sampled box, in world units.",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Inclusive upper corner of the sampled box, in world units.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.003,
        help="Grid spacing in world units. Sample count scales cubically.",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=1.2e8,
        help="Refuse to run above this many queries; guards against a typo "
        "in --voxel-size turning into an hours-long job.",
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=0.005,
        help="Step size used to convert density to occupancy (alpha).",
    )
    parser.add_argument(
        "--time-value",
        type=float,
        default=0.0,
        help="Normalized time fed to the field. Static scenes use 0.",
    )
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def grid_axes(
    bbox_min: np.ndarray, bbox_max: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inclusive axes covering the box at (at most) ``voxel_size`` spacing."""
    if voxel_size <= 0:
        raise ValueError(f"--voxel-size must be positive, got {voxel_size}")
    if not np.all(bbox_max > bbox_min):
        raise ValueError(f"--bbox-max must exceed --bbox-min: {bbox_min} {bbox_max}")
    axes = []
    for lo, hi in zip(bbox_min, bbox_max):
        count = int(np.floor((hi - lo) / voxel_size)) + 1
        axes.append(np.linspace(lo, lo + (count - 1) * voxel_size, count))
    return tuple(axes)  # type: ignore[return-value]


def main() -> int:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    local_args = parse_local_args(sys.argv[1:separator])
    evc_args = sys.argv[separator + 1 :] if separator < len(sys.argv) else []

    output_path = local_args.output_path.resolve()
    if output_path.exists() and not local_args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing path: {output_path}")

    bbox_min = np.asarray(local_args.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(local_args.bbox_max, dtype=np.float64)
    xs, ys, zs = grid_axes(bbox_min, bbox_max, float(local_args.voxel_size))
    total = xs.size * ys.size * zs.size
    if total > local_args.max_samples:
        raise ValueError(
            f"Grid would need {total:,} queries "
            f"({xs.size}x{ys.size}x{zs.size}), above --max-samples "
            f"{local_args.max_samples:,.0f}. Raise --voxel-size or the limit."
        )
    print(f"grid {xs.size}x{ys.size}x{zs.size} = {total:,} queries")

    # Configure EasyVolcap before importing its registries, matching the
    # upstream extract_mesh.py entrypoint ordering.
    sys.path.append(".")
    sys.argv = [sys.argv[0], "-t", "test", *evc_args]

    import torch
    from easyvolcap.engine import cfg
    from easyvolcap.scripts.main import test
    from easyvolcap.utils.data_utils import add_iter, to_cuda

    runner = test(cfg, dry_run=True)
    epoch = int(runner.load_network())
    runner.model.eval()

    # MultilevelNetwork keeps the coarse proposal networks first; the last one
    # is the fine field that actually carries the geometry.
    network = runner.model.network
    if hasattr(network, "networks"):
        network = network.networks[-1]
    network = network.cuda()
    network.eval()

    # The embedders read per-batch context, so hand them a real batch rather
    # than a synthetic one. This is the same loader the depth exporter uses.
    batch = next(iter(runner.val_dataloader))
    batch = add_iter(batch, epoch * runner.ep_iter - 1, runner.total_iter)
    batch = to_cuda(batch)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1
    ).astype(np.float32)

    density = np.empty(len(points), dtype=np.float32)
    occupancy = np.empty(len(points), dtype=np.float32)
    chunk = max(1, int(local_args.chunk))
    used_compute_geometry = True

    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        xyz = torch.from_numpy(points[start:stop]).cuda()
        t = torch.full((len(xyz), 1), float(local_args.time_value), device=xyz.device)
        with torch.inference_mode(), torch.no_grad():
            if used_compute_geometry:
                try:
                    geometry = network.compute_geometry(
                        xyz, t, float(local_args.dist), batch
                    )
                    occ_chunk = geometry.occ
                    density_chunk = geometry.density
                except (AttributeError, KeyError, TypeError):
                    # Fall back to the public helpers; costs a second forward
                    # pass per chunk but keeps the export working.
                    used_compute_geometry = False
            if not used_compute_geometry:
                occ_chunk = network.occ(xyz, t, float(local_args.dist), batch)
                density_chunk = network.density(xyz, t, float(local_args.dist), batch)
        occupancy[start:stop] = _numpy(occ_chunk).reshape(-1)
        density[start:stop] = _numpy(density_chunk).reshape(-1)
        if start % (chunk * 200) == 0:
            print(f"  {stop:,}/{len(points):,}", flush=True)

    shape = (xs.size, ys.size, zs.size)
    metadata = {
        "format": "l3mhet-density-grid-v1",
        "axis_order": "ij: volume[i, j, k] <-> (xs[i], ys[j], zs[k])",
        "frame_world": "dataset world frame; use_aligned_cameras must be false",
        "checkpoint_epoch": epoch,
        "dist": float(local_args.dist),
        "time_value": float(local_args.time_value),
        "used_compute_geometry": used_compute_geometry,
    }
    _atomic_savez(
        output_path,
        density=density.reshape(shape).astype(np.float16),
        occupancy=occupancy.reshape(shape).astype(np.float16),
        xs=xs.astype(np.float64),
        ys=ys.astype(np.float64),
        zs=zs.astype(np.float64),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size=np.asarray(float(local_args.voxel_size)),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    finite = np.isfinite(density)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "shape": list(shape),
                "queries": int(total),
                "density_percentiles": {
                    str(p): float(np.percentile(density[finite], p))
                    for p in (50, 90, 99, 99.9, 100)
                },
                "occupancy_ge_0_5_fraction": float(np.mean(occupancy >= 0.5)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
