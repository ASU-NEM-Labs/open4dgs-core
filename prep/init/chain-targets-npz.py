#!/usr/bin/env python3
"""Add long-range chained motion targets (chain_targets.py) to a training NPZ.

Chains are linked through the DENSE prep NPZ (all rows, so each link has its
nearest neighbours available) for every dynamic row of the TARGET NPZ, which
may be the same file or a row subset of it (e.g. a target-count downsample).
Target rows are matched to dense rows by source frame and exact position.

Adds two fields to the target NPZ and writes the result:
  chain_pos            [N, S, 3] float32, chained position at each offset, NaN
                       where the chain had already stopped (and for static rows)
  chain_offsets_frames [S] int32, offsets in video frames (-window..window,
                       keyframe step apart, 0 excluded)
The trainer uses them with --config.init-chain-fit.

Example:
  python chain-targets-npz.py dense.npz train-500k.npz train-500k-chains.npz --window-frames 21
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain_targets as ct  # noqa: E402


def _per_frame_motion(z, step: int):
    units = str(z["velocity_units"]) if "velocity_units" in z.files else "world_per_video_frame"
    vel = np.asarray(z["velocities"], np.float64)
    acc = np.asarray(z["accelerations"], np.float64) if "accelerations" in z.files else np.zeros_like(vel)
    if "has_acceleration" in z.files:
        acc = np.where(np.asarray(z["has_acceleration"]).reshape(-1, 1).astype(bool), acc, 0.0)
    if "world_per_video_frame" in units:
        return vel, acc
    if "world_per_keyframe_interval" in units:
        return vel / step, acc / (step * step)
    raise SystemExit(f"unknown velocity_units {units!r}")


def _row_keys(pos: np.ndarray, frames: np.ndarray) -> np.ndarray:
    rec = np.empty(len(pos), dtype=[("f", np.int64), ("x", np.float32), ("y", np.float32), ("z", np.float32)])
    rec["f"] = frames; rec["x"], rec["y"], rec["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dense", help="Full prep NPZ (every row) used to link chains.")
    p.add_argument("target", help="NPZ to annotate: the dense file itself or a row subset of it.")
    p.add_argument("output")
    p.add_argument("--window-frames", type=int, default=21, help="Furthest chain offset, in video frames.")
    p.add_argument("--tau", type=float, default=3.0, help="Link tolerance in the linked keyframe's point spacings.")
    p.add_argument("--no-cycle-check", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    dz = np.load(a.dense, allow_pickle=False)
    step = int(dz["keyframe_step"]) if "keyframe_step" in dz.files else 1
    d_pos = np.asarray(dz["positions"], np.float32)
    d_frames = np.asarray(dz["source_frames"]).reshape(-1).astype(np.int64)
    d_dyn = ~np.asarray(dz["is_static"]).reshape(-1).astype(bool) if "is_static" in dz.files else np.ones(len(d_pos), bool)
    d_vel, d_acc = _per_frame_motion(dz, step)
    print(f"dense: {len(d_pos):,} rows ({int(d_dyn.sum()):,} dynamic), keyframe step {step} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    tz = np.load(a.target, allow_pickle=False)
    compressed = any(i.compress_type != zipfile.ZIP_STORED for i in zipfile.ZipFile(a.target).infolist())
    data = {k: np.asarray(tz[k]) for k in tz.files}
    n = len(data["positions"])
    t_frames = data["source_frames"].reshape(-1).astype(np.int64)
    t_dyn = ~data["is_static"].reshape(-1).astype(bool) if "is_static" in data else np.ones(n, bool)

    d_keys = _row_keys(d_pos[d_dyn], d_frames[d_dyn])
    d_idx = np.nonzero(d_dyn)[0]
    order = np.argsort(d_keys, kind="stable")
    t_keys = _row_keys(data["positions"][t_dyn].astype(np.float32), t_frames[t_dyn])
    loc = np.searchsorted(d_keys[order], t_keys)
    loc = np.clip(loc, 0, len(order) - 1)
    found = d_keys[order][loc] == t_keys
    if not found.all():
        raise SystemExit(f"{int((~found).sum()):,} dynamic target rows are not in the dense NPZ "
                         "(same source frame and exact position required)")
    query = d_idx[order[loc]]
    print(f"target: {n:,} rows, {len(query):,} dynamic matched to dense rows [{time.time() - t0:.0f}s]", flush=True)

    params = ct.ChainParams(window_frames=a.window_frames, tau_spacings=a.tau, cycle_check=not a.no_cycle_check)
    offsets, chains = ct.build_chains(d_pos, d_vel, d_acc, d_frames, d_dyn, query, step, params)
    chain_pos = np.full((n, len(offsets), 3), np.nan, np.float32)
    chain_pos[np.nonzero(t_dyn)[0]] = chains
    fwd, bwd = ct.contiguous_span(~np.isnan(chains[..., 0]), offsets)
    half = np.minimum(fwd, bwd)
    print(f"chains [{time.time() - t0:.0f}s]: forward span p50/p90 {np.median(fwd):.0f}/{np.percentile(fwd, 90):.0f} "
          f"frames, backward {np.median(bwd):.0f}/{np.percentile(bwd, 90):.0f}; both sides >= "
          + ", ".join(f"{o}: {np.mean(half >= o):.0%}" for o in offsets[offsets > 0]), flush=True)

    data["chain_pos"] = chain_pos
    data["chain_offsets_frames"] = offsets
    (np.savez_compressed if compressed else np.savez)(a.output, **data)
    print(f"wrote {a.output} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
