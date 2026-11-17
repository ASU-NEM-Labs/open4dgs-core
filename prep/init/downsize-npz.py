#!/usr/bin/env python3
"""Downsize a prep NPZ to the rows the trainer would keep, so training startup
doesn't build (and then throw away) the whole prep cloud: example capture's
27.7M rows -> 1.5M (1.85 GB -> 62 MB), with the trainer's init built on the
rows it actually trains (splat sizes and neighbourhoods from the kept rows).

Selection mirrors trigs.cli.train_from_colmap._apply_split_budget_to_prebuilt_params:
static rows are capped at --static-budget (random), and dynamic rows get
--target-count minus the static share. That share is split across keyframes in
proportion to each keyframe's row count (floor, at least 1, remainder to the
largest fractions), and rows are picked at random within each keyframe.
--even instead keeps one row per voxel cell per keyframe (cell size bisected to
hit the keyframe's quota), for roughly even spacing. --keyframe-stride N first
drops dynamic rows off an N-frame grid from the first dynamic keyframe (match
the trainer's --keyframe-step N --config.keyframe-stride N); still rows stay.

--slow-keep F < 1 seeds slow dynamic rows (prep's motion-budget test: the
neighbourhood's median speed under --min-speed spacings per frame) at F of a
moving row's weight, in each keyframe's share of the budget and within it, and
gives the difference to moving rows. Trained slow rows live far longer than
moving ones (example capture results-24, 10k: under 0.05 spacings/frame ~20-560
frames vs 3-6 for fast rows), so seeding them at every keyframe is mostly
redundant, while 64% of a proportional 500k budget went to them.

Every per-row field (first dim == row count) is subset with the same indices;
metadata fields are copied unchanged. The output uses the input's compression.

Run it before chain-targets-npz.py, which links chains through the full file.
Then train with the same --config.target-count (the trainer keeps every row).

Example:
  python downsize-npz.py in.npz out.npz --target-count 500000 [--even] [--keyframe-stride 9]
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_budget as mb  # noqa: E402


def keyframe_quotas(counts: np.ndarray, budget: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-keyframe row quotas summing to ``budget``, proportional to ``weights``
    (default: the row counts), never above a keyframe's row count."""
    w = counts if weights is None else weights
    # float32 like the trainer's torch quotas, so floors land on the same rows counts
    desired = np.asarray(w).astype(np.float32) / np.float32(np.sum(w)) * np.float32(budget)
    quotas = np.minimum(np.maximum(np.floor(desired).astype(np.int64), 1), counts)
    frac = desired - np.floor(desired)
    remainder = int(budget - quotas.sum())
    if remainder > 0:
        for j in np.argsort(-frac, kind="stable"):
            if remainder <= 0:
                break
            add = min(int(counts[j] - quotas[j]), remainder)
            if add > 0:
                quotas[j] += add
                remainder -= add
    elif remainder < 0:
        for j in np.argsort(frac, kind="stable"):
            if remainder >= 0:
                break
            take = min(max(0, int(quotas[j]) - 1), -remainder)
            quotas[j] -= take
            remainder += take
    return quotas


def _voxel_keys(pos: np.ndarray, edge: float) -> np.ndarray:
    c = np.floor((pos - pos.min(axis=0)) / edge).astype(np.int64)
    return (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)


def pick_even(pos: np.ndarray, quota: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pos)
    if quota >= n:
        return np.arange(n)
    perm = rng.permutation(n)
    p = pos[perm].astype(np.float64)
    lo, hi = 1e-9, float((p.max(axis=0) - p.min(axis=0)).max()) + 1e-9
    for _ in range(40):  # occupied-cell count falls as the cell grows
        mid = 0.5 * (lo + hi)
        if len(np.unique(_voxel_keys(p, mid))) >= quota:
            lo = mid
        else:
            hi = mid
    _, first = np.unique(_voxel_keys(p, lo), return_index=True)
    if len(first) > quota:
        first = rng.choice(first, size=quota, replace=False)
    return np.sort(perm[first])


def _pick(rows: np.ndarray, q: int, pos: np.ndarray | None, even: bool, rng: np.random.Generator) -> np.ndarray:
    if q >= len(rows):
        return rows
    if q <= 0:
        return rows[:0]
    if even:
        return rows[pick_even(pos[rows], int(q), rng)]
    return np.sort(rng.choice(rows, int(q), replace=False))


def select_rows(static: np.ndarray, group: np.ndarray, pos: np.ndarray | None, target: int,
                static_budget: int, seed: int, even: bool,
                moving: np.ndarray | None = None, slow_keep: float = 1.0) -> np.ndarray:
    """Rows to keep. With ``moving`` (per row) and ``slow_keep`` < 1, a slow row
    weighs ``slow_keep`` of a moving row both in its keyframe's share of the
    budget and in the split inside the keyframe."""
    rng = np.random.default_rng(seed)
    static_idx = np.nonzero(static)[0]
    dyn_idx = np.nonzero(~static)[0]
    s_budget = min(static_budget, len(static_idx))
    d_budget = min(target - s_budget, len(dyn_idx))
    if d_budget <= 0:
        raise SystemExit(f"dynamic budget collapsed: target {target} <= static budget {s_budget}")
    static_keep = static_idx if len(static_idx) <= s_budget else np.sort(rng.choice(static_idx, s_budget, replace=False))
    keys, inverse, counts = np.unique(group[dyn_idx], return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    split = moving is not None and slow_keep < 1.0
    if split:
        mov = moving[dyn_idx][order]
        n_fast = np.add.reduceat(mov.astype(np.int64), bounds[:-1])
        n_slow = counts - n_fast
        quotas = keyframe_quotas(counts, d_budget, weights=n_fast + slow_keep * n_slow)
    else:
        quotas = keyframe_quotas(counts, d_budget)
    parts, kept_fast, kept_slow = [], 0, 0
    for j, q in enumerate(quotas):
        rows = dyn_idx[order[bounds[j]:bounds[j + 1]]]
        if not split:
            parts.append(_pick(rows, int(q), pos, even, rng))
            continue
        fast, slow = rows[moving[rows]], rows[~moving[rows]]
        w = len(fast) + slow_keep * len(slow)
        q_slow = int(round(q * slow_keep * len(slow) / w)) if w > 0 else 0
        q_fast = int(q) - q_slow
        if q_fast > len(fast):                       # not enough movers: the rest goes to slow rows
            q_slow += q_fast - len(fast); q_fast = len(fast)
        q_slow = min(q_slow, len(slow))
        parts += [_pick(fast, q_fast, pos, even, rng), _pick(slow, q_slow, pos, even, rng)]
        kept_fast += min(q_fast, len(fast)); kept_slow += q_slow
    keep = np.sort(np.concatenate([static_keep] + parts))
    print(f"static {len(static_idx):,} -> {len(static_keep):,} | dynamic {len(dyn_idx):,} -> {sum(len(p) for p in parts):,} "
          f"over {len(keys)} keyframes (per keyframe {int(quotas.min()):,}-{int(quotas.max()):,})", flush=True)
    if split:
        n_mov = int(moving[dyn_idx].sum())
        print(f"slow-keep {slow_keep:g}: moving rows {n_mov:,} -> {kept_fast:,} "
              f"(proportional would keep {round(d_budget * n_mov / len(dyn_idx)):,}) | slow rows "
              f"{len(dyn_idx) - n_mov:,} -> {kept_slow:,}", flush=True)
    return keep


def moving_rows(z, rows: np.ndarray, min_speed: float, knn: int, spacing_points: int) -> np.ndarray:
    """Prep's motion-budget test per keyframe: a row moves when its neighbourhood's
    median speed is at least ``min_speed`` x the keyframe's spacing at
    ``spacing_points`` rows (motion_budget.py)."""
    units = str(z["velocity_units"]) if "velocity_units" in z.files else "world_per_video_frame"
    per_frame = 1.0 if "world_per_video_frame" in units else 1.0 / max(int(z["keyframe_step"]), 1)
    pos = np.asarray(z["positions"])[rows].astype(np.float64)
    speed = np.linalg.norm(np.asarray(z["velocities"])[rows], axis=1) * per_frame
    has_v = (np.asarray(z["has_velocity"]).reshape(-1)[rows].astype(bool) if "has_velocity" in z.files
             else np.ones(len(rows), bool))
    frames = np.asarray(z["source_frames"]).reshape(-1)[rows]
    out = np.zeros(len(rows), bool)
    for f in np.unique(frames):
        k = np.nonzero(frames == f)[0]
        sp = mb.keyframe_spacing(pos[k], spacing_points)
        out[k] = mb.neighbourhood_speed(pos[k], speed[k], has_v[k], knn) >= min_speed * sp
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--target-count", type=int, required=True)
    ap.add_argument("--static-budget", type=int, default=100000, help="trainer default: 100000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--even", action="store_true", help="voxel-even selection per keyframe instead of random")
    ap.add_argument("--keyframe-stride", type=int, default=0,
                    help="keep dynamic rows only every N source frames from the first dynamic keyframe (0 = all)")
    ap.add_argument("--slow-keep", type=float, default=1.0,
                    help="weight of a slow dynamic row relative to a moving one (1 = off). Slow rows live "
                         "far longer once trained, so fewer seeds cover them; their share goes to moving rows")
    ap.add_argument("--min-speed", type=float, default=0.2,
                    help="moving threshold in spacings per video frame (prep --motion-budget-min-speed)")
    ap.add_argument("--knn", type=int, default=16, help="neighbourhood for the speed median (prep's default)")
    ap.add_argument("--spacing-points", type=int, default=70000,
                    help="density at which the threshold's spacing is measured (prep's default)")
    a = ap.parse_args()

    t0 = time.time()
    compressed = any(i.compress_type != zipfile.ZIP_STORED for i in zipfile.ZipFile(a.src).infolist())
    z = np.load(a.src, allow_pickle=False)
    n = int(z["positions"].shape[0])
    static = (z["is_static"].reshape(-1) > 0.5) if "is_static" in z.files else np.zeros(n, bool)
    group = z["times"].reshape(-1) if "times" in z.files else z["source_frames"].reshape(-1)
    eligible = np.ones(n, bool)
    if a.keyframe_stride > 0:
        sf = z["source_frames"].reshape(-1)
        first = int(sf[~static].min())
        eligible = static | (((sf - first) % a.keyframe_stride) == 0)
        print(f"keyframe stride {a.keyframe_stride}: {int((~static & eligible).sum()):,} of {int((~static).sum()):,} "
              f"dynamic rows on the grid ({len(np.unique(sf[~static & eligible]))} keyframes)", flush=True)
    pos = z["positions"] if a.even else None
    sub = np.nonzero(eligible)[0]
    moving = None
    if a.slow_keep < 1.0:
        moving = np.zeros(len(sub), bool)
        dyn = np.nonzero(~static[sub])[0]
        moving[dyn] = moving_rows(z, sub[dyn], a.min_speed, a.knn, a.spacing_points)
        print(f"moving rows (>= {a.min_speed:g} spacings/frame, neighbourhood median): {int(moving.sum()):,} "
              f"of {len(dyn):,} dynamic [{time.time() - t0:.0f}s]", flush=True)
    keep = sub[select_rows(static[sub], group[sub], None if pos is None else pos[sub],
                           a.target_count, a.static_budget, a.seed, a.even, moving=moving, slow_keep=a.slow_keep)]
    del pos

    out = {}
    for name in z.files:
        v = z[name]
        out[name] = v[keep] if (v.ndim >= 1 and v.shape[0] == n) else v
        tag = "rows" if out[name] is not v else "meta"
        print(f"  {name:24s} {tag} {tuple(v.shape)} -> {tuple(out[name].shape)}", flush=True)
    (np.savez_compressed if compressed else np.savez)(a.dst, **out)
    print(f"wrote {a.dst}: {n:,} -> {len(keep):,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
