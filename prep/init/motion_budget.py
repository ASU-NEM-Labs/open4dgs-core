"""Finalize-time motion-aware point cap per keyframe: never thin moving areas.

Each keyframe's still rows are thinned to ``still_per_frame`` (prep's
--target-points-per-frame) by priority voxel dedup, as the per-frame cap always
did, while its moving rows are all kept (or capped separately by
``moving_per_frame``; 0 = no cap). Keyframes with motion therefore exceed the
cap, and the trainer's --target-count keeps every keyframe's rows in
proportion, so those frames also get a larger share of the training budget.

Measured on another capture (2026-10-19): with training per frame held fixed, the
dancer's render sharpness rose with splats per frame (1.8k -> 0.62,
5.3k -> 0.70, 16k -> 0.81, the single-frame ceiling).

A row moves when the median speed of its neighbourhood (itself plus ``knn``
nearest rows of the same keyframe, tracked rows only) is at least
``min_speed_spacings`` point spacings per video frame. The neighbourhood median
keeps one bad track from flipping a row's class, and lets untracked rows on a
moving part (often blurred limbs) count as moving. The spacing is the one the
keyframe would have at ``spacing_points`` rows (default 70k, the usual
per-frame cap), independent of the still cap, so changing the cap never moves
the speed threshold. Tracking noise there measured p99 0.04 (example capture,
60 fps) to ~0.15 (another capture calm keyframes, 30 fps) spacings per frame.

Static rows (``is_static``) are left untouched.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

# thin_fn(positions, colors, priorities, target) -> local indices to keep
ThinFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class MotionBudgetParams:
    moving_per_frame: int = 0           # 0 = keep every moving row
    still_per_frame: int = 70000        # 0 = keep every still row
    min_speed_spacings: float = 0.2
    knn: int = 16
    seed: int = 0
    # Density at which the spacing in min_speed_spacings is measured. Fixed, not
    # the still cap, so lowering the cap does not raise the speed threshold.
    spacing_points: int = 70000


def keyframe_spacing(positions: np.ndarray, n_points: int) -> float:
    """Point spacing the keyframe would have if thinned evenly to n_points rows.

    Median nearest-neighbour distance of all rows, scaled by sqrt(rows / n_points):
    the clouds are surfaces, so spacing grows with the square root of thinning.
    (A random subsample would not do: it leaves clumps whose median gap stays
    at the original spacing.)
    """
    n = len(positions)
    if n < 2:
        return 0.0
    d = cKDTree(positions).query(positions, k=2)[0][:, 1]
    return float(np.median(d)) * float(np.sqrt(max(1.0, n / max(int(n_points), 1))))


def neighbourhood_speed(
    positions: np.ndarray, speed: np.ndarray, has_velocity: np.ndarray, knn: int,
) -> np.ndarray:
    """Median speed of each row's neighbourhood (self + knn), over tracked rows only.

    Rows whose whole neighbourhood is untracked get 0.
    """
    n = len(positions)
    if n == 0:
        return np.zeros(0, np.float64)
    k = int(min(knn + 1, n))
    nn = cKDTree(positions).query(positions, k=k)[1].reshape(n, k)
    s = np.where(has_velocity, speed, np.nan)[nn]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN neighbourhoods
        med = np.nanmedian(s, axis=1)
    return np.nan_to_num(med, nan=0.0)


def motion_budget_select(
    *,
    positions: np.ndarray,
    colors: np.ndarray,
    priorities: np.ndarray,
    speed_per_frame: np.ndarray,
    has_velocity: np.ndarray,
    source_frames: np.ndarray,
    is_static: np.ndarray,
    params: MotionBudgetParams,
    thin_fn: ThinFn,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return sorted indices of the rows to keep, and per-keyframe stats."""
    positions = np.asarray(positions, np.float64)
    source_frames = np.asarray(source_frames).reshape(-1)
    is_static = np.asarray(is_static, bool).reshape(-1)
    has_velocity = np.asarray(has_velocity, bool).reshape(-1)
    speed_per_frame = np.asarray(speed_per_frame, np.float64).reshape(-1)
    rng = np.random.default_rng(int(params.seed))

    keep = [np.nonzero(is_static)[0]]
    per_kf = []
    dyn = ~is_static
    for kf in np.unique(source_frames[dyn]):
        rows = np.nonzero(dyn & (source_frames == kf))[0]
        pos = positions[rows]
        spacing = keyframe_spacing(pos, int(params.spacing_points) or len(rows))
        med = neighbourhood_speed(pos, speed_per_frame[rows], has_velocity[rows], params.knn)
        moving = med >= float(params.min_speed_spacings) * spacing if spacing > 0 else np.zeros(len(rows), bool)
        kept_kf = []
        for mask, quota in ((moving, int(params.moving_per_frame)), (~moving, int(params.still_per_frame))):
            cls = rows[mask]
            if len(cls) == 0:
                continue
            if quota <= 0 or len(cls) <= quota:                # 0 = no cap
                kept_kf.append(cls)
                continue
            sel = cls[np.asarray(thin_fn(positions[cls], colors[cls], priorities[cls], quota), np.int64)]
            if len(sel) > quota:
                # Voxel thinning stops at its max voxel (e.g. near-planar clouds);
                # enforce the quota, keeping higher-priority rows first.
                order = np.lexsort((rng.random(len(sel)), -np.asarray(priorities)[sel]))
                sel = sel[order[:quota]]
            kept_kf.append(sel)
        kept_kf = np.concatenate(kept_kf) if kept_kf else np.zeros(0, np.int64)
        keep.append(kept_kf)
        n_mov_kept = int(np.isin(kept_kf, rows[moving]).sum())
        per_kf.append({
            "keyframe": int(kf), "rows_in": int(len(rows)), "moving_in": int(moving.sum()),
            "moving_kept": n_mov_kept, "still_kept": int(len(kept_kf) - n_mov_kept),
            "spacing": spacing,
        })
    keep_idx = np.sort(np.concatenate(keep).astype(np.int64))
    kept = np.array([p["moving_kept"] + p["still_kept"] for p in per_kf]) if per_kf else np.zeros(1)
    stats = {
        "rows_in": int(len(positions)), "rows_out": int(len(keep_idx)),
        "static_rows": int(is_static.sum()), "keyframes": len(per_kf),
        "kept_per_keyframe_min_p50_max": [int(kept.min()), int(np.median(kept)), int(kept.max())],
        "per_keyframe": per_kf,
    }
    return keep_idx, stats
