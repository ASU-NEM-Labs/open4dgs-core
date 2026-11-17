"""Long-range motion targets for dynamic rows, chained across keyframes.

Prep measures each point's motion only one step forward and back (centred
velocity and acceleration at its own keyframe), so the trainer's starting
trajectories are constrained only near each row's keyframe. Measured on
example capture (2026-10-24), trained splats stay on the moving girl to about
+-9 frames and their lifetimes settle there, while the same rows can be chained
through the dense prep cloud much further (cycle-checked, 3 spacings: 55-60%
of girl points to +-21 frames in the better direction).

A chain predicts where a row is one keyframe later from its own velocity
(per video frame) and acceleration (per frame^2), ``x + v d + a d^2 / 2``,
links to the nearest dynamic row of that keyframe within ``tau`` of that
keyframe's point spacing, and continues from the linked row's own motion. With
``cycle_check`` the linked row's backward prediction must also land within
``tau`` spacings of the row it came from, which rejects links that slide onto
an unrelated surface. A chain stops at its first failed link.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ChainParams:
    window_frames: int = 21
    tau_spacings: float = 3.0
    cycle_check: bool = True
    spacing_sample: int = 20000
    workers: int = -1


def predict(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, d: float) -> np.ndarray:
    return pos + vel * d + 0.5 * acc * d * d


class KeyframeIndex:
    """KD-trees and median point spacing of each keyframe's dynamic rows, built on demand."""

    def __init__(self, pos: np.ndarray, frames: np.ndarray, dynamic: np.ndarray, params: ChainParams):
        self.pos = pos
        self.params = params
        dyn_idx = np.nonzero(dynamic)[0]
        order = np.argsort(frames[dyn_idx], kind="stable")
        keys, starts, counts = np.unique(frames[dyn_idx][order], return_index=True, return_counts=True)
        self.rows = {int(k): dyn_idx[order[s:s + c]] for k, s, c in zip(keys, starts, counts)}
        self._tree: Dict[int, cKDTree] = {}
        self._spacing: Dict[int, float] = {}

    def has(self, frame: int) -> bool:
        return frame in self.rows

    def tree(self, frame: int) -> cKDTree:
        if frame not in self._tree:
            self._tree[frame] = cKDTree(self.pos[self.rows[frame]])
        return self._tree[frame]

    def spacing(self, frame: int) -> float:
        if frame not in self._spacing:
            rows = self.rows[frame]
            sample = rows[:: max(1, len(rows) // self.params.spacing_sample)]
            d = self.tree(frame).query(self.pos[sample], k=2, workers=self.params.workers)[0][:, 1]
            self._spacing[frame] = float(np.median(d)) if len(d) else 0.0
        return self._spacing[frame]


def chain_offsets(step: int, window_frames: int) -> np.ndarray:
    n = max(int(window_frames) // int(step), 0)
    return np.concatenate([-np.arange(n, 0, -1), np.arange(1, n + 1)]).astype(np.int32) * int(step)


def build_chains(
    pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, frames: np.ndarray, dynamic: np.ndarray,
    query: np.ndarray, step: int, params: ChainParams = ChainParams(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Chained positions for rows ``query`` (indices into the dense arrays).

    Returns ``(offsets [S] frames, chain_pos [Q, S, 3])``; entries past a chain's
    first failed link are NaN. Velocities are per video frame and accelerations
    per frame^2; ``step`` is the keyframe spacing in frames.
    """
    pos = np.asarray(pos, np.float64); vel = np.asarray(vel, np.float64); acc = np.asarray(acc, np.float64)
    frames = np.asarray(frames).reshape(-1).astype(np.int64)
    index = KeyframeIndex(pos, frames, np.asarray(dynamic, bool), params)
    offsets = chain_offsets(step, params.window_frames)
    out = np.full((len(query), len(offsets), 3), np.nan, np.float32)
    col = {int(o): i for i, o in enumerate(offsets)}
    for direction in (1, -1):
        cur = np.asarray(query, np.int64).copy()
        alive = np.ones(len(cur), bool)
        for j in range(1, len(offsets) // 2 + 1):
            d = direction * step
            nxt = cur.copy()
            ok = np.zeros(len(cur), bool)
            target = frames[cur] + d
            for f in np.unique(target[alive]):
                sel = np.nonzero(alive & (target == f))[0]
                if not index.has(int(f)):
                    continue
                pred = predict(pos[cur[sel]], vel[cur[sel]], acc[cur[sel]], d)
                dist, k = index.tree(int(f)).query(pred, workers=params.workers)
                cand = index.rows[int(f)][k]
                tol = params.tau_spacings * index.spacing(int(f))
                good = dist <= tol
                if params.cycle_check:
                    back = predict(pos[cand], vel[cand], acc[cand], -d)
                    good &= np.linalg.norm(back - pos[cur[sel]], axis=1) <= tol
                nxt[sel] = cand
                ok[sel] = good
            alive &= ok
            cur = np.where(alive, nxt, cur)
            out[alive, col[direction * j * step]] = pos[cur[alive]]
            if not alive.any():
                break
    return offsets, out


def contiguous_span(valid: np.ndarray, offsets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per row, the furthest forward and backward offset (frames) reached without a gap."""
    offsets = np.asarray(offsets)
    fwd = np.zeros(len(valid), np.int64); bwd = np.zeros(len(valid), np.int64)
    alive_f = np.ones(len(valid), bool); alive_b = np.ones(len(valid), bool)
    for o in sorted(int(x) for x in offsets if x > 0):
        alive_f &= valid[:, int(np.nonzero(offsets == o)[0][0])]
        fwd = np.where(alive_f, o, fwd)
    for o in sorted((int(x) for x in offsets if x < 0), reverse=True):
        alive_b &= valid[:, int(np.nonzero(offsets == o)[0][0])]
        bwd = np.where(alive_b, -o, bwd)
    return fwd, bwd
