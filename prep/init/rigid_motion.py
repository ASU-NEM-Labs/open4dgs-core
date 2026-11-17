"""Piecewise-rigid motion repair for tracked keyframe points (``--rigid-cluster-motion``).

Per-pixel temporal matching (RoMa or optical flow) can fail on fast, blurred parts in a
way that is consistent across views: a track that sticks to the background or stays put
triangulates cleanly, so multi-view and reprojection gates accept it. Measured on
xr-lab-02 (fast forearm reaching a stop), ~25% of all-camera flow tracks were wrong like
this, while a rigid fit to the limb's points recovered the motion to ~1.3 prep spacings.

This pass works on the assembled keyframe rows. For each keyframe k and each neighbour
keyframe k +/- d (d = temporal delta = keyframe step), in small spatial clusters:

  1. Surface gate: a tracked endpoint must land within ``gate_spacings`` of a point
     reconstructed at the neighbour keyframe (a stuck track left where the part used
     to be lands in empty space).
  2. RANSAC rigid fit (Kabsch) to the gated tracks, then trimmed ICP (xyz + colour)
     against the neighbour keyframe's points. Fits that barely move are skipped, so a
     static consensus never overwrites real motion.
  3. Every row of the cluster whose endpoint is missing or disagrees with the fit gets
     the fitted endpoint; consistent tracks are kept as measured.

Velocities and accelerations are then rebuilt from the endpoints exactly as the
centred-velocity tracker does: v = (E+ - E-) / 2d, a = (E+ - 2X + E-) / d^2.

The only modelling assumption is local rigidity over one keyframe step, the same one
the TRiGS motion model makes per primitive. Non-rigid clusters (cloth, hair) fail the
inlier test and keep their per-point tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

TRACK_FWD = np.uint8(1)     # endpoint at k + d measured by the tracker
TRACK_BWD = np.uint8(2)     # endpoint at k - d measured by the tracker
RIGID_FWD = np.uint8(4)     # endpoint at k + d replaced by the cluster's rigid fit
RIGID_BWD = np.uint8(8)     # endpoint at k - d replaced by the cluster's rigid fit


@dataclass
class RigidMotionParams:
    cluster_size: float = 0.0            # world units; 0 = cluster_size_spacings x spacing
    cluster_size_spacings: float = 10.0
    min_cluster_points: int = 30
    min_tracks: int = 12
    gate_spacings: float = 3.0
    inlier_spacings: float = 2.0
    min_inlier_frac: float = 0.3
    min_motion_spacings: float = 1.0
    ransac_iters: int = 256
    icp_iters: int = 20
    icp_keep: float = 0.7
    colour_spacings_per_0p1: float = 1.0  # a colour difference of 0.1 counts as this many spacings
    fill_hops: int = 3                     # surface-growth steps (<= 2 spacings each) for leftover rows
    seed: int = 0
    # When a row's fit is rejected by the local-support vote (its neighbours agree more with the
    # row's current motion than with the fit), keep that current endpoint for the velocity rebuild
    # instead of rebuilding the row from its other endpoint alone. Off = the original behaviour.
    keep_inherited_on_vote: bool = False


@dataclass
class RigidMotionStats:
    keyframes: int = 0
    clusters: int = 0
    fits: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    skipped_static: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    skipped_low_inliers: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    gated_out: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    replaced: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    filled_untracked: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    kept_on_vote: Dict[str, int] = field(default_factory=lambda: {"fwd": 0, "bwd": 0})
    rows_changed: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "keyframes": self.keyframes, "clusters": self.clusters, "fits": self.fits,
            "skipped_static": self.skipped_static, "skipped_low_inliers": self.skipped_low_inliers,
            "gated_out": self.gated_out, "replaced": self.replaced,
            "filled_untracked": self.filled_untracked, "kept_on_vote": self.kept_on_vote,
            "rows_changed": self.rows_changed,
        }


def kabsch(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid R, t minimising |R a + t - b| (proper rotation)."""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def kabsch_batch(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Kabsch: a, b [N, m, 3] -> R [N, 3, 3], t [N, 3]."""
    ca, cb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
    h = np.einsum("nmi,nmj->nij", a - ca, b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", vt, u)))
    dm = np.zeros((len(a), 3, 3)); dm[:, 0, 0] = 1; dm[:, 1, 1] = 1; dm[:, 2, 2] = d
    r = np.einsum("nji,njk,nlk->nil", vt, dm, u)
    t = cb[:, 0] - np.einsum("nij,nj->ni", r, ca[:, 0])
    return r, t


def median_spacing(points: np.ndarray, cap: int = 20000, seed: int = 0) -> float:
    if len(points) < 2:
        return 0.0
    pts = points
    if len(pts) > cap:
        pts = pts[np.random.default_rng(seed).choice(len(pts), cap, replace=False)]
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


def _fit_cluster(
    x: np.ndarray, e: np.ndarray, xcol: np.ndarray,
    target_tree: cKDTree, target_xyzc: np.ndarray,
    spacing: float, colour_w: float, p: RigidMotionParams, rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """RANSAC rigid fit of x -> e (gated tracks), then trimmed ICP to the target cloud."""
    n = len(x)
    inlier_thr = p.inlier_spacings * spacing
    if n < max(3, p.min_tracks):
        return None, None, "few"
    idx = np.stack([rng.choice(n, 3, replace=False) for _ in range(p.ransac_iters)])
    r_h, t_h = kabsch_batch(x[idx], e[idx])
    res = np.linalg.norm(np.einsum("hij,nj->hni", r_h, x) + t_h[:, None, :] - e[None], axis=2)
    counts = (res < inlier_thr).sum(1)
    best = int(np.argmax(counts))
    inl = res[best] < inlier_thr
    if inl.sum() < max(3, p.min_inlier_frac * n):
        return None, None, "low_inliers"
    r, t = kabsch(x[inl], e[inl])
    if float(np.median(np.linalg.norm(x @ r.T + t - x, axis=1))) < p.min_motion_spacings * spacing:
        return None, None, "static"
    # Trimmed ICP against the neighbour keyframe's points (geometry + colour).
    r_icp, t_icp = r, t
    q = np.c_[x, xcol * colour_w]
    rms = np.inf
    for _ in range(int(p.icp_iters)):
        xp = x @ r_icp.T + t_icp
        d, j = target_tree.query(np.c_[xp, q[:, 3:]])
        keep = d <= np.quantile(d, p.icp_keep)
        if keep.sum() < 3:
            break
        r_icp, t_icp = kabsch(x[keep], target_xyzc[j[keep], :3])
        rms = float(np.sqrt(np.mean(np.sort(d)[: max(1, int(p.icp_keep * len(d)))] ** 2)))
    moved = float(np.median(np.linalg.norm(x @ r_icp.T + t_icp - x, axis=1)))
    if np.isfinite(rms) and rms <= inlier_thr and moved >= p.min_motion_spacings * spacing:
        return r_icp, t_icp, "ok"
    return r, t, "ok"


def rigid_cluster_motion(
    *,
    positions: np.ndarray,          # [N,3]
    colors: np.ndarray,             # [N,3] in 0..1
    velocities: np.ndarray,         # [N,3] world units per frame
    accelerations: np.ndarray,      # [N,3] world units per frame^2
    has_acceleration: np.ndarray,   # [N] bool
    track_flags: np.ndarray,        # [N] uint8 (TRACK_FWD | TRACK_BWD)
    source_frames: np.ndarray,      # [N] keyframe of each row
    is_static: np.ndarray,          # [N] bool
    frame_delta: int,
    params: RigidMotionParams = RigidMotionParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RigidMotionStats]:
    """Returns (velocities, accelerations, has_acceleration, track_flags, changed_mask, stats)."""
    p = params
    d = float(frame_delta)
    pos = np.asarray(positions, np.float64)
    col = np.asarray(colors, np.float64)
    vel = np.array(velocities, np.float64, copy=True)
    acc = np.array(accelerations, np.float64, copy=True)
    has_acc = np.array(has_acceleration, bool, copy=True)
    flags = np.array(track_flags, np.uint8, copy=True)
    sf = np.asarray(source_frames).reshape(-1).astype(np.int64)
    dyn = ~np.asarray(is_static, bool).reshape(-1)
    changed = np.zeros(len(pos), bool)
    stats = RigidMotionStats()
    rng = np.random.default_rng(p.seed)
    keyframes = np.unique(sf)
    kf_set = set(int(k) for k in keyframes)
    rows_of = {int(k): np.nonzero(sf == k)[0] for k in keyframes}

    for k in keyframes:
        k = int(k)
        rows = rows_of[k]
        rows = rows[dyn[rows]]
        if len(rows) < p.min_cluster_points:
            continue
        spacing = median_spacing(pos[rows], seed=p.seed)
        if spacing <= 0:
            continue
        colour_w = p.colour_spacings_per_0p1 * spacing / 0.1
        size = p.cluster_size if p.cluster_size > 0 else p.cluster_size_spacings * spacing
        x_all = pos[rows]
        e_fwd = x_all + d * vel[rows] + 0.5 * acc[rows] * d * d
        e_bwd = x_all - d * vel[rows] + 0.5 * acc[rows] * d * d
        f = flags[rows]
        have = {"fwd": (f & TRACK_FWD) > 0, "bwd": (f & TRACK_BWD) > 0}
        ends = {"fwd": e_fwd.copy(), "bwd": e_bwd.copy()}
        avail = {"fwd": have["fwd"].copy(), "bwd": have["bwd"].copy()}
        lost_vote = {"fwd": np.zeros(len(rows), bool), "bwd": np.zeros(len(rows), bool)}
        targets = {}
        for name, nb in (("fwd", k + frame_delta), ("bwd", k - frame_delta)):
            if nb in kf_set:
                t_rows = rows_of[nb]
                xyzc = np.c_[pos[t_rows], col[t_rows] * colour_w]
                targets[name] = (cKDTree(pos[t_rows]), cKDTree(xyzc), xyzc)
        if not targets:
            continue
        stats.keyframes += 1
        fitted = {"fwd": [], "bwd": []}      # (member row indices into `rows`, R, t)

        # Local evidence: the measured tracks of each row's neighbours (as tracked,
        # before any replacement), counting only neighbours whose own endpoint lands on
        # the neighbour keyframe's surface. A row takes a fit only if more of them
        # agree with the fit than with the row's current motion. A stuck point on a
        # moving part is surrounded by points that move with the fit (its stuck
        # neighbours mostly land in empty space and do not vote); a still surface next
        # to it is surrounded by points that stay put -- even when the moving part's
        # motion would carry the still surface onto itself (a torso behind an arm
        # swinging parallel to it), which the surface gate alone cannot catch.
        local = {}
        for name in targets:
            tr = np.nonzero(have[name])[0]
            if len(tr) == 0:
                local[name] = None
                continue
            on = targets[name][0].query(ends[name][tr])[0] < p.gate_spacings * spacing
            local[name] = (cKDTree(x_all[tr]), tr, ends[name][tr].copy(), on)

        def fit_supported(name: str, sel: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
            info = local[name]
            if info is None or len(sel) == 0:
                return np.ones(len(sel), bool)
            tree_tr, tr, e_tr, on_tr = info
            kq = min(9, len(tr))
            dist, j = tree_tr.query(x_all[sel], k=kq)
            dist, j = dist.reshape(len(sel), kq), j.reshape(len(sel), kq)
            vote = (dist <= 3.0 * spacing) & (tr[j] != sel[:, None]) & on_tr[j]
            thr = p.inlier_spacings * spacing
            x_n = x_all[tr[j]]
            for_fit = np.linalg.norm(e_tr[j] - (x_n @ r.T + t), axis=2) <= thr
            own_disp = (ends[name][sel] - x_all[sel])[:, None, :]       # the row's current motion
            for_own = np.linalg.norm((e_tr[j] - x_n) - own_disp, axis=2) <= thr
            a = (for_fit & vote).sum(1)
            b = (for_own & ~for_fit & vote).sum(1)
            return a >= b
        keys = np.floor(x_all / size).astype(np.int64)
        _, cluster_id = np.unique(keys, axis=0, return_inverse=True)
        cluster_id = cluster_id.reshape(-1)
        order = np.argsort(cluster_id, kind="stable")
        bounds = np.flatnonzero(np.diff(cluster_id[order])) + 1
        for members in np.split(order, bounds):
            if len(members) < p.min_cluster_points:
                continue
            stats.clusters += 1
            for name, (surf_tree, tree6, xyzc) in targets.items():
                tracked = members[have[name][members]]
                if len(tracked) < p.min_tracks:
                    continue
                e = ends[name][tracked]
                gate = surf_tree.query(e)[0] < p.gate_spacings * spacing
                stats.gated_out[name] += int((~gate).sum())
                moving = np.linalg.norm(e[gate] - x_all[tracked][gate], axis=1) >= p.min_motion_spacings * spacing
                if int(moving.sum()) < p.min_tracks:
                    continue
                r, t, why = _fit_cluster(
                    x_all[tracked][gate], e[gate], col[rows][tracked][gate],
                    tree6, xyzc, spacing, colour_w, p, rng,
                )
                if r is None:
                    if why == "static":
                        stats.skipped_static[name] += 1
                    elif why == "low_inliers":
                        stats.skipped_low_inliers[name] += 1
                    continue
                stats.fits[name] += 1
                fitted[name].append((members, r, t))
                pred = x_all[members] @ r.T + t
                cur = ends[name][members]
                # Keep a measured endpoint if it agrees with the fit and lands on the
                # neighbour keyframe's surface. Otherwise take the fit -- but only where
                # the fit itself lands on that surface: a cluster can straddle a moving
                # part and a still one, and a still point carried by the moving part's
                # motion ends up in empty space, so its own endpoint is left alone.
                keep = (
                    avail[name][members]
                    & (np.linalg.norm(cur - pred, axis=1) <= p.inlier_spacings * spacing)
                    & (surf_tree.query(cur)[0] < p.gate_spacings * spacing)
                )
                pred_on = surf_tree.query(pred)[0] < p.gate_spacings * spacing
                replace = ~keep & pred_on
                if replace.any():
                    sup = fit_supported(name, members[replace], r, t)
                    lost_vote[name][members[replace][~sup]] = True
                    replace[replace] = sup
                stats.replaced[name] += int((replace & avail[name][members]).sum())
                stats.filled_untracked[name] += int((replace & ~avail[name][members]).sum())
                tgt = members[replace]
                ends[name][tgt] = pred[replace]
                avail[name][tgt] = True
                flags[rows[tgt]] |= RIGID_FWD if name == "fwd" else RIGID_BWD
                changed[rows[tgt]] = True

        # Rows touching a fitted cluster (within 2 spacings of one of its points) but
        # left out of it -- voxel boundaries, clusters too small to fit -- are judged
        # by that cluster's fit with the same rule as its members: a measured endpoint
        # is kept only if it agrees with the fit and lands on the neighbour surface;
        # otherwise the row takes the fit. Growth continues outward along the
        # contiguous surface for a few hops of <= 2 spacings.
        for name, fits in fitted.items():
            if not fits:
                continue
            surf_tree = targets[name][0]
            rigid_bit = RIGID_FWD if name == "fwd" else RIGID_BWD
            member_idx = np.concatenate([m for m, _, _ in fits])
            fit_of = np.concatenate([np.full(len(m), i) for i, (m, _, _) in enumerate(fits)])
            in_fit = np.zeros(len(rows), bool)
            in_fit[member_idx] = True
            candidate = ~in_fit
            for _ in range(max(1, int(p.fill_hops))):
                left = np.nonzero(candidate)[0]
                if len(left) == 0:
                    break
                dist, j = cKDTree(x_all[member_idx]).query(x_all[left])
                near = dist <= 2.0 * spacing
                if not near.any():
                    break
                new_members, new_fit = [], []
                for i_fit in np.unique(fit_of[j[near]]):
                    sel = left[near][fit_of[j[near]] == i_fit]
                    _, r, t = fits[int(i_fit)]
                    pred = x_all[sel] @ r.T + t
                    cur = ends[name][sel]
                    keep = (
                        avail[name][sel]
                        & (np.linalg.norm(cur - pred, axis=1) <= p.inlier_spacings * spacing)
                        & (surf_tree.query(cur)[0] < p.gate_spacings * spacing)
                    )
                    pred_on = surf_tree.query(pred)[0] < p.gate_spacings * spacing
                    take = ~keep & pred_on
                    if take.any():
                        sup = fit_supported(name, sel[take], r, t)
                        lost_vote[name][sel[take][~sup]] = True
                        take[take] = sup
                    rep = sel[take]
                    stats.replaced[name] += int((avail[name][rep]).sum())
                    stats.filled_untracked[name] += int((~avail[name][rep]).sum())
                    ends[name][rep] = pred[take]
                    avail[name][rep] = True
                    flags[rows[rep]] |= rigid_bit
                    changed[rows[rep]] = True
                    candidate[sel] = False
                    # Only rows that move with this fit (kept as consistent, or taken)
                    # carry the growth further; a still neighbour stops it.
                    grow = sel[keep | take]
                    new_members.append(grow); new_fit.append(np.full(len(grow), i_fit))
                member_idx = np.concatenate([member_idx] + new_members)
                fit_of = np.concatenate([fit_of] + new_fit)

        if p.keep_inherited_on_vote:
            # The vote favoured the row's current endpoint over the fit: keep it for the rebuild
            # (ends[name] still holds it) rather than extrapolating from the other endpoint.
            for name in ("fwd", "bwd"):
                kept = lost_vote[name] & ~avail[name]
                stats.kept_on_vote[name] += int(kept.sum())
                avail[name] |= kept

        # Rebuild velocity / acceleration for changed rows from their endpoints.
        ch = changed[rows]
        if not ch.any():
            continue
        both = ch & avail["fwd"] & avail["bwd"]
        only_f = ch & avail["fwd"] & ~avail["bwd"]
        only_b = ch & avail["bwd"] & ~avail["fwd"]
        x = x_all
        r_idx = rows[both]
        vel[r_idx] = (ends["fwd"][both] - ends["bwd"][both]) / (2.0 * d)
        acc[r_idx] = (ends["fwd"][both] - 2.0 * x[both] + ends["bwd"][both]) / (d * d)
        has_acc[r_idx] = True
        r_idx = rows[only_f]
        vel[r_idx] = (ends["fwd"][only_f] - x[only_f]) / d
        acc[r_idx] = 0.0
        has_acc[r_idx] = False
        r_idx = rows[only_b]
        vel[r_idx] = (x[only_b] - ends["bwd"][only_b]) / d
        acc[r_idx] = 0.0
        has_acc[r_idx] = False

    stats.rows_changed = int(changed.sum())
    return (
        vel.astype(np.float32), acc.astype(np.float32), has_acc, flags, changed, stats,
    )


def track_flags_for_rows(
    row_positions: np.ndarray,
    tracked_positions: np.ndarray,
    tracked_has_accel: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """TRACK_FWD for rows that are tracker outputs (exact X(t) match), plus TRACK_BWD
    where the backward track also succeeded (has_acceleration)."""
    n = len(row_positions)
    flags = np.zeros((n,), np.uint8)
    if n == 0 or len(tracked_positions) == 0:
        return flags
    dist, idx = cKDTree(np.asarray(tracked_positions, np.float64)).query(
        np.asarray(row_positions, np.float64), k=1
    )
    hit = dist <= tol
    flags[hit] |= TRACK_FWD
    bwd = np.zeros((n,), bool)
    bwd[hit] = np.asarray(tracked_has_accel, bool)[idx[hit]]
    flags[bwd] |= TRACK_BWD
    return flags
