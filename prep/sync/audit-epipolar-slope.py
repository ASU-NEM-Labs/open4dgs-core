#!/usr/bin/env python3
"""Audit cross-camera SPATIAL calibration and TEMPORAL skew from one signal.

Why this exists
---------------
`solve_camera_offsets` (prep/init/prep-roma-trigs-memory.py) is circular: it
needs triangulated points and 3D velocities to measure timing, but you need
timing to triangulate moving points. It is therefore weakest exactly where the
answer matters (fast motion) and silently confident where there is no signal
at all (low motion).

This audit removes the circularity. For a matched pair of pixels between
cameras A and B at nominal frame f, RoMa matches the same *surface point*, but
A imaged it at t + tau_A and B at t + tau_B. So B sees it displaced by
V * (tau_B - tau_A). Projected into B's image that displacement is
v_B * dtau, and the component perpendicular to the epipolar line is exactly
the epipolar residual:

    d_signed(match) ~= dtau_frames * v_perp(match)                        (1)

where v_perp is B's own image-space velocity (px/frame) projected onto the
epipolar-line normal. No triangulation. No 3D velocity. One robust line fit
per camera pair gives:

    slope     -> dtau_AB = tau_B - tau_A, in frames  (TEMPORAL skew)
    intercept -> systematic epipolar bias            (SPATIAL calib bias)
    residual scatter at v_perp ~= 0 -> calibration noise floor

The fundamental matrix comes from the REFERENCE CALIBRATION, never from
RANSAC on the matches. Fitting F to the matches would absorb the very signal
being measured, which is also why the pipeline's pair RANSAC discards fast
matches as outliers instead of reporting them.

What it reports
---------------
1. STATIC epipolar residual per camera / per frame. Static content cannot be
   affected by timing, so this isolates spatial calibration. A trend across
   frames is rig drift; a step is a bumped stand; a stripe on one camera is
   that camera's calibration.
2. TEMPORAL skew per pair -> graph solve -> per-camera offsets (seconds,
   mean-zero, PLUS convention), directly writable as sync.json.
3. MEASURABILITY per pair. A pair with no perpendicular image motion is
   reported as unmeasured rather than given a fabricated number.
4. ALIASING rate: forward/reverse warp round-trip failures among fast matches.
   Repeating structure (guitar frets, ladders, brick) matched across a
   sync-skewed pair aliases to the wrong repeat; those matches are confidently
   wrong and survive epipolar gating when the structure runs along the
   epipolar line. High rates here mean the geometry is not the whole story.

Usage
-----
    python prep/sync/audit-epipolar-slope.py \
        --reference-json /path/to/reference_export.json \
        --images-root /path/to/images \
        --image-pattern '{frame:06d}.png' \
        --frames 120,124,128,132 \
        --fps 60 \
        --out-dir /tmp/sync_audit

Pick --frames ON the fast motion you care about (the strum, the swing). The
audit needs image motion to measure timing; static frames only exercise
report (1).

Add --write-sync-json PATH to emit a PLUS-convention sync.json from the solve
(only cameras that were actually constrained get non-zero values).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Import the (hyphenated) prep module so Camera/RoMa/geometry code is shared
# ---------------------------------------------------------------------------

# Loaded lazily: it pulls in torch/cv2/RoMa, which the pure-geometry helpers
# below (and their synthetic tests) do not need.
_PREP = None


def prep():
    """Import prep-roma-trigs-memory.py, reusing its Camera/RoMa/IO code."""
    global _PREP
    if _PREP is not None:
        return _PREP
    init_dir = Path(__file__).resolve().parent.parent / "init"
    target = init_dir / "prep-roma-trigs-memory.py"
    if not target.exists():
        raise FileNotFoundError(f"cannot locate prep module at {target}")
    # roma_backends.py sits beside it and is imported by bare name.
    if str(init_dir) not in sys.path:
        sys.path.insert(0, str(init_dir))
    spec = importlib.util.spec_from_file_location("prep_roma_trigs_memory", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PREP = mod
    return mod


# ---------------------------------------------------------------------------
# Geometry: F from calibration (undistorted pixel coordinates)
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def relative_pose(cam_a, cam_b) -> Tuple[np.ndarray, np.ndarray]:
    """A-frame -> B-frame, given both cameras' world->camera (R, t)."""
    R_rel = cam_b.R @ cam_a.R.T
    t_rel = cam_b.t - R_rel @ cam_a.t
    return R_rel, t_rel


def fundamental_from_calibration(cam_a, cam_b) -> np.ndarray:
    """F such that x_b^T F x_a = 0 for UNDISTORTED pixel coordinates."""
    R_rel, t_rel = relative_pose(cam_a, cam_b)
    E = skew(t_rel) @ R_rel
    Ka_inv = np.linalg.inv(cam_a._K)
    Kb_inv_T = np.linalg.inv(cam_b._K).T
    return Kb_inv_T @ E @ Ka_inv


def to_undistorted_px(cam, uv: np.ndarray) -> np.ndarray:
    """Distorted image pixels -> undistorted pixels using the camera model."""
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) == 0:
        return uv.reshape(-1, 2)
    normalized = cam.unproject_image_points(uv)  # [N, 2], z == 1
    homo = np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)
    projected = homo @ cam._K.T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)


def pair_geometry(cam_a, cam_b, scene_center: np.ndarray) -> Tuple[float, float]:
    """Baseline (world units) and the angle the two cameras subtend at the
    scene center, in degrees.

    Near-opposing pairs (large angle) see largely DIFFERENT surfaces, so a
    dense matcher returns confident correspondences that are not the same
    physical point. Their epipolar residuals are matcher failure, not
    calibration error, and must not be pooled with near-neighbour pairs.
    """
    Ca, Cb = cam_a.center_world, cam_b.center_world
    baseline = float(np.linalg.norm(Ca - Cb))
    va, vb = scene_center - Ca, scene_center - Cb
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return baseline, 0.0
    cos = float(np.clip(va @ vb / (na * nb), -1.0, 1.0))
    return baseline, float(np.degrees(np.arccos(cos)))


def signed_epipolar_and_normal(
    F: np.ndarray, pa_u: np.ndarray, pb_u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed distance from pb_u to its epipolar line in B, plus the unit
    normal of that line. Sign convention matches equation (1) above."""
    a_h = np.concatenate([pa_u, np.ones((len(pa_u), 1))], axis=1)
    b_h = np.concatenate([pb_u, np.ones((len(pb_u), 1))], axis=1)
    lines_b = a_h @ F.T                                      # [N, 3]
    norm = np.linalg.norm(lines_b[:, :2], axis=1)
    norm = np.maximum(norm, 1e-12)
    d_signed = np.sum(b_h * lines_b, axis=1) / norm
    normals = lines_b[:, :2] / norm[:, None]
    return d_signed, normals


# ---------------------------------------------------------------------------
# Robust line fit: RANSAC on slope, then least squares on the inliers
# ---------------------------------------------------------------------------

def robust_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    band_px: float,
    iters: int = 2000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Fit y = slope * x + intercept, robust to a high outlier fraction.

    Aliased matches are confidently wrong, so they form structured outlier
    clusters that reweighting schemes latch onto. Consensus counting does not.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 30:
        return None

    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1

    # Sample pairs with enough x-separation to define a stable slope.
    x_span = float(np.percentile(x, 95) - np.percentile(x, 5))
    min_sep = max(0.25 * x_span, 1e-6)

    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if abs(x[i] - x[j]) < min_sep:
            continue
        slope = (y[j] - y[i]) / (x[j] - x[i])
        intercept = y[i] - slope * x[i]
        inliers = np.abs(y - (slope * x + intercept)) <= band_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 30:
        return None

    xi, yi = x[best_inliers], y[best_inliers]
    A = np.stack([xi, np.ones_like(xi)], axis=1)
    coef, *_ = np.linalg.lstsq(A, yi, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])

    residuals = yi - (slope * xi + intercept)
    dof = max(len(xi) - 2, 1)
    sigma2 = float(residuals @ residuals) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        slope_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        slope_se = float("inf")

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "n_inliers": int(len(xi)),
        "n_total": int(n),
        "inlier_frac": float(len(xi) / n),
        "resid_rms_px": float(np.sqrt(sigma2)),
    }


# ---------------------------------------------------------------------------
# Graph solve: pairwise dtau_ab = tau_b - tau_a  ->  per-camera tau
# ---------------------------------------------------------------------------

def solve_offsets_from_pairs(
    measurements: Sequence[Tuple[int, int, float, float]],
    n_cams: int,
    ridge: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted least squares on the difference graph, gauge-fixed mean-zero.

    measurements: (cam_a, cam_b, dtau_frames, weight)
    Returns (tau_frames, constrained_mask).
    """
    L = np.zeros((n_cams, n_cams), dtype=np.float64)
    rhs = np.zeros(n_cams, dtype=np.float64)
    degree = np.zeros(n_cams, dtype=np.int64)

    for a, b, dtau, w in measurements:
        L[a, a] += w
        L[b, b] += w
        L[a, b] -= w
        L[b, a] -= w
        rhs[a] -= w * dtau
        rhs[b] += w * dtau
        degree[a] += 1
        degree[b] += 1

    constrained = degree > 0
    tau = np.zeros(n_cams, dtype=np.float64)
    if not constrained.any():
        return tau, constrained

    idx = np.flatnonzero(constrained)
    L_sub = L[np.ix_(idx, idx)]
    rhs_sub = rhs[idx]
    # Gauge fix: pin the mean of the constrained set to zero.
    L_sub = L_sub + ridge * np.trace(L_sub) / len(idx) * np.eye(len(idx))
    L_sub = L_sub + np.ones((len(idx), len(idx))) / len(idx)
    try:
        tau_sub = np.linalg.solve(L_sub, rhs_sub)
    except np.linalg.LinAlgError:
        tau_sub = np.linalg.lstsq(L_sub, rhs_sub, rcond=None)[0]
    tau_sub -= tau_sub.mean()
    tau[idx] = tau_sub
    return tau, constrained


# ---------------------------------------------------------------------------
# Per-pair measurement
# ---------------------------------------------------------------------------

def run_frame(
    model,
    cameras: Sequence,
    pairs: Sequence[Tuple[int, int]],
    images_root: Path,
    frame: int,
    scene_center: np.ndarray,
    args,
) -> Dict[str, object]:
    fps = float(args.fps)
    delta = int(args.temporal_frame_delta)

    paths, images = prep().load_frame_images(
        images_root, cameras, frame, args.image_pattern
    )
    paths_next, images_next = prep().load_frame_images(
        images_root, cameras, frame + delta, args.image_pattern
    )

    # Temporal warps t -> t+delta, one per camera, reused by every pair.
    temporal_warps: Dict[int, object] = {}
    for ci in range(len(cameras)):
        try:
            temporal_warps[ci] = prep().build_roma_warp(
                model, paths[ci], paths_next[ci],
                images[ci], images_next[ci], args.device,
            )
        except Exception as exc:
            print(f"  [warn] temporal warp failed cam {cameras[ci].folder}: {exc}")
            temporal_warps[ci] = None

    pair_records: List[Dict[str, object]] = []

    for (ia, ib) in pairs:
        cam_a, cam_b = cameras[ia], cameras[ib]
        label = f"{cam_a.folder}<->{cam_b.folder}"
        baseline_m, tri_angle_deg = pair_geometry(cam_a, cam_b, scene_center)
        if tri_angle_deg > float(args.max_pair_angle_deg):
            print(f"  {label:>13s}  SKIPPED: {tri_angle_deg:.0f} deg apart "
                  f"(baseline {baseline_m:.2f} m) -- near-opposing views "
                  f"cannot be densely matched")
            continue

        try:
            dense = prep().build_roma_warp(
                model, paths[ia], paths[ib], images[ia], images[ib], args.device,
            )
            uv_a, uv_b, cert = model.sample_points_from_dense(
                dense, num=int(args.roma_num_matches)
            )
        except Exception as exc:
            print(f"  [warn] cross warp failed {label}: {exc}")
            continue

        uv_a = np.asarray(uv_a, dtype=np.float64)
        uv_b = np.asarray(uv_b, dtype=np.float64)
        cert = np.asarray(cert, dtype=np.float64).reshape(-1)

        keep = (
            (cert >= float(args.roma_min_certainty))
            & np.all(np.isfinite(uv_a), axis=1)
            & np.all(np.isfinite(uv_b), axis=1)
        )
        uv_a, uv_b, cert = uv_a[keep], uv_b[keep], cert[keep]
        if len(uv_a) < 50:
            continue

        # --- image velocity in camera B, in UNDISTORTED px/frame -----------
        warp_b = temporal_warps.get(ib)
        if warp_b is None:
            continue
        flow_b = prep().sample_flow_per_frame(
            model, warp_b, uv_b.astype(np.float32), delta,
            min_cert=float(args.flow_min_certainty),
        )
        moved = np.linalg.norm(flow_b, axis=1) > 1e-9

        pb_u = to_undistorted_px(cam_b, uv_b)
        pb_next_u = to_undistorted_px(cam_b, uv_b + flow_b)
        v_b_u = pb_next_u - pb_u                       # undistorted px/frame

        pa_u = to_undistorted_px(cam_a, uv_a)

        F = fundamental_from_calibration(cam_a, cam_b)
        d_signed, normals = signed_epipolar_and_normal(F, pa_u, pb_u)
        v_perp = np.sum(v_b_u * normals, axis=1)

        finite = np.isfinite(d_signed) & np.isfinite(v_perp)
        # Gross-mismatch gate only. Deliberately generous: a tight gate here
        # would remove the fast matches that carry the timing signal.
        finite &= np.abs(d_signed) <= float(args.max_abs_epipolar_px)
        d_signed, v_perp, moved = d_signed[finite], v_perp[finite], moved[finite]
        v_mag = np.linalg.norm(v_b_u[finite], axis=1)
        if len(d_signed) < 50:
            continue

        # --- (1) static-only residual = spatial calibration health ---------
        static = v_mag < float(args.static_flow_px)
        static_med = float(np.median(np.abs(d_signed[static]))) if static.sum() >= 20 else float("nan")
        static_p90 = float(np.percentile(np.abs(d_signed[static]), 90)) if static.sum() >= 20 else float("nan")
        static_bias = float(np.median(d_signed[static])) if static.sum() >= 20 else float("nan")

        # --- (2) timing slope ---------------------------------------------
        band = float(args.ransac_band_px)
        if np.isfinite(static_p90):
            band = max(band, 1.5 * static_p90)

        fit = None
        usable = moved & (v_mag >= float(args.min_flow_px))
        v_spread = 0.0
        if usable.sum() >= 30:
            xv, yv = v_perp[usable], d_signed[usable]
            v_spread = float(np.percentile(np.abs(xv), 95))
            if v_spread >= float(args.min_vperp_spread_px):
                fit = robust_line_fit(xv, yv, band_px=band, seed=int(args.seed))

        # --- (4) aliasing: forward/reverse round-trip on fast matches ------
        alias_frac = float("nan")
        fast_sel = np.flatnonzero(usable)
        if len(fast_sel) >= 30 and getattr(model, "supports_reverse", False):
            try:
                sample_idx = fast_sel[: int(args.alias_sample)]
                uv_a_fast = uv_a[finite][sample_idx].astype(np.float32)
                fwd, c_fwd = prep().sample_warp_at_uvs(model, dense, uv_a_fast)
                back, c_back = prep().sample_warp_at_uvs(
                    model, dense, np.asarray(fwd, dtype=np.float32), reverse=True
                )
                cyc = np.linalg.norm(np.asarray(back) - uv_a_fast, axis=1)
                ok = np.isfinite(cyc) & (np.asarray(c_fwd).reshape(-1) > 0) \
                    & (np.asarray(c_back).reshape(-1) > 0)
                if ok.sum() >= 20:
                    alias_frac = float(
                        np.mean(cyc[ok] > float(args.alias_cycle_px))
                    )
            except Exception:
                pass

        record: Dict[str, object] = {
            "frame": int(frame),
            "cam_a": cam_a.folder,
            "cam_b": cam_b.folder,
            "idx_a": int(ia),
            "idx_b": int(ib),
            "baseline_m": float(baseline_m),
            "tri_angle_deg": float(tri_angle_deg),
            "n_matches": int(len(d_signed)),
            "n_static": int(static.sum()),
            "static_abs_med_px": static_med,
            "static_abs_p90_px": static_p90,
            "static_signed_bias_px": static_bias,
            "n_fast": int(usable.sum()),
            "vperp_p95_px_per_frame": v_spread,
            "ransac_band_px": band,
            "alias_frac_fast": alias_frac,
            "measured": False,
        }

        if fit is not None:
            dtau_frames = fit["slope"]
            record.update({
                "measured": True,
                "dtau_frames": float(dtau_frames),
                "dtau_ms": float(dtau_frames / fps * 1000.0),
                "dtau_se_frames": float(fit["slope_se"]),
                "dtau_se_ms": float(fit["slope_se"] / fps * 1000.0),
                "calib_intercept_px": float(fit["intercept"]),
                "fit_inliers": int(fit["n_inliers"]),
                "fit_inlier_frac": float(fit["inlier_frac"]),
                "fit_resid_rms_px": float(fit["resid_rms_px"]),
            })

        pair_records.append(record)

        tag = (
            f"dtau {record['dtau_ms']:+7.3f} +/- {record['dtau_se_ms']:.3f} ms"
            if record["measured"] else "UNMEASURED (no perpendicular motion)"
        )
        alias_tag = (
            f" alias {alias_frac * 100:.0f}%" if np.isfinite(alias_frac) else ""
        )
        print(
            f"  {label:>13s}  static |d| {static_med:5.2f}px  "
            f"fast n={record['n_fast']:5d} vperp_p95={v_spread:6.2f}  {tag}{alias_tag}"
        )

    return {"frame": int(frame), "pairs": pair_records}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(
    frame_results: Sequence[Dict[str, object]],
    cameras: Sequence,
    args,
) -> Dict[str, object]:
    fps = float(args.fps)
    n_cams = len(cameras)

    all_pairs: List[Dict[str, object]] = []
    for fr in frame_results:
        all_pairs.extend(fr["pairs"])  # type: ignore[arg-type]

    # ---- (1) spatial calibration: static residual per frame and per camera
    per_frame_static: List[Dict[str, float]] = []
    for fr in frame_results:
        vals = [
            p["static_abs_med_px"] for p in fr["pairs"]  # type: ignore[index]
            if np.isfinite(p.get("static_abs_med_px", float("nan")))  # type: ignore[union-attr]
        ]
        per_frame_static.append({
            "frame": float(fr["frame"]),  # type: ignore[arg-type]
            "static_abs_med_px": float(np.median(vals)) if vals else float("nan"),
            "n_pairs": float(len(vals)),
        })

    cam_static: Dict[str, List[float]] = defaultdict(list)
    for p in all_pairs:
        v = p.get("static_abs_med_px", float("nan"))
        if np.isfinite(v):  # type: ignore[arg-type]
            cam_static[str(p["cam_a"])].append(float(v))  # type: ignore[arg-type]
            cam_static[str(p["cam_b"])].append(float(v))  # type: ignore[arg-type]
    per_camera_static = {
        cam: {
            "static_abs_med_px": float(np.median(v)),
            "n_pair_obs": int(len(v)),
        }
        for cam, v in sorted(cam_static.items())
    }

    # ---- (2) temporal: graph solve over measured pairs -------------------
    measurements: List[Tuple[int, int, float, float]] = []
    for p in all_pairs:
        if not p.get("measured"):
            continue
        se = float(p["dtau_se_frames"])  # type: ignore[arg-type]
        if not np.isfinite(se) or se <= 0:
            continue
        if se / fps * 1000.0 > float(args.max_pair_se_ms):
            continue
        measurements.append((
            int(p["idx_a"]),  # type: ignore[arg-type]
            int(p["idx_b"]),  # type: ignore[arg-type]
            float(p["dtau_frames"]),  # type: ignore[arg-type]
            1.0 / (se * se),
        ))

    tau_frames, constrained = solve_offsets_from_pairs(measurements, n_cams)
    tau_sec = tau_frames / fps

    # Residual of the graph fit: how self-consistent are the pair measurements
    graph_resid_ms: List[float] = []
    for a, b, dtau, _w in measurements:
        pred = tau_frames[b] - tau_frames[a]
        graph_resid_ms.append(abs(pred - dtau) / fps * 1000.0)

    # ---- (2b) CROSS-FRAME consistency: the real trust metric -------------
    # A camera's time offset is a constant. If the same pair measured at two
    # frames disagrees by many sigma, the slope is absorbing something that is
    # not a constant offset, and the pooled graph solve will hide it.
    by_pair: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for p in all_pairs:
        if p.get("measured"):
            by_pair[(int(p["idx_a"]), int(p["idx_b"]))].append(p)  # type: ignore[arg-type]

    consistency: List[Dict[str, object]] = []
    for (ia, ib), ps in sorted(by_pair.items()):
        if len(ps) < 2:
            continue
        vals = np.array([float(q["dtau_ms"]) for q in ps])          # type: ignore[arg-type]
        ses = np.array([float(q["dtau_se_ms"]) for q in ps])        # type: ignore[arg-type]
        spread = float(vals.max() - vals.min())
        pooled_se = float(np.sqrt(np.sum(ses ** 2)) / len(ses))
        consistency.append({
            "cam_a": str(ps[0]["cam_a"]),
            "cam_b": str(ps[0]["cam_b"]),
            "n_frames": int(len(ps)),
            "dtau_ms": [float(v) for v in vals],
            "spread_ms": spread,
            "pooled_se_ms": pooled_se,
            "n_sigma": float(spread / pooled_se) if pooled_se > 0 else float("inf"),
        })

    alias_vals = [
        float(p["alias_frac_fast"]) for p in all_pairs  # type: ignore[arg-type]
        if np.isfinite(p.get("alias_frac_fast", float("nan")))  # type: ignore[arg-type]
    ]

    n_measured = sum(1 for p in all_pairs if p.get("measured"))
    report = {
        "config": {
            "frames": [int(fr["frame"]) for fr in frame_results],  # type: ignore[arg-type]
            "fps": fps,
            "temporal_frame_delta": int(args.temporal_frame_delta),
            "static_flow_px": float(args.static_flow_px),
            "min_flow_px": float(args.min_flow_px),
            "min_vperp_spread_px": float(args.min_vperp_spread_px),
        },
        "spatial_calibration": {
            "per_frame": per_frame_static,
            "per_camera": per_camera_static,
            "overall_static_abs_med_px": float(np.nanmedian([
                f["static_abs_med_px"] for f in per_frame_static
            ])) if per_frame_static else float("nan"),
        },
        "temporal": {
            "n_pair_measurements": int(len(measurements)),
            "n_pairs_total": int(len(all_pairs)),
            "n_pairs_measured": int(n_measured),
            "n_cameras_constrained": int(constrained.sum()),
            "graph_resid_ms_median": float(np.median(graph_resid_ms)) if graph_resid_ms else float("nan"),
            "graph_resid_ms_p90": float(np.percentile(graph_resid_ms, 90)) if graph_resid_ms else float("nan"),
            "offsets_ms": {
                cameras[i].folder: float(tau_sec[i] * 1000.0)
                for i in range(n_cams)
            },
            "constrained": {
                cameras[i].folder: bool(constrained[i]) for i in range(n_cams)
            },
            "spread_ms": float(
                tau_sec[constrained].max() - tau_sec[constrained].min()
            ) * 1000.0 if constrained.any() else float("nan"),
            "cross_frame_consistency": consistency,
            "cross_frame_max_n_sigma": float(
                max(c["n_sigma"] for c in consistency)  # type: ignore[type-var]
            ) if consistency else float("nan"),
            "cross_frame_median_spread_ms": float(
                np.median([c["spread_ms"] for c in consistency])
            ) if consistency else float("nan"),
        },
        "matching": {
            "alias_frac_fast_median": float(np.median(alias_vals)) if alias_vals else float("nan"),
            "alias_frac_fast_p90": float(np.percentile(alias_vals, 90)) if alias_vals else float("nan"),
        },
        "pairs": all_pairs,
    }
    return report


def print_verdict(report: Dict[str, object], cameras: Sequence, args) -> None:
    spatial = report["spatial_calibration"]  # type: ignore[index]
    temporal = report["temporal"]            # type: ignore[index]
    matching = report["matching"]            # type: ignore[index]

    print("\n" + "=" * 72)
    print("  SPATIAL CALIBRATION  (static content only -- timing cannot affect it)")
    print("=" * 72)
    overall = spatial["overall_static_abs_med_px"]  # type: ignore[index]
    print(f"  median |epipolar residual| on static matches: {overall:.3f} px")
    print("\n  per frame (a trend = rig drift; a step = bumped stand):")
    for f in spatial["per_frame"]:  # type: ignore[index]
        print(f"    frame {int(f['frame']):6d}   {f['static_abs_med_px']:6.3f} px"
              f"   ({int(f['n_pairs'])} pairs)")

    worst = sorted(
        spatial["per_camera"].items(),  # type: ignore[index]
        key=lambda kv: -kv[1]["static_abs_med_px"],
    )[:8]
    print("\n  worst cameras:")
    for cam, v in worst:
        print(f"    cam {cam:>6s}   {v['static_abs_med_px']:6.3f} px"
              f"   ({v['n_pair_obs']} pair-observations)")

    if np.isfinite(overall):  # type: ignore[arg-type]
        if overall < 0.5:  # type: ignore[operator]
            print("\n  VERDICT: spatial calibration is HEALTHY. Static content is")
            print("           sub-half-pixel consistent, so per-frame extrinsics")
            print("           refinement is NOT your bottleneck.")
        elif overall < 1.5:  # type: ignore[operator]
            print("\n  VERDICT: spatial calibration is MARGINAL. Worth a multi-frame")
            print("           re-solve, but check the temporal section first.")
        else:
            print("\n  VERDICT: spatial calibration is BAD. Fix this before anything")
            print("           temporal -- check the per-frame trend for drift vs bias.")

    print("\n" + "=" * 72)
    print("  TEMPORAL SKEW  (slope of epipolar residual vs perpendicular speed)")
    print("=" * 72)
    print(f"  pairs measured: {temporal['n_pairs_measured']}/{temporal['n_pairs_total']}"  # type: ignore[index]
          f"   cameras constrained: {temporal['n_cameras_constrained']}/{len(cameras)}")  # type: ignore[index]
    print(f"  graph self-consistency residual: median "
          f"{temporal['graph_resid_ms_median']:.3f} ms, "  # type: ignore[index]
          f"p90 {temporal['graph_resid_ms_p90']:.3f} ms")  # type: ignore[index]
    print(f"  offset spread across cameras: {temporal['spread_ms']:.3f} ms"  # type: ignore[index]
          f"   (one frame @ {args.fps:g}fps = {1000.0 / float(args.fps):.2f} ms)")

    offsets = temporal["offsets_ms"]      # type: ignore[index]
    constrained = temporal["constrained"]  # type: ignore[index]
    print("\n  per-camera offset (PLUS convention: actual = nominal + offset):")
    for cam in sorted(offsets, key=lambda c: -abs(offsets[c])):  # type: ignore[index,arg-type]
        flag = "" if constrained[cam] else "   [UNCONSTRAINED]"  # type: ignore[index]
        print(f"    cam {cam:>6s}   {offsets[cam]:+8.3f} ms{flag}")  # type: ignore[index]

    cons = temporal["cross_frame_consistency"]  # type: ignore[index]
    if cons:
        print("\n  cross-frame consistency (a true offset is CONSTANT over frames):")
        for c in sorted(cons, key=lambda x: -x["n_sigma"])[:8]:  # type: ignore[arg-type,index]
            vals = "  ".join(f"{v:+7.3f}" for v in c["dtau_ms"])  # type: ignore[index]
            print(f"    {c['cam_a']}<->{c['cam_b']}  [{vals}] ms   "  # type: ignore[index]
                  f"spread {c['spread_ms']:6.3f} ms = {c['n_sigma']:6.1f} sigma")  # type: ignore[index]
        worst = temporal["cross_frame_max_n_sigma"]  # type: ignore[index]
        if worst > 3.0:  # type: ignore[operator]
            print(f"\n  VERDICT: NOT MEASURABLE at these frames. The same pair disagrees")
            print(f"           across frames by up to {worst:.0f} sigma, so the slope is")
            print("           absorbing something that is not a constant camera offset")
            print("           (rolling shutter, or timing signal buried in the calibration")
            print("           floor because the scene is moving too slowly). Do NOT write")
            print("           these to sync.json. Re-run on genuinely fast frames.")
        else:
            print("\n  VERDICT: offsets are CONSISTENT across frames. Real sub-frame skew --")
            print("           write to sync.json and re-run prep.")
    else:
        print("\n  VERDICT: no pair was measured at 2+ frames, so cross-frame")
        print("           consistency could not be checked. Treat the offsets as")
        print("           UNVERIFIED and add more frames.")

    print("\n" + "=" * 72)
    print("  MATCH RELIABILITY  (forward/reverse round-trip on fast matches)")
    print("=" * 72)
    am = matching["alias_frac_fast_median"]  # type: ignore[index]
    ap = matching["alias_frac_fast_p90"]     # type: ignore[index]
    if np.isfinite(am):  # type: ignore[arg-type]
        print(f"  cycle-inconsistent fast matches: median {am * 100:.1f}%, "  # type: ignore[operator]
              f"p90 {ap * 100:.1f}%")  # type: ignore[operator]
        if am > 0.25 or ap > 0.75:  # type: ignore[operator]
            print("\n  VERDICT: matching itself is UNRELIABLE on moving content. Repeating")
            print("           structure (frets, rungs, brick) aliases to the wrong repeat")
            print("           and no amount of calibration will fix it. Sub-frame sync")
            print("           makes this worse by displacing the pattern ~1 period.")
        else:
            print("\n  VERDICT: matching is holding up on fast content.")
    else:
        print("  not enough fast matches to assess (add frames with real motion)")
    print()


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reference-json", type=str, required=True)
    p.add_argument("--images-root", type=str, required=True)
    p.add_argument("--image-pattern", type=str, default="{frame:06d}.png")
    p.add_argument("--frames", type=str, required=True,
                   help="Comma-separated frames to audit. Pick frames ON the "
                        "fast motion you care about.")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--write-sync-json", type=str, default=None,
                   help="Also write a PLUS-convention sync.json from the solve.")

    p.add_argument("--camera-neighbors", type=int, default=4)
    p.add_argument("--wide-baseline-per-camera", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Cap pairs per frame (0 = no cap). Useful for a fast look.")

    p.add_argument("--temporal-frame-delta", type=int, default=1)
    p.add_argument("--static-flow-px", type=float, default=0.5,
                   help="|flow| below this counts as static content.")
    p.add_argument("--min-flow-px", type=float, default=1.5,
                   help="|flow| above this is used for the timing regression.")
    p.add_argument("--min-vperp-spread-px", type=float, default=3.0,
                   help="p95(|v_perp|) must exceed this or the pair is reported "
                        "UNMEASURED rather than given a fabricated slope.")
    p.add_argument("--max-pair-angle-deg", type=float, default=70.0,
                   help="Skip pairs whose cameras subtend more than this angle "
                        "at the scene center. Near-opposing views see different "
                        "surfaces, so their matches are wrong regardless of "
                        "calibration; pooling them corrupts every statistic.")
    p.add_argument("--max-abs-epipolar-px", type=float, default=60.0,
                   help="Gross-mismatch gate. Keep generous.")
    p.add_argument("--ransac-band-px", type=float, default=1.5)
    p.add_argument("--max-pair-se-ms", type=float, default=2.0,
                   help="Drop pair measurements whose slope SE exceeds this.")

    p.add_argument("--alias-sample", type=int, default=2000)
    p.add_argument("--alias-cycle-px", type=float, default=3.0)

    p.add_argument("--roma-backend", choices=("v1", "v2"), default="v1")
    p.add_argument("--roma-model", choices=("outdoor", "indoor", "tiny-outdoor"),
                   default="outdoor")
    p.add_argument("--roma-v2-setting", type=str, default="precise")
    p.add_argument("--roma-num-matches", type=int, default=8000)
    p.add_argument("--roma-min-certainty", type=float, default=0.20)
    p.add_argument("--flow-min-certainty", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = [int(v.strip()) for v in args.frames.split(",") if v.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras, _scene_meta = prep().load_reference_export(
        Path(args.reference_json).expanduser().resolve()
    )
    print(f"[info] loaded {len(cameras)} cameras")

    pairs = prep().choose_camera_pairs(
        cameras,
        neighbors_per_camera=int(args.camera_neighbors),
        wide_baseline_per_camera=int(args.wide_baseline_per_camera),
    )
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    print(f"[info] {len(pairs)} camera pairs, {len(frames)} frames")

    model = prep().create_roma_backend(
        backend=args.roma_backend,
        device=args.device,
        model=args.roma_model,
        v2_setting=args.roma_v2_setting,
    )
    print(f"[info] matcher: {getattr(model, 'name', 'unknown')}")

    images_root = Path(args.images_root).expanduser().resolve()
    frame_results = []
    for frame in frames:
        print(f"\n[frame {frame}]")
        frame_results.append(
            run_frame(model, cameras, pairs, images_root, frame,
                      scene_center, args)
        )

    report = summarize(frame_results, cameras, args)
    print_verdict(report, cameras, args)

    report_path = out_dir / "epipolar_slope_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[info] wrote {report_path}")

    if args.write_sync_json:
        temporal = report["temporal"]  # type: ignore[index]
        offsets_ms = temporal["offsets_ms"]      # type: ignore[index]
        constrained = temporal["constrained"]    # type: ignore[index]
        sync: Dict[str, float] = {}
        for cam in offsets_ms:  # type: ignore[union-attr]
            key = f"{int(cam):04d}" if str(cam).isdigit() else str(cam)
            sync[key] = (
                float(offsets_ms[cam]) / 1000.0  # type: ignore[index]
                if constrained[cam] else 0.0     # type: ignore[index]
            )
        sync = {k: sync[k] for k in sorted(sync)}
        sync_path = Path(args.write_sync_json).expanduser().resolve()
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path.write_text(json.dumps(sync, indent=4) + "\n", encoding="utf-8")
        print(f"[info] wrote {sync_path} (PLUS convention, seconds)")


if __name__ == "__main__":
    main()
