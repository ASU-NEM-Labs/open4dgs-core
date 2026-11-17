#!/usr/bin/env python3
"""Measure rolling-shutter readout (rho) geometrically, from images alone.

The idea
--------
A pixel's capture time depends on its COLUMN, not just its camera:

    t_c(u) = t_nominal + tau_c + s_c * rho * (u - u_ref_c) / W

`s_c = +/-1` because the rig records landscape and the extract rotates +/-90 per
camera, so readout runs along image columns with a per-camera sign. `tau_c` is
the ordinary per-camera sub-frame offset. `rho` is the full-span readout time.

For a RoMa match between cameras A and B, both see the same surface point but at
different times, so it is displaced by V*dt in the world. Projected into B and
taken perpendicular to the epipolar line, that displacement IS the signed
epipolar residual (see audit-epipolar-slope.py for the derivation):

    d_signed ~= dt * v_perp
    dt       =  dtau_AB + rho * g,
    g        = [ s_B*(u_b - W/2) - s_A*(u_a - W/2) ] / W

Substituting gives a model that is LINEAR in the unknowns:

    d_signed ~= dtau_AB * v_perp  +  rho * (g * v_perp)              (*)

so a single least-squares fit over every match in every pair recovers one
global `rho` alongside a nuisance `dtau` per (pair, frame).

Why this is identifiable
------------------------
`dtau` is fit per (pair, frame), so it absorbs ANY per-camera-per-frame
constant. That includes the choice of `u_ref_c` -- which is why this script
just uses W/2 and does not need the flow-weighted reference column at all.
What survives is only the term that varies WITH COLUMN INSIDE a single image.
Nothing except rolling shutter does that, which is what makes a non-zero `rho`
hard to explain away.

It also means the estimate does NOT require reconstructing from a camera
subset. Every camera contributes; the L/R sign split is a regressor, not a
partition of the rig.

Null tests
----------
A fitted number is worthless without a null, so the report also runs:

  * PERMUTATION -- reshuffle the per-camera l/r signs and refit. Rolling
    shutter lives in the SIGN STRUCTURE, so a real signal must collapse when
    the signs are scrambled. The true `rho` is reported as a percentile of
    that null distribution.
  * UNSIGNED control -- refit with every sign forced to +1. A real signal must
    be weaker without the correct signs.
  * SPLIT agreement -- fit `rho` on same-sign pairs alone and on opposite-sign
    pairs alone. Both subsets must land on the same `rho`.

(An earlier version of this script compared residual scatter between same-sign
and opposite-sign pairs. That has no power: the sign cancellation happens in
the MEAN of the column term, which the per-group `dtau` absorbs by
construction. The synthetic test in tests/ demonstrates the null result.)

Two stages
----------
`collect` runs RoMa (GPU, slow) and writes every match's
(pair, frame, u_a, u_b, v_perp, d_signed, s_a, s_b, flow) to an npz.
`fit` (CPU, seconds) does the regression. Iterate on `fit` without paying for
RoMa again.

    # stage 1 -- GPU, once
    python prep/sync/audit-rolling-shutter.py --stage collect \
        --reference-json reference_export.json --images-root images \
        --image-pattern '{frame:06d}.jpg' --rotation-json camera_rotations.json \
        --frames 148,149,150,151,152,263,264,265,266,267 --fps 60 \
        --out-dir rs_audit

    # stage 2 -- CPU, re-runnable
    python prep/sync/audit-rolling-shutter.py --stage fit \
        --out-dir rs_audit --fps 60

Pick frames where the subject actually MOVES. The estimator needs image-plane
speed; on near-static frames it will (correctly) report no measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Reuse the epipolar-slope audit's geometry (and, through it, prep's RoMa)
# --------------------------------------------------------------------------

def load_audit():
    here = Path(__file__).resolve().parent
    target = here / "audit-epipolar-slope.py"
    if not target.exists():
        raise FileNotFoundError(f"expected sibling script at {target}")
    spec = importlib.util.spec_from_file_location("audit_epipolar_slope", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AUDIT = load_audit()


# --------------------------------------------------------------------------
# Rotation map -> per-camera readout sign
# --------------------------------------------------------------------------

def load_rs_signs(
    rotation_json: Path, cameras: Sequence, flip: bool = False
) -> np.ndarray:
    """Map each camera to s_c = +1 ('l') or -1 ('r').

    Accepts keys by folder ("01"), zero-padded folder ("0001"), camera name,
    or "Camera 1" style, matching the convention prep-roma-trigs-memory.py
    uses for --rotation-json.
    """
    raw = json.loads(rotation_json.read_text(encoding="utf-8"))
    table = {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}

    signs = np.zeros(len(cameras), dtype=np.float64)
    missing: List[str] = []
    for i, cam in enumerate(cameras):
        folder = str(cam.folder)
        keys = [folder, folder.lower(), str(cam.name).lower()]
        if folder.isdigit():
            n = int(folder)
            # Bare numeric forms, then prefixed forms. Both zero-padded and
            # unpadded, because "Camera 01" and "Camera 1" are both in the wild
            # and only one of them matches a given rig's folder naming.
            numeric = [f"{n:04d}", f"{n:02d}", str(n)]
            keys += numeric
            for prefix in ("camera ", "cam ", "camera", "cam"):
                keys += [f"{prefix}{v}" for v in numeric]
        val = None
        for k in keys:
            if k in table:
                val = table[k]
                break
        if val is None:
            missing.append(folder)
            signs[i] = 0.0
            continue
        if val.startswith("l"):
            signs[i] = +1.0
        elif val.startswith("r"):
            signs[i] = -1.0
        else:
            raise ValueError(f"camera {folder}: expected 'l'/'r', got {val!r}")

    if flip:
        signs = -signs
    if missing:
        raise KeyError(
            f"rotation-json is missing {len(missing)} cameras: {missing[:10]}. "
            "Every camera needs an l/r entry or the sign split is meaningless."
        )
    n_l = int((signs > 0).sum())
    print(f"[rs] rotation signs: {n_l} 'l' (+1), {len(signs) - n_l} 'r' (-1)")
    return signs


# --------------------------------------------------------------------------
# Stage 1: collect per-match observations
# --------------------------------------------------------------------------

def collect(args) -> Path:
    prep = AUDIT.prep()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras, scene_meta = prep.load_reference_export(
        Path(args.reference_json).expanduser().resolve()
    )
    scene_center = np.asarray(
        (scene_meta or {}).get("scene_center", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    signs = load_rs_signs(
        Path(args.rotation_json).expanduser().resolve(), cameras, flip=args.flip_sign
    )

    pairs = prep.choose_camera_pairs(
        cameras,
        neighbors_per_camera=int(args.camera_neighbors),
        wide_baseline_per_camera=int(args.wide_baseline_per_camera),
    )
    # Near-opposing views cannot be densely matched; their "matches" are wrong
    # regardless of calibration and would pollute the regression.
    kept = []
    for (ia, ib) in pairs:
        _, ang = AUDIT.pair_geometry(cameras[ia], cameras[ib], scene_center)
        if ang <= float(args.max_pair_angle_deg):
            kept.append((ia, ib))
    pairs = kept
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    print(f"[rs] {len(cameras)} cameras, {len(pairs)} usable pairs, "
          f"{len(args.frames.split(','))} frames")

    model = prep.create_roma_backend(
        backend=args.roma_backend, device=args.device, model=args.roma_model,
        v2_setting=args.roma_v2_setting,
    )
    print(f"[rs] matcher: {getattr(model, 'name', '?')}")

    images_root = Path(args.images_root).expanduser().resolve()
    frames = [int(v) for v in args.frames.split(",") if v.strip()]
    delta = int(args.temporal_frame_delta)

    cols: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("pair_id", "frame", "u_a", "u_b", "v_perp",
                        "d_signed", "s_a", "s_b", "flow_mag", "cam_a", "cam_b")
    }
    pair_index: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(pairs)}

    for frame in frames:
        print(f"\n[frame {frame}]")
        paths, images = prep.load_frame_images(
            images_root, cameras, frame, args.image_pattern
        )
        paths_n, images_n = prep.load_frame_images(
            images_root, cameras, frame + delta, args.image_pattern
        )
        warps = {}
        for ci in range(len(cameras)):
            try:
                warps[ci] = prep.build_roma_warp(
                    model, paths[ci], paths_n[ci], images[ci], images_n[ci], args.device
                )
            except Exception as exc:
                print(f"  [warn] temporal warp cam {cameras[ci].folder}: {exc}")
                warps[ci] = None

        for (ia, ib) in pairs:
            cam_a, cam_b = cameras[ia], cameras[ib]
            if warps.get(ib) is None:
                continue
            try:
                dense = prep.build_roma_warp(
                    model, paths[ia], paths[ib], images[ia], images[ib], args.device
                )
                uv_a, uv_b, cert = model.sample_points_from_dense(
                    dense, num=int(args.roma_num_matches)
                )
            except Exception as exc:
                print(f"  [warn] {cam_a.folder}<->{cam_b.folder}: {exc}")
                continue

            uv_a = np.asarray(uv_a, np.float64)
            uv_b = np.asarray(uv_b, np.float64)
            cert = np.asarray(cert, np.float64).reshape(-1)
            keep = (cert >= float(args.roma_min_certainty)) \
                & np.all(np.isfinite(uv_a), axis=1) & np.all(np.isfinite(uv_b), axis=1)
            uv_a, uv_b = uv_a[keep], uv_b[keep]
            if len(uv_a) < 50:
                continue

            flow_b = prep.sample_flow_per_frame(
                model, warps[ib], uv_b.astype(np.float32), delta,
                min_cert=float(args.flow_min_certainty),
            )
            pb_u = AUDIT.to_undistorted_px(cam_b, uv_b)
            pb_next = AUDIT.to_undistorted_px(cam_b, uv_b + flow_b)
            v_b = pb_next - pb_u
            pa_u = AUDIT.to_undistorted_px(cam_a, uv_a)

            F = AUDIT.fundamental_from_calibration(cam_a, cam_b)
            d_signed, normals = AUDIT.signed_epipolar_and_normal(F, pa_u, pb_u)
            v_perp = np.sum(v_b * normals, axis=1)
            flow_mag = np.linalg.norm(v_b, axis=1)

            ok = np.isfinite(d_signed) & np.isfinite(v_perp) & np.isfinite(flow_mag)
            ok &= np.abs(d_signed) <= float(args.max_abs_epipolar_px)
            ok &= flow_mag >= float(args.min_flow_px)
            n = int(ok.sum())
            if n < 30:
                continue

            pid = pair_index[(ia, ib)]
            cols["pair_id"].append(np.full(n, pid, np.int32))
            cols["frame"].append(np.full(n, frame, np.int32))
            cols["u_a"].append(uv_a[ok, 0])
            cols["u_b"].append(uv_b[ok, 0])
            cols["v_perp"].append(v_perp[ok])
            cols["d_signed"].append(d_signed[ok])
            cols["s_a"].append(np.full(n, signs[ia], np.float64))
            cols["s_b"].append(np.full(n, signs[ib], np.float64))
            cols["flow_mag"].append(flow_mag[ok])
            cols["cam_a"].append(np.full(n, ia, np.int32))
            cols["cam_b"].append(np.full(n, ib, np.int32))
            print(f"  {cam_a.folder}<->{cam_b.folder}  n={n:6d}  "
                  f"|vperp|p95={np.percentile(np.abs(v_perp[ok]),95):6.2f}  "
                  f"sign {int(signs[ia]):+d}/{int(signs[ib]):+d}")

    if not cols["pair_id"]:
        raise RuntimeError("no usable matches collected -- try faster frames")

    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["width"] = np.array([cameras[0].width], np.float64)
    data["pair_a"] = np.array([p[0] for p in pairs], np.int32)
    data["pair_b"] = np.array([p[1] for p in pairs], np.int32)
    data["folders"] = np.array([str(c.folder) for c in cameras])
    out = out_dir / "rs_observations.npz"
    np.savez_compressed(out, **data)
    print(f"\n[rs] wrote {out}  ({len(data['pair_id']):,} matches)")
    return out


# --------------------------------------------------------------------------
# Stage 2: joint fit of one global rho + per-(pair, frame) dtau
# --------------------------------------------------------------------------

def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def joint_fit(
    group: np.ndarray,      # (N,) group index in [0, G)
    x_tau: np.ndarray,      # (N,) v_perp
    x_rho: np.ndarray,      # (N,) g * v_perp
    y: np.ndarray,          # (N,) d_signed
    n_groups: int,
    fit_rho: bool,
    huber_px: float,
    iters: int = 12,
) -> Dict[str, object]:
    """IRLS for  y = dtau[group]*x_tau + rho*x_rho.

    Solved by eliminating the per-group nuisance analytically (each group's
    dtau has a closed form given rho), which keeps the system 1x1 in rho no
    matter how many groups there are.
    """
    w = np.ones_like(y)
    rho = 0.0
    dtau = np.zeros(n_groups)

    for _ in range(iters):
        # per-group weighted normal equations for dtau given current rho
        resid_rho = y - rho * x_rho
        num = np.bincount(group, weights=w * x_tau * resid_rho, minlength=n_groups)
        den = np.bincount(group, weights=w * x_tau * x_tau, minlength=n_groups)
        dtau = num / np.maximum(den, 1e-12)

        if fit_rho:
            # rho given dtau
            resid_tau = y - dtau[group] * x_tau
            rho = float(
                np.sum(w * x_rho * resid_tau) / max(np.sum(w * x_rho * x_rho), 1e-12)
            )

        r = y - (dtau[group] * x_tau + rho * x_rho)
        w = huber_weights(r, huber_px)

    r = y - (dtau[group] * x_tau + rho * x_rho)
    wr = huber_weights(r, huber_px)
    dof = max(len(y) - n_groups - (1 if fit_rho else 0), 1)
    sigma2 = float(np.sum(wr * r * r) / dof)
    se_rho = (
        float(np.sqrt(sigma2 / max(np.sum(wr * x_rho * x_rho), 1e-12)))
        if fit_rho else float("nan")
    )
    return {
        "rho_frames": rho,
        "se_rho_frames": se_rho,
        "dtau_frames": dtau,
        "resid_rms_px": float(np.sqrt(np.mean(r * r))),
        "resid_robust_rms_px": float(np.sqrt(sigma2)),
        "n": int(len(y)),
        "n_groups": int(n_groups),
    }


def per_camera_fit(
    group: np.ndarray, v: np.ndarray, ub_t: np.ndarray, ua_t: np.ndarray,
    cam_a: np.ndarray, cam_b: np.ndarray, y: np.ndarray,
    n_groups: int, n_cams: int, huber_px: float, iters: int = 15,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve a FREE readout coefficient per camera -- sign and magnitude both
    unconstrained -- from

        y ~ dtau_g * v  +  rho_B * (v*ub~/W)  -  rho_A * (v*ua~/W)

    This is the strongest form of the test. The l/r map is never given to the
    solver, so if the recovered signs line up with it anyway, that agreement is
    real evidence rather than an assumption being echoed back. It is also far
    better conditioned than the sign-imposed global fit, because each camera's
    coefficient draws on every pair it appears in rather than on the column
    CONTRAST within one pair (which collapses whenever both cameras happen to
    frame the subject at the same column).

    Returns (rho_per_cam, se_per_cam, resid_rms_before, resid_rms_after).
    """
    xb, xa = v * ub_t, -v * ua_t
    rho = np.zeros(n_cams)
    w = np.ones_like(y)
    A = np.zeros((n_cams, n_cams))
    for _ in range(iters):
        r = y - (rho[cam_b] * xb + rho[cam_a] * xa)
        num = np.bincount(group, weights=w * v * r, minlength=n_groups)
        den = np.bincount(group, weights=w * v * v, minlength=n_groups)
        dtau = num / np.maximum(den, 1e-12)
        rr = y - dtau[group] * v
        A[:] = 0.0
        b = np.zeros(n_cams)
        np.add.at(b, cam_b, w * xb * rr)
        np.add.at(b, cam_a, w * xa * rr)
        np.add.at(A, (cam_b, cam_b), w * xb * xb)
        np.add.at(A, (cam_a, cam_a), w * xa * xa)
        np.add.at(A, (cam_b, cam_a), w * xb * xa)
        np.add.at(A, (cam_a, cam_b), w * xb * xa)
        A += 1e-6 * np.trace(A) / n_cams * np.eye(n_cams)
        rho = np.linalg.solve(A, b)
        res = y - (dtau[group] * v + rho[cam_b] * xb + rho[cam_a] * xa)
        w = huber_weights(res, huber_px)

    before = float(np.sqrt(np.mean((y - dtau[group] * v) ** 2)))
    after = float(np.sqrt(np.mean(res ** 2)))
    sig2 = float(np.sum(w * res * res) / max(len(y) - n_groups - n_cams, 1))
    se = np.sqrt(np.clip(np.diag(sig2 * np.linalg.inv(A)), 0.0, None))
    return rho, se, before, after


def fit(args) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    d = np.load(out_dir / "rs_observations.npz", allow_pickle=False)
    fps = float(args.fps)
    W = float(d["width"][0])

    pair_id, frame = d["pair_id"], d["frame"]
    u_a, u_b = d["u_a"], d["u_b"]
    v_perp, y = d["v_perp"], d["d_signed"]
    s_a, s_b = d["s_a"], d["s_b"]

    # Keep matches with real perpendicular motion -- that is the only place
    # timing information lives.
    m = np.abs(v_perp) >= float(args.min_vperp_px)
    pair_id, frame, u_a, u_b, v_perp, y, s_a, s_b = (
        arr[m] for arr in (pair_id, frame, u_a, u_b, v_perp, y, s_a, s_b)
    )
    print(f"[fit] {len(y):,} matches with |v_perp| >= {args.min_vperp_px} px/frame")
    if len(y) < 1000:
        print("[fit] WARNING: very little usable motion; treat results as weak")

    # Nuisance group = (pair, frame). Absorbs every per-camera-per-frame
    # constant, including the u_ref choice, leaving rho identified purely by
    # WITHIN-IMAGE column variation.
    key = pair_id.astype(np.int64) * 1_000_000 + frame.astype(np.int64)
    _, group = np.unique(key, return_inverse=True)
    n_groups = int(group.max()) + 1

    g = (s_b * (u_b - W / 2.0) - s_a * (u_a - W / 2.0)) / W
    x_rho = g * v_perp

    # Leverage check: rho is only identified if g VARIES inside groups.
    gvar = np.sqrt(np.bincount(group, weights=(g - np.bincount(
        group, weights=g, minlength=n_groups)[group]) ** 2,
        minlength=n_groups) / np.maximum(np.bincount(group, minlength=n_groups), 1))
    print(f"[fit] within-group std of column term g: median {np.median(gvar):.4f} "
          f"(needs to be well above 0 for rho to be identifiable)")

    huber = float(args.huber_px)
    m0 = joint_fit(group, v_perp, x_rho, y, n_groups, False, huber)
    m1 = joint_fit(group, v_perp, x_rho, y, n_groups, True, huber)

    rho_ms = m1["rho_frames"] / fps * 1000.0
    se_ms = m1["se_rho_frames"] / fps * 1000.0

    print("\n" + "=" * 70)
    print("  ROLLING SHUTTER: joint fit of one global readout constant")
    print("=" * 70)
    print(f"  matches {m1['n']:,}   nuisance groups (pair x frame) {m1['n_groups']}")
    print(f"\n  rho = 0 (no RS)      robust residual RMS = {m0['resid_robust_rms_px']:.4f} px")
    print(f"  rho free             robust residual RMS = {m1['resid_robust_rms_px']:.4f} px")
    impr = (1.0 - m1["resid_robust_rms_px"] / max(m0["resid_robust_rms_px"], 1e-12)) * 100
    print(f"  improvement          {impr:+.2f}%")
    print(f"\n  rho (full-span readout) = {rho_ms:+.3f} +/- {se_ms:.3f} ms"
          f"   ({abs(rho_ms) / max(se_ms, 1e-9):.1f} sigma from zero)")
    print(f"  one frame @ {fps:g} fps = {1000.0 / fps:.2f} ms")

    if args.bootstrap > 0:
        rng = np.random.default_rng(0)
        pids = np.unique(pair_id)
        boots = []
        for _ in range(int(args.bootstrap)):
            take = rng.choice(pids, size=len(pids), replace=True)
            sel = np.concatenate([np.flatnonzero(pair_id == p) for p in take])
            _, gb = np.unique(key[sel], return_inverse=True)
            fb = joint_fit(gb, v_perp[sel], x_rho[sel], y[sel],
                           int(gb.max()) + 1, True, huber, iters=6)
            boots.append(fb["rho_frames"] / fps * 1000.0)
        boots = np.array(boots)
        print(f"  bootstrap over pairs ({args.bootstrap}x): "
              f"median {np.median(boots):+.3f} ms  "
              f"CI95 [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")

    # ---- null tests: does the SIGN STRUCTURE actually matter? -------------
    cam_a, cam_b = d["cam_a"][m], d["cam_b"][m]
    folders = d["folders"] if "folders" in d.files else None
    n_cams = int(max(cam_a.max(), cam_b.max())) + 1
    if folders is None or len(folders) < n_cams:
        folders = np.array([str(i) for i in range(n_cams)])
    signs = np.zeros(n_cams)
    signs[cam_a] = s_a
    signs[cam_b] = s_b

    def refit(sg: np.ndarray) -> Tuple[float, float]:
        gg = (sg[cam_b] * (u_b - W / 2.0) - sg[cam_a] * (u_a - W / 2.0)) / W
        f = joint_fit(group, v_perp, gg * v_perp, y, n_groups, True, huber, iters=8)
        i = (1.0 - f["resid_robust_rms_px"] / max(m0["resid_robust_rms_px"], 1e-12)) * 100
        return f["rho_frames"] / fps * 1000.0, i

    print("\n" + "=" * 70)
    print("  NULL TESTS: is the signal actually in the l/r sign structure?")
    print("=" * 70)

    rho_uns, impr_uns = refit(np.ones(n_cams))
    print(f"  unsigned control (all +1):  rho {rho_uns:+7.3f} ms   resid {impr_uns:+6.2f}%")

    if args.permutations > 0:
        rng = np.random.default_rng(1234)
        null_rho, null_impr = [], []
        for _ in range(int(args.permutations)):
            r, i = refit(rng.permutation(signs))
            null_rho.append(abs(r))
            null_impr.append(i)
        null_rho = np.array(null_rho)
        null_impr = np.array(null_impr)
        pval = float(np.mean(null_rho >= abs(rho_ms)))
        print(f"  permuted signs ({args.permutations}x):     |rho| null median "
              f"{np.median(null_rho):.3f} ms, p95 {np.percentile(null_rho, 95):.3f} ms")
        print(f"                              resid null median {np.median(null_impr):+.2f}%")
        print(f"  --> permutation p-value for |rho| = {pval:.4f}"
              f"   ({'significant' if pval < 0.05 else 'NOT significant'})")
    else:
        pval = float("nan")

    print("\n  split agreement (both subsets must give the same rho):")
    opp = (s_a * s_b) < 0
    split_rho = {}
    for lbl, sel in (("same-sign pairs", ~opp), ("opposite-sign pairs", opp)):
        if sel.sum() < 500:
            print(f"    {lbl:22s} too few matches ({int(sel.sum())})")
            continue
        _, gs = np.unique(key[m][sel] if False else group[sel], return_inverse=True)
        f = joint_fit(gs, v_perp[sel], x_rho[sel], y[sel],
                      int(gs.max()) + 1, True, huber, iters=8)
        r = f["rho_frames"] / fps * 1000.0
        se_s = f["se_rho_frames"] / fps * 1000.0
        split_rho[lbl] = r
        print(f"    {lbl:22s} n={int(sel.sum()):8,d}   rho {r:+7.3f} +/- {se_s:.3f} ms")
    if len(split_rho) == 2:
        a, b = list(split_rho.values())
        print(f"    difference = {abs(a - b):.3f} ms "
              f"({'consistent' if abs(a - b) < 2.0 else 'INCONSISTENT -- do not trust rho'})")

    # ---- primary test: free per-camera readout, l/r map NOT imposed -------
    print("\n" + "=" * 70)
    print("  PRIMARY TEST: free readout coefficient per camera")
    print("=" * 70)
    rho_c, se_c, rms_before, rms_after = per_camera_fit(
        group, v_perp, (u_b - W / 2.0) / W, (u_a - W / 2.0) / W,
        cam_a, cam_b, y, n_groups, n_cams, huber,
    )
    rho_c_ms, se_c_ms = rho_c / fps * 1000.0, se_c / fps * 1000.0
    agree = int(np.sum(np.sign(rho_c_ms) == np.sign(signs)))
    # one-sided binomial tail under a fair coin
    from math import comb
    p_sign = sum(comb(n_cams, k) for k in range(agree, n_cams + 1)) / 2 ** n_cams
    print(f"  residual RMS {rms_before:.4f} -> {rms_after:.4f} px")
    print(f"  SIGN AGREEMENT with the l/r map: {agree}/{n_cams}  "
          f"(binomial p = {p_sign:.3e})")
    mag = np.abs(rho_c_ms)
    print(f"  |rho| across cameras: median {np.median(mag):.2f} ms  "
          f"IQR {np.percentile(mag, 25):.2f}-{np.percentile(mag, 75):.2f}  "
          f"(frame period {1000.0 / fps:.2f} ms)")
    if args.verbose_per_camera:
        print("\n    folder  l/r      rho_c (ms)")
        for i in np.argsort(-rho_c_ms):
            print(f"      {folders[i]:>4s}    {'l' if signs[i] > 0 else 'r'}   "
                  f"{rho_c_ms[i]:+9.2f} +/- {se_c_ms[i]:6.2f}")
    sign_ok = p_sign < 1e-3

    print("\n" + "=" * 70)
    split_ok = (len(split_rho) < 2) or (
        abs(list(split_rho.values())[0] - list(split_rho.values())[1]) < 2.0
    )
    verdict_ok = (
        abs(rho_ms) > 3.0 * max(se_ms, 1e-9)
        and impr > 1.0
        and abs(rho_ms) < 1000.0 / fps
        and (not np.isfinite(pval) or pval < 0.05)
    )
    # A global rho whose own subsets disagree is not a measurement. Report the
    # sign result (which does not depend on magnitude) and withhold the number.
    if sign_ok and not (verdict_ok and split_ok):
        print("  VERDICT: rolling shutter CONFIRMED by the sign test "
              f"({agree}/{n_cams}, p={p_sign:.1e}) --")
        print("           per-camera readout direction follows the l/r map even")
        print("           though the solver was never given it.")
        print(f"           MAGNITUDE IS NOT PINNED: per-camera |rho| spans "
              f"{np.percentile(mag, 10):.1f}-{np.percentile(mag, 90):.1f} ms")
        print(f"           and the sign-imposed subsets disagree. Use the median "
              f"({np.median(mag):.0f} ms) as a working value, not a calibration.")
        verdict_ok = False
        sign_only = True
    else:
        sign_only = False
    if not sign_ok and not verdict_ok:
        pass
    if sign_only:
        pass
    elif verdict_ok:
        print(f"  VERDICT: rolling shutter DETECTED, rho ~ {abs(rho_ms):.1f} ms full-span.")
        print("           Per-camera scalar time offsets cannot represent this")
        print("           (it varies with column INSIDE each image), so it is an")
        print("           irreducible error on moving content until the renderer")
        print("           evaluates per-column capture time.")
    elif abs(rho_ms) >= 1000.0 / fps:
        print("  VERDICT: implausible rho (>= one frame). The fit is unidentifiable")
        print("           here -- check the within-group g spread above and use")
        print("           frames with more, and more varied, subject motion.")
    else:
        print("  VERDICT: NO rolling-shutter signal. rho is within noise of zero and")
        print("           the sign split shows no excess. Look elsewhere.")
    print("=" * 70 + "\n")

    (out_dir / "rs_fit.json").write_text(json.dumps({
        "rho_ms": rho_ms, "se_ms": se_ms,
        "resid_rms_px_rho0": m0["resid_robust_rms_px"],
        "resid_rms_px_rhofree": m1["resid_robust_rms_px"],
        "improvement_pct": impr,
        "n_matches": int(m1["n"]), "n_groups": int(m1["n_groups"]),
        "permutation_p": pval,
        "rho_ms_unsigned_control": rho_uns,
        "rho_ms_by_split": split_rho,
        "per_camera": {
            str(folders[i]): {"rho_ms": float(rho_c_ms[i]), "se_ms": float(se_c_ms[i]),
                              "map_sign": ("l" if signs[i] > 0 else "r")}
            for i in range(n_cams)
        },
        "sign_agreement": f"{agree}/{n_cams}",
        "sign_agreement_p": p_sign,
        "rho_ms_median_abs_per_camera": float(np.median(mag)),
    }, indent=2), encoding="utf-8")
    print(f"[fit] wrote {out_dir / 'rs_fit.json'}")


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", choices=("collect", "fit", "both"), default="both")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--fps", type=float, default=60.0)

    # collect
    p.add_argument("--reference-json", type=str)
    p.add_argument("--images-root", type=str)
    p.add_argument("--image-pattern", type=str, default="{frame:06d}.jpg")
    p.add_argument("--rotation-json", type=str,
                   help="JSON mapping each camera to 'l'/'r' extract rotation, "
                        "same format prep-roma-trigs-memory.py --rotation-json takes.")
    p.add_argument("--flip-sign", action="store_true",
                   help="Invert the l/r -> readout-direction convention.")
    p.add_argument("--frames", type=str,
                   help="Comma-separated frames. Use frames where the subject MOVES.")
    p.add_argument("--camera-neighbors", type=int, default=6)
    p.add_argument("--wide-baseline-per-camera", type=int, default=0)
    p.add_argument("--max-pair-angle-deg", type=float, default=70.0)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--temporal-frame-delta", type=int, default=1)
    p.add_argument("--min-flow-px", type=float, default=1.0)
    p.add_argument("--max-abs-epipolar-px", type=float, default=60.0)
    p.add_argument("--roma-backend", choices=("v1", "v2"), default="v2")
    p.add_argument("--roma-model", choices=("outdoor", "indoor", "tiny-outdoor"),
                   default="outdoor")
    p.add_argument("--roma-v2-setting", type=str, default="precise")
    p.add_argument("--roma-num-matches", type=int, default=10000)
    p.add_argument("--roma-min-certainty", type=float, default=0.14)
    p.add_argument("--flow-min-certainty", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")

    # fit
    p.add_argument("--min-vperp-px", type=float, default=1.0,
                   help="Drop matches below this |v_perp| (px/frame); they carry "
                        "no timing signal and only add noise.")
    p.add_argument("--huber-px", type=float, default=2.0)
    p.add_argument("--permutations", type=int, default=200,
                   help="Permutation resamples of the per-camera l/r signs, for a "
                        "null distribution on rho. 0 disables.")
    p.add_argument("--verbose-per-camera", action="store_true",
                   help="Print every camera's fitted readout coefficient.")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="Bootstrap resamples over camera pairs for a rho CI. 0 disables.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("collect", "both"):
        for req in ("reference_json", "images_root", "rotation_json", "frames"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} is required for collect")
        collect(args)
    if args.stage in ("fit", "both"):
        fit(args)


if __name__ == "__main__":
    main()
