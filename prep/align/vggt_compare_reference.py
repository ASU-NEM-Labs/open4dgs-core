#!/usr/bin/env python3
"""Structural sanity gate for a VGGT solve, against a reference export.

Why this exists
---------------
Grading a solve properly costs a matcher run and a held-out third-view test.
This gate costs nothing -- pure numpy on two JSON files -- and it can reject a
failed VGGT run before any of that is spent. It answers one question: did VGGT
produce a coherent rig at all?

What it is NOT
--------------
This is not an accuracy measurement. The 4DV manifest is itself 1-1.5 degrees
out per camera (measured: 281 px p50 held-out three-view, vs 3.06 px for a
solved control). So disagreement of roughly 1-2 degrees is the EXPECTED result
for a good VGGT solve, and near-zero agreement would be the suspicious outcome,
not the good one -- VGGT never saw the manifest and cannot reproduce it.

Read it as a gate, not a score:

  centres form a ring, residual a few % of radius, rotations within a few deg
      -> coherent. Proceed to a real held-out referee.
  centres collapsed or scrambled, rotations tens of degrees
      -> VGGT failed on this rig. Stop; do not spend a matcher run.

The focal test
--------------
This rig has a 3.42x focal spread (fx 3542-12107) and 26-42 deg FOV, which is
unusual input for a model trained on ordinary imagery. VGGT predicts field of
view rather than reading it, so the failure mode to look for is regression to
the training mean: predicted focals pulled toward a central value, which shows
up as a strongly negative slope of log(ratio) against log(fx_reference).

The manifest's intrinsics are trustworthy in a way its poses are not -- frames
are confirmed rectified (sagitta p50 0.159 px across fx 3542-12107) and cx is
exactly (W-1)/2 for all 96 cameras -- so this comparison grades VGGT, not the
manifest.

Gauge
-----
VGGT's world frame is arbitrary (camera 0 at the origin, scale unresolved), so
centres are compared through a similarity transform fitted by Umeyama. Because
the reference contains known-bad cameras, a trimmed refit on the best-fitting
80% is also reported; if the two disagree a lot, a few cameras are dominating.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vggt", required=True, help="JSON from vggt_solve.py")
    p.add_argument("--reference", required=True,
                   help="reference_export-style JSON to compare against "
                        "(e.g. reference_export-4dv-manifest.json)")
    p.add_argument("--out", default=None, help="Optional JSON report path")
    p.add_argument("--trim", type=float, default=0.8,
                   help="Fraction of best-fitting cameras used for the trimmed refit")
    p.add_argument("--worst", type=int, default=10, help="How many worst cameras to list")
    return p.parse_args()


def load_cameras(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    out = {}
    for cam in data["cameras"]:
        R = np.asarray(cam["R"], dtype=np.float64)
        t = np.asarray(cam["t"], dtype=np.float64)
        out[cam["folder"]] = {
            "R": R,
            "t": t,
            "C": -R.T @ t,
            "params": np.asarray(cam["params"], dtype=np.float64),
        }
    return out


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform mapping src -> dst. Returns (scale, R, t)."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    C = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (X ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def rot_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    d = A @ B.T
    return float(np.degrees(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0))))


def evaluate(folders, vggt, ref, scale, Ra, ta):
    Cv = np.array([vggt[f]["C"] for f in folders])
    Cr = np.array([ref[f]["C"] for f in folders])
    pred = (scale * (Ra @ Cv.T).T) + ta
    resid = np.linalg.norm(pred - Cr, axis=1)
    ang = np.array([rot_angle_deg(ref[f]["R"], vggt[f]["R"] @ Ra.T) for f in folders])
    return resid, ang


def stats(x: np.ndarray) -> dict:
    return {
        "p50": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def main() -> None:
    args = parse_args()
    vggt = load_cameras(args.vggt)
    ref = load_cameras(args.reference)

    folders = sorted(set(vggt) & set(ref))
    if len(folders) < 4:
        sys.exit(f"only {len(folders)} shared cameras; need at least 4")
    missing = sorted(set(ref) - set(vggt))
    print(f"comparing {len(folders)} shared cameras "
          f"(vggt {len(vggt)}, reference {len(ref)})")
    if missing:
        print(f"  not in vggt solve: {', '.join(missing[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")

    Cv = np.array([vggt[f]["C"] for f in folders])
    Cr = np.array([ref[f]["C"] for f in folders])

    scale, Ra, ta = umeyama(Cv, Cr)
    resid, ang = evaluate(folders, vggt, ref, scale, Ra, ta)

    keep = max(4, int(round(args.trim * len(folders))))
    idx = np.argsort(resid)[:keep]
    sc2, Ra2, ta2 = umeyama(Cv[idx], Cr[idx])
    resid2, ang2 = evaluate(folders, vggt, ref, sc2, Ra2, ta2)

    # Ring geometry, measured independently in each frame.
    ring_ref = np.linalg.norm(Cr - Cr.mean(0), axis=1)
    ring_vggt = np.linalg.norm(Cv - Cv.mean(0), axis=1) * scale

    print()
    print("=== camera centres (similarity-aligned) ===")
    print(f"  reference ring radius   p50 {np.median(ring_ref):.3f} m   "
          f"spread p10-p90 {np.percentile(ring_ref, 10):.3f}-{np.percentile(ring_ref, 90):.3f}")
    print(f"  vggt ring radius        p50 {np.median(ring_vggt):.3f} m   "
          f"spread p10-p90 {np.percentile(ring_vggt, 10):.3f}-{np.percentile(ring_vggt, 90):.3f}")
    r = float(np.median(ring_ref))
    for label, res in (("all cameras", resid), (f"trimmed {keep}/{len(folders)}", resid2)):
        s = stats(res)
        print(f"  residual {label:<22} p50 {s['p50']:.3f} m  p90 {s['p90']:.3f}  max {s['max']:.3f}"
              f"   ({100 * s['p50'] / r:.1f}% of ring radius at p50)")

    print()
    print("=== camera rotations ===")
    print("  (manifest is itself ~1-1.5 deg out, so 1-2 deg here is the expected good result)")
    for label, a in (("all cameras", ang), (f"trimmed {keep}/{len(folders)}", ang2)):
        s = stats(a)
        print(f"  angular diff {label:<18} p50 {s['p50']:.2f} deg  p90 {s['p90']:.2f}  max {s['max']:.2f}")
    for thr in (2.0, 5.0, 15.0):
        print(f"    within {thr:>4.0f} deg: {int((ang2 <= thr).sum())}/{len(folders)}")

    print()
    print("=== intrinsics (regression-to-mean test) ===")
    fx_v = np.array([vggt[f]["params"][0] for f in folders])
    fx_r = np.array([ref[f]["params"][0] for f in folders])
    ratio = fx_v / fx_r
    lr = np.log(ratio)
    lf = np.log(fx_r)
    slope = float(np.polyfit(lf, lr, 1)[0])
    corr = float(np.corrcoef(lf, lr)[0, 1]) if lr.std() > 1e-12 else float("nan")
    print(f"  reference fx  {fx_r.min():.0f} .. {fx_r.max():.0f}  ({fx_r.max() / fx_r.min():.2f}x spread)")
    print(f"  vggt fx       {fx_v.min():.0f} .. {fx_v.max():.0f}  ({fx_v.max() / fx_v.min():.2f}x spread)")
    print(f"  ratio vggt/reference   p50 {np.median(ratio):.3f}   "
          f"p10-p90 {np.percentile(ratio, 10):.3f}-{np.percentile(ratio, 90):.3f}")
    print(f"  slope d(log ratio)/d(log fx) = {slope:+.3f}   corr {corr:+.3f}")
    print("    slope ~  0  -> focal spread tracked correctly")
    print("    slope ~ -1  -> focals collapsed to a constant (regression to training mean)")

    # This rig uses only a handful of discrete zoom settings. Breaking the gate
    # down by setting separates "VGGT failed on this capture" from "VGGT handled
    # the wide cameras and struggled on the telephoto tail", which are different
    # problems with different fixes.
    print()
    print("=== by zoom setting ===")
    groups = {}
    for i, f in enumerate(folders):
        groups.setdefault(int(round(fx_r[i])), []).append(i)
    print(f"  {'fx':>7} {'HFOV':>6} {'n':>4} {'centre_p50':>11} {'rot_p50':>9} {'rot_p90':>9} {'fx_ratio':>9}")
    for key in sorted(groups):
        g = np.array(groups[key])
        hfov = 2 * np.degrees(np.arctan(0.5 * 3840.0 / key))
        print(f"  {key:>7} {hfov:>5.1f}d {len(g):>4} {np.median(resid2[g]):>11.3f} "
              f"{np.median(ang2[g]):>9.2f} {np.percentile(ang2[g], 90):>9.2f} "
              f"{np.median(ratio[g]):>9.3f}")

    print()
    print(f"=== {args.worst} worst cameras (trimmed alignment) ===")
    order = np.argsort(-resid2)[:args.worst]
    print(f"  {'cam':<6} {'centre_err_m':>13} {'rot_err_deg':>12} {'fx_ratio':>9}")
    for i in order:
        print(f"  {folders[i]:<6} {resid2[i]:>13.3f} {ang2[i]:>12.2f} {ratio[i]:>9.3f}")

    if args.out:
        report = {
            "n_cameras": len(folders),
            "alignment_scale": scale,
            "alignment_scale_trimmed": sc2,
            "ring_radius_reference_p50": float(np.median(ring_ref)),
            "ring_radius_vggt_p50": float(np.median(ring_vggt)),
            "centre_residual_m": stats(resid),
            "centre_residual_m_trimmed": stats(resid2),
            "rotation_deg": stats(ang),
            "rotation_deg_trimmed": stats(ang2),
            "fx_ratio_p50": float(np.median(ratio)),
            "fx_ratio_log_slope": slope,
            "fx_ratio_log_corr": corr,
            "per_camera": {
                folders[i]: {
                    "centre_err_m": float(resid2[i]),
                    "rotation_err_deg": float(ang2[i]),
                    "fx_ratio": float(ratio[i]),
                }
                for i in range(len(folders))
            },
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
