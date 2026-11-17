#!/usr/bin/env python3
"""Decompose third-view error into pose error and observation noise.

Why this exists
---------------
referee_third_view.py reports a single number, and the plan for reaching
sub-pixel assumes that number is dominated by observation noise rather than by
the cameras. That assumption has never been tested, and it determines whether
weeks of work on better correspondences would pay for itself.

There is a second, sharper reason. **The referee measures using RoMa
correspondences**, so its result already contains RoMa's own localisation error.
Triangulate-then-reproject amplifies observation noise, so there is a floor
below which this metric cannot read even with perfect cameras. Chasing
"sub-pixel" without knowing that floor risks chasing a target that does not
exist.

The model
---------
For a third-view test, error contributions add roughly in quadrature:

    err(sigma)^2  =  a^2  +  (k * sigma)^2

    a      pose + structure error, in px, what better correspondences CANNOT fix
    k      geometric amplification of observation noise (typically 1.5-2)
    sigma  observation noise of the correspondences used to measure

Two independent measurements pin both unknowns.

**Measurement 1 -- synthetic noise sweep.** Inject known Gaussian noise of
increasing sigma into the correspondences and refit. The slope gives k and the
intercept gives (a^2 + k^2 sigma_roma^2), i.e. the current reading.

**Measurement 2 -- a second correspondence set with different noise.** Grade the
same poses on the VGGT/LightGlue track observations, whose noise is larger by a
known factor (they are tracked at 1024 on 3840 imagery, so ~3.75x). That second
equation separates a from sigma_roma.

How to read the result
----------------------
    a  much smaller than the current reading
        -> observation-limited. Better correspondences pay off, and the floor
           for this metric is k * sigma_roma.
    a  close to the current reading
        -> pose-limited. Better correspondences will NOT help; the cameras are
           the problem and Steps 1-3 of the plan are not worth running.

Nothing here is fitted to the data being graded: poses come in fixed, and the
sweep only ever adds noise to observations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referee_third_view import (  # noqa: E402
    load_reference, load_pair_matches, triangulate_two, project, undistort_pixel)
from bundle_adjust_static import build_tracks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--reference", required=True,
                   help="The solve to probe (poses held fixed throughout)")
    p.add_argument("--sparse", default=None,
                   help="Optional COLMAP model whose track observations form the "
                        "second, noisier correspondence set")
    p.add_argument("--noise-ratio", type=float, default=3.75,
                   help="How much noisier --sparse observations are than the RoMa ones. "
                        "Default 3.75 = tracking at 1024 on 3840 imagery.")
    p.add_argument("--report-json", default="dynamic_calibration_report.json")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--min-certainty", type=float, default=0.6)
    p.add_argument("--max-points-per-pair", type=int, default=800)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--max-tracks", type=int, default=8000)
    p.add_argument("--sigmas", default="0,0.5,1,2,4",
                   help="Injected noise levels in px")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def score(tracks, triples, cams_by_idx, sigma, rng):
    """Third-view p50 with sigma px of Gaussian noise injected into observations."""
    errs = []
    for tr_i, (i_a, i_b, i_c) in triples:
        tr = tracks[tr_i]
        (ca, ua), (cb, ub), (cc, uc) = tr[i_a], tr[i_b], tr[i_c]
        A, B, C = cams_by_idx[ca], cams_by_idx[cb], cams_by_idx[cc]
        if A is None or B is None or C is None:
            continue
        if sigma > 0:
            n = rng.normal(0.0, sigma, 6)
            ua = np.asarray(ua, float) + n[0:2]
            ub = np.asarray(ub, float) + n[2:4]
            uc = np.asarray(uc, float) + n[4:6]
        pa, pb = undistort_pixel(ua, A), undistort_pixel(ub, B)
        if pa is None or pb is None:
            continue
        X = triangulate_two(pa, A, pb, B)
        if X is None:
            continue
        p = project(X, C)
        if p is None:
            continue
        errs.append(float(np.hypot(p[0] - uc[0], p[1] - uc[1])))
    return np.asarray(errs)


def main() -> int:
    args = parse_args()
    root = args.data_dir
    rpath = args.reference if os.path.isabs(args.reference) else os.path.join(root, args.reference)
    cams = load_reference(rpath, True)
    print(f"probing {os.path.basename(rpath)}  ({len(cams)} cameras, poses held fixed)\n")

    folders = sorted(cams)
    pm, n_obs = load_pair_matches(root, args.report_json, args.cache_dir, folders,
                                  args.min_certainty, args.max_points_per_pair)
    tracks = build_tracks(pm, len(folders), 1.0, args.min_views)
    rng = np.random.default_rng(args.seed)
    if len(tracks) > args.max_tracks:
        tracks = [tracks[i] for i in rng.choice(len(tracks), args.max_tracks, replace=False)]
    triples = [(i, tuple(rng.choice(len(t), 3, replace=False).tolist()))
               for i, t in enumerate(tracks)]
    cbi = [(cams[f]["R"], cams[f]["t"], cams[f]["k"], cams[f]["d"]) for f in folders]
    print(f"RoMa correspondences: {n_obs} in {len(pm)} pairs -> "
          f"{len(tracks)} tracks, {len(triples)} triples\n")

    sigmas = [float(s) for s in args.sigmas.split(",")]
    print("=== measurement 1: synthetic noise sweep ===")
    print(f"  {'sigma_px':>9} {'p50':>8} {'p90':>9} {'n':>7}")
    res = []
    for s in sigmas:
        e = score(tracks, triples, cbi, s, np.random.default_rng(args.seed + 1))
        res.append((s, float(np.median(e))))
        print(f"  {s:>9.2f} {np.median(e):>8.3f} {np.percentile(e, 90):>9.3f} {e.size:>7}")

    # err^2 = C + k^2 sigma^2  -- least squares in sigma^2
    S = np.array([r[0] for r in res]) ** 2
    E = np.array([r[1] for r in res]) ** 2
    k2, C = np.polyfit(S, E, 1)
    k = float(np.sqrt(max(k2, 1e-12)))
    C = float(C)
    base = res[0][1]
    print(f"\n  fit err^2 = {C:.3f} + {k2:.3f} * sigma^2")
    print(f"  amplification k = {k:.2f}   (noise-free reading = {np.sqrt(max(C, 0)):.3f} px)")

    if not args.sparse:
        print("\nNo --sparse given, so a and sigma_roma cannot be separated.")
        print("Re-run with --sparse <colmap model> to complete the decomposition.")
        return 0

    # ---- measurement 2: the noisier correspondence set ----------------------
    import pycolmap
    rec = pycolmap.Reconstruction(args.sparse)
    print(f"\n=== measurement 2: {os.path.basename(args.sparse)} track observations ===")
    fidx = {f: i for i, f in enumerate(folders)}
    vt = []
    for pt in rec.points3D.values():
        obs = []
        for el in pt.track.elements:
            img = rec.images[el.image_id]
            f = os.path.splitext(img.name)[0]
            if f in fidx:
                obs.append((fidx[f], np.asarray(img.points2D[el.point2D_idx].xy, float)))
        if len(obs) >= 3:
            vt.append(obs)
    rng2 = np.random.default_rng(args.seed)
    if len(vt) > args.max_tracks:
        vt = [vt[i] for i in rng2.choice(len(vt), args.max_tracks, replace=False)]
    vtrip = [(i, tuple(rng2.choice(len(t), 3, replace=False).tolist())) for i, t in enumerate(vt)]
    e2 = score(vt, vtrip, cbi, 0.0, rng2)
    b2 = float(np.median(e2))
    print(f"  {len(vt)} tracks, {e2.size} triples -> p50 {b2:.3f} px  "
          f"p90 {np.percentile(e2, 90):.3f}")

    # base^2 = a^2 + k^2 sr^2 ;  b2^2 = a^2 + k^2 (ratio*sr)^2
    r = args.noise_ratio
    denom = k * k * (r * r - 1.0)
    print("\n=== decomposition ===")
    if denom <= 0 or b2 <= base:
        print(f"  cannot solve: second set is not measurably noisier "
              f"({b2:.3f} vs {base:.3f} px). Check --noise-ratio.")
        return 0
    sr2 = (b2 * b2 - base * base) / denom
    a2 = base * base - k * k * sr2
    sr = float(np.sqrt(max(sr2, 0.0)))
    a = float(np.sqrt(max(a2, 0.0)))
    print(f"  observation noise of RoMa correspondences   sigma_roma = {sr:.3f} px")
    print(f"  pose + structure error                              a  = {a:.3f} px")
    print(f"  measurement floor of the referee itself        k*sigma = {k * sr:.3f} px")
    print()
    frac = 100.0 * (k * k * sr2) / (base * base) if base > 0 else 0.0
    print(f"  current reading {base:.3f} px is {frac:.0f}% observation noise, "
          f"{100 - frac:.0f}% pose error")
    # sigma_roma is the noise of the correspondences doing the MEASURING, which
    # is a property of the ruler and cannot be improved by a better solve.
    # a is the solve's own error, and IS reducible -- by better correspondences
    # feeding the solve. The two are different correspondence sets; do not read
    # a large sigma_roma as "better correspondences will not help".
    print(f"\n  FLOOR: this metric cannot read below {k * sr:.2f} px. A solve with")
    print(f"  perfect cameras would still read {k * sr:.2f}, not 0 -- that is the ruler,")
    print("  not the solve. Sub-pixel on THIS metric is not a reachable target.")
    print(f"\n  HEADROOM: {base:.2f} -> {k * sr:.2f} px is the whole available gain "
          f"({100 * (1 - k * sr / base):.0f}%),")
    print(f"  and it comes from removing the {a:.2f} px of pose error, which better")
    print("  correspondences FEEDING THE SOLVE do reduce.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"reference": os.path.basename(rpath), "sweep": res,
                       "k": k, "noise_free_reading": float(np.sqrt(max(C, 0))),
                       "second_set_p50": b2, "noise_ratio": r,
                       "sigma_roma": sr, "pose_error_a": a,
                       "referee_floor": k * sr, "baseline": base}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
