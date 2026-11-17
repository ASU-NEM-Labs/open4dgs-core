#!/usr/bin/env python3
"""Held-out third-view referee. Grades any reference_export without fitting it.

Why this exists
---------------
Every accuracy number in the calibration work so far was validated against RoMa
correspondences while the solve being graded was itself fit on RoMa
correspondences. Different frames and content, same matcher -- not a neutral
referee. When COLMAP's SIFT tracks and RoMa disagreed it was impossible to
adjudicate.

This is the cleanest test available: triangulate a static clean-plate point from
two cameras, then reproject it into a THIRD camera that took no part in that
triangulation. Nothing is fitted here, so a solve cannot score well by having
absorbed the data it is graded on.

Why it is comparable across solves
----------------------------------
Reprojection error is invariant to similarity transforms of the world. A solve
in an arbitrary frame with arbitrary scale (VGGT) is therefore directly
comparable against one in metric manifest coordinates -- no alignment step, no
gauge choice, nothing to tune. That is what makes this the right adjudicator.

Tracks are built from 2D correspondences only, so the track set and the chosen
triples are identical for every reference graded in one invocation. The
comparison is controlled: same points, same triples, only the cameras differ.

Deliberately no pose-dependent filtering
----------------------------------------
bundle_adjust_static.py drops tracks whose initial triangulation reprojects
badly. That is correct when seeding a solve and WRONG here -- it would advantage
whichever pose set did the filtering. Every track with enough views is graded.

Static content only
-------------------
Correspondences come from the clean-plate match cache. A correspondence is
geometrically valid iff both pixels image the same 3D point; for time-invariant
geometry that holds regardless of when each camera exposed, so free-running
cameras introduce no error here and no timing assumption enters the measurement.

Reference points (measured previously, same test)
-------------------------------------------------
    4DV manifest        p50  281.46 px    p90 1069.38    triples <10px   0.8%
    RoMa solve          p50    3.06 px    p90   20.47    triples <10px  88.2%
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle_adjust_static import build_tracks, rodrigues, inv_rodrigues  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Dataset root")
    p.add_argument("--reference", required=True, action="append",
                   help="reference_export-style JSON to grade. Repeat to compare several "
                        "on identical tracks and triples.")
    p.add_argument("--report-json", default="dynamic_calibration_report.json",
                   help="Match report listing accepted pairs and their cache files")
    p.add_argument("--cache-dir", default=None, help="Override the directory holding .npz caches")
    p.add_argument("--min-certainty", type=float, default=0.6)
    p.add_argument("--max-points-per-pair", type=int, default=800)
    p.add_argument("--track-quant-px", type=float, default=1.0)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--max-tracks", type=int, default=25000)
    p.add_argument("--triples-per-track", type=int, default=1,
                   help="How many held-out triples to draw from each track")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional JSON report path")
    p.add_argument("--per-camera", action="store_true",
                   help="Break the result down by held-out camera. The rig-wide p50 can "
                        "sit near the metric floor while a minority of cameras carry a "
                        "large tail -- that distinction changes what to fix.")
    p.add_argument("--no-distortion", action="store_true",
                   help="Grade every reference as pure PINHOLE, discarding distortion "
                        "coefficients. Off by default: a solve whose poses were optimised "
                        "jointly with distortion must be graded WITH it, or part of its "
                        "model is stripped and its error inflated.")
    return p.parse_args()


def load_reference(path: str, use_distortion: bool):
    with open(path) as f:
        data = json.load(f)
    cams, n_dist = {}, 0
    for c in data["cameras"]:
        p = c["params"]
        d = np.asarray(p[4:], dtype=np.float64) if len(p) > 4 else np.zeros(0)
        if d.size and np.abs(d).max() > 1e-12:
            n_dist += 1
        cams[c["folder"]] = {
            "R": np.asarray(c["R"], dtype=np.float64),
            "t": np.asarray(c["t"], dtype=np.float64),
            "k": np.array([p[0], p[1], p[2], p[3]], dtype=np.float64),
            "d": d if use_distortion else np.zeros(0),
        }
    if n_dist:
        state = "APPLIED" if use_distortion else "IGNORED (graded as PINHOLE)"
        print(f"  {n_dist}/{len(cams)} cameras carry distortion coefficients -- {state}")
    return cams


def distort(x: float, y: float, d: np.ndarray):
    """OpenCV radial-tangential forward model on normalised coords."""
    if d.size == 0:
        return x, y
    k1 = d[0] if d.size > 0 else 0.0
    k2 = d[1] if d.size > 1 else 0.0
    p1 = d[2] if d.size > 2 else 0.0
    p2 = d[3] if d.size > 3 else 0.0
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_pixel(u, cam, tol_px: float = 1.0):
    """Invert the forward model to recover an ideal-pinhole pixel.

    Returns None when the model cannot be inverted at this point. Some solved
    references on this rig carry coefficients like k2 = -46.98, whose radial
    curve is non-monotonic; the forward model is then not injective and no
    unique inverse exists. Those observations are counted, not silently
    replaced with garbage. Uses cv2.undistortPoints so behaviour matches
    bundle_adjust_static.py, then validates by re-distorting.
    """
    R, t, k, d = cam
    if d.size == 0 or np.abs(d).max() < 1e-12:
        return u
    K = np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        out = cv2.undistortPoints(np.asarray(u, np.float64).reshape(1, 1, 2), K, d, P=K)
        uu = np.asarray(out, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(uu)):
            return None
        xd, yd = distort((uu[0] - k[2]) / k[0], (uu[1] - k[3]) / k[1], d)
        if not (np.isfinite(xd) and np.isfinite(yd)):
            return None
        back = np.array([k[0] * xd + k[2], k[1] * yd + k[3]])
        if np.hypot(back[0] - u[0], back[1] - u[1]) > tol_px:
            return None
    return uu


def load_pair_matches(root, report_json, cache_dir, folders, min_certainty, max_pts):
    fidx = {f: i for i, f in enumerate(folders)}
    with open(os.path.join(root, report_json)) as f:
        rep = json.load(f)
    pair_matches, n_obs, n_missing = [], 0, 0
    for m in rep["matches"]:
        if not m.get("accepted"):
            continue
        a, b = m["folder_a"], m["folder_b"]
        if a not in fidx or b not in fidx:
            continue
        cp = m["cache"]
        path = os.path.join(cache_dir, os.path.basename(cp)) if cache_dir else os.path.join(root, cp)
        if not os.path.exists(path):
            n_missing += 1
            continue
        d = np.load(path)
        keep = d["certainty"] > min_certainty
        pa, pb = d["points_a"][keep], d["points_b"][keep]
        if len(pa) < 30:
            continue
        if len(pa) > max_pts:
            sel = np.linspace(0, len(pa) - 1, max_pts).astype(int)
            pa, pb = pa[sel], pb[sel]
        pair_matches.append((fidx[a], fidx[b], pa.astype(np.float64), pb.astype(np.float64)))
        n_obs += len(pa)
    if n_missing:
        print(f"  {n_missing} accepted pairs had no cache file on disk")
    return pair_matches, n_obs


def triangulate_two(u1, cam1, u2, cam2):
    """DLT from two views on ideal-pinhole pixels. cam = (R, t, k, d)."""
    rows = []
    for (u, v), (R, t, k, _d) in ((u1, cam1), (u2, cam2)):
        P = np.hstack([R, t.reshape(3, 1)])
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
        P = K @ P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def project(X, cam):
    """Project into real image pixels, including this camera's distortion."""
    R, t, k, d = cam
    c = R @ X + t
    if c[2] <= 1e-6:
        return None
    xd, yd = distort(c[0] / c[2], c[1] / c[2], d)
    return np.array([k[0] * xd + k[2], k[1] * yd + k[3]])


def grade(tracks, triples, cams_by_idx):
    """Triangulate from two views, reproject into a held-out third.

    Observations are undistorted before triangulation so the DLT sees ideal
    pinhole geometry, then the 3D point is projected back through the third
    camera's full model and compared against the ORIGINAL observed pixel. Error
    is therefore always measured in real image pixels and stays comparable
    across references that carry different distortion.
    """
    errs, n_cheirality, n_degenerate, n_uninvertible = [], 0, 0, 0
    per_cam = {}
    for tr_i, (i_a, i_b, i_c) in triples:
        tr = tracks[tr_i]
        (ca, ua), (cb, ub), (cc, uc) = tr[i_a], tr[i_b], tr[i_c]
        cam_a, cam_b, cam_c = cams_by_idx[ca], cams_by_idx[cb], cams_by_idx[cc]
        if cam_a is None or cam_b is None or cam_c is None:
            n_degenerate += 1
            continue
        pa, pb = undistort_pixel(ua, cam_a), undistort_pixel(ub, cam_b)
        if pa is None or pb is None:
            n_uninvertible += 1
            continue
        X = triangulate_two(pa, cam_a, pb, cam_b)
        if X is None:
            n_degenerate += 1
            continue
        p = project(X, cam_c)
        if p is None:
            n_cheirality += 1
            continue
        e = float(np.hypot(p[0] - uc[0], p[1] - uc[1]))
        errs.append(e)
        per_cam.setdefault(cc, []).append(e)
    return np.asarray(errs), n_cheirality, n_degenerate, n_uninvertible, per_cam


def main() -> int:
    args = parse_args()
    root = args.data_dir

    refs = {}
    for path in args.reference:
        full = path if os.path.isabs(path) else os.path.join(root, path)
        if not os.path.exists(full):
            sys.exit(f"missing reference: {full}")
        print(f"loading {os.path.basename(full)}")
        refs[os.path.basename(full)] = load_reference(full, not args.no_distortion)

    folders = sorted(set.intersection(*[set(r) for r in refs.values()]))
    print(f"\n{len(folders)} cameras common to all {len(refs)} references")

    pair_matches, n_obs = load_pair_matches(
        root, args.report_json, args.cache_dir, folders, args.min_certainty, args.max_points_per_pair)
    print(f"static pairs: {len(pair_matches)}  correspondences: {n_obs}")
    if not pair_matches:
        sys.exit("no usable pairs -- check --report-json and --cache-dir")

    tracks = build_tracks(pair_matches, len(folders), args.track_quant_px, args.min_views)
    print(f"tracks with >= {args.min_views} views: {len(tracks)}")
    rng = np.random.default_rng(args.seed)
    if len(tracks) > args.max_tracks:
        tracks = [tracks[i] for i in rng.choice(len(tracks), args.max_tracks, replace=False)]
        print(f"  subsampled to {len(tracks)}")

    lens = np.array([len(t) for t in tracks])
    print(f"track length: p50 {np.median(lens):.0f}  p90 {np.percentile(lens, 90):.0f}  max {lens.max()}")

    # Triples are chosen once, from 2D structure only, so every reference is
    # graded on exactly the same held-out problems.
    triples = []
    for ti, tr in enumerate(tracks):
        n = len(tr)
        if n < 3:
            continue
        for _ in range(args.triples_per_track):
            triples.append((ti, tuple(rng.choice(n, 3, replace=False).tolist())))
    print(f"held-out triples: {len(triples)}\n")

    results = {}
    print(f"{'reference':<44} {'p50':>9} {'p90':>10} {'p99':>10} {'<5px':>7} {'<10px':>7} {'<20px':>7}")
    print("-" * 100)
    for name, cams in refs.items():
        cams_by_idx = [
            (cams[f]["R"], cams[f]["t"], cams[f]["k"], cams[f]["d"]) if f in cams else None
            for f in folders
        ]
        errs, n_ch, n_deg, n_uninv, per_cam = grade(tracks, triples, cams_by_idx)
        if errs.size == 0:
            print(f"{name:<44} {'no valid triples':>50}")
            continue
        p50, p90, p99 = np.percentile(errs, [50, 90, 99])
        f5 = float((errs < 5).mean()) * 100
        f10 = float((errs < 10).mean()) * 100
        f20 = float((errs < 20).mean()) * 100
        print(f"{name:<44} {p50:>9.2f} {p90:>10.2f} {p99:>10.2f} "
              f"{f5:>6.1f}% {f10:>6.1f}% {f20:>6.1f}%")
        results[name] = {
            "n_triples_scored": int(errs.size),
            "n_behind_camera": int(n_ch),
            "n_degenerate": int(n_deg),
            "n_uninvertible_distortion": int(n_uninv),
            "p50": float(p50), "p90": float(p90), "p99": float(p99),
            "frac_under_5px": f5, "frac_under_10px": f10, "frac_under_20px": f20,
            "mean": float(errs.mean()),
            "per_camera": {folders[c]: {"n": len(v), "p50": float(np.median(v)),
                                        "p90": float(np.percentile(v, 90))}
                           for c, v in sorted(per_cam.items())},
        }

    print()
    for name, r in results.items():
        if r["n_behind_camera"] or r["n_degenerate"] or r["n_uninvertible_distortion"]:
            print(f"  {name}: {r['n_behind_camera']} behind camera, "
                  f"{r['n_degenerate']} degenerate, "
                  f"{r['n_uninvertible_distortion']} uninvertible-distortion "
                  f"(all excluded from percentiles)")

    if args.per_camera:
        for name, r in results.items():
            pc = r["per_camera"]
            med = np.array([v["p50"] for v in pc.values()])
            print(f"\n=== per-camera, {name} ===")
            print(f"  camera medians: p50 {np.median(med):.2f}  p90 {np.percentile(med, 90):.2f}"
                  f"  max {med.max():.2f}")
            for thr in (5.0, 10.0, 20.0):
                print(f"    cameras with median > {thr:>4.0f} px: "
                      f"{int((med > thr).sum())}/{len(med)}")
            worst = sorted(pc.items(), key=lambda kv: -kv[1]["p50"])[:12]
            print(f"  {'cam':<6} {'n':>7} {'p50':>8} {'p90':>9}")
            for cam, v in worst:
                print(f"  {cam:<6} {v['n']:>7} {v['p50']:>8.2f} {v['p90']:>9.2f}")

    print("\nreference points from the same test, measured previously:")
    print("  4DV manifest      p50  281.46 px   p90 1069.38   <10px   0.8%")
    print("  RoMa solve        p50    3.06 px   p90   20.47   <10px  88.2%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "n_cameras": len(folders),
                "n_pairs": len(pair_matches),
                "n_correspondences": n_obs,
                "n_tracks": len(tracks),
                "n_triples": len(triples),
                "min_views": args.min_views,
                "min_certainty": args.min_certainty,
                "seed": args.seed,
                "results": results,
            }, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
