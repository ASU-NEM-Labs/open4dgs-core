#!/usr/bin/env python3
"""Pose-guided match-pair selection, prioritised by which cameras are starved.

Why this exists
---------------
Measured on example capture: per-camera third-view error is governed almost
entirely by how many accepted match pairs a camera has.

    accepted pairs   cameras   median error
         0-4            1        34.36 px
         4-8           28         8.63 px
        8-12           47         2.92 px
         12+           20         2.21 px

corr(log triples, log error) = -0.860. There is a threshold near 8 pairs; above
it cameras sit at the metric's own noise floor, below it they fall apart. And the
starvation is not spread evenly -- it is the narrow-FOV cameras:

    56.9 deg  45 cams   2.19 px    3/45 over 5 px   11 accepted pairs
    25.9 deg  11 cams   9.62 px    8/11             6
    22.1 deg   9 cams  10.67 px    9/9              7
    18.0 deg   7 cams  12.87 px    7/7              7

Every camera in the two tightest groups is misregistered, and those are the
cameras that carry fine detail. Meanwhile only 708 of 4560 possible pairs are
attempted at all (15.5%). The telephotos are not failing to match so much as not
being asked to.

What this does
--------------
Uses an existing solve to predict co-visibility from real triangulated structure
rather than frustum heuristics, then ranks candidate pairs by how much they would
relieve the starved cameras. Emits, per pair:

  * predicted shared points, and the fraction of each image they cover
  * scale ratio between the two views, so cross-zoom pairs can be matched at
    comparable scale instead of across a 3.4x mismatch
  * a crop box per image bounding the shared region, which is what makes
    scale-normalised matching possible

Chicken-and-egg
---------------
Selection uses the current poses, which are imperfect -- but pair *selection* only
needs to be approximately right, and the current solve is 2.90 px held-out, far
better than the 1-1.5 degree prior that presumably produced the existing list.
Nothing here feeds back into the poses; it only decides which images to match.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referee_third_view import load_reference, distort  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--reference", required=True, help="Solve providing poses/intrinsics")
    p.add_argument("--points-from", required=True,
                   help="COLMAP sparse model supplying the 3D points used as the "
                        "co-visibility probe")
    p.add_argument("--existing-report", default="dynamic_calibration_report.json",
                   help="Report listing pairs already attempted/accepted")
    p.add_argument("--out", required=True, help="Output JSON pair list")
    p.add_argument("--out-pairs-arg", default=None,
                   help="Also write the selection as a comma-separated A:B string, ready "
                        "to pass to solve_4dv_dynamic_calibration.py --camera-pairs")
    p.add_argument("--target-pairs", type=int, default=14,
                   help="Accepted pairs each camera should end up with. The measured "
                        "threshold is ~8; default aims comfortably past it.")
    p.add_argument("--min-covis", type=int, default=40,
                   help="Minimum shared points for a pair to be worth attempting")
    p.add_argument("--max-new-pairs", type=int, default=1200)
    p.add_argument("--no-synthetic-fill", action="store_true",
                   help="Do not add back-projected probe points. Off by default: a "
                        "tracker's 3D points do not tile every camera's view, and a "
                        "camera with no probe points gets no pairs proposed at all.")
    p.add_argument("--grid", type=int, default=24,
                   help="Back-projected grid per camera for the synthetic probe")
    p.add_argument("--max-scale-ratio", type=float, default=6.0,
                   help="Reject pairs whose scale mismatch exceeds this even after "
                        "cropping; no matcher will survive them")
    return p.parse_args()


def read_points(sparse_dir: str) -> np.ndarray:
    """Read 3D points from a COLMAP model without needing pycolmap."""
    import struct
    path = os.path.join(sparse_dir, "points3D.bin")
    pts = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)
            f.read(8)
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(tl * 8)
            pts.append(xyz)
    return np.asarray(pts, dtype=np.float64)


def synthetic_probe(cams, folders, W, H, grid):
    """Back-project an image grid from every camera to the scene-centre depth.

    Real tracker points leave holes -- on example capture eight cameras saw none
    of VGGT's 38,908 points despite having good poses, so no pair could ever be
    proposed for them. This guarantees every camera has probe points in its own
    view by construction. Depths are approximate, which is fine: pair SELECTION
    only needs to be roughly right, and a wrong pair merely fails to match.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for f in folders:
        R = cams[f]["R"]
        C = -R.T @ cams[f]["t"]
        d = R[2] / np.linalg.norm(R[2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    centre = np.linalg.solve(A, b)

    out = []
    for f in folders:
        c = cams[f]
        R, k = c["R"], c["k"]
        C = -R.T @ c["t"]
        depth = float(np.linalg.norm(C - centre))
        us = np.linspace(0.05 * W, 0.95 * W, grid)
        vs = np.linspace(0.05 * H, 0.95 * H, grid)
        uu, vv = np.meshgrid(us, vs)
        xn = (uu.ravel() - k[2]) / k[0]
        yn = (vv.ravel() - k[3]) / k[1]
        dirs = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        out.append(dirs * depth @ R + C)
    return np.concatenate(out, axis=0), centre


def project_all(X: np.ndarray, cam) -> tuple:
    """Project N points. Returns (uv Nx2, valid mask, depth N)."""
    R, t, k, d = cam
    C = X @ R.T + t
    z = C[:, 2]
    ok = z > 1e-6
    xn = np.zeros(len(X))
    yn = np.zeros(len(X))
    xn[ok] = C[ok, 0] / z[ok]
    yn[ok] = C[ok, 1] / z[ok]
    if d.size and np.abs(d).max() > 1e-12:
        k1 = d[0] if d.size > 0 else 0.0
        k2 = d[1] if d.size > 1 else 0.0
        p1 = d[2] if d.size > 2 else 0.0
        p2 = d[3] if d.size > 3 else 0.0
        r2 = xn * xn + yn * yn
        rad = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    else:
        xd, yd = xn, yn
    uv = np.stack([k[0] * xd + k[2], k[1] * yd + k[3]], axis=1)
    return uv, ok, z


def main() -> int:
    args = parse_args()
    root = args.data_dir
    rpath = args.reference if os.path.isabs(args.reference) else os.path.join(root, args.reference)
    cams = load_reference(rpath, True)
    folders = sorted(cams)
    W = H = None

    with open(rpath) as f:
        for c in json.load(f)["cameras"]:
            W, H = int(c["width"]), int(c["height"])
            break
    print(f"{len(folders)} cameras from {os.path.basename(rpath)}, {W}x{H}")

    X = read_points(args.points_from)
    print(f"{len(X)} probe points from {args.points_from}")
    if not args.no_synthetic_fill:
        S, centre = synthetic_probe(cams, folders, W, H, args.grid)
        print(f"+ {len(S)} back-projected points (scene centre {np.round(centre, 3)}) "
              f"so every camera has coverage")
        X = np.concatenate([X, S], axis=0)

    # ---- visibility of every probe point in every camera -------------------
    vis, UV, DEP = {}, {}, {}
    for fdr in folders:
        c = cams[fdr]
        cam = (c["R"], c["t"], c["k"], c["d"])
        uv, ok, z = project_all(X, cam)
        inb = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        vis[fdr] = inb
        UV[fdr] = uv
        DEP[fdr] = z
    nvis = np.array([vis[f].sum() for f in folders])
    print(f"probe points visible per camera: p50 {np.median(nvis):.0f}  "
          f"min {nvis.min()}  max {nvis.max()}")

    # ---- what already exists ------------------------------------------------
    attempted, accepted = set(), set()
    acc_count = Counter()
    rep_path = os.path.join(root, args.existing_report)
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            rep = json.load(f)
        for m in rep["matches"]:
            key = tuple(sorted((m["folder_a"], m["folder_b"])))
            attempted.add(key)
            if m.get("accepted"):
                accepted.add(key)
                acc_count[key[0]] += 1
                acc_count[key[1]] += 1
        print(f"existing report: {len(attempted)} attempted, {len(accepted)} accepted "
              f"of {len(folders) * (len(folders) - 1) // 2} possible")
    have = np.array([acc_count[f] for f in folders])
    print(f"accepted pairs per camera: p50 {np.median(have):.0f}  min {have.min()}  "
          f"max {have.max()}   cameras under 8: {int((have < 8).sum())}/{len(folders)}")

    # ---- score every candidate pair ----------------------------------------
    need = {f: max(0, args.target_pairs - acc_count[f]) for f in folders}
    cands = []
    for a, b in itertools.combinations(folders, 2):
        key = (a, b)
        if key in attempted:
            continue
        both = vis[a] & vis[b]
        n = int(both.sum())
        if n < args.min_covis:
            continue
        ua, ub = UV[a][both], UV[b][both]
        za, zb = DEP[a][both], DEP[b][both]
        # local image scale of a world feature is fx/depth
        sa = cams[a]["k"][0] / np.maximum(za, 1e-9)
        sb = cams[b]["k"][0] / np.maximum(zb, 1e-9)
        ratio = float(np.median(sa / sb))
        ratio = max(ratio, 1.0 / ratio)
        if ratio > args.max_scale_ratio:
            continue

        def box(u):
            lo = np.percentile(u, 2, axis=0)
            hi = np.percentile(u, 98, axis=0)
            return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]

        ba, bb = box(ua), box(ub)
        cov_a = (ba[2] - ba[0]) * (ba[3] - ba[1]) / (W * H)
        cov_b = (bb[2] - bb[0]) * (bb[3] - bb[1]) / (W * H)
        # a pair is worth more when it feeds a camera that still needs pairs
        starve = need[a] + need[b]
        if starve == 0:
            continue
        score = float(n) * (1.0 + starve)
        cands.append({
            "folder_a": a, "folder_b": b,
            "n_covis": n,
            "scale_ratio": round(ratio, 3),
            "coverage_a": round(cov_a, 4), "coverage_b": round(cov_b, 4),
            "crop_a": [round(v, 1) for v in ba], "crop_b": [round(v, 1) for v in bb],
            "needs": starve, "score": round(score, 1),
        })

    cands.sort(key=lambda d: -d["score"])
    print(f"\n{len(cands)} candidate pairs pass covis>={args.min_covis} and "
          f"scale<={args.max_scale_ratio}")

    # ---- greedily take pairs until every camera reaches target -------------
    chosen, got = [], Counter(acc_count)
    for c in cands:
        if len(chosen) >= args.max_new_pairs:
            break
        a, b = c["folder_a"], c["folder_b"]
        if got[a] >= args.target_pairs and got[b] >= args.target_pairs:
            continue
        chosen.append(c)
        got[a] += 1
        got[b] += 1

    after = np.array([got[f] for f in folders])
    print(f"selected {len(chosen)} new pairs to attempt")
    print(f"projected pairs per camera: p50 {np.median(after):.0f}  min {after.min()}  "
          f"cameras under 8: {int((after < 8).sum())}/{len(folders)}")

    still = [folders[i] for i in range(len(folders)) if after[i] < 8]
    if still:
        print(f"  still short of 8 even after selection: {', '.join(still)}")
        print("  (these have genuinely little co-visible structure; they need "
              "resection against the solved rig, not more matching)")

    print(f"\n{'cam':<6} {'fx':>7} {'before':>7} {'after':>6}   most starved first")
    order = np.argsort(have)[:14]
    for i in order:
        f = folders[i]
        print(f"  {f:<6} {cams[f]['k'][0]:>7.0f} {have[i]:>6} {after[i]:>6}")

    hi = [c for c in chosen if c["scale_ratio"] > 2.0]
    print(f"\n{len(hi)} of the selected pairs exceed 2x scale mismatch -- match these "
          f"through the crop boxes,")
    print("otherwise the matcher sees a 2-6x scale change and will reject them the "
          "same way it did before.")

    with open(args.out, "w") as f:
        json.dump({
            "reference": os.path.basename(rpath),
            "points_from": os.path.abspath(args.points_from),
            "target_pairs": args.target_pairs,
            "n_existing_accepted": len(accepted),
            "n_new_pairs": len(chosen),
            "image_width": W, "image_height": H,
            "pairs": chosen,
        }, f, indent=1)
    print(f"\nwrote {args.out}")

    if args.out_pairs_arg:
        # choose_pairs() unions explicit pairs with its nearest-centre selection,
        # so passing these ADDS to the existing set rather than replacing it, and
        # the match cache makes already-computed pairs free to re-list.
        with open(args.out_pairs_arg, "w") as f:
            f.write(",".join(f"{c['folder_a']}:{c['folder_b']}" for c in chosen))
        print(f"wrote {args.out_pairs_arg} "
              f"({len(chosen)} pairs, for --camera-pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
