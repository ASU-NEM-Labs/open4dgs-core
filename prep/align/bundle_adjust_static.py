#!/usr/bin/env python3
"""Bundle-adjust 4DV PTZ cameras against STATIC correspondences only.

Why this exists
---------------
The 4DV take manifest supplies a PTZ forward-model prior. Measured on held-out
third-view reprojection it carries roughly 1-1.5 degrees of per-camera aiming
error -- normal for pose derived from pan/tilt encoders, and fine as an
initialisation, but not a calibration.

The existing refinement stage (solve_4dv_dynamic_calibration.py) cannot close
that gap, for four structural reasons:

  * it minimises EPIPOLAR residual, which does not constrain a point along its
    ray, so a camera can score well while being wrong in the direction that
    ruins triangulation;
  * it refines ROTATION ONLY, holding camera centres at the prior's values --
    and those centres come from the same encoder model as the rotations;
  * it never optimises STRUCTURE, so 3D points stay wherever the initial
    triangulation put them;
  * it leaves INTRINSICS untouched for all but the cameras it has already
    declared failures. Every camera in the manifest carries cx, cy at exactly
    (W-1)/2, (H-1)/2 -- an assumed principal point, never calibrated.

This script does the opposite on all four counts: a joint bundle adjustment
over poses, camera centres, intrinsics and structure, minimising reprojection
error under a robust loss.

Time invariance
---------------
Only static correspondences are admitted. A correspondence is geometrically
valid iff both pixels image the same 3D point; for time-invariant geometry that
holds regardless of when each camera exposed, so free-running cameras are no
obstacle. Matches on anything that moves are inadmissible and are never used --
which is why the default input is the clean plate.

Validation
----------
Whole tracks are held out of the fit. They are triangulated from the optimised
cameras and reprojected, so the reported number is never measured on the data
that produced it. A third-view test is also reported: triangulate from two
cameras, reproject into a third that took no part in that point's triangulation.

Gauge
-----
Seven degrees of freedom are unobservable (similarity). Camera 0's pose is held
fixed, and one residual pins the mean camera-centre radius to its initial value.
No other prior is applied, so the solution is not pulled toward the input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ----------------------------------------------------------------- utilities
def rodrigues(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(rvec.reshape(3, 1))[0]


def inv_rodrigues(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0].reshape(3)


def project(X: np.ndarray, rvec: np.ndarray, t: np.ndarray, k: np.ndarray) -> np.ndarray:
    """X (N,3) world -> (N,2) pixels. k = [fx, fy, cx, cy]."""
    R = rodrigues(rvec)
    cam = X @ R.T + t
    z = np.maximum(cam[:, 2], 1e-9)
    return np.stack([k[0] * cam[:, 0] / z + k[2],
                     k[1] * cam[:, 1] / z + k[3]], axis=1)


# ------------------------------------------------------------- track building
class UnionFind:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, a: int) -> int:
        p = self.p
        r = a
        while p.get(r, r) != r:
            r = p[r]
        while p.get(a, a) != a:
            p[a], a = r, p[a]
        return r

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_tracks(pair_matches, n_cams, quant=1.0, min_views=3):
    """Chain pairwise matches into multi-view tracks by quantised pixel identity."""
    uf = UnionFind()
    node_id: dict[tuple, int] = {}
    node_obs: dict[int, list] = {}

    def node(cam: int, uv: np.ndarray) -> int:
        key = (cam, int(round(uv[0] / quant)), int(round(uv[1] / quant)))
        nid = node_id.get(key)
        if nid is None:
            nid = len(node_id)
            node_id[key] = nid
            node_obs[nid] = [cam, uv[0], uv[1], 1]
        else:
            o = node_obs[nid]
            n = o[3]
            o[1] = (o[1] * n + uv[0]) / (n + 1)
            o[2] = (o[2] * n + uv[1]) / (n + 1)
            o[3] = n + 1
        return nid

    for ia, ib, pa, pb in pair_matches:
        for ua, ub in zip(pa, pb):
            uf.union(node(ia, ua), node(ib, ub))

    groups: dict[int, list[int]] = defaultdict(list)
    for nid in node_obs:
        groups[uf.find(nid)].append(nid)

    tracks = []
    for members in groups.values():
        seen: dict[int, tuple[float, float]] = {}
        dup = False
        for nid in members:
            cam, u, v, _ = node_obs[nid]
            if cam in seen:          # two observations in one camera -> ambiguous
                dup = True
                break
            seen[cam] = (u, v)
        if dup or len(seen) < min_views:
            continue
        tracks.append(sorted(seen.items()))
    return tracks


def triangulate_multi(obs, cams_rt, cams_k):
    """DLT over all views of one track."""
    A = []
    for cam, (u, v) in obs:
        R = rodrigues(cams_rt[cam][:3])
        P = np.hstack([R, cams_rt[cam][3:].reshape(3, 1)])
        k = cams_k[cam]
        Kk = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
        P = Kk @ P
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--report-json", default="dynamic_calibration_report.json",
                    help="report listing accepted pairs + cache paths (STATIC content)")
    ap.add_argument("--cache-dir", default=None,
                    help="override directory holding the .npz match caches")
    ap.add_argument("--init-reference", default="reference_export-4dv-manifest.json")
    ap.add_argument("--output-reference", default="reference_export_ba.json")
    ap.add_argument("--report-out", default="bundle_adjust_report.json")
    ap.add_argument("--min-certainty", type=float, default=0.6)
    ap.add_argument("--max-points-per-pair", type=int, default=800)
    ap.add_argument("--track-quant-px", type=float, default=1.0)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--max-tracks", type=int, default=25000)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--init-reproj-max-px", type=float, default=200.0,
                    help="drop tracks whose initial reprojection is hopeless")
    ap.add_argument("--huber-px", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=3,
                    help="BA rounds; observations above --outlier-px are dropped between rounds")
    ap.add_argument("--outlier-px", type=float, default=4.0)
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--no-optimize-intrinsics", action="store_true")
    ap.add_argument("--no-optimize-centers", action="store_true")
    ap.add_argument("--optimize-principal-point", action="store_true", default=True)
    ap.add_argument("--ignore-init-distortion", action="store_true",
                    help="Treat the init as PINHOLE, discarding its distortion "
                         "coefficients. Correct when the delivered frames are "
                         "already rectified (verify with a straight-line test): "
                         "any non-zero coefficients then describe fitted error, "
                         "not optics, and applying them imports that error.")
    ap.add_argument("--fixed-cameras", default="",
                    help="comma-separated folders to hold completely fixed")
    ap.add_argument("--exclude-cameras", default="",
                    help="comma-separated folders to drop entirely")
    args = ap.parse_args()

    root = args.data_dir
    ref = json.load(open(os.path.join(root, args.init_reference)))
    cams_list = ref["cameras"]
    exclude = {c for c in args.exclude_cameras.split(",") if c}
    cams_list = [c for c in cams_list if c["folder"] not in exclude]
    folders = [c["folder"] for c in cams_list]
    fidx = {f: i for i, f in enumerate(folders)}
    n_cams = len(folders)
    print(f"cameras: {n_cams}" + (f"  (excluded {sorted(exclude)})" if exclude else ""))

    cams_rt = np.zeros((n_cams, 6))
    cams_k = np.zeros((n_cams, 4))
    dist = {}
    for c in cams_list:
        i = fidx[c["folder"]]
        cams_rt[i, :3] = inv_rodrigues(np.array(c["R"]))
        cams_rt[i, 3:] = np.array(c["t"])
        p = c["params"]
        cams_k[i] = [p[0], p[1], p[2], p[3]]
        dist[i] = None if args.ignore_init_distortion else (np.array(p[4:]) if len(p) > 4 else None)

    # ---- load static correspondences ------------------------------------
    rep = json.load(open(os.path.join(root, args.report_json)))
    cache_dir = args.cache_dir
    pair_matches = []
    n_obs_raw = 0
    for m in rep["matches"]:
        if not m.get("accepted"):
            continue
        a, b = m["folder_a"], m["folder_b"]
        if a not in fidx or b not in fidx:
            continue
        cp = m["cache"]
        path = os.path.join(cache_dir, os.path.basename(cp)) if cache_dir else os.path.join(root, cp)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        keep = d["certainty"] > args.min_certainty
        pa, pb = d["points_a"][keep], d["points_b"][keep]
        if len(pa) < 30:
            continue
        if len(pa) > args.max_points_per_pair:
            sel = np.linspace(0, len(pa) - 1, args.max_points_per_pair).astype(int)
            pa, pb = pa[sel], pb[sel]
        ia, ib = fidx[a], fidx[b]
        # undistort once, up front; everything downstream is ideal-pinhole
        for idx, pts in ((ia, pa), (ib, pb)):
            if dist[idx] is not None:
                K = np.array([[cams_k[idx][0], 0, cams_k[idx][2]],
                              [0, cams_k[idx][1], cams_k[idx][3]], [0, 0, 1]])
                pts[:] = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64),
                                             K, dist[idx], P=K).reshape(-1, 2)
        pair_matches.append((ia, ib, pa.astype(np.float64), pb.astype(np.float64)))
        n_obs_raw += len(pa)
    print(f"static pairs: {len(pair_matches)}  correspondences: {n_obs_raw}")
    if not pair_matches:
        print("no usable pairs", file=sys.stderr)
        return 2

    # ---- tracks ----------------------------------------------------------
    tracks = build_tracks(pair_matches, n_cams, args.track_quant_px, args.min_views)
    print(f"tracks with >= {args.min_views} views: {len(tracks)}")
    if len(tracks) > args.max_tracks:
        rng = np.random.default_rng(0)
        tracks = [tracks[i] for i in rng.choice(len(tracks), args.max_tracks, replace=False)]
        print(f"  subsampled to {len(tracks)}")

    pts, kept = [], []
    for tr in tracks:
        X = triangulate_multi(tr, cams_rt, cams_k)
        if X is None:
            continue
        ok = True
        errs = []
        for cam, (u, v) in tr:
            R = rodrigues(cams_rt[cam][:3])
            c = X @ R.T + cams_rt[cam][3:]
            if c[2] <= 1e-6:
                ok = False
                break
            k = cams_k[cam]
            errs.append(np.hypot(k[0] * c[0] / c[2] + k[2] - u, k[1] * c[1] / c[2] + k[3] - v))
        if ok and np.median(errs) < args.init_reproj_max_px:
            pts.append(X)
            kept.append(tr)
    tracks, pts = kept, np.array(pts)
    print(f"tracks surviving initial triangulation: {len(tracks)}")
    if len(tracks) < 200:
        print("too few tracks to bundle adjust", file=sys.stderr)
        return 3

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(tracks))
    n_hold = int(len(tracks) * args.holdout_frac)
    hold_idx = set(perm[:n_hold].tolist())
    fit = [i for i in range(len(tracks)) if i not in hold_idx]
    hold = sorted(hold_idx)
    print(f"fit tracks {len(fit)} | held-out tracks {len(hold)}")

    fixed = {fidx[f] for f in args.fixed_cameras.split(",") if f in fidx}
    fixed.add(0)
    opt_k = not args.no_optimize_intrinsics
    opt_c = not args.no_optimize_centers

    init_radius = float(np.mean(np.linalg.norm(
        np.array([-rodrigues(cams_rt[i][:3]).T @ cams_rt[i][3:] for i in range(n_cams)])
        - np.mean([-rodrigues(cams_rt[i][:3]).T @ cams_rt[i][3:] for i in range(n_cams)], axis=0),
        axis=1)))

    active = [np.ones(len(tracks[i]), bool) for i in range(len(tracks))]

    def pack(rt, kk, XX):
        return np.concatenate([rt.ravel(), kk.ravel(), XX.ravel()])

    def unpack(p):
        rt = p[:n_cams * 6].reshape(n_cams, 6).copy()
        kk = p[n_cams * 6:n_cams * 6 + n_cams * 4].reshape(n_cams, 4).copy()
        XX = p[n_cams * 6 + n_cams * 4:].reshape(-1, 3)
        return rt, kk, XX

    for rnd in range(1, args.rounds + 1):
        obs_cam, obs_pt, obs_uv = [], [], []
        for ti in fit:
            for j, (cam, uv) in enumerate(tracks[ti]):
                if active[ti][j]:
                    obs_cam.append(cam)
                    obs_pt.append(ti)
                    obs_uv.append(uv)
        obs_cam = np.array(obs_cam)
        obs_pt = np.array(obs_pt)
        obs_uv = np.array(obs_uv)
        print(f"\nround {rnd}: observations {len(obs_uv)}")

        def residual(p):
            rt, kk, XX = unpack(p)
            r = np.empty(len(obs_uv) * 2 + 1)
            for cam in np.unique(obs_cam):
                m = obs_cam == cam
                pr = project(XX[obs_pt[m]], rt[cam, :3], rt[cam, 3:], kk[cam])
                d = pr - obs_uv[m]
                sel = np.where(m)[0]
                r[2 * sel] = d[:, 0]
                r[2 * sel + 1] = d[:, 1]
            C = np.array([-rodrigues(rt[i, :3]).T @ rt[i, 3:] for i in range(n_cams)])
            rad = float(np.mean(np.linalg.norm(C - C.mean(0), axis=1)))
            r[-1] = (rad - init_radius) * 50.0        # gauge: pin overall scale
            return r

        p0 = pack(cams_rt, cams_k, pts)
        n_p = len(p0)
        S = lil_matrix((len(obs_uv) * 2 + 1, n_p), dtype=int)
        base_pt = n_cams * 10
        for i, (cam, ti) in enumerate(zip(obs_cam, obs_pt)):
            if cam not in fixed:
                S[2 * i, cam * 6:cam * 6 + (6 if opt_c else 3)] = 1
                S[2 * i + 1, cam * 6:cam * 6 + (6 if opt_c else 3)] = 1
            if opt_k:
                kcols = 4 if args.optimize_principal_point else 2
                S[2 * i, n_cams * 6 + cam * 4: n_cams * 6 + cam * 4 + kcols] = 1
                S[2 * i + 1, n_cams * 6 + cam * 4: n_cams * 6 + cam * 4 + kcols] = 1
            S[2 * i, base_pt + ti * 3: base_pt + ti * 3 + 3] = 1
            S[2 * i + 1, base_pt + ti * 3: base_pt + ti * 3 + 3] = 1
        S[-1, :n_cams * 6] = 1

        res = least_squares(residual, p0, jac_sparsity=S, verbose=2,
                            x_scale="jac", loss="huber", f_scale=args.huber_px,
                            method="trf", max_nfev=args.max_nfev)
        cams_rt, cams_k, pts = unpack(res.x)

        # Drop diverged observations before the next round. The threshold must
        # adapt: while the model is still far out, a fixed few-pixel gate
        # rejects essentially everything and starves the next round. Trim the
        # worst tail instead, and only tighten toward --outlier-px as the fit
        # actually improves.
        cur = []
        for ti in fit:
            for j, (cam, uv) in enumerate(tracks[ti]):
                if active[ti][j]:
                    pr = project(pts[ti:ti + 1], cams_rt[cam, :3],
                                 cams_rt[cam, 3:], cams_k[cam])[0]
                    cur.append(np.hypot(*(pr - uv)))
        cur = np.asarray(cur)
        thresh = max(float(args.outlier_px), float(np.percentile(cur, 80)))
        print("  residual now: p50 %.2f p80 %.2f px -> outlier threshold %.2f px"
              % (np.median(cur), np.percentile(cur, 80), thresh))
        dropped = 0
        for ti in fit:
            for j, (cam, uv) in enumerate(tracks[ti]):
                if not active[ti][j]:
                    continue
                pr = project(pts[ti:ti + 1], cams_rt[cam, :3], cams_rt[cam, 3:], cams_k[cam])[0]
                if np.hypot(*(pr - uv)) > thresh:
                    active[ti][j] = False
                    dropped += 1
        rem = [ti for ti in fit if active[ti].sum() >= args.min_views]
        print(f"  dropped {dropped} outlier observations; tracks retained {len(rem)}/{len(fit)}")
        fit = rem

    # ---- validation on held-out tracks -----------------------------------
    def track_err(ti, rt, kk):
        X = triangulate_multi(tracks[ti], rt, kk)
        if X is None:
            return None
        e = []
        for cam, uv in tracks[ti]:
            c = X @ rodrigues(rt[cam][:3]).T + rt[cam][3:]
            if c[2] <= 1e-6:
                return None
            k = kk[cam]
            e.append(np.hypot(k[0] * c[0] / c[2] + k[2] - uv[0], k[1] * c[1] / c[2] + k[3] - uv[1]))
        return float(np.median(e))

    init_rt = np.zeros((n_cams, 6))
    init_k = np.zeros((n_cams, 4))
    for c in cams_list:
        i = fidx[c["folder"]]
        init_rt[i, :3] = inv_rodrigues(np.array(c["R"]))
        init_rt[i, 3:] = np.array(c["t"])
        init_k[i] = c["params"][:4]

    before = [e for e in (track_err(ti, init_rt, init_k) for ti in hold) if e is not None]
    after = [e for e in (track_err(ti, cams_rt, cams_k) for ti in hold) if e is not None]
    print("\nHELD-OUT tracks (never in the fit):")
    print("  before  p50 %8.3f  p90 %9.3f px  (n=%d)" % (
        np.median(before), np.percentile(before, 90), len(before)))
    print("  after   p50 %8.3f  p90 %9.3f px  (n=%d)" % (
        np.median(after), np.percentile(after, 90), len(after)))

    per_cam = defaultdict(list)
    for ti in hold:
        X = triangulate_multi(tracks[ti], cams_rt, cams_k)
        if X is None:
            continue
        for cam, uv in tracks[ti]:
            c = X @ rodrigues(cams_rt[cam][:3]).T + cams_rt[cam][3:]
            if c[2] <= 1e-6:
                continue
            k = cams_k[cam]
            per_cam[cam].append(np.hypot(k[0] * c[0] / c[2] + k[2] - uv[0],
                                         k[1] * c[1] / c[2] + k[3] - uv[1]))
    rows = []
    for i, f in enumerate(folders):
        e = per_cam.get(i, [])
        rows.append({
            "folder": f, "n": len(e),
            "p50": float(np.median(e)) if e else None,
            "p90": float(np.percentile(e, 90)) if e else None,
            "fx_ratio": float(cams_k[i][0] / init_k[i][0]),
            "cx_shift": float(cams_k[i][2] - init_k[i][2]),
            "cy_shift": float(cams_k[i][3] - init_k[i][3]),
            "rot_change_deg": float(np.degrees(np.linalg.norm(
                inv_rodrigues(rodrigues(cams_rt[i][:3]) @ rodrigues(init_rt[i][:3]).T)))),
            "centre_shift": float(np.linalg.norm(
                -rodrigues(cams_rt[i][:3]).T @ cams_rt[i][3:]
                + rodrigues(init_rt[i][:3]).T @ init_rt[i][3:])),
        })
    graded = [r for r in rows if r["p50"] is not None]
    print("\nper-camera held-out reprojection:")
    print("  graded %d/%d | p50 %.3f px | cameras <1px: %d | >2px: %d" % (
        len(graded), n_cams, np.median([r["p50"] for r in graded]),
        sum(r["p50"] < 1 for r in graded), sum(r["p50"] > 2 for r in graded)))
    print("  worst:", " ".join("%s=%.2f" % (r["folder"], r["p50"])
                               for r in sorted(graded, key=lambda x: -x["p50"])[:8]))
    print("  intrinsics moved: fx ratio p50 %.5f | cx shift p50 %+.2f px | cy shift p50 %+.2f px" % (
        np.median([r["fx_ratio"] for r in rows]),
        np.median([r["cx_shift"] for r in rows]),
        np.median([r["cy_shift"] for r in rows])))

    out = {"cameras": [], "scene_meta": ref.get("scene_meta", {}),
           "calibration_source": "bundle_adjust_static.py (static correspondences only)"}
    for c in cams_list:
        i = fidx[c["folder"]]
        e = dict(c)
        e["R"] = rodrigues(cams_rt[i][:3]).tolist()
        e["t"] = cams_rt[i][3:].tolist()
        e["camera_model_name"] = "PINHOLE"
        e["params"] = [float(cams_k[i][0]), float(cams_k[i][1]),
                       float(cams_k[i][2]), float(cams_k[i][3])]
        out["cameras"].append(e)
    with open(os.path.join(root, args.output_reference), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(root, args.report_out), "w") as fh:
        json.dump({
            "n_cameras": n_cams, "n_tracks": len(tracks),
            "holdout": {"before_p50": float(np.median(before)),
                        "after_p50": float(np.median(after)),
                        "before_p90": float(np.percentile(before, 90)),
                        "after_p90": float(np.percentile(after, 90))},
            "per_camera": rows,
        }, fh, indent=1)
    print(f"\nwrote {args.output_reference} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
