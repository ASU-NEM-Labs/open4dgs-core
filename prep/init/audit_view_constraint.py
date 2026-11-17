#!/usr/bin/env python3
"""Measure how well-constrained each primitive's DEPTH is by the camera rig.

Read-only. No training, no images -- pure projective geometry from the COLMAP
model plus the trained means.

Why
---
A primitive's position along a viewing ray is pinned only by cameras that see it
from DIFFERENT directions. Observed by many cameras over a wide arc, exactly one
depth satisfies them all. Observed by few cameras, or by cameras clustered in a
narrow arc, an entire RANGE of depths is equally good -- so background-coloured
mass can sit at the wrong distance and cost nothing photometrically.

Metric
------
NOT max pairwise angle -- that is wrong. Two cameras on OPPOSITE sides give a
180 deg angle but nearly COLLINEAR rays, and every point along that shared line
projects to the same pixel in both, so 180 deg is as degenerate as 0 deg. The
best-conditioned pair is at 90 deg.

The correct measure is the conditioning of the ray-intersection system:

    M = sum_c (I - d_c d_c^T)   over cameras that OBSERVE the point

lambda_min(M) is how strongly the worst-constrained direction is pinned.
Normalised by n_visible it lands in [0,1]:

    single ray            -> 0     (free along the ray)
    two rays at 180 deg   -> 0     (free along the common line)
    two rays at 90 deg    -> 0.5
    wide camera ring      -> 0.5
    8 cams over a 14 deg arc -> 0.0067  (effectively free)

Occlusion matters too: a camera only constrains a point it can actually SEE.
Pass --occluders to filter cameras whose line of sight is blocked by real
geometry; without it, "in frame" overstates the constraint badly for anything
sitting behind a subject.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _quat_xyzw_to_R(quat) -> np.ndarray:
    """[x,y,z,w] -> 3x3. Matches trigs/io/dataset.py's new-API path exactly."""
    x, y, z, w = (float(v) for v in quat)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _load_colmap(path: Path):
    """Return (centers [C,3], R [C,3,3], t [C,3], K [C,3,3], wh [C,2]).

    Pose/intrinsic extraction mirrors trigs/io/dataset.py, which is the path the
    trainer itself uses, so this agrees with training by construction.
    """
    try:
        import pycolmap
    except ImportError:  # pragma: no cover
        sys.exit("pycolmap is required:  uv pip install pycolmap")

    rec = pycolmap.Reconstruction(str(path))
    centers, Rs, ts, Ks, whs = [], [], [], [], []
    api = None
    for _, img in rec.images.items():
        cam = rec.cameras[img.camera_id]

        if api is None:
            try:
                _ = img.cam_from_world()
                api = "new"
            except Exception:
                api = "old"

        if api == "new":
            pose = img.cam_from_world()          # method, returns Rigid3d
            Rm = _quat_xyzw_to_R(pose.rotation.quat)
            tv = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        else:
            Rm = np.asarray(img.R(), dtype=np.float64)
            tv = np.asarray(img.tvec, dtype=np.float64).reshape(3)

        centers.append(-Rm.T @ tv)
        Rs.append(Rm)
        ts.append(tv)

        try:
            fx = float(cam.focal_length_x)
            fy = float(cam.focal_length_y)
            cx = float(cam.principal_point_x)
            cy = float(cam.principal_point_y)
        except AttributeError:
            p = np.asarray(cam.params, dtype=np.float64)
            model = str(cam.model).split(".")[-1].upper()
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
        whs.append([cam.width, cam.height])

    print(f"  pycolmap API: {api}")
    return (np.array(centers), np.array(Rs), np.array(ts),
            np.array(Ks), np.array(whs, dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--colmap", required=True, type=Path,
                    help="COLMAP sparse model dir, e.g. sparse/dynamic_4dv_v8")
    ap.add_argument("--indices", required=True, type=Path,
                    help=".npy of flagged row indices from "
                         "audit_checkpoint_floaters.py --dump-indices")
    ap.add_argument("--sample", type=int, default=20000,
                    help="Random sample of the comparison group (speed).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occluders", type=Path, default=None,
                    help="NPZ of clean geometry. If given, a camera only counts "
                         "when its line of sight to the point is NOT blocked by "
                         "that geometry. This is the difference between 'in "
                         "frame' and 'actually observed'.")
    ap.add_argument("--occl-radius", type=float, default=0.03,
                    help="Metres; a sample this close to occluder geometry "
                         "counts as blocked.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    means = ckpt["params"]["means"].detach().cpu().numpy().astype(np.float64)
    n = means.shape[0]

    flag_idx = np.load(args.indices).astype(np.int64)
    flag = np.zeros((n,), dtype=bool)
    flag[flag_idx] = True

    centers, Rs, ts, Ks, whs = _load_colmap(args.colmap)
    C = centers.shape[0]
    print(f"checkpoint : {args.checkpoint}")
    print(f"  primitives: {n:,}   flagged: {int(flag.sum()):,}")
    print(f"COLMAP     : {args.colmap}  ({C} cameras)")

    occl_tree = None
    occl_radius = float(args.occl_radius)
    if args.occluders is not None:
        from scipy.spatial import cKDTree
        _z = np.load(args.occluders, allow_pickle=True)
        occl_tree = cKDTree(np.asarray(_z["positions"], dtype=np.float64))
        print(f"  occluders : {len(_z['positions']):,} pts, "
              f"radius {occl_radius} m  (line-of-sight filtering ON)")

    rng = np.random.default_rng(args.seed)
    rest_idx = np.flatnonzero(~flag)
    if rest_idx.size > args.sample:
        rest_idx = rng.choice(rest_idx, args.sample, replace=False)
    groups = {"FLAGGED": flag_idx, "REST": rest_idx}

    results = {}
    for name, idx in groups.items():
        P = means[idx]                                   # [M,3]
        M = P.shape[0]
        vis = np.zeros((C, M), dtype=bool)
        dirs = np.zeros((C, M, 3), dtype=np.float64)

        for c in range(C):
            Pc = (Rs[c] @ P.T).T + ts[c]                 # camera coords
            z = Pc[:, 2]
            ok = z > 1e-6
            zz = np.where(ok, z, 1.0)
            u = Ks[c][0, 0] * Pc[:, 0] / zz + Ks[c][0, 2]
            v = Ks[c][1, 1] * Pc[:, 1] / zz + Ks[c][1, 2]
            vis[c] = ok & (u >= 0) & (u < whs[c][0]) & (v >= 0) & (v < whs[c][1])
            d = P - centers[c][None, :]
            dirs[c] = d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)

        # --- occlusion: a camera only constrains a point if it can SEE it ----
        if occl_tree is not None:
            K = 24
            for c in range(C):
                idx_v = np.flatnonzero(vis[c])
                if idx_v.size == 0:
                    continue
                A = centers[c][None, :]
                B = P[idx_v]
                # Sample the segment from the camera up to just short of the
                # point; a hit means solid geometry blocks the line of sight.
                fr = np.linspace(0.02, 0.94, K)[None, :, None]
                S = A[:, None, :] + (B - A)[:, None, :] * fr        # [m,K,3]
                d_hit, _ = occl_tree.query(S.reshape(-1, 3), k=1, workers=-1)
                blocked = (d_hit.reshape(-1, K) < occl_radius).any(axis=1)
                vis[c, idx_v[blocked]] = False

        n_vis = vis.sum(axis=0).astype(np.int32)

        # --- depth conditioning ---------------------------------------------
        # M_p = sum_c (I - d d^T) over cameras that SEE the point. Its smallest
        # eigenvalue is how well the worst-constrained direction is pinned.
        # One ray  -> lambda_min = 0 (free along the ray).
        # Two rays at 180 deg (collinear) -> lambda_min = 0 (degenerate!).
        # Two rays at 90 deg -> lambda_min = 1. Wide spread -> grows with n.
        # Normalising by n_vis gives [0,1]: 0 = a free direction, higher =
        # genuinely pinned in every direction.
        S3 = np.zeros((M, 3, 3))
        for c in range(C):
            w = vis[c].astype(np.float64)[:, None, None]
            d = dirs[c]
            S3 += w * (d[:, :, None] * d[:, None, :])
        Mp = n_vis[:, None, None].astype(np.float64) * np.eye(3)[None] - S3
        eig = np.linalg.eigvalsh(Mp)                      # ascending
        lam_min = eig[:, 0]
        lam_norm = lam_min / np.maximum(n_vis, 1)

        results[name] = (n_vis, lam_min, lam_norm)

    print(f"\n{'':<34}{'FLAGGED':>14}{'REST':>14}")

    def row(label, fn, n=4):
        a = fn(results["FLAGGED"]); b = fn(results["REST"])
        print(f"  {label:<32} {a:>14.{n}g} {b:>14.{n}g}")

    seen = "cameras that SEE the point" if occl_tree is not None else \
           "cameras with the point in frame"
    print(f"\n-- {seen} --")
    for lbl, q in (("median", 50), ("p10", 10), ("p25", 25), ("p90", 90)):
        row(f"n_visible {lbl}", lambda r, q=q: np.percentile(r[0], q))
    row("n_visible < 5  %", lambda r: 100.0 * float((r[0] < 5).mean()))
    row("n_visible < 2  %", lambda r: 100.0 * float((r[0] < 2).mean()))

    print("\n-- depth conditioning: lambda_min / n_visible  (0 = free direction) --")
    for lbl, q in (("median", 50), ("p10", 10), ("p25", 25), ("p90", 90)):
        row(f"cond {lbl}", lambda r, q=q: np.percentile(r[2], q))
    row("cond < 0.05  %", lambda r: 100.0 * float((r[2] < 0.05).mean()))
    row("cond < 0.15  %", lambda r: 100.0 * float((r[2] < 0.15).mean()))

    print("\n-- raw lambda_min (absolute stiffness) --")
    for lbl, q in (("median", 50), ("p10", 10)):
        row(f"lambda_min {lbl}", lambda r, q=q: np.percentile(r[1], q))

    print("\nRead: cond near 0 means the rig leaves a direction along which the"
          "\nposition is free -- mass can sit anywhere along it for no cost. If"
          "\nFLAGGED and REST have similar cond, the rig pins those positions"
          "\nfine and depth ambiguity is NOT the cause.")


if __name__ == "__main__":
    main()
