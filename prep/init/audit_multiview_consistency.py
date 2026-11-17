#!/usr/bin/env python3
"""Do the observing cameras AGREE about what is at each primitive's position?

Read-only. Uses the COLMAP model + the real source images.

Why this and not depth conditioning
-----------------------------------
Conditioning (audit_view_constraint.py) asks "given these poses, is depth
determined?" -- it takes the poses as TRUTH, so it detects AMBIGUITY but is
blind to BIAS. If calibration is off, rays still converge crisply, just at the
wrong place. Well-conditioned and wrong is precisely what a calibration error
looks like.

This test is pose-error-sensitive. For each primitive, project into every camera
that observes it and sample the actual GT pixel:

  * A REAL surface point -> all views sample the same surface -> LOW colour
    spread (up to shading/specularity).
  * A point at the WRONG depth -> each view's ray continues to a different piece
    of background -> HIGH colour spread.

So if the flagged floaters show LOW spread, the cameras genuinely agree there is
background-coloured matter at that depth: the geometry is telling the trainer to
put it there, and the fault is upstream in calibration/triangulation. If they
show HIGH spread, the position is not photometrically supported and the question
returns to why the loss tolerates it.

Also reports how well each primitive's own learned colour matches the consensus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required:  uv pip install pillow")

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    sys.exit("SciPy is required:  uv pip install scipy")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_view_constraint import _load_colmap, _quat_xyzw_to_R  # noqa: E402


def _pcts(x, qs=(10, 25, 50, 75, 90)):
    return {q: float(np.percentile(x, q)) for q in qs}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--indices", required=True, type=Path,
                    help=".npy from audit_checkpoint_floaters.py --dump-indices")
    ap.add_argument("--images-root", required=True, type=Path,
                    help="Root such that <root>/<colmap image name> resolves, "
                         "e.g. 'images' when COLMAP names are '0095/000001.jpg'.")
    ap.add_argument("--occluders", type=Path, default=None,
                    help="NPZ of clean geometry for line-of-sight filtering.")
    ap.add_argument("--occl-radius", type=float, default=0.03)
    ap.add_argument("--sample", type=int, default=4000,
                    help="Size of the comparison group.")
    ap.add_argument("--min-views", type=int, default=4,
                    help="Skip points seen by fewer cameras than this.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    means = params["means"].detach().cpu().numpy().astype(np.float64)
    n = means.shape[0]

    sh0 = params.get("sh0")
    if sh0 is not None:
        own = sh0.detach().cpu().numpy().reshape(n, -1)[:, :3] * 0.2820947917738781 + 0.5
        own = np.clip(own, 0.0, 1.0)
    else:
        own = None

    flag_idx = np.load(args.indices).astype(np.int64)
    flag = np.zeros((n,), dtype=bool)
    flag[flag_idx] = True

    rng = np.random.default_rng(args.seed)
    rest_idx = np.flatnonzero(~flag)
    if rest_idx.size > args.sample:
        rest_idx = rng.choice(rest_idx, args.sample, replace=False)

    sel = np.concatenate([flag_idx, rest_idx])
    is_flag = np.concatenate([np.ones(len(flag_idx), bool),
                              np.zeros(len(rest_idx), bool)])
    P = means[sel]
    M = P.shape[0]

    centers, Rs, ts, Ks, whs, names = _load_colmap_named(args.colmap)
    C = len(centers)
    print(f"checkpoint : {args.checkpoint}")
    print(f"  points    : {M:,}  (flagged {int(is_flag.sum()):,}, "
          f"rest {int((~is_flag).sum()):,})")
    print(f"COLMAP     : {args.colmap}  ({C} cameras)")

    occl_tree = None
    if args.occluders is not None:
        z = np.load(args.occluders, allow_pickle=True)
        occl_tree = cKDTree(np.asarray(z["positions"], dtype=np.float64))
        print(f"  occluders : {len(z['positions']):,} pts "
              f"(line-of-sight filtering ON)")

    sum_c = np.zeros((M, 3))
    sum_c2 = np.zeros((M, 3))
    cnt = np.zeros(M)

    for c in range(C):
        path = args.images_root / names[c]
        if not path.exists():
            continue
        im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        H, W = im.shape[:2]

        Pc = (Rs[c] @ P.T).T + ts[c]
        z_ = Pc[:, 2]
        ok = z_ > 1e-6
        zz = np.where(ok, z_, 1.0)
        # COLMAP intrinsics are for the original resolution; rescale if the
        # image on disk was resized.
        sx, sy = W / whs[c][0], H / whs[c][1]
        u = (Ks[c][0, 0] * Pc[:, 0] / zz + Ks[c][0, 2]) * sx
        v = (Ks[c][1, 1] * Pc[:, 1] / zz + Ks[c][1, 2]) * sy
        inb = ok & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

        if occl_tree is not None and inb.any():
            idx_v = np.flatnonzero(inb)
            K = 24
            fr = np.linspace(0.02, 0.94, K)[None, :, None]
            A = centers[c][None, :]
            S = A[:, None, :] + (P[idx_v] - A)[:, None, :] * fr
            dh, _ = occl_tree.query(S.reshape(-1, 3), k=1, workers=-1)
            blocked = (dh.reshape(-1, K) < args.occl_radius).any(axis=1)
            inb[idx_v[blocked]] = False

        if not inb.any():
            continue
        ui = np.clip(u[inb].astype(np.int64), 0, W - 1)
        vi = np.clip(v[inb].astype(np.int64), 0, H - 1)
        col = im[vi, ui, :]
        sum_c[inb] += col
        sum_c2[inb] += col * col
        cnt[inb] += 1

    valid = cnt >= args.min_views
    mean_c = np.zeros((M, 3))
    mean_c[valid] = sum_c[valid] / cnt[valid, None]
    var = np.zeros((M, 3))
    var[valid] = np.maximum(sum_c2[valid] / cnt[valid, None] - mean_c[valid] ** 2, 0)
    spread = np.sqrt(var).mean(axis=1)     # mean per-channel std over views

    print(f"\nviews sampled per point: median {np.median(cnt):.0f}   "
          f"points with >= {args.min_views} views: {int(valid.sum()):,}/{M:,}")

    print(f"\n{'':<36}{'FLAGGED':>13}{'REST':>13}")

    def cmp(label, arr, fn, n=4):
        a = arr[valid & is_flag]; b = arr[valid & ~is_flag]
        av = fn(a) if a.size else float("nan")
        bv = fn(b) if b.size else float("nan")
        print(f"  {label:<34} {av:>13.{n}g} {bv:>13.{n}g}")

    print("\n-- multi-view GT colour spread (std across observing views) --")
    print("   LOW  = the cameras agree there IS a surface at this position")
    print("   HIGH = each view sees different content -> wrong depth")
    for lbl, q in (("p10", 10), ("median", 50), ("p90", 90)):
        cmp(f"colour spread {lbl}", spread, lambda x, q=q: np.percentile(x, q))
    for thr in (0.05, 0.10, 0.20):
        cmp(f"spread < {thr:.2f}  %", spread,
            lambda x, t=thr: 100.0 * float((x < t).mean()))

    if own is not None:
        err = np.abs(own[sel] - mean_c).mean(axis=1)
        print("\n-- |learned colour - multi-view consensus| --")
        for lbl, q in (("median", 50), ("p90", 90)):
            cmp(f"colour error {lbl}", err, lambda x, q=q: np.percentile(x, q))

    print("\n-- consensus brightness sampled at these positions --")
    lum = mean_c.mean(axis=1)
    for lbl, q in (("p10", 10), ("median", 50), ("p90", 90)):
        cmp(f"GT luminance {lbl}", lum, lambda x, q=q: np.percentile(x, q))

    print("\nRead: if FLAGGED spread is LOW and comparable to REST, the cameras"
          "\nagree there is matter at that depth -- the calibration is putting it"
          "\nthere and the fault is upstream of the trainer. If FLAGGED spread is"
          "\nHIGH, the position is not photometrically supported by the images.")


def _load_colmap_named(path: Path):
    """Same as _load_colmap but also returns each camera's COLMAP image name."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(path))
    centers, Rs, ts, Ks, whs, names = [], [], [], [], [], []
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
            pose = img.cam_from_world()
            Rm = _quat_xyzw_to_R(pose.rotation.quat)
            tv = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        else:
            Rm = np.asarray(img.R(), dtype=np.float64)
            tv = np.asarray(img.tvec, dtype=np.float64).reshape(3)
        centers.append(-Rm.T @ tv); Rs.append(Rm); ts.append(tv)
        try:
            fx = float(cam.focal_length_x); fy = float(cam.focal_length_y)
            cx = float(cam.principal_point_x); cy = float(cam.principal_point_y)
        except AttributeError:
            p = np.asarray(cam.params, dtype=np.float64)
            if len(p) == 3:
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
        whs.append([cam.width, cam.height])
        names.append(img.name)
    return (np.array(centers), np.array(Rs), np.array(ts),
            np.array(Ks), np.array(whs, dtype=np.float64), names)


if __name__ == "__main__":
    main()
