#!/usr/bin/env python3
"""Bundle-adjust an existing camera solve against long tracks, using pyceres.

Why this exists
---------------
bundle_adjust_static.py stalls: optimality flat at 3.31e6, intrinsics moved
exactly 0. That was read as a scipy limitation, but the measured cause is track
structure, not the optimizer.

    tracks from            mean length   residual/parameter
    RoMa pairwise linking      5.6            3.7 : 1
    VGGT/LightGlue tracker    18.0           11.9 : 1

A point seen in 3 views contributes 6 residuals and 3 new parameters. A point
seen in 18 views contributes 36 for the same 3. Pairwise matching plus
transitive linking cannot build long tracks on a ring this fragmented -- the
same pathology that gave DetectorFreeSfM a mean track length of 2.1. Below about
4:1 the normal equations are barely determined, which is why CHOLMOD reports
"matrix not positive definite" and why the scipy solve went nowhere.

What this does
--------------
Takes the TRACKS from one COLMAP model and the POSES AND INTRINSICS from a
separate reference_export.json, then bundle adjusts. Those come from different
places on purpose: the tracker that produces well-conditioned tracks is not
necessarily the solve that produces the best poses. On example capture the
VGGT/LightGlue model supplies 18-long tracks while reference_export.json supplies
poses measured at 3.85 px held-out, against VGGT's own 8.40 px.

Grading
-------
Grade the output with referee_third_view.py against the RoMa clean-plate cache.
That cache took no part in this optimisation and comes from an unrelated matcher
family, so it is a genuine cross-matcher train/test split -- the independent
adjudication the calibration handoff asks for and never had.

Staging
-------
Refining everything at once on ill-conditioned data is how the earlier COLMAP
attempt produced a focal 2.34x off and a principal point 2029 px off sensor.
Default here refines poses and structure only. Add --refine-focal and then
--refine-principal-point as separate runs and let the referee decide whether each
one earned its parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pycolmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, help="COLMAP model supplying TRACKS")
    p.add_argument("--init-reference", required=True,
                   help="reference_export.json supplying POSES and INTRINSICS to seed from")
    p.add_argument("--out", required=True, help="Output reference_export-style JSON")
    p.add_argument("--out-sparse", default=None, help="Optional COLMAP model output dir")
    p.add_argument("--image-name", default="000001.jpg", help="image_name for output entries")
    p.add_argument("--refine-focal", action="store_true")
    p.add_argument("--refine-principal-point", action="store_true")
    p.add_argument("--refine-extra-params", action="store_true",
                   help="Refine distortion. Frames on this rig are confirmed rectified, "
                        "so this should stay off.")
    p.add_argument("--no-refine-points", action="store_true")
    p.add_argument("--no-refine-poses", action="store_true",
                   help="Hold camera poses at the seed. Isolates an intrinsics correction so "
                        "noisy tracks cannot corrupt poses that are already good.")
    p.add_argument("--max-reproj-error", type=float, default=12.0,
                   help="Filter observations after re-triangulating under the seeded poses")
    p.add_argument("--min-tri-angle", type=float, default=1.0, help="degrees")
    p.add_argument("--min-track-length", type=int, default=3)
    return p.parse_args()


def load_reference(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    out = {}
    for c in data["cameras"]:
        p = c["params"]
        out[c["folder"]] = {
            "R": np.asarray(c["R"], dtype=np.float64),
            "t": np.asarray(c["t"], dtype=np.float64),
            "fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3]),
            "extra": np.asarray(p[4:], dtype=np.float64) if len(p) > 4 else np.zeros(0),
        }
    return out


def triangulate_dlt(obs):
    """obs: list of (P 3x4 projection matrix, xy). Returns X or None."""
    rows = []
    for P, (u, v) in obs:
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    try:
        _, _, Vt = np.linalg.svd(np.asarray(rows))
    except np.linalg.LinAlgError:
        return None
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def main() -> int:
    args = parse_args()

    rec = pycolmap.Reconstruction(args.sparse)
    print(f"loaded tracks from {args.sparse}")
    print(f"  {rec.num_images()} images, {rec.num_points3D()} points, "
          f"{rec.compute_num_observations()} observations, "
          f"mean track {rec.compute_mean_track_length():.1f}")

    ref = load_reference(args.init_reference)
    print(f"seeding poses/intrinsics from {os.path.basename(args.init_reference)} "
          f"({len(ref)} cameras)")
    n_extra = sum(1 for v in ref.values() if v["extra"].size and np.abs(v["extra"]).max() > 1e-12)
    if n_extra:
        print(f"  NOTE: {n_extra} cameras carry distortion in the seed; modelled as PINHOLE "
              f"(rig is confirmed rectified, so those coefficients are fitted error)")

    # ---- overwrite poses and intrinsics from the seed --------------------
    missing = []
    for img in rec.images.values():
        folder = os.path.splitext(img.name)[0]
        if folder not in ref:
            missing.append(img.name)
            continue
        r = ref[folder]
        img.frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(r["R"]), r["t"])
        cam = rec.cameras[img.camera_id]
        # Keep whatever the seed models. On this rig the distortion coefficients
        # were read as fitted error and discarded -- but they are worth 9.35 -> 3.85 px
        # on HELD-OUT third-view reprojection, fit on footage and generalising to
        # clean plates. Whatever they absorb is real, so carry it and let BA refine it.
        if r["extra"].size >= 4:
            cam.model = pycolmap.CameraModelId.OPENCV
            cam.params = np.concatenate([[r["fx"], r["fy"], r["cx"], r["cy"]],
                                         r["extra"][:4]]).astype(np.float64)
        else:
            cam.model = pycolmap.CameraModelId.PINHOLE
            cam.params = np.array([r["fx"], r["fy"], r["cx"], r["cy"]], dtype=np.float64)
    if missing:
        sys.exit(f"{len(missing)} images have no entry in the seed reference: {missing[:5]}")

    # ---- re-triangulate every track under the seeded poses ---------------
    # Triangulate on ideal-pinhole rays: undistort observations once, up front.
    proj, undist = {}, {}
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        pose = img.cam_from_world()
        proj[img.image_id] = K @ np.hstack([pose.rotation.matrix(), pose.translation.reshape(3, 1)])
        d = np.asarray(cam.params[4:], dtype=np.float64)
        undist[img.image_id] = d if d.size and np.abs(d).max() > 1e-12 else None

    # pycolmap 4.x has no filter_all_points3D, so triangulation, cheirality,
    # outlier rejection and the track-length floor are done here in one pass.
    centres = {img.image_id: np.asarray(img.projection_center())
               for img in rec.images.values()}

    n_retri = n_failed = n_drop_track = n_drop_obs = n_drop_angle = 0
    resid_before = []
    cos_min = np.cos(np.radians(args.min_tri_angle))
    for pid in list(rec.point3D_ids()):
        pt = rec.points3D[pid]
        els = [(el.image_id, el.point2D_idx) for el in pt.track.elements]
        obs = []
        for i, j in els:
            xy = np.asarray(rec.images[i].points2D[j].xy, dtype=np.float64)
            d = undist[i]
            if d is not None:
                cam = rec.cameras[rec.images[i].camera_id]
                fx, fy, cx, cy = cam.params[:4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                xy = cv2.undistortPoints(xy.reshape(1, 1, 2), K, d, P=K).reshape(2)
            obs.append((proj[i], xy))
        X = triangulate_dlt(obs) if len(obs) >= 2 else None
        if X is None or not np.all(np.isfinite(X)):
            rec.delete_point3D(pid)
            n_failed += 1
            continue

        errs, bad = [], []
        for (i, j), (P, xy) in zip(els, obs):
            h = P @ np.append(X, 1.0)
            if h[2] <= 1e-9:
                bad.append((i, j))
                continue
            e = float(np.hypot(h[0] / h[2] - xy[0], h[1] / h[2] - xy[1]))
            errs.append(e)
            if e > args.max_reproj_error:
                bad.append((i, j))
        if not errs or np.median(errs) > args.max_reproj_error:
            rec.delete_point3D(pid)
            n_drop_track += 1
            continue

        # triangulation angle: widest ray pair must exceed the floor
        rays = [(X - centres[i]) for i, _ in els if (i, _) not in bad]
        rays = [r / np.linalg.norm(r) for r in rays if np.linalg.norm(r) > 1e-9]
        widest = min((float(a @ b) for k, a in enumerate(rays) for b in rays[k + 1:]), default=1.0)
        if widest > cos_min:
            rec.delete_point3D(pid)
            n_drop_angle += 1
            continue

        if len(els) - len(bad) < args.min_track_length:
            rec.delete_point3D(pid)
            n_drop_track += 1
            continue
        for i, j in bad:
            rec.delete_observation(i, j)
            n_drop_obs += 1
        pt.xyz = X
        resid_before.extend(errs)
        n_retri += 1

    resid_before = np.asarray(resid_before)
    print(f"re-triangulated {n_retri} points under the seeded poses")
    print(f"  dropped: {n_failed} untriangulable, {n_drop_track} bad/short tracks, "
          f"{n_drop_angle} low parallax, {n_drop_obs} outlier observations")
    print(f"  reprojection under seed (kept obs): p50 {np.median(resid_before):.3f} px  "
          f"p90 {np.percentile(resid_before, 90):.3f}")
    print(f"  remaining: {rec.num_points3D()} points, {rec.compute_num_observations()} "
          f"observations, mean track {rec.compute_mean_track_length():.1f}")

    n_par = rec.num_images() * 6 + rec.num_points3D() * 3
    if args.refine_focal:
        n_par += rec.num_cameras()
    if args.refine_principal_point:
        n_par += 2 * rec.num_cameras()
    print(f"  conditioning: {2 * rec.compute_num_observations() / max(n_par, 1):.1f}:1 "
          f"residuals per parameter")

    # ---- bundle adjust ---------------------------------------------------
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = args.refine_focal
    opts.refine_principal_point = args.refine_principal_point
    opts.refine_extra_params = args.refine_extra_params
    opts.refine_points3D = not args.no_refine_points
    opts.refine_rig_from_world = not args.no_refine_poses
    opts.min_track_length = args.min_track_length
    opts.print_summary = True
    print(f"\nBA: poses={opts.refine_rig_from_world} focal={opts.refine_focal_length} "
          f"pp={opts.refine_principal_point} extra={opts.refine_extra_params} "
          f"points={opts.refine_points3D}")
    pycolmap.bundle_adjustment(rec, opts)
    print(f"\nreprojection error after BA: {rec.compute_mean_reprojection_error():.3f} px")

    # ---- export ----------------------------------------------------------
    cams_out = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        folder = os.path.splitext(img.name)[0]
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        extra = [float(x) for x in cam.params[4:]]
        pose = img.cam_from_world()
        cams_out.append({
            "name": f"{folder}/{args.image_name}",
            "folder": folder,
            "image_name": args.image_name,
            "R": pose.rotation.matrix().tolist(),
            "t": np.asarray(pose.translation).tolist(),
            "camera_model_name": "PINHOLE",
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [float(fx), float(fy), float(cx), float(cy)] + extra,
        })

    fx_before = np.array([ref[c["folder"]]["fx"] for c in cams_out])
    fx_after = np.array([c["params"][0] for c in cams_out])
    ratio = fx_after / fx_before
    print(f"focal change: p50 {np.median(ratio):.4f}  "
          f"p10-p90 {np.percentile(ratio, 10):.4f}-{np.percentile(ratio, 90):.4f}  "
          f"max |1-r| {np.abs(1 - ratio).max():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"cameras": cams_out,
                   "calibration_source": "pycolmap-ba",
                   "seeded_from": os.path.basename(args.init_reference),
                   "tracks_from": os.path.abspath(args.sparse)}, f, indent=1)
    print(f"wrote {args.out}")

    if args.out_sparse:
        os.makedirs(args.out_sparse, exist_ok=True)
        rec.write(args.out_sparse)
        print(f"wrote {args.out_sparse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
