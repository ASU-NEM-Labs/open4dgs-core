#!/usr/bin/env python3
"""Measure a camera solve with COLMAP SIFT reprojection error.

Builds a COLMAP model from the poses/intrinsics in a reference_export JSON,
triangulates the existing SIFT tracks in a COLMAP database against those fixed
poses, and reports the reprojection error.

The point of using SIFT rather than a dense matcher: detector keypoints are
sub-pixel refined and localise to roughly 0.2-0.5 px, where RoMa's dense flow
localises to ~0.86 px. A measurement built on RoMa correspondences cannot read
sub-pixel because its own noise floor sits above that; SIFT can.

Poses are held fixed throughout. Nothing is bundle adjusted, so the reported
error is the solve's, not a fit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pycolmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", required=True, help="COLMAP database.db with SIFT features and matches")
    p.add_argument("--image-path", required=True, help="Root the database image names are relative to")
    p.add_argument("--reference", required=True, action="append",
                   help="reference_export JSON to measure. Repeat to compare several.")
    p.add_argument("--data-dir", default="", help="Prefix for relative --reference paths")
    p.add_argument("--out", default=None)
    p.add_argument("--keep-model", default=None, help="Directory to keep triangulated models in")
    return p.parse_args()


def camera_params(c):
    """OPENCV params from a reference_export camera: fx, fy, cx, cy, k1, k2, p1, p2.

    The database's cameras are OPENCV, and triangulate_points asserts the model
    matches, so everything is expressed in that model with zeros where a
    reference carries no distortion.
    """
    p = c["params"]
    extra = [float(x) for x in p[4:8]]
    extra += [0.0] * (4 - len(extra))
    return np.array([float(p[0]), float(p[1]), float(p[2]), float(p[3])] + extra,
                    dtype=np.float64)


def build_model(ref_path: str, db_images: dict) -> tuple:
    """Reconstruction with the JSON's cameras/poses and no points."""
    with open(ref_path) as f:
        data = json.load(f)
    rec = pycolmap.Reconstruction()
    by_folder = {c["folder"]: c for c in data["cameras"]}
    params_by_cam = {}
    for image_id, (name, camera_id) in sorted(db_images.items()):
        folder = name.split("/")[0]
        if folder not in by_folder:
            continue
        c = by_folder[folder]
        params = camera_params(c)
        params_by_cam[camera_id] = params
        cam = pycolmap.Camera.create_from_model_name(
            camera_id, "OPENCV", float(params[0]), int(c["width"]), int(c["height"]))
        cam.camera_id = camera_id
        cam.params = params
        # pycolmap 4.x keeps the pose on the frame and Image.cam_from_world is
        # read-only, so these go in through the trivial rig/frame wrappers.
        rec.add_camera_with_trivial_rig(cam)
        img = pycolmap.Image(name=name, camera_id=camera_id, image_id=image_id)
        rec.add_image_with_trivial_frame(img)
        rec.images[image_id].frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.asarray(c["R"], dtype=np.float64)),
            np.asarray(c["t"], dtype=np.float64))
    # A frame carrying a pose is still not "registered", and triangulate_points
    # refuses to run until it is.
    for frame_id in list(rec.frames):
        rec.register_frame(frame_id)
    return rec, params_by_cam


def patch_database_cameras(db_path: str, params_by_cam: dict) -> None:
    """Write the same intrinsics into the database so the two cannot disagree."""
    import sqlite3
    con = sqlite3.connect(db_path)
    for camera_id, params in params_by_cam.items():
        con.execute("update cameras set model=4, params=?, prior_focal_length=1 "
                    "where camera_id=?",
                    (np.asarray(params, dtype=np.float64).tobytes(), int(camera_id)))
    con.commit()
    con.close()


def main() -> int:
    args = parse_args()
    import sqlite3
    con = sqlite3.connect(args.database)
    db_images = {int(i): (n, int(c)) for i, n, c in
                 con.execute("select image_id,name,camera_id from images")}
    nkp = con.execute("select sum(rows) from keypoints").fetchone()[0]
    npairs = con.execute("select count(*) from two_view_geometries where rows>0").fetchone()[0]
    nver = con.execute("select sum(rows) from two_view_geometries").fetchone()[0]
    con.close()
    print(f"database: {len(db_images)} images, {nkp} SIFT keypoints, "
          f"{npairs} verified pairs, {nver} verified matches\n")

    results = {}
    print(f"{'reference':<46} {'mean px':>9} {'points':>9} {'obs':>10} {'track':>7}")
    print("-" * 86)
    for ref in args.reference:
        path = ref if os.path.isabs(ref) else os.path.join(args.data_dir, ref)
        rec, params_by_cam = build_model(path, db_images)
        work_db = os.path.join(tempfile.gettempdir(), "sift_measure_work.db")
        shutil.copyfile(args.database, work_db)
        patch_database_cameras(work_db, params_by_cam)
        outdir = (os.path.join(args.keep_model, os.path.basename(path).replace(".json", ""))
                  if args.keep_model else tempfile.mkdtemp(prefix="sift-tri-"))
        os.makedirs(outdir, exist_ok=True)
        pycolmap.triangulate_points(rec, work_db, args.image_path, outdir,
                                    clear_points=True)
        tri = pycolmap.Reconstruction(outdir)
        err = tri.compute_mean_reprojection_error()
        npts = tri.num_points3D()
        nobs = tri.compute_num_observations()
        tl = tri.compute_mean_track_length()
        name = os.path.basename(path)
        print(f"{name:<46} {err:>9.3f} {npts:>9} {nobs:>10} {tl:>7.2f}")
        results[name] = {"mean_reproj_px": float(err), "num_points3D": int(npts),
                         "num_observations": int(nobs), "mean_track_length": float(tl)}
        if not args.keep_model:
            shutil.rmtree(outdir, ignore_errors=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"database": args.database, "results": results}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
