#!/usr/bin/env python3
"""Convert a COLMAP sparse model to reference_export.json schema.

Why this exists
---------------
Solvers that emit COLMAP models (demo_colmap.py, COLMAP itself) need to be
graded with the same tooling as everything else in this pipeline, which speaks
reference_export.json. This does that conversion and nothing else.

Conventions
-----------
COLMAP stores world-to-camera as a quaternion (w, x, y, z) plus translation,
which is the same convention reference_export.json uses for R and t, so no
inversion or axis flip is applied. Camera models are expanded to PINHOLE
[fx, fy, cx, cy]; a SIMPLE_PINHOLE model has one focal, so fx and fy are set
equal to it.

The `folder` key is taken from the image filename stem, which is how the flat
image directory fed to demo_colmap.py encodes the camera id.

Distortion
----------
Frames on this rig are confirmed rectified (sagitta p50 0.159 px across the
whole fx range), so any non-zero distortion coefficient in a solved model is
fitted error rather than optics. Models carrying distortion are converted on
their pinhole part and the dropped coefficients are reported, not silently
discarded.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

# model_id -> (name, num_params, indices of (fx, fy, cx, cy) within params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, (0, 0, 1, 2)),
    1: ("PINHOLE", 4, (0, 1, 2, 3)),
    2: ("SIMPLE_RADIAL", 4, (0, 0, 1, 2)),
    3: ("RADIAL", 5, (0, 0, 1, 2)),
    4: ("OPENCV", 8, (0, 1, 2, 3)),
    5: ("OPENCV_FISHEYE", 8, (0, 1, 2, 3)),
    6: ("FULL_OPENCV", 12, (0, 1, 2, 3)),
    7: ("FOV", 5, (0, 1, 2, 3)),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, (0, 0, 1, 2)),
    9: ("RADIAL_FISHEYE", 5, (0, 0, 1, 2)),
    10: ("THIN_PRISM_FISHEYE", 12, (0, 1, 2, 3)),
}
PINHOLE_LIKE = {0, 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, help="COLMAP sparse model dir (with cameras.bin, images.bin)")
    p.add_argument("--out", required=True, help="Output reference_export-style JSON")
    p.add_argument("--image-name", default="000001.jpg",
                   help="image_name written into each entry (the pipeline's per-camera frame)")
    p.add_argument("--source", default="vggt-ba", help="Value for calibration_source")
    return p.parse_args()


def _read(f, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("truncated COLMAP file")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: str) -> dict:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            cam_id, model_id, width, height = _read(f, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                sys.exit(f"unknown COLMAP camera model id {model_id}")
            name, n_params, idx = CAMERA_MODELS[model_id]
            params = _read(f, "<" + "d" * n_params)
            cams[cam_id] = {"model": name, "model_id": model_id, "width": width,
                            "height": height, "params": np.array(params), "idx": idx}
    return cams


def read_images_binary(path: str) -> dict:
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            image_id = _read(f, "<I")[0]
            qvec = np.array(_read(f, "<dddd"))       # w, x, y, z
            tvec = np.array(_read(f, "<ddd"))
            cam_id = _read(f, "<I")[0]
            chars = []
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                chars.append(c)
            name = b"".join(chars).decode()
            (n_pts,) = _read(f, "<Q")
            f.seek(n_pts * 24, os.SEEK_CUR)          # x, y, point3D_id per observation
            imgs[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id,
                              "name": name, "n_points2D": n_pts}
    return imgs


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def main() -> None:
    args = parse_args()
    cams = read_cameras_binary(os.path.join(args.sparse, "cameras.bin"))
    imgs = read_images_binary(os.path.join(args.sparse, "images.bin"))
    print(f"read {len(imgs)} registered images, {len(cams)} camera models from {args.sparse}")

    models = sorted({c["model"] for c in cams.values()})
    print(f"camera models present: {', '.join(models)}")
    dropped = {c["model"] for c in cams.values() if c["model_id"] not in PINHOLE_LIKE}
    if dropped:
        print(f"WARNING: dropping distortion coefficients from {', '.join(sorted(dropped))} "
              f"-- frames on this rig are confirmed rectified, so those are fitted error")

    out = []
    for image_id in sorted(imgs, key=lambda i: imgs[i]["name"]):
        im = imgs[image_id]
        cam = cams[im["camera_id"]]
        fx_i, fy_i, cx_i, cy_i = cam["idx"]
        p = cam["params"]
        folder = os.path.splitext(im["name"])[0]
        out.append({
            "name": f"{folder}/{args.image_name}",
            "folder": folder,
            "image_name": args.image_name,
            "R": qvec_to_rotmat(im["qvec"]).tolist(),
            "t": im["tvec"].tolist(),
            "camera_model_name": "PINHOLE",
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "params": [float(p[fx_i]), float(p[fy_i]), float(p[cx_i]), float(p[cy_i])],
            "n_points2D": int(im["n_points2D"]),
        })

    wh = {(c["width"], c["height"]) for c in cams.values()}
    print(f"image sizes in model: {sorted(wh)}")
    fx = np.array([c["params"][0] for c in out])
    print(f"fx range {fx.min():.0f} .. {fx.max():.0f}  ({fx.max() / fx.min():.2f}x spread)")
    npts = np.array([c["n_points2D"] for c in out])
    print(f"observations per image: p50 {np.median(npts):.0f}  min {npts.min()}  max {npts.max()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"cameras": out, "calibration_source": args.source}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
