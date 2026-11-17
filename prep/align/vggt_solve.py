#!/usr/bin/env python3
"""Feed-forward camera solve for the 4DV ring using VGGT.

Why this exists
---------------
Every pairwise-matching route into this capture has failed the same way: the
static background does not connect all 96 cameras into one rigid component.
DetectorFreeSfM fragmented into 4 components (41/31/25/23, mean track length
2.1); the iterated resection audit could not grow the trusted set past 37, and
those 37 leave a 70 degree azimuthal gap; COLMAP's bundle_adjuster reported
rank-deficient normal equations.

None of that is a matching-quality problem. It is that pairwise matching has to
CHAIN overlap around a ring, and the chain breaks -- narrow FOV (26-42 deg) with
a 3.42x zoom spread means adjacent cameras are framed very differently on the
background, even though all 96 are framed on the subject.

VGGT does not chain. It attends over all input images jointly in one forward
pass, so connectivity is never built out of pairwise overlap. That is the exact
failure mode, attacked at its cause.

What this is and is not
-----------------------
VGGT works at 518 px on the long edge. From 3840x2160 that is a 7.4x downscale,
so one VGGT pixel is ~7.4 native pixels. At fx~5124 that is ~0.083 deg per VGGT
pixel. The manifest prior is 1-1.5 deg out, so even a sloppy 2 px VGGT solve
would be an order of magnitude better initialisation -- but it is still an
INITIALISER. It cannot be a sub-pixel calibration and must be refined.

Static content only
-------------------
The default input is reference_images/ -- the clean plates, one frame per
camera. A correspondence is geometrically valid iff both pixels image the same
3D point; for time-invariant geometry that holds regardless of when each camera
exposed. Free-running cameras are therefore no obstacle here, and no timing
assumption enters the solve. Do not point this at images/ -- that content moves
and the cameras are up to 16.3 ms apart.

Output
------
JSON in the same schema as reference_export.json (cameras[] with R, t and
PINHOLE params at native 3840x2160), so it drops into the existing tooling
unchanged. Intrinsics are un-transformed from VGGT's processed resolution back
to native using per-axis scales, because the round-to-multiple-of-14 in VGGT's
preprocessing makes the horizontal and vertical scales differ slightly.

VGGT's world frame is arbitrary: camera 0 is the origin and overall scale is
unresolved. Compare against any reference with a similarity transform --
see vggt_compare_reference.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-root", required=True,
                   help="Directory of per-camera folders, e.g. <dataset>/reference_images")
    p.add_argument("--image-name", default="000001.jpg",
                   help="Filename to take from inside each camera folder")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--npz", default=None,
                   help="Optional path to also dump raw VGGT extrinsics/intrinsics")
    p.add_argument("--model", default="facebook/VGGT-1B", help="HuggingFace model id")
    p.add_argument("--mode", default="crop", choices=["crop", "pad"],
                   help="VGGT preprocessing. 'crop' keeps the full 16:9 FOV at 518x294; "
                        "'pad' letterboxes to 518x518")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N cameras (for a memory/behaviour probe)")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth camera (e.g. 4 -> 24 cameras evenly around the ring)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def collect_images(root: str, image_name: str, limit, stride: int):
    folders = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    folders = folders[::stride]
    if limit is not None:
        folders = folders[:limit]
    entries = []
    for folder in folders:
        path = os.path.join(root, folder, image_name)
        if not os.path.isfile(path):
            sys.exit(f"missing image: {path}")
        entries.append((folder, path))
    if not entries:
        sys.exit(f"no camera folders found under {root}")
    return entries


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    entries = collect_images(args.images_root, args.image_name, args.limit, args.stride)
    paths = [p for _, p in entries]
    print(f"[vggt] {len(entries)} cameras from {args.images_root}", flush=True)

    native_w, native_h = Image.open(paths[0]).size
    for _, p in entries:
        if Image.open(p).size != (native_w, native_h):
            sys.exit(f"mixed native resolutions; {p} differs from {native_w}x{native_h}")
    print(f"[vggt] native resolution {native_w}x{native_h}", flush=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available; VGGT on CPU is not practical at this frame count")
    cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    if device == "cuda":
        print(f"[vggt] {torch.cuda.get_device_name(0)}  sm_{cap}x  dtype={dtype}", flush=True)

    t0 = time.time()
    model = VGGT.from_pretrained(args.model).to(device).eval()
    print(f"[vggt] model loaded in {time.time() - t0:.1f}s", flush=True)

    images = load_and_preprocess_images(paths, mode=args.mode).to(device)
    proc_h, proc_w = int(images.shape[-2]), int(images.shape[-1])
    print(f"[vggt] processed resolution {proc_w}x{proc_h}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
            batch = images[None]
            tokens, ps_idx = model.aggregator(batch)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), (proc_h, proc_w))
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    print(f"[vggt] forward pass {elapsed:.1f}s   peak GPU {peak:.1f} GiB", flush=True)

    extrinsic = extrinsic[0].cpu().numpy().astype(np.float64)   # (S, 3, 4) world-to-camera
    intrinsic = intrinsic[0].cpu().numpy().astype(np.float64)   # (S, 3, 3) in processed pixels

    # VGGT preprocessing rounds the short side to a multiple of 14, so the
    # horizontal and vertical scales are not identical. Un-transform per axis.
    sx = native_w / proc_w
    sy = native_h / proc_h
    print(f"[vggt] intrinsic rescale sx={sx:.6f} sy={sy:.6f}", flush=True)

    cameras = []
    for i, (folder, _) in enumerate(entries):
        K = intrinsic[i]
        R = extrinsic[i, :, :3]
        t = extrinsic[i, :, 3]
        cameras.append({
            "name": f"{folder}/{args.image_name}",
            "folder": folder,
            "image_name": args.image_name,
            "R": R.tolist(),
            "t": t.tolist(),
            "camera_model_name": "PINHOLE",
            "width": native_w,
            "height": native_h,
            "params": [K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy],
        })

    payload = {
        "cameras": cameras,
        "calibration_source": "vggt",
        "vggt": {
            "model": args.model,
            "mode": args.mode,
            "n_images": len(entries),
            "processed_width": proc_w,
            "processed_height": proc_h,
            "native_width": native_w,
            "native_height": native_h,
            "scale_x": sx,
            "scale_y": sy,
            "dtype": str(dtype),
            "forward_seconds": elapsed,
            "peak_gpu_gib": peak,
            "images_root": os.path.abspath(args.images_root),
            "note": "world frame is arbitrary: camera 0 is the origin and scale is unresolved",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[vggt] wrote {args.out}", flush=True)

    if args.npz:
        np.savez(args.npz, extrinsic=extrinsic, intrinsic=intrinsic,
                 folders=np.array([f for f, _ in entries]))
        print(f"[vggt] wrote {args.npz}", flush=True)


if __name__ == "__main__":
    main()
