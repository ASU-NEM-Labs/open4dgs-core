#!/usr/bin/env python3
"""Repair holey subject masks using a verified triangulated point cloud.

The masks are mostly right (90-98% of verified subject points land inside
them) but have holes. The NPZ subject cloud is multi-view verified geometry,
so its projection into a camera is a reliable, if sparse, subject stencil.
The repaired mask is:

    close( original_mask  OR  dilate(splatted subject points) )

The original silhouette edge is kept wherever the original mask already
covered the subject; the point splat only adds coverage, never removes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--subject-center", type=float, nargs=3, required=True, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--subject-radius",
        type=float,
        default=1.6,
        help="Points beyond this distance from the centre are scene, not subject.",
    )
    parser.add_argument(
        "--dilate-px",
        type=int,
        default=9,
        help="Grow each splatted point by this radius before closing. Must "
        "exceed half the projected point spacing or the splat stays speckle.",
    )
    parser.add_argument("--close-px", type=int, default=7)
    parser.add_argument("--mask-pattern", type=str, default="{frame:06d}.png")
    parser.add_argument("--threshold", type=int, default=127)
    return parser


def disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def main() -> int:
    args = build_parser().parse_args()
    reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
    with np.load(args.npz_path, allow_pickle=False) as archive:
        points = np.asarray(archive["positions"], np.float64).reshape(-1, 3)
    center = np.asarray(args.subject_center, np.float64)
    subject = points[np.linalg.norm(points - center, axis=1) < float(args.subject_radius)]
    if len(subject) < 1000:
        raise ValueError(f"Only {len(subject)} subject points selected; check --subject-center")

    dilate = disk(int(args.dilate_px))
    close = disk(int(args.close_px))
    mask_name = args.mask_pattern.format(frame=int(args.frame))
    report = []
    for camera in reference["cameras"]:
        folder = str(camera["folder"])
        mask_path = args.masks_root / folder / mask_name
        original = np.asarray(Image.open(mask_path).convert("L"), np.uint8)
        height, width = original.shape
        params = np.asarray(camera["params"], np.float64)
        R = np.asarray(camera["R"], np.float64)
        t = np.asarray(camera["t"], np.float64).reshape(3)
        cam = (R @ subject.T).T + t
        ok = cam[:, 2] > 1e-6
        u = np.rint(params[0] * cam[ok, 0] / cam[ok, 2] + params[2]).astype(int)
        v = np.rint(params[1] * cam[ok, 1] / cam[ok, 2] + params[3]).astype(int)
        keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        splat = np.zeros((height, width), dtype=bool)
        splat[v[keep], u[keep]] = True
        splat = binary_dilation(splat, structure=dilate)
        base = original > int(args.threshold)
        repaired = binary_closing(base | splat, structure=close)
        repaired = binary_fill_holes(repaired)
        out_path = args.output_root / folder / mask_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((repaired * 255).astype(np.uint8)).save(out_path)
        added = float((repaired & ~base).mean())
        report.append(
            {
                "camera": folder,
                "fg_before": round(float(base.mean()), 4),
                "fg_after": round(float(repaired.mean()), 4),
                "added_fraction_of_frame": round(added, 4),
            }
        )
        print(
            f"{folder}: fg {base.mean()*100:5.2f}% -> {repaired.mean()*100:5.2f}%"
            f"  (+{added*100:.2f}% of frame)"
        )
    (args.output_root / "repair-report.json").write_text(
        json.dumps({"frame": int(args.frame), "cameras": report}, indent=1) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
