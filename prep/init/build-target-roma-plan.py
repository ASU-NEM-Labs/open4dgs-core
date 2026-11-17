#!/usr/bin/env python3
"""Plan high-resolution RoMa crop passes over a known 3D target region.

Full-frame matching gives a small subject region a proportional share of the
sample budget, so a head occupying 1.6% of a 4K frame gets 1.6% of the matches.
This planner spends the whole budget on the target instead: it projects a world
sphere into every camera, keeps the cameras that actually contain it, and emits
a crop pair for every viable camera pair.

Crucially the pair set is built here rather than inherited, so it is not limited
to the geometric-neighbour pairs used for full-frame matching -- those are
chosen without regard to whether both cameras can see the target, which is the
failure mode this exists to fix.

The plan contains only rectangles and camera pairs. No geometry is imported and
no downstream gate is relaxed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PlanCamera:
    def __init__(self, index: int, item: dict[str, Any]):
        if str(item["camera_model_name"]).upper() != "PINHOLE":
            raise ValueError("Target plans require PINHOLE input images")
        params = np.asarray(item["params"], dtype=np.float64)
        self.index = index
        self.folder = str(item["folder"])
        self.width = int(item["width"])
        self.height = int(item["height"])
        self.fx, self.fy, self.cx, self.cy = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        )
        self.R = np.asarray(item["R"], dtype=np.float64)
        self.t = np.asarray(item["t"], dtype=np.float64).reshape(3)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project_sphere(
        self, center: np.ndarray, radius: float
    ) -> tuple[float, float, float, float] | None:
        """Image position and pixel radius of a world sphere, or None if behind."""
        camera = self.R @ np.asarray(center, dtype=np.float64).reshape(3) + self.t
        depth = float(camera[2])
        if depth <= radius:
            return None
        u = self.fx * camera[0] / depth + self.cx
        v = self.fy * camera[1] / depth + self.cy
        # Small-angle radius is enough here; the crop is padded afterwards.
        radius_px = max(self.fx, self.fy) * radius / depth
        return float(u), float(v), float(radius_px), depth


def target_view(
    camera: PlanCamera,
    center: np.ndarray,
    radius: float,
    min_visible_fraction: float,
) -> dict[str, Any] | None:
    """How much of the projected target survives the sensor rectangle.

    A clipped target is still worth matching -- most of this rig images the
    head hanging off the top edge -- so this reports the visible fraction
    rather than demanding full containment.
    """
    projected = camera.project_sphere(center, radius)
    if projected is None:
        return None
    u, v, radius_px, depth = projected
    x0, y0 = u - radius_px, v - radius_px
    x1, y1 = u + radius_px, v + radius_px
    clip_x0, clip_y0 = max(x0, 0.0), max(y0, 0.0)
    clip_x1 = min(x1, float(camera.width))
    clip_y1 = min(y1, float(camera.height))
    if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
        return None
    visible = ((clip_x1 - clip_x0) * (clip_y1 - clip_y0)) / max(
        (x1 - x0) * (y1 - y0), 1e-12
    )
    if visible < float(min_visible_fraction):
        return None
    return {
        "u": round(u, 1),
        "v": round(v, 1),
        "radius_px": round(radius_px, 1),
        "depth": round(depth, 4),
        "visible_fraction": round(float(visible), 4),
        "clip": (clip_x0, clip_y0, clip_x1, clip_y1),
    }


def square_crop(
    camera: PlanCamera,
    view: dict[str, Any],
    padding: float,
    minimum: int,
    maximum: int,
) -> list[int] | None:
    """A square crop around the visible part of the target, inside the image."""
    clip_x0, clip_y0, clip_x1, clip_y1 = view["clip"]
    extent = max(clip_x1 - clip_x0, clip_y1 - clip_y0)
    size = int(math.ceil(extent * (1.0 + padding)))
    size = max(size, minimum)
    size = min(size, maximum, camera.width, camera.height)
    if size < 8:
        return None
    x0 = int(round(0.5 * (clip_x0 + clip_x1) - size / 2.0))
    y0 = int(round(0.5 * (clip_y0 + clip_y1) - size / 2.0))
    x0 = min(max(x0, 0), camera.width - size)
    y0 = min(max(y0, 0), camera.height - size)
    return [x0, y0, x0 + size, y0 + size]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    reference_json = args.reference_json.resolve()
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    cameras = [PlanCamera(i, item) for i, item in enumerate(reference["cameras"])]

    center = np.asarray(args.target_center, dtype=np.float64)
    radius = float(args.target_radius)
    if radius <= 0:
        raise ValueError(f"--target-radius must be positive, got {radius}")

    excluded = {int(index) for index in (args.exclude_cameras or [])}
    visible: list[tuple[PlanCamera, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for camera in cameras:
        if camera.index in excluded:
            rejected.append(
                {
                    "camera_index": camera.index,
                    "camera": camera.folder,
                    "reason": "excluded",
                }
            )
            continue
        view = target_view(camera, center, radius, float(args.min_visible_fraction))
        if view is None:
            rejected.append({"camera_index": camera.index, "camera": camera.folder})
            continue
        visible.append((camera, view))

    if len(visible) < 2:
        raise ValueError(
            f"Only {len(visible)} camera(s) contain the target, so no pair can "
            "be formed. Check --target-center, or lower --coverage."
        )

    # Score every candidate first, then decide which to keep. A camera on the
    # far side of the target is far in angle from everything else, so a global
    # angle cap starves exactly the cameras covering the back and crown -- the
    # regions nothing else can reach.
    candidates: list[tuple[float, int, int, dict[str, Any], dict[str, Any]]] = []
    for (cam_a, info_a), (cam_b, info_b) in itertools.combinations(visible, 2):
        view_a = cam_a.center - center
        view_b = cam_b.center - center
        view_a = view_a / max(float(np.linalg.norm(view_a)), 1e-12)
        view_b = view_b / max(float(np.linalg.norm(view_b)), 1e-12)
        angle = math.degrees(
            math.acos(float(np.clip(np.dot(view_a, view_b), -1.0, 1.0)))
        )
        if angle < float(args.min_view_angle_deg):
            continue
        candidates.append((angle, cam_a.index, cam_b.index, info_a, info_b))

    lookup = {cam.index: cam for cam, _view in visible}
    within_cap = {
        (a, b)
        for angle, a, b, _ia, _ib in candidates
        if angle <= float(args.max_view_angle_deg)
    }
    keep = set(within_cap)
    angle_rejected = len(candidates) - len(within_cap)

    # Top every starved camera back up with its narrowest remaining baselines.
    minimum = int(args.min_pairs_per_camera)
    if minimum > 0:
        by_camera: dict[int, list[tuple[float, int, int]]] = {
            index: [] for index in lookup
        }
        for angle, a, b, _ia, _ib in candidates:
            by_camera[a].append((angle, a, b))
            by_camera[b].append((angle, a, b))
        for options in by_camera.values():
            options.sort(key=lambda row: abs(row[0] - float(args.preferred_angle_deg)))
        counts = {index: 0 for index in lookup}
        for a, b in keep:
            counts[a] += 1
            counts[b] += 1
        for index in sorted(lookup, key=lambda i: counts[i]):
            for angle, a, b in by_camera[index]:
                if counts[index] >= minimum:
                    break
                if (a, b) in keep:
                    continue
                keep.add((a, b))
                counts[a] += 1
                counts[b] += 1

    entries: list[dict[str, Any]] = []
    for angle, index_a, index_b, info_a, info_b in candidates:
        if (index_a, index_b) not in keep:
            continue
        cam_a, cam_b = lookup[index_a], lookup[index_b]
        crop_a = square_crop(
            cam_a,
            info_a,
            float(args.crop_padding),
            int(args.min_crop_size_px),
            int(args.max_crop_size_px),
        )
        crop_b = square_crop(
            cam_b,
            info_b,
            float(args.crop_padding),
            int(args.min_crop_size_px),
            int(args.max_crop_size_px),
        )
        if crop_a is None or crop_b is None:
            continue
        # Rank by how much of the target both cameras actually hold and how
        # close the baseline is to the preferred one: RoMa degrades on wide
        # baselines, and a smooth low-texture surface degrades faster.
        weakest = min(info_a["visible_fraction"], info_b["visible_fraction"])
        entries.append(
            {
                "id": f"{cam_a.index:04d}-{cam_b.index:04d}",
                "cam_a": cam_a.index,
                "cam_b": cam_b.index,
                "camera_a": cam_a.folder,
                "camera_b": cam_b.folder,
                "crop_a": crop_a,
                "crop_b": crop_b,
                "view_angle_deg": round(angle, 3),
                "visible_a": info_a["visible_fraction"],
                "visible_b": info_b["visible_fraction"],
                "score": round(
                    weakest / (1.0 + abs(angle - float(args.preferred_angle_deg))), 6
                ),
            }
        )

    entries.sort(key=lambda row: row["score"], reverse=True)
    if int(args.max_pairs) > 0:
        entries = entries[: int(args.max_pairs)]

    return {
        "format": "trigs-depth-guided-roma-plan-v1",
        "purpose": "target-region high-resolution RoMa crops",
        "reference_json": reference_json.as_posix(),
        "reference_sha256": _sha256(reference_json),
        "target_frame": int(args.target_frame),
        "target_center": [float(v) for v in center],
        "target_radius": radius,
        "thresholds": {
            "min_visible_fraction": float(args.min_visible_fraction),
            "min_view_angle_deg": float(args.min_view_angle_deg),
            "max_view_angle_deg": float(args.max_view_angle_deg),
            "crop_padding": float(args.crop_padding),
            "min_crop_size_px": int(args.min_crop_size_px),
            "max_crop_size_px": int(args.max_crop_size_px),
        },
        "summary": {
            "cameras": len(cameras),
            "cameras_containing_target": len(visible),
            "candidate_pairs": len(visible) * (len(visible) - 1) // 2,
            "angle_rejected_pairs": angle_rejected,
            "entries": len(entries),
        },
        "cameras_containing_target": [
            {
                "camera_index": cam.index,
                "camera": cam.folder,
                **{k: v for k, v in view.items() if k != "clip"},
            }
            for cam, view in visible
        ],
        "cameras_rejected": rejected,
        "entries": entries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument(
        "--target-center", type=float, nargs=3, required=True, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--target-radius", type=float, required=True)
    parser.add_argument("--target-frame", type=int, required=True)
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.25,
        help="Fraction of the target's projected extent that must fall inside "
        "the sensor. A clipped target still matches, so this is well below 1.",
    )
    parser.add_argument("--min-view-angle-deg", type=float, default=4.0)
    parser.add_argument("--max-view-angle-deg", type=float, default=40.0)
    parser.add_argument(
        "--preferred-angle-deg",
        type=float,
        default=15.0,
        help="Pairs nearest this baseline sort first, so a --max-pairs cut "
        "keeps the ones RoMa matches most reliably.",
    )
    parser.add_argument(
        "--exclude-cameras",
        type=int,
        nargs="+",
        help="Camera indices to keep out of the plan entirely. Use for cameras "
        "whose calibration is known bad: they consume pairs and return nothing.",
    )
    parser.add_argument(
        "--min-pairs-per-camera",
        type=int,
        default=6,
        help="Guarantee every visible camera this many partners, adding its "
        "narrowest baselines past --max-view-angle-deg if needed. Without it "
        "the cap starves back and crown cameras, which nothing else covers.",
    )
    parser.add_argument("--crop-padding", type=float, default=0.35)
    parser.add_argument("--min-crop-size-px", type=int, default=384)
    parser.add_argument("--max-crop-size-px", type=int, default=1024)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    plan = build_plan(parsed)
    output = parsed.output_plan.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_plan": str(output), **plan["summary"]}, indent=2))
