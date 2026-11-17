#!/usr/bin/env python3
"""Recover image-consistent 4DV PTZ camera poses from multi-view footage.

The raw 4DV capture bundle contains two complementary calibration records:

* the bound snapshot contains PTZ-adjusted extrinsics but only base intrinsics;
* the capture manifest contains zoom-aware focal metadata and camera settings;
* the image-calibrated reference contains full distortion and optical centres.

Neither file contains the image-refined calibration produced by 4DV's dynamic
calibration service.  This tool recreates the missing rotation-refinement stage
without allowing an unconstrained SfM solve to move cameras arbitrarily:

1. align the snapshot PTZ rotations into an existing reference-export world;
2. retain the reference's image-calibrated centres and distortion models;
3. undistort RoMa observations and fit independent fundamental matrices;
4. recover relative poses and solve absolute image-derived rotation seeds from
   the known camera-centre baseline directions;
5. jointly refine every camera rotation, then repair failing cameras' pose and
   per-axis focal scale against only geometrically consistent training edges;
6. validate every repaired camera on the original, unpruned held-out
   correspondences and write a new reference export only when all gates pass.

The match cache is resumable.  A candidate is not written unless the camera
graph is connected and the held-out validation gates pass (unless explicitly
overridden with ``--write-failed-candidate``).
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _load_module(path: Path, name: str):
    # Several of the existing prep entry points use sibling imports such as
    # ``from roma_backends import ...`` because they are normally executed as
    # scripts.  Preserve that execution environment when loading one through
    # importlib from another directory.
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_prep_module():
    return _load_module(
        Path(__file__).parents[1] / "init" / "prep-roma-trigs-memory.py",
        "_trigs_prep_roma_for_dynamic_calibration",
    )


def _load_candidate_module():
    return _load_module(
        Path(__file__).with_name("build_4dv_snapshot_pose_candidate.py"),
        "_trigs_4dv_pose_candidate_for_dynamic_calibration",
    )


def _load_colmap_io_module():
    return _load_module(
        trigs_file("io/colmap.py"),
        "_trigs_colmap_io_for_dynamic_calibration",
    )


def parse_frames(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive comma-separated frames")
    return tuple(dict.fromkeys(values))


def parse_camera_pairs(text: str) -> Tuple[Tuple[str, str], ...]:
    result: List[Tuple[str, str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [value.strip() for value in item.split(":")]
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise argparse.ArgumentTypeError(
                "expected comma-separated pairs such as 0026:0079"
            )
        result.append(tuple(sorted(parts)))
    return tuple(dict.fromkeys(result))


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -np.asarray(rotation, dtype=np.float64).T @ np.asarray(
        translation, dtype=np.float64
    )


def capture_intrinsics(capture: dict) -> Dict[str, np.ndarray]:
    """Return capture-time, zoom-aware K matrices keyed by four-digit folder."""
    result: Dict[str, np.ndarray] = {}
    for row in capture.get("intri", {}).get("data", []):
        camera_id = str(row["camera_id"])
        matrix = np.asarray(row["intrinsic"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{camera_id}: invalid capture intrinsic matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError(f"{camera_id}: non-positive focal length")
        result[f"{int(camera_id.removeprefix('JCam')):04d}"] = matrix
    if not result:
        raise ValueError("capture JSON contains no intrinsics")
    return result


def build_effective_prior(
    base: dict,
    capture: dict,
    snapshot: dict,
    pose_prior_source: str = "base",
) -> Tuple[dict, dict]:
    """Build a pose prior while retaining image-calibrated projection geometry.

    The 4DV records identify a successful mechanical calibration session, but
    they do not contain the image-refined dynamic calibration.  In particular,
    the snapshot/capture rotations can disagree substantially with the recorded
    pixels even when their camera centres and PTZ metadata are internally
    consistent.  The image-derived base is therefore the safe default pose
    prior.  Snapshot rotations remain available as an explicit A/B option.

    The capture manifest has zoom-aware focal lengths but no distortion.  The
    extracted frames are still distorted, so all modes retain the base
    reference's full OPENCV projection model.
    """
    if pose_prior_source not in {"base", "snapshot"}:
        raise ValueError("pose_prior_source must be 'base' or 'snapshot'")
    candidate_module = _load_candidate_module()
    folders = tuple(str(camera["folder"]) for camera in base["cameras"])
    snapshot_prior, diagnostics = candidate_module.build_candidate(
        base, snapshot, folders, replace_projection=False
    )
    prior = (
        copy.deepcopy(base)
        if pose_prior_source == "base"
        else snapshot_prior
    )
    intrinsics = capture_intrinsics(capture)
    missing = [folder for folder in folders if folder not in intrinsics]
    if missing:
        raise ValueError(
            "capture JSON is missing intrinsics for: " + ", ".join(missing[:12])
        )
    base_by_folder = {
        str(camera["folder"]): camera for camera in base["cameras"]
    }
    focal_ratio_rows = []
    for camera in prior["cameras"]:
        folder = str(camera["folder"])
        base_camera = base_by_folder[folder]
        if pose_prior_source == "snapshot":
            rotation = np.asarray(camera["R"], dtype=np.float64)
            base_rotation = np.asarray(base_camera["R"], dtype=np.float64)
            base_translation = np.asarray(base_camera["t"], dtype=np.float64)
            center = camera_center(base_rotation, base_translation)
            camera["t"] = (-rotation @ center).tolist()
        for key in ("camera_model_name", "width", "height", "params"):
            camera[key] = base_camera[key]
        matrix = intrinsics[folder]
        params = np.asarray(base_camera["params"], dtype=np.float64)
        focal_ratio_rows.append({
            "folder": folder,
            "base_over_capture_fx": float(params[0] / matrix[0, 0]),
            "base_over_capture_fy": float(params[1] / matrix[1, 1]),
        })
    diagnostics["projection_comparison"] = focal_ratio_rows
    prior["dynamic_calibration_prior"] = {
        "pose_source": (
            "base_reference_image_calibration"
            if pose_prior_source == "base"
            else "4dv_bound_snapshot_mechanical_extrinsics"
        ),
        "snapshot_pose_available_as_optional_prior": True,
        "center_source": "base_reference_image_calibration",
        "projection_source": "base_reference_full_distortion_model",
        "capture_intrinsics_used_for_diagnostics_only": True,
        "optical_centres_fixed": True,
        "similarity": diagnostics,
    }
    return prior, diagnostics


def choose_pairs(
    centers: np.ndarray,
    neighbors_per_camera: int,
    wide_baseline_per_camera: int,
    explicit_pairs: Sequence[Tuple[str, str]],
    folders: Sequence[str],
) -> List[Tuple[int, int]]:
    count = len(centers)
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    result = set()
    near = min(max(int(neighbors_per_camera), 1), count - 1)
    wide = min(max(int(wide_baseline_per_camera), 0), count - 1)
    for index in range(count):
        order = np.argsort(distances[index])
        for other in order[:near]:
            result.add(tuple(sorted((index, int(other)))))
        if wide:
            finite_order = order[np.isfinite(distances[index, order])]
            for other in finite_order[-wide:]:
                result.add(tuple(sorted((index, int(other)))))
    by_folder = {folder: index for index, folder in enumerate(folders)}
    for folder_a, folder_b in explicit_pairs:
        missing = [x for x in (folder_a, folder_b) if x not in by_folder]
        if missing:
            raise ValueError("unknown explicit camera(s): " + ", ".join(missing))
        result.add(tuple(sorted((by_folder[folder_a], by_folder[folder_b]))))
    return sorted(result)


def connected_components(count: int, pairs: Iterable[Tuple[int, int]]) -> List[List[int]]:
    adjacency = [set() for _ in range(count)]
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(count))
    components: List[List[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for other in adjacency[node]:
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def fit_fundamental_inliers(
    points_a: np.ndarray, points_b: np.ndarray, threshold_px: float
) -> Tuple[np.ndarray | None, np.ndarray]:
    import cv2

    if len(points_a) < 8:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix = mask = None
    try:
        matrix, mask = cv2.findFundamentalMat(
            np.asarray(points_a, dtype=np.float32),
            np.asarray(points_b, dtype=np.float32),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=float(threshold_px),
            confidence=0.999,
            maxIters=10000,
        )
    except cv2.error:
        matrix, mask = cv2.findFundamentalMat(
            np.asarray(points_a, dtype=np.float32),
            np.asarray(points_b, dtype=np.float32),
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=float(threshold_px),
            confidence=0.999,
        )
    if matrix is None or mask is None or np.asarray(matrix).size < 9:
        return None, np.zeros(len(points_a), dtype=bool)
    matrix = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)[0]
    keep = np.asarray(mask).reshape(-1).astype(bool)
    if len(keep) != len(points_a):
        return None, np.zeros(len(points_a), dtype=bool)
    return matrix, keep


def recover_relative_pose(
    fundamental: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    inliers: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
    """Recover camera-A to camera-B motion from a fitted fundamental matrix."""
    import cv2

    if fundamental is None or int(np.sum(inliers)) < 8:
        return None, None, 0
    essential = matrix_b.T @ np.asarray(fundamental) @ matrix_a
    # Project the noisy calibrated matrix onto the essential manifold.
    u, singular, vt = np.linalg.svd(essential)
    if np.linalg.det(u @ vt) < 0.0:
        vt[-1] *= -1.0
    scale = 0.5 * float(singular[0] + singular[1])
    essential = u @ np.diag([scale, scale, 0.0]) @ vt
    normalized_a = normalized_rays(points_a, matrix_a)
    normalized_b = normalized_rays(points_b, matrix_b)
    normalized_a = normalized_a[:, :2] / normalized_a[:, 2:3]
    normalized_b = normalized_b[:, :2] / normalized_b[:, 2:3]
    pose_mask = np.asarray(inliers, dtype=np.uint8).reshape(-1, 1)
    try:
        count, rotation, translation, _ = cv2.recoverPose(
            essential,
            np.asarray(normalized_a, dtype=np.float64),
            np.asarray(normalized_b, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            mask=pose_mask,
        )
    except cv2.error:
        return None, None, 0
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or abs(np.linalg.det(rotation) - 1.0) > 1e-3
    ):
        return None, None, 0
    translation /= max(float(np.linalg.norm(translation)), 1e-12)
    return rotation, translation, int(count)


def normalized_rays(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rays = np.empty((len(points), 3), dtype=np.float64)
    rays[:, 0] = (points[:, 0] - matrix[0, 2]) / matrix[0, 0]
    rays[:, 1] = (points[:, 1] - matrix[1, 2]) / matrix[1, 1]
    rays[:, 2] = 1.0
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    return rays


def camera_rays(camera: Any, points: np.ndarray) -> np.ndarray:
    """Unproject raw distorted pixels to unit camera-coordinate rays."""
    normalized = np.asarray(
        camera.unproject_image_points(points), dtype=np.float64
    )
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    return rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)


@dataclass
class MatchBlock:
    frame: int
    camera_a: int
    camera_b: int
    rays_a: np.ndarray
    rays_b: np.ndarray
    certainty: np.ndarray
    train_mask: np.ndarray
    raw_matches: int
    fitted_ratio: float
    relative_rotation: np.ndarray | None
    relative_translation: np.ndarray | None
    pose_inliers: int


def cache_key(
    path_a: Path, path_b: Path, args: argparse.Namespace
) -> str:
    payload = {
        "a": str(path_a.resolve()), "b": str(path_b.resolve()),
        "a_stat": [path_a.stat().st_size, path_a.stat().st_mtime_ns],
        "b_stat": [path_b.stat().st_size, path_b.stat().st_mtime_ns],
        "backend": args.roma_backend, "setting": args.roma_v2_setting,
        "matches": args.roma_num_matches,
        "certainty": args.roma_min_certainty,
        "spatial_grid": args.roma_v2_spatial_grid,
        "spatial_fraction": args.roma_v2_spatial_fraction,
        "ransac": args.pair_ransac_px,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def gather_matches(
    args: argparse.Namespace,
    cameras,
    matrices: Sequence[np.ndarray],
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[MatchBlock], List[dict]]:
    prep = _load_prep_module()
    model = prep.create_roma_backend(
        backend=args.roma_backend,
        device=args.device,
        model=args.roma_model,
        v2_setting=args.roma_v2_setting,
        symmetric=True,
        upsample_preds=True,
        v2_mask_oversample=args.roma_v2_mask_oversample,
        v2_spatial_grid=args.roma_v2_spatial_grid,
        v2_spatial_fraction=args.roma_v2_spatial_fraction,
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    @functools.lru_cache(maxsize=max(2, int(args.image_cache_size)))
    def image(path_text: str) -> np.ndarray:
        return load_rgb(Path(path_text))

    blocks: List[MatchBlock] = []
    rows: List[dict] = []
    rng = np.random.default_rng(int(args.seed))
    work = [(frame, left, right) for frame in args.frames for left, right in pairs]
    for work_index, (frame, left, right) in enumerate(work, start=1):
        camera_a, camera_b = cameras[left], cameras[right]
        path_a = Path(args.images_root) / camera_a.folder / args.image_pattern.format(frame=frame)
        path_b = Path(args.images_root) / camera_b.folder / args.image_pattern.format(frame=frame)
        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(f"missing calibration image: {path_a} or {path_b}")
        key = cache_key(path_a, path_b, args)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            points_a = np.asarray(cached["points_a"], dtype=np.float32)
            points_b = np.asarray(cached["points_b"], dtype=np.float32)
            certainty = np.asarray(cached["certainty"], dtype=np.float32)
        else:
            points_a, points_b, certainty = prep.run_roma_pair(
                model, path_a, path_b, image(str(path_a)), image(str(path_b)),
                args.roma_num_matches, args.roma_min_certainty,
                args.pair_ransac_px, 8, args.device,
                use_fundamental_ransac=False,
            )
            np.savez_compressed(
                cache_path, points_a=points_a, points_b=points_b,
                certainty=certainty,
            )
        raw_count = int(len(points_a))
        # RoMa coordinates refer to the raw, distorted frames.  Fit the
        # fundamental/Essential geometry on each camera's ideal pixel plane;
        # one F matrix cannot model radial distortion in raw image space.
        geometry_a = camera_a.undistort_image_points_pixel(points_a)
        geometry_b = camera_b.undistort_image_points_pixel(points_b)
        fundamental, fitted = fit_fundamental_inliers(
            geometry_a, geometry_b, args.pair_ransac_px
        )
        fitted_count = int(np.sum(fitted))
        fitted_ratio = float(fitted_count / max(raw_count, 1))
        accepted = (
            fitted_count >= int(args.min_pair_inliers)
            and fitted_ratio >= float(args.min_pair_fit_ratio)
        )
        relative_rotation = relative_translation = None
        pose_inliers = 0
        if accepted:
            relative_rotation, relative_translation, pose_inliers = recover_relative_pose(
                fundamental,
                geometry_a,
                geometry_b,
                matrices[left],
                matrices[right],
                fitted,
            )
        row = {
            "frame": int(frame), "camera_a": int(left), "camera_b": int(right),
            "folder_a": camera_a.folder, "folder_b": camera_b.folder,
            "raw_matches": raw_count, "fitted_inliers": fitted_count,
            "fitted_ratio": fitted_ratio, "accepted": bool(accepted),
            "pose_recovered": relative_rotation is not None,
            "pose_inliers": int(pose_inliers),
            "distortion_aware_geometry": True,
            "cache": str(cache_path),
        }
        rows.append(row)
        print(
            f"[{work_index}/{len(work)}] f{frame} {camera_a.folder}<->{camera_b.folder}: "
            f"matches={raw_count} fit={fitted_count} ({100*fitted_ratio:.1f}%) "
            f"{'accepted' if accepted else 'rejected'}"
        )
        if not accepted:
            continue
        indices = np.flatnonzero(fitted)
        if len(indices) > int(args.max_pair_inliers):
            # Certainty-biased but not certainty-exclusive: retain the strongest
            # half and sample the rest so spatially uncommon surfaces survive.
            order = indices[np.argsort(certainty[indices])[::-1]]
            strong_count = int(args.max_pair_inliers) // 2
            strong = order[:strong_count]
            remainder = order[strong_count:]
            sampled = rng.choice(
                remainder,
                size=int(args.max_pair_inliers) - len(strong),
                replace=False,
            )
            indices = np.sort(np.concatenate([strong, sampled]))
        split_rng = np.random.default_rng(
            int(args.seed) + frame * 1000003 + left * 1009 + right
        )
        train_mask = split_rng.random(len(indices)) >= float(args.validation_fraction)
        if not np.any(train_mask) or np.all(train_mask):
            train_mask[: max(1, len(train_mask) - 1)] = True
            train_mask[-1:] = False
        blocks.append(MatchBlock(
            frame=int(frame), camera_a=int(left), camera_b=int(right),
            rays_a=camera_rays(camera_a, points_a[indices]),
            rays_b=camera_rays(camera_b, points_b[indices]),
            certainty=np.asarray(certainty[indices], dtype=np.float64),
            train_mask=train_mask,
            raw_matches=raw_count, fitted_ratio=fitted_ratio,
            relative_rotation=relative_rotation,
            relative_translation=relative_translation,
            pose_inliers=int(pose_inliers),
        ))
    return blocks, rows


def _skew_torch(vectors, torch):
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(-1)
    return torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(vectors.shape[:-1] + (3, 3))


def so3_exp_torch(vectors, torch):
    theta2 = torch.sum(vectors * vectors, dim=-1, keepdim=True)
    small = theta2 < 1e-8
    # torch.where evaluates both branches.  Keep sqrt away from exactly zero as
    # well as avoiding 0/0: sqrt'(0) is infinite and can poison the otherwise
    # unused regular branch during the first backward pass.
    safe_theta2 = torch.where(small, torch.ones_like(theta2), theta2)
    safe_theta = torch.sqrt(safe_theta2)
    a = torch.where(
        small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(safe_theta) / safe_theta,
    )
    b = torch.where(
        small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(safe_theta)) / safe_theta2,
    )
    skew = _skew_torch(vectors, torch)
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    identity = identity.expand(vectors.shape[:-1] + (3, 3))
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def bounded_rotation_delta(raw_delta, max_angle: float, torch):
    """Map unconstrained vectors to an axis-angle ball with radius max_angle."""
    norm2 = torch.sum(raw_delta * raw_delta, dim=-1, keepdim=True)
    small = norm2 < 1e-12
    safe_norm = torch.sqrt(torch.where(small, torch.ones_like(norm2), norm2))
    scale = torch.where(
        small,
        float(max_angle) * (1.0 - norm2 / 3.0),
        float(max_angle) * torch.tanh(safe_norm) / safe_norm,
    )
    return raw_delta * scale


def bounded_vector_delta(raw_delta, max_norm: float, torch):
    norms = torch.sqrt(torch.sum(raw_delta * raw_delta, dim=-1, keepdim=True) + 1e-16)
    scale = float(max_norm) * torch.tanh(norms) / norms
    return raw_delta * scale


def pack_observations(
    blocks: Sequence[MatchBlock],
    centers: np.ndarray,
    matrices: Sequence[np.ndarray],
    train: bool,
) -> Dict[str, np.ndarray]:
    packed: Dict[str, List[np.ndarray]] = {
        key: [] for key in (
            "camera_a", "camera_b", "rays_a", "rays_b", "baseline",
            "focal", "weight", "pair_id",
        )
    }
    for pair_id, block in enumerate(blocks):
        select = block.train_mask if train else ~block.train_mask
        count = int(np.sum(select))
        if count == 0:
            continue
        left, right = block.camera_a, block.camera_b
        baseline = centers[right] - centers[left]
        baseline /= max(float(np.linalg.norm(baseline)), 1e-12)
        focal = math.sqrt(
            math.sqrt(matrices[left][0, 0] * matrices[left][1, 1])
            * math.sqrt(matrices[right][0, 0] * matrices[right][1, 1])
        )
        certainty = np.clip(block.certainty[select], 1e-4, None)
        certainty /= max(float(np.median(certainty)), 1e-6)
        certainty = np.clip(certainty, 0.25, 4.0)
        packed["camera_a"].append(np.full(count, left, dtype=np.int64))
        packed["camera_b"].append(np.full(count, right, dtype=np.int64))
        packed["rays_a"].append(block.rays_a[select])
        packed["rays_b"].append(block.rays_b[select])
        packed["baseline"].append(np.repeat(baseline[None, :], count, axis=0))
        packed["focal"].append(np.full(count, focal, dtype=np.float64))
        packed["weight"].append(certainty)
        packed["pair_id"].append(np.full(count, pair_id, dtype=np.int64))
    if not packed["camera_a"]:
        raise RuntimeError("no observations survived packing")
    return {
        key: np.concatenate(values, axis=0) for key, values in packed.items()
    }


def rotation_delta_degrees(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left) @ np.asarray(right).T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def weighted_wahba_rotation(
    world_vectors: np.ndarray,
    camera_vectors: np.ndarray,
    base_weights: np.ndarray,
    iterations: int = 8,
) -> Tuple[np.ndarray | None, dict]:
    """Robustly solve camera_vector ~= R @ world_vector."""
    world_vectors = np.asarray(world_vectors, dtype=np.float64)
    camera_vectors = np.asarray(camera_vectors, dtype=np.float64)
    base_weights = np.asarray(base_weights, dtype=np.float64)
    if len(world_vectors) < 3:
        return None, {"reason": "fewer_than_three_directions"}
    weights = base_weights / max(float(np.median(base_weights)), 1e-12)
    rotation = None
    singular = np.zeros(3)
    angles = np.full(len(world_vectors), np.inf)
    for _ in range(int(iterations)):
        cross_covariance = (
            camera_vectors.T @ (weights[:, None] * world_vectors)
        )
        u, singular, vt = np.linalg.svd(cross_covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ correction @ vt
        predicted = (rotation @ world_vectors.T).T
        dots = np.clip(np.sum(predicted * camera_vectors, axis=1), -1.0, 1.0)
        angles = np.arccos(dots)
        scale = max(float(np.median(angles)) * 2.0, math.radians(2.0))
        weights = base_weights / (1.0 + (angles / scale) ** 2)
    if rotation is None or singular[1] <= max(singular[0], 1e-12) * 1e-4:
        return None, {
            "reason": "baseline_directions_are_rank_deficient",
            "singular_values": singular.tolist(),
        }
    return rotation, {
        "directions": int(len(world_vectors)),
        "singular_values": singular.tolist(),
        "angular_residual_p50_deg": float(np.degrees(np.quantile(angles, 0.50))),
        "angular_residual_p90_deg": float(np.degrees(np.quantile(angles, 0.90))),
    }


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation to a 3x3 matrix."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def synchronize_relative_rotations(
    camera_indices: Sequence[int],
    blocks: Sequence[MatchBlock],
    block_weights: np.ndarray,
    iterations: int = 20,
) -> Tuple[Dict[int, np.ndarray], dict]:
    """Robustly synchronize R_ba ~= S_b S_a^T inside one graph component."""
    indices = set(int(value) for value in camera_indices)
    raw = [
        (block, float(weight))
        for block, weight in zip(blocks, block_weights)
        if block.camera_a in indices and block.camera_b in indices
    ]
    if not raw:
        return {}, {"reason": "component_has_no_relative_rotations"}

    # Collapse the repeated frame measurements for each edge first.  A medoid
    # seed prevents one bad frame from pulling the chordal rotation average to
    # a solution that none of the frames actually supports.
    grouped: Dict[Tuple[int, int], List[Tuple[np.ndarray, float]]] = {}
    for block, weight in raw:
        key = (block.camera_a, block.camera_b)
        grouped.setdefault(key, []).append((
            np.asarray(block.relative_rotation, dtype=np.float64), weight
        ))
    edges = []
    for (left, right), rows in grouped.items():
        rotations = [row[0] for row in rows]
        weights = np.asarray([row[1] for row in rows], dtype=np.float64)
        costs = np.asarray([
            np.sum(weights * np.asarray([
                rotation_delta_degrees(candidate, other)
                for other in rotations
            ]))
            for candidate in rotations
        ])
        medoid = rotations[int(np.argmin(costs))]
        residuals = np.asarray([
            rotation_delta_degrees(rotation, medoid) for rotation in rotations
        ])
        scale = max(float(np.median(residuals)) * 2.0, 2.0)
        robust = weights / (1.0 + (residuals / scale) ** 2)
        aggregate = project_to_rotation(sum(
            weight * rotation for weight, rotation in zip(robust, rotations)
        ))
        spread = float(np.quantile(residuals, 0.90))
        confidence = float(np.sum(weights)) / (1.0 + (spread / 5.0) ** 2)
        edges.append({
            "left": left, "right": right, "rotation": aggregate,
            "weight": confidence, "measurements": len(rows),
            "frame_spread_p90_deg": spread,
        })

    edge_map = {
        (edge["left"], edge["right"]): edge["rotation"] for edge in edges
    }

    def edge_rotation(source: int, target: int) -> np.ndarray | None:
        key = tuple(sorted((source, target)))
        rotation = edge_map.get(key)
        if rotation is None:
            return None
        return rotation if source < target else rotation.T

    # Score each edge by triangle consistency before building the tree.  This
    # stops a high-confidence but wrong pair from defining an entire camera's
    # initial orientation.
    for edge in edges:
        left, right = edge["left"], edge["right"]
        cycle_residuals = []
        for other in indices - {left, right}:
            left_to_other = edge_rotation(left, other)
            right_to_other = edge_rotation(right, other)
            if left_to_other is None or right_to_other is None:
                continue
            predicted = right_to_other.T @ left_to_other
            cycle_residuals.append(rotation_delta_degrees(
                edge["rotation"], predicted
            ))
        cycle_error = (
            float(np.median(cycle_residuals)) if cycle_residuals else 30.0
        )
        edge["cycle_error_deg"] = cycle_error
        edge["synchronization_weight"] = edge["weight"] / (
            1.0 + (cycle_error / 5.0) ** 2
        )

    ordered = sorted(indices)
    local_index = {camera: position for position, camera in enumerate(ordered)}
    root = ordered[0]

    def spectral_solution(edge_weights: np.ndarray) -> Dict[int, np.ndarray]:
        count = len(ordered)
        connection = np.zeros((3 * count, 3 * count), dtype=np.float64)
        degrees = np.zeros(count, dtype=np.float64)
        for edge, weight in zip(edges, edge_weights):
            left = local_index[edge["left"]]
            right = local_index[edge["right"]]
            relative = edge["rotation"]
            connection[
                3 * right:3 * right + 3, 3 * left:3 * left + 3
            ] += weight * relative
            connection[
                3 * left:3 * left + 3, 3 * right:3 * right + 3
            ] += weight * relative.T
            degrees[left] += weight
            degrees[right] += weight
        if np.any(degrees <= 1e-12):
            return {}
        repeated_degree = np.repeat(degrees, 3)
        normalized = connection / np.sqrt(
            repeated_degree[:, None] * repeated_degree[None, :]
        )
        _, eigenvectors = np.linalg.eigh(normalized)
        embedding = eigenvectors[:, -3:]
        raw_blocks = {
            camera: embedding[3 * position:3 * position + 3] /
            math.sqrt(degrees[position])
            for position, camera in enumerate(ordered)
        }
        root_inverse = np.linalg.pinv(raw_blocks[root])
        return {
            camera: project_to_rotation(block @ root_inverse)
            for camera, block in raw_blocks.items()
        }

    robust_weights = np.asarray([
        edge["synchronization_weight"] for edge in edges
    ], dtype=np.float64)
    residuals = np.zeros(len(edges), dtype=np.float64)
    for _ in range(int(iterations)):
        synchronized = spectral_solution(robust_weights)
        if len(synchronized) != len(indices):
            return synchronized, {
                "reason": "spectral_rotation_graph_is_disconnected",
                "initialized": len(synchronized),
                "cameras": len(indices),
            }
        for position, edge in enumerate(edges):
            predicted = (
                synchronized[edge["right"]]
                @ synchronized[edge["left"]].T
            )
            residuals[position] = rotation_delta_degrees(
                edge["rotation"], predicted
            )
        scale = max(float(np.median(residuals)) * 2.0, 2.0)
        robust_weights = np.asarray([
            edge["synchronization_weight"] / (1.0 + (residual / scale) ** 2)
            for edge, residual in zip(edges, residuals)
        ])

    synchronized = spectral_solution(robust_weights)

    for position, edge in enumerate(edges):
        predicted = synchronized[edge["right"]] @ synchronized[edge["left"]].T
        residuals[position] = rotation_delta_degrees(
            edge["rotation"], predicted
        )
    return synchronized, {
        "cameras": len(indices),
        "blocks": len(raw),
        "unique_edges": len(edges),
        "solver": "cycle_weighted_irls_spectral_synchronization",
        "rotation_residual_p50_deg": float(np.quantile(residuals, 0.50)),
        "rotation_residual_p90_deg": float(np.quantile(residuals, 0.90)),
        "cycle_error_p50_deg": float(np.quantile(
            [edge["cycle_error_deg"] for edge in edges], 0.50
        )),
        "cycle_error_p90_deg": float(np.quantile(
            [edge["cycle_error_deg"] for edge in edges], 0.90
        )),
    }


def initialize_rotations_from_pair_poses(
    blocks: Sequence[MatchBlock],
    centers: np.ndarray,
    prior_rotations: np.ndarray,
    min_pose_inliers: int,
) -> Tuple[np.ndarray, dict]:
    """Synchronize relative rotations, then recover each component's gauge."""
    usable = [
        block for block in blocks
        if block.relative_rotation is not None
        and block.relative_translation is not None
        and block.pose_inliers >= int(min_pose_inliers)
    ]
    weights = np.asarray([
        math.sqrt(float(block.pose_inliers)) * max(block.fitted_ratio, 1e-3)
        for block in usable
    ], dtype=np.float64)
    components = connected_components(
        len(prior_rotations),
        ((block.camera_a, block.camera_b) for block in usable),
    )
    initialized = np.asarray(prior_rotations, dtype=np.float64).copy()
    component_rows = []
    solved_cameras = set()
    for component in components:
        component_set = set(component)
        component_blocks = [
            block for block in usable
            if block.camera_a in component_set and block.camera_b in component_set
        ]
        component_weights = np.asarray([
            math.sqrt(float(block.pose_inliers)) * max(block.fitted_ratio, 1e-3)
            for block in component_blocks
        ], dtype=np.float64)
        synchronized, sync_diagnostics = synchronize_relative_rotations(
            component, component_blocks, component_weights
        )
        row = {"camera_indices": component, "synchronization": sync_diagnostics}
        if len(synchronized) != len(component) or len(component) < 2:
            row["initialized"] = False
            row["reason"] = "relative_rotation_synchronization_failed"
            component_rows.append(row)
            continue

        world_directions = []
        synchronized_directions = []
        gauge_weights = []
        relative_residuals = []
        for block, base_weight in zip(component_blocks, component_weights):
            left, right = block.camera_a, block.camera_b
            baseline = centers[right] - centers[left]
            baseline_norm = float(np.linalg.norm(baseline))
            if baseline_norm <= 1e-12:
                continue
            relative = np.asarray(block.relative_rotation, dtype=np.float64)
            predicted_relative = synchronized[right] @ synchronized[left].T
            relative_residual = rotation_delta_degrees(relative, predicted_relative)
            relative_residuals.append(relative_residual)
            rotation_scale = max(
                float(sync_diagnostics.get("rotation_residual_p50_deg", 2.0)) * 2.0,
                2.0,
            )
            weight = base_weight / (1.0 + (relative_residual / rotation_scale) ** 2)
            world_a_to_b = baseline / baseline_norm
            world_b_to_a = -world_a_to_b
            camera_b_to_a = np.asarray(
                block.relative_translation, dtype=np.float64
            )
            camera_a_to_b = -relative.T @ camera_b_to_a
            # R_i = S_i G, so S_i^T t_i = G (C_other-C_i).
            world_directions.extend((world_b_to_a, world_a_to_b))
            synchronized_directions.extend((
                synchronized[right].T @ camera_b_to_a,
                synchronized[left].T @ camera_a_to_b,
            ))
            gauge_weights.extend((weight, weight))
        gauge, gauge_diagnostics = weighted_wahba_rotation(
            np.asarray(world_directions),
            np.asarray(synchronized_directions),
            np.asarray(gauge_weights),
            iterations=12,
        )
        row["gauge"] = gauge_diagnostics
        if gauge is None:
            row["initialized"] = False
            row["reason"] = "world_gauge_recovery_failed"
            component_rows.append(row)
            continue
        for camera_index in component:
            initialized[camera_index] = synchronized[camera_index] @ gauge
            solved_cameras.add(camera_index)
        row["initialized"] = True
        component_rows.append(row)

    camera_rows = [{
        "camera_index": int(camera_index),
        "initialized": camera_index in solved_cameras,
        "delta_from_4dv_prior_deg": rotation_delta_degrees(
            initialized[camera_index], prior_rotations[camera_index]
        ),
    } for camera_index in range(len(prior_rotations))]
    return initialized, {
        "method": "relative_rotation_synchronization_plus_global_baseline_gauge",
        "relative_pose_blocks": int(len(usable)),
        "initialized_cameras": int(len(solved_cameras)),
        "fallback_cameras": int(len(prior_rotations) - len(solved_cameras)),
        "components": component_rows,
        "per_camera": camera_rows,
    }


def torch_residuals(rotations, batch: Dict[str, Any], torch):
    left = rotations[batch["camera_a"]]
    right = rotations[batch["camera_b"]]
    ray_a = torch.einsum("nij,nj->ni", left.transpose(1, 2), batch["rays_a"])
    ray_b = torch.einsum("nij,nj->ni", right.transpose(1, 2), batch["rays_b"])
    normal_a = torch.linalg.cross(batch["baseline"], ray_a, dim=-1)
    normal_b = torch.linalg.cross(batch["baseline"], ray_b, dim=-1)
    numerator = torch.abs(torch.sum(normal_a * ray_b, dim=-1))
    # sqrt(sum(x^2) + eps) has a finite derivative at a zero epipolar normal;
    # clamp(norm(x)) does not, because norm's derivative is already undefined.
    normal_a_norm = torch.sqrt(torch.sum(normal_a * normal_a, dim=-1) + 1e-16)
    normal_b_norm = torch.sqrt(torch.sum(normal_b * normal_b, dim=-1) + 1e-16)
    sine_a = numerator / normal_a_norm
    sine_b = numerator / normal_b_norm
    return 0.5 * (sine_a + sine_b) * batch["focal"]


def torch_residuals_with_centers(rotations, centers, batch, torch):
    dynamic = dict(batch)
    dynamic["baseline"] = (
        centers[batch["camera_b"]] - centers[batch["camera_a"]]
    )
    return torch_residuals(rotations, dynamic, torch)


def scale_camera_rays_torch(rays, camera_ids, focal_scales, torch):
    scale = focal_scales[camera_ids]
    xy = rays[:, :2] / torch.clamp(rays[:, 2:3], min=1e-8)
    adjusted = torch.cat([xy / scale, torch.ones_like(scale[:, :1])], dim=-1)
    return adjusted / torch.sqrt(
        torch.sum(adjusted * adjusted, dim=-1, keepdim=True) + 1e-16
    )


def torch_residuals_with_focal_scales(
    rotations, centers, focal_scales, batch, torch
):
    dynamic = dict(batch)
    dynamic["baseline"] = (
        centers[batch["camera_b"]] - centers[batch["camera_a"]]
    )
    dynamic["rays_a"] = scale_camera_rays_torch(
        batch["rays_a"], batch["camera_a"], focal_scales, torch
    )
    dynamic["rays_b"] = scale_camera_rays_torch(
        batch["rays_b"], batch["camera_b"], focal_scales, torch
    )
    geometric_scale = torch.sqrt(torch.prod(focal_scales, dim=-1))
    dynamic["focal"] = batch["focal"] * torch.sqrt(
        geometric_scale[batch["camera_a"]]
        * geometric_scale[batch["camera_b"]]
    )
    return torch_residuals(rotations, dynamic, torch)


def to_torch(packed: Dict[str, np.ndarray], device: str, torch):
    result = {}
    for key, value in packed.items():
        if key in ("camera_a", "camera_b", "pair_id"):
            result[key] = torch.as_tensor(value, dtype=torch.long, device=device)
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float64, device=device)
    return result


def optimize_rotations(
    prior_rotations: np.ndarray,
    train: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    import torch

    device = str(args.optimizer_device or args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"optimizer requested {device}, but CUDA is unavailable")
    torch.manual_seed(int(args.seed))
    prior = torch.as_tensor(prior_rotations, dtype=torch.float64, device=device)
    packed = to_torch(train, device, torch)
    raw_delta = torch.nn.Parameter(
        torch.zeros((len(prior_rotations), 3), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([raw_delta], lr=float(args.learning_rate))
    max_angle = math.radians(float(args.max_rotation_delta_deg))
    sigma = math.radians(float(args.rotation_prior_sigma_deg))
    observation_count = len(train["camera_a"])
    batch_size = min(int(args.batch_size), observation_count)
    history = []
    for iteration in range(int(args.iterations)):
        optimizer.zero_grad(set_to_none=True)
        if batch_size < observation_count:
            indices = torch.randint(
                observation_count, (batch_size,), device=device
            )
            batch = {key: value[indices] for key, value in packed.items()}
        else:
            batch = packed
        delta = bounded_rotation_delta(raw_delta, max_angle, torch)
        rotations = so3_exp_torch(delta, torch) @ prior
        residual = torch_residuals(rotations, batch, torch)
        scaled = residual / max(float(args.robust_scale_px), 1e-6)
        robust = torch.sqrt(1.0 + scaled * scaled) - 1.0
        data_loss = torch.sum(batch["weight"] * robust) / torch.sum(batch["weight"])
        prior_loss = torch.mean(torch.sum((delta / max(sigma, 1e-8)) ** 2, dim=-1))
        loss = data_loss + float(args.rotation_prior_weight) * prior_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite calibration loss at iteration {iteration + 1}; "
                "the RoMa cache is intact and can be reused"
            )
        loss.backward()
        if raw_delta.grad is None or not bool(torch.all(torch.isfinite(raw_delta.grad)).item()):
            raise FloatingPointError(
                f"non-finite calibration gradient at iteration {iteration + 1}; "
                "the RoMa cache is intact and can be reused"
            )
        torch.nn.utils.clip_grad_norm_([raw_delta], max_norm=10.0)
        optimizer.step()
        if not bool(torch.all(torch.isfinite(raw_delta)).item()):
            raise FloatingPointError(
                f"non-finite camera rotation at iteration {iteration + 1}"
            )
        if iteration % int(args.log_every) == 0 or iteration + 1 == int(args.iterations):
            with torch.no_grad():
                median = float(torch.median(residual).item())
                p90 = float(torch.quantile(residual, 0.90).item())
                max_delta_now = float(
                    torch.max(torch.linalg.norm(delta, dim=-1)).item() * 180.0 / math.pi
                )
            item = {
                "iteration": iteration + 1, "loss": float(loss.item()),
                "batch_median_px": median, "batch_p90_px": p90,
                "max_rotation_delta_deg": max_delta_now,
            }
            history.append(item)
            print(
                f"opt {iteration + 1}/{args.iterations}: loss={item['loss']:.6f} "
                f"median={median:.3f}px p90={p90:.3f}px "
                f"max_delta={max_delta_now:.3f}deg"
            )
    with torch.no_grad():
        delta = bounded_rotation_delta(raw_delta, max_angle, torch)
        rotations = (so3_exp_torch(delta, torch) @ prior).cpu().numpy()
        delta_np = delta.cpu().numpy()
    return rotations, {
        "history": history,
        "rotation_delta_deg": (
            np.linalg.norm(delta_np, axis=1) * 180.0 / math.pi
        ).tolist(),
    }


def optimize_failed_camera_poses(
    initial_rotations: np.ndarray,
    initial_centers: np.ndarray,
    train: Dict[str, np.ndarray],
    camera_indices: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Refine only failed cameras while the validated camera network is fixed."""
    import torch

    selected = np.asarray(sorted(set(int(value) for value in camera_indices)))
    if len(selected) == 0:
        return initial_rotations, initial_centers, {"enabled": False}
    incident = np.isin(train["camera_a"], selected) | np.isin(
        train["camera_b"], selected
    )
    target_train = {key: value[incident] for key, value in train.items()}
    if len(target_train["camera_a"]) == 0:
        raise ValueError("failed-camera refinement has no incident observations")

    device = str(args.optimizer_device or args.device)
    base_rotations = torch.as_tensor(
        initial_rotations, dtype=torch.float64, device=device
    )
    base_centers = torch.as_tensor(
        initial_centers, dtype=torch.float64, device=device
    )
    target_ids = torch.as_tensor(selected, dtype=torch.long, device=device)
    packed = to_torch(target_train, device, torch)
    raw_rotation = torch.nn.Parameter(
        torch.zeros((len(selected), 3), dtype=torch.float64, device=device)
    )
    raw_center = torch.nn.Parameter(
        torch.zeros((len(selected), 3), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam(
        [raw_rotation, raw_center], lr=float(args.failed_pose_learning_rate)
    )
    observation_count = len(target_train["camera_a"])
    batch_size = min(int(args.batch_size), observation_count)
    max_rotation = math.radians(float(args.failed_pose_max_rotation_delta_deg))
    rotation_sigma = math.radians(float(args.failed_pose_rotation_sigma_deg))
    history = []
    for iteration in range(int(args.failed_pose_iterations)):
        optimizer.zero_grad(set_to_none=True)
        if batch_size < observation_count:
            indices = torch.randint(observation_count, (batch_size,), device=device)
            batch = {key: value[indices] for key, value in packed.items()}
        else:
            batch = packed
        rotation_delta = bounded_rotation_delta(raw_rotation, max_rotation, torch)
        center_delta = bounded_vector_delta(
            raw_center, float(args.failed_pose_max_center_delta), torch
        )
        full_rotation_delta = torch.zeros_like(base_centers).index_copy(
            0, target_ids, rotation_delta
        )
        full_center_delta = torch.zeros_like(base_centers).index_copy(
            0, target_ids, center_delta
        )
        rotations = so3_exp_torch(full_rotation_delta, torch) @ base_rotations
        centers = base_centers + full_center_delta
        residual = torch_residuals_with_centers(rotations, centers, batch, torch)
        scaled = residual / max(float(args.robust_scale_px), 1e-6)
        robust = torch.sqrt(1.0 + scaled * scaled) - 1.0
        data_loss = torch.sum(batch["weight"] * robust) / torch.sum(batch["weight"])
        rotation_prior = torch.mean(torch.sum(
            (rotation_delta / max(rotation_sigma, 1e-8)) ** 2, dim=-1
        ))
        center_prior = torch.mean(torch.sum(
            (center_delta / max(float(args.failed_pose_center_sigma), 1e-8)) ** 2,
            dim=-1,
        ))
        loss = data_loss + float(args.failed_pose_prior_weight) * (
            rotation_prior + center_prior
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite failed-camera pose loss at iteration {iteration + 1}"
            )
        loss.backward()
        if any(
            parameter.grad is None
            or not bool(torch.all(torch.isfinite(parameter.grad)).item())
            for parameter in (raw_rotation, raw_center)
        ):
            raise FloatingPointError(
                f"non-finite failed-camera pose gradient at iteration {iteration + 1}"
            )
        torch.nn.utils.clip_grad_norm_([raw_rotation, raw_center], max_norm=10.0)
        optimizer.step()
        if iteration % int(args.log_every) == 0 or iteration + 1 == int(
            args.failed_pose_iterations
        ):
            with torch.no_grad():
                item = {
                    "iteration": iteration + 1,
                    "loss": float(loss.item()),
                    "median_px": float(torch.median(residual).item()),
                    "p90_px": float(torch.quantile(residual, 0.90).item()),
                    "max_rotation_delta_deg": float(
                        torch.max(torch.linalg.norm(rotation_delta, dim=-1)).item()
                        * 180.0 / math.pi
                    ),
                    "max_center_delta": float(
                        torch.max(torch.linalg.norm(center_delta, dim=-1)).item()
                    ),
                }
            history.append(item)
            print(
                f"failed-pose opt {iteration + 1}/{args.failed_pose_iterations}: "
                f"median={item['median_px']:.3f}px p90={item['p90_px']:.3f}px "
                f"rot={item['max_rotation_delta_deg']:.3f}deg "
                f"center={item['max_center_delta']:.4f}"
            )
    with torch.no_grad():
        rotation_delta = bounded_rotation_delta(raw_rotation, max_rotation, torch)
        center_delta = bounded_vector_delta(
            raw_center, float(args.failed_pose_max_center_delta), torch
        )
        full_rotation_delta = torch.zeros_like(base_centers).index_copy(
            0, target_ids, rotation_delta
        )
        full_center_delta = torch.zeros_like(base_centers).index_copy(
            0, target_ids, center_delta
        )
        rotations = (so3_exp_torch(full_rotation_delta, torch) @ base_rotations).cpu().numpy()
        centers = (base_centers + full_center_delta).cpu().numpy()
        rotation_delta_np = rotation_delta.cpu().numpy()
        center_delta_np = center_delta.cpu().numpy()
    return rotations, centers, {
        "enabled": True,
        "camera_indices": selected.tolist(),
        "observations": int(observation_count),
        "history": history,
        "rotation_delta_deg": (
            np.linalg.norm(rotation_delta_np, axis=1) * 180.0 / math.pi
        ).tolist(),
        "center_delta": center_delta_np.tolist(),
        "center_delta_norm": np.linalg.norm(center_delta_np, axis=1).tolist(),
    }


def optimize_failed_camera_focals(
    rotations: np.ndarray,
    centers: np.ndarray,
    train: Dict[str, np.ndarray],
    camera_indices: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    """Refine per-axis PTZ focal scale after pose refinement still fails."""
    import torch

    selected = np.asarray(sorted(set(int(value) for value in camera_indices)))
    focal_scales_result = np.ones((len(rotations), 2), dtype=np.float64)
    if len(selected) == 0:
        return focal_scales_result, {"enabled": False}
    incident = np.isin(train["camera_a"], selected) | np.isin(
        train["camera_b"], selected
    )
    target_train = {key: value[incident] for key, value in train.items()}
    observation_count = len(target_train["camera_a"])
    if observation_count == 0:
        raise ValueError("failed-camera focal refinement has no incident observations")

    device = str(args.optimizer_device or args.device)
    fixed_rotations = torch.as_tensor(rotations, dtype=torch.float64, device=device)
    fixed_centers = torch.as_tensor(centers, dtype=torch.float64, device=device)
    target_ids = torch.as_tensor(selected, dtype=torch.long, device=device)
    packed = to_torch(target_train, device, torch)
    raw_log_scale = torch.nn.Parameter(
        torch.zeros((len(selected), 2), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam(
        [raw_log_scale], lr=float(args.failed_focal_learning_rate)
    )
    batch_size = min(int(args.batch_size), observation_count)
    max_log_scale = math.log(float(args.failed_focal_max_scale))
    history = []
    for iteration in range(int(args.failed_focal_iterations)):
        optimizer.zero_grad(set_to_none=True)
        if batch_size < observation_count:
            indices = torch.randint(observation_count, (batch_size,), device=device)
            batch = {key: value[indices] for key, value in packed.items()}
        else:
            batch = packed
        log_scale = max_log_scale * torch.tanh(raw_log_scale)
        full_log_scale = torch.zeros(
            (len(rotations), 2), dtype=torch.float64, device=device
        ).index_copy(0, target_ids, log_scale)
        focal_scales = torch.exp(full_log_scale)
        residual = torch_residuals_with_focal_scales(
            fixed_rotations, fixed_centers, focal_scales, batch, torch
        )
        scaled = residual / max(float(args.robust_scale_px), 1e-6)
        robust = torch.sqrt(1.0 + scaled * scaled) - 1.0
        data_loss = torch.sum(batch["weight"] * robust) / torch.sum(batch["weight"])
        focal_prior = torch.mean(
            (log_scale / float(args.failed_focal_log_sigma)) ** 2
        )
        loss = data_loss + float(args.failed_focal_prior_weight) * focal_prior
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite failed-camera focal loss at iteration {iteration + 1}"
            )
        loss.backward()
        if raw_log_scale.grad is None or not bool(
            torch.all(torch.isfinite(raw_log_scale.grad)).item()
        ):
            raise FloatingPointError(
                f"non-finite failed-camera focal gradient at iteration {iteration + 1}"
            )
        torch.nn.utils.clip_grad_norm_([raw_log_scale], max_norm=10.0)
        optimizer.step()
        if iteration % int(args.log_every) == 0 or iteration + 1 == int(
            args.failed_focal_iterations
        ):
            with torch.no_grad():
                item = {
                    "iteration": iteration + 1,
                    "loss": float(loss.item()),
                    "median_px": float(torch.median(residual).item()),
                    "p90_px": float(torch.quantile(residual, 0.90).item()),
                    "focal_scales_xy": (
                        torch.exp(log_scale).cpu().numpy().tolist()
                    ),
                }
            history.append(item)
            print(
                f"failed-focal opt {iteration + 1}/{args.failed_focal_iterations}: "
                f"median={item['median_px']:.3f}px p90={item['p90_px']:.3f}px "
                "scales_xy="
                + ",".join(
                    f"({value[0]:.5f},{value[1]:.5f})"
                    for value in item["focal_scales_xy"]
                )
            )
    with torch.no_grad():
        log_scale = max_log_scale * torch.tanh(raw_log_scale)
        solved_scales = torch.exp(log_scale).cpu().numpy()
    focal_scales_result[selected] = solved_scales
    return focal_scales_result, {
        "enabled": True,
        "camera_indices": selected.tolist(),
        "observations": int(observation_count),
        "focal_scales_xy": solved_scales.tolist(),
        "history": history,
    }


def residuals_numpy(
    rotations: np.ndarray, packed: Dict[str, np.ndarray]
) -> np.ndarray:
    left = rotations[packed["camera_a"]]
    right = rotations[packed["camera_b"]]
    ray_a = np.einsum("nji,nj->ni", left, packed["rays_a"])
    ray_b = np.einsum("nji,nj->ni", right, packed["rays_b"])
    normal_a = np.cross(packed["baseline"], ray_a)
    normal_b = np.cross(packed["baseline"], ray_b)
    numerator = np.abs(np.sum(normal_a * ray_b, axis=-1))
    sine_a = numerator / np.maximum(np.linalg.norm(normal_a, axis=-1), 1e-8)
    sine_b = numerator / np.maximum(np.linalg.norm(normal_b, axis=-1), 1e-8)
    return 0.5 * (sine_a + sine_b) * packed["focal"]


def packed_with_centers(
    packed: Dict[str, np.ndarray], centers: np.ndarray
) -> Dict[str, np.ndarray]:
    updated = dict(packed)
    updated["baseline"] = (
        centers[packed["camera_b"]] - centers[packed["camera_a"]]
    )
    return updated


def scale_camera_rays_numpy(
    rays: np.ndarray, camera_ids: np.ndarray, focal_scales: np.ndarray
) -> np.ndarray:
    scale = focal_scales[camera_ids]
    xy = rays[:, :2] / np.maximum(rays[:, 2:3], 1e-8)
    adjusted = np.column_stack([xy / scale, np.ones(len(rays))])
    return adjusted / np.maximum(
        np.linalg.norm(adjusted, axis=-1, keepdims=True), 1e-12
    )


def packed_with_calibration(
    packed: Dict[str, np.ndarray],
    centers: np.ndarray,
    focal_scales: np.ndarray,
) -> Dict[str, np.ndarray]:
    updated = packed_with_centers(packed, centers)
    updated["rays_a"] = scale_camera_rays_numpy(
        packed["rays_a"], packed["camera_a"], focal_scales
    )
    updated["rays_b"] = scale_camera_rays_numpy(
        packed["rays_b"], packed["camera_b"], focal_scales
    )
    geometric_scale = np.sqrt(np.prod(focal_scales, axis=-1))
    updated["focal"] = packed["focal"] * np.sqrt(
        geometric_scale[packed["camera_a"]]
        * geometric_scale[packed["camera_b"]]
    )
    return updated


def subset_packed(
    packed: Dict[str, np.ndarray], mask: np.ndarray
) -> Dict[str, np.ndarray]:
    return {key: value[mask] for key, value in packed.items()}


def select_consistent_failed_camera_neighbors(
    rotations: np.ndarray,
    centers: np.ndarray,
    train: Dict[str, np.ndarray],
    blocks: Sequence[MatchBlock],
    folders: Sequence[str],
    camera_indices: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    """Choose neighbor edges on training data; held-out data remains untouched."""
    selected_cameras = sorted(set(int(value) for value in camera_indices))
    calibrated_train = packed_with_centers(train, centers)
    residual = residuals_numpy(rotations, calibrated_train)
    selected_pair_ids: set[int] = set()
    camera_reports = []
    for camera_index in selected_cameras:
        by_neighbor: Dict[int, dict] = {}
        for pair_id, block in enumerate(blocks):
            if camera_index not in (block.camera_a, block.camera_b):
                continue
            neighbor = (
                block.camera_b if block.camera_a == camera_index else block.camera_a
            )
            mask = train["pair_id"] == pair_id
            count = int(np.sum(mask))
            if count == 0:
                continue
            row = by_neighbor.setdefault(neighbor, {
                "neighbor_index": int(neighbor),
                "pair_ids": [],
                "observations": 0,
                "inliers": 0,
                "residuals": [],
            })
            values = residual[mask]
            row["pair_ids"].append(int(pair_id))
            row["observations"] += count
            row["inliers"] += int(np.sum(values <= 2.5))
            row["residuals"].append(values)
        candidates = []
        for row in by_neighbor.values():
            values = np.concatenate(row.pop("residuals"))
            row["folder"] = folders[row["neighbor_index"]]
            row["inlier_ratio_2_5px"] = float(
                row.pop("inliers") / max(row["observations"], 1)
            )
            row["residual"] = quantiles(values)
            candidates.append(row)
        candidates.sort(
            key=lambda row: (
                -row["inlier_ratio_2_5px"],
                row["residual"]["p50"] if row["residual"]["p50"] is not None else math.inf,
            )
        )
        chosen = [
            row for row in candidates
            if row["inlier_ratio_2_5px"]
            >= float(args.failed_pose_min_neighbor_inlier_ratio)
            and row["residual"]["p90"] is not None
            and row["residual"]["p90"]
            <= float(args.failed_pose_max_neighbor_p90_px)
        ]
        # Never force a geometrically inconsistent edge merely to reach the
        # requested neighbor count.  The old fallback did exactly that for
        # camera 0026: it admitted an edge with a 28.8 px training p90, then
        # hid the rejected edges from final validation.  One trustworthy edge
        # plus the pose prior is safer than optimizing against a known-bad one;
        # the untouched full held-out set below decides whether the result is
        # actually publishable.
        minimum = min(int(args.failed_pose_min_neighbors), len(candidates))
        chosen_ids = {
            pair_id for row in chosen for pair_id in row["pair_ids"]
        }
        selected_pair_ids.update(chosen_ids)
        camera_reports.append({
            "camera_index": int(camera_index),
            "folder": folders[camera_index],
            "selected_neighbors": [row["folder"] for row in chosen],
            "selected_pair_ids": sorted(chosen_ids),
            "minimum_neighbors": int(minimum),
            "minimum_neighbors_satisfied": len(chosen) >= minimum,
            "neighbors": candidates,
        })
    incident = np.isin(train["camera_a"], selected_cameras) | np.isin(
        train["camera_b"], selected_cameras
    )
    keep = ~incident | np.isin(train["pair_id"], sorted(selected_pair_ids))
    return keep, {
        "enabled": True,
        "selection_source": "training_observations_only",
        "selected_pair_ids": sorted(selected_pair_ids),
        "kept_observations": int(np.sum(keep)),
        "removed_observations": int(np.sum(~keep)),
        "per_camera": camera_reports,
    }


def quantiles(values: np.ndarray) -> Dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def format_pixels(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.3f}px"


def validation_report(
    prior_rotations: np.ndarray,
    solved_rotations: np.ndarray,
    validation: Dict[str, np.ndarray],
    blocks: Sequence[MatchBlock],
    folders: Sequence[str],
    args: argparse.Namespace,
    before_validation: Dict[str, np.ndarray] | None = None,
) -> dict:
    before_source = validation if before_validation is None else before_validation
    before = residuals_numpy(prior_rotations, before_source)
    after = residuals_numpy(solved_rotations, validation)
    explicit_pairs = {tuple(sorted(pair)) for pair in args.camera_pairs}
    pair_rows = []
    for pair_id, block in enumerate(blocks):
        select = validation["pair_id"] == pair_id
        if not np.any(select):
            continue
        values_before, values_after = before[select], after[select]
        pair_rows.append({
            "frame": block.frame,
            "folder_a": folders[block.camera_a],
            "folder_b": folders[block.camera_b],
            "explicit": tuple(sorted((
                folders[block.camera_a], folders[block.camera_b]
            ))) in explicit_pairs,
            "observations": int(np.sum(select)),
            "before": quantiles(values_before), "after": quantiles(values_after),
            "before_inlier_ratio_2_5px": float(np.mean(values_before <= 2.5)),
            "after_inlier_ratio_2_5px": float(np.mean(values_after <= 2.5)),
        })
    camera_rows = []
    for camera_index, folder in enumerate(folders):
        select = (
            (validation["camera_a"] == camera_index)
            | (validation["camera_b"] == camera_index)
        )
        values_before, values_after = before[select], after[select]
        camera_rows.append({
            "camera_index": camera_index, "folder": folder,
            "observations": int(np.sum(select)),
            "before": quantiles(values_before), "after": quantiles(values_after),
            "before_inlier_ratio_2_5px": (
                float(np.mean(values_before <= 2.5)) if len(values_before) else 0.0
            ),
            "after_inlier_ratio_2_5px": (
                float(np.mean(values_after <= 2.5)) if len(values_after) else 0.0
            ),
        })
    overall_before = quantiles(before)
    overall_after = quantiles(after)
    overall_before_ratio = float(np.mean(before <= 2.5))
    overall_after_ratio = float(np.mean(after <= 2.5))
    observed_camera_rows = [
        row for row in camera_rows
        if row["observations"] >= int(args.min_camera_validation_observations)
    ]
    max_camera_p90_px = float(
        getattr(args, "max_camera_validation_p90_px", math.inf)
    )
    failed_cameras = []
    for row in observed_camera_rows:
        reasons = []
        if row["after_inlier_ratio_2_5px"] < float(
            args.min_camera_validation_inlier_ratio
        ):
            reasons.append("inlier_ratio")
        camera_p90 = row["after"]["p90"]
        if camera_p90 is None or camera_p90 > max_camera_p90_px:
            reasons.append("p90")
        if reasons:
            failed_cameras.append({**row, "failure_reasons": reasons})
    failed_explicit_pair_frames = [
        row for row in pair_rows
        if row["explicit"]
        and row["after_inlier_ratio_2_5px"]
        < float(args.min_explicit_pair_validation_inlier_ratio)
    ]
    observed_explicit_pairs = {
        tuple(sorted((row["folder_a"], row["folder_b"])))
        for row in pair_rows if row["explicit"]
    }
    missing_explicit_validation_pairs = sorted(
        explicit_pairs - observed_explicit_pairs
    )
    gates = {
        "overall_inlier_ratio": (
            overall_after_ratio >= float(args.min_validation_inlier_ratio)
        ),
        "overall_p90": (
            overall_after["p90"] is not None
            and overall_after["p90"] <= float(args.max_validation_p90_px)
        ),
        "material_improvement": (
            overall_after_ratio >= overall_before_ratio
            + float(args.min_validation_improvement)
        ),
        "per_camera": len(failed_cameras) == 0,
        "explicit_pairs": (
            len(failed_explicit_pair_frames) == 0
            and len(missing_explicit_validation_pairs) == 0
        ),
    }
    return {
        "observations": int(len(after)),
        "before": overall_before, "after": overall_after,
        "before_inlier_ratio_2_5px": overall_before_ratio,
        "after_inlier_ratio_2_5px": overall_after_ratio,
        "gates": gates, "passed": bool(all(gates.values())),
        "failed_cameras": failed_cameras,
        "failed_explicit_pair_frames": failed_explicit_pair_frames,
        "missing_explicit_validation_pairs": [
            list(pair) for pair in missing_explicit_validation_pairs
        ],
        "per_camera": camera_rows, "per_pair_frame": pair_rows,
    }


def write_solved_reference(
    prior: dict, rotations: np.ndarray, centers: np.ndarray,
    focal_scales: np.ndarray, optimization: dict, validation: dict, output: Path,
) -> None:
    for index, camera in enumerate(prior["cameras"]):
        rotation = rotations[index]
        camera["R"] = rotation.tolist()
        camera["t"] = (-rotation @ centers[index]).tolist()
        scale_x, scale_y = (
            float(focal_scales[index, 0]), float(focal_scales[index, 1])
        )
        if not (np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0)):
            if str(camera.get("camera_model_name")) != "OPENCV":
                raise ValueError(
                    f"{camera.get('folder')}: focal refinement requires OPENCV"
                )
            parameters = list(camera["params"])
            parameters[0] = float(parameters[0]) * scale_x
            parameters[1] = float(parameters[1]) * scale_y
            camera["params"] = parameters
    prior["dynamic_calibration"] = {
        "method": "multiframe_epipolar_pose_refinement",
        "optimization": optimization,
        "validation_summary": {
            key: validation[key] for key in (
                "observations", "before", "after",
                "before_inlier_ratio_2_5px", "after_inlier_ratio_2_5px",
                "gates", "passed",
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")


def write_solved_colmap(
    input_sparse: Path,
    output_sparse: Path,
    reference: dict,
) -> dict:
    """Write a COLMAP model whose cameras exactly match a solved reference.

    The existing sparse points and tracks are retained only as scene metadata;
    image poses and intrinsics are replaced.  A fresh camera is emitted for
    every image so shared COLMAP camera IDs cannot silently mix two 4DV views.
    """
    colmap = _load_colmap_io_module()
    loaded = colmap.read_model(str(input_sparse), ext="")
    if loaded is None:
        raise ValueError(f"could not read COLMAP model from {input_sparse}")
    _, input_images, points3d = loaded
    reference_by_folder = {
        str(camera["folder"]): camera for camera in reference["cameras"]
    }
    image_by_folder = {}
    for image in input_images.values():
        folder = Path(str(image.name)).parent.name
        if folder in image_by_folder:
            raise ValueError(f"COLMAP model has multiple images for folder {folder}")
        image_by_folder[folder] = image
    missing_images = sorted(set(reference_by_folder) - set(image_by_folder))
    extra_images = sorted(set(image_by_folder) - set(reference_by_folder))
    if missing_images or extra_images:
        raise ValueError(
            "reference/COLMAP folder mismatch: "
            f"missing={missing_images[:12]} extra={extra_images[:12]}"
        )
    if output_sparse.exists() and any(output_sparse.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty COLMAP output {output_sparse}"
        )
    output_sparse.mkdir(parents=True, exist_ok=True)
    output_cameras = {}
    output_images = {}
    for camera_id, folder in enumerate(sorted(reference_by_folder), start=1):
        camera = reference_by_folder[folder]
        model = str(camera["camera_model_name"]).upper()
        params = np.asarray(camera["params"], dtype=np.float64)
        output_cameras[camera_id] = colmap.Camera(
            id=camera_id,
            model=model,
            width=int(camera["width"]),
            height=int(camera["height"]),
            params=params,
        )
        old_image = image_by_folder[folder]
        rotation = np.asarray(camera["R"], dtype=np.float64)
        translation = np.asarray(camera["t"], dtype=np.float64)
        output_images[old_image.id] = colmap.Image(
            id=old_image.id,
            qvec=colmap.rotmat2qvec(rotation),
            tvec=translation,
            camera_id=camera_id,
            name=old_image.name,
            xys=old_image.xys,
            point3D_ids=old_image.point3D_ids,
        )
    colmap.write_model(
        output_cameras, output_images, points3d, str(output_sparse), ext=".bin"
    )
    verification = colmap.read_model(str(output_sparse), ext=".bin")
    if verification is None:
        raise RuntimeError(f"could not read back written COLMAP model {output_sparse}")
    verify_cameras, verify_images, verify_points = verification
    if len(verify_cameras) != len(reference_by_folder):
        raise RuntimeError("written COLMAP camera count failed verification")
    if len(verify_images) != len(reference_by_folder):
        raise RuntimeError("written COLMAP image count failed verification")
    return {
        "input_sparse": str(input_sparse),
        "output_sparse": str(output_sparse),
        "cameras": len(verify_cameras),
        "images": len(verify_images),
        "points3D": len(verify_points),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reference", default="reference_export.json")
    parser.add_argument("--capture-json", required=True)
    parser.add_argument("--snapshot-record", required=True)
    parser.add_argument(
        "--pose-prior-source",
        choices=("base", "snapshot"),
        default="base",
        help=(
            "Pose seed for image refinement. 'base' uses the image-derived "
            "reference (default); 'snapshot' is retained only for controlled "
            "A/B comparison of the mechanical 4DV record."
        ),
    )
    parser.add_argument("--images-root", default="images")
    parser.add_argument("--image-pattern", default="{frame:06d}.jpg")
    parser.add_argument("--frames", type=parse_frames, default=(1, 13, 25))
    parser.add_argument("--camera-neighbors", type=int, default=8)
    parser.add_argument("--wide-baseline-per-camera", type=int, default=1)
    parser.add_argument("--camera-pairs", type=parse_camera_pairs, default=())
    parser.add_argument(
        "--optional-camera-pairs", action="store_true",
        help=(
            "Treat --camera-pairs as candidates rather than requirements. By default "
            "any explicit pair that fails pair-fit filtering aborts the run, which is "
            "correct when the pairs are hand-picked and known-good but wrong when they "
            "are proposals from select_pairs.py -- there roughly half are expected to "
            "fail, and failing one should not discard the whole match set."
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer-device", default=None)
    parser.add_argument("--roma-backend", choices=("v1", "v2"), default="v2")
    parser.add_argument("--roma-model", default="outdoor")
    parser.add_argument("--roma-v2-setting", default="precise")
    parser.add_argument("--roma-v2-mask-oversample", type=float, default=2.0)
    parser.add_argument("--roma-v2-spatial-grid", type=int, default=8)
    parser.add_argument("--roma-v2-spatial-fraction", type=float, default=0.5)
    parser.add_argument("--roma-num-matches", type=int, default=6000)
    parser.add_argument("--roma-min-certainty", type=float, default=0.05)
    parser.add_argument("--pair-ransac-px", type=float, default=1.25)
    parser.add_argument("--min-pair-inliers", type=int, default=100)
    parser.add_argument("--min-pair-fit-ratio", type=float, default=0.30)
    parser.add_argument("--max-pair-inliers", type=int, default=512)
    parser.add_argument("--min-relative-pose-inliers", type=int, default=40)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--cache-dir", default="dynamic_calibration_cache")
    parser.add_argument("--image-cache-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--robust-scale-px", type=float, default=2.0)
    parser.add_argument("--rotation-prior-sigma-deg", type=float, default=3.0)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.002)
    parser.add_argument("--max-rotation-delta-deg", type=float, default=15.0)
    parser.add_argument(
        "--disable-failed-camera-pose-refinement", action="store_true",
        help="Do not refine rotation and optical center for per-camera failures.",
    )
    parser.add_argument("--failed-pose-iterations", type=int, default=1500)
    parser.add_argument("--failed-pose-learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--failed-pose-max-rotation-delta-deg", type=float, default=10.0
    )
    parser.add_argument("--failed-pose-max-center-delta", type=float, default=0.30)
    parser.add_argument(
        "--failed-pose-rotation-sigma-deg", type=float, default=3.0
    )
    parser.add_argument("--failed-pose-center-sigma", type=float, default=0.10)
    parser.add_argument("--failed-pose-prior-weight", type=float, default=0.002)
    parser.add_argument(
        "--failed-pose-min-neighbor-inlier-ratio", type=float, default=0.35
    )
    parser.add_argument(
        "--failed-pose-max-neighbor-p90-px",
        type=float,
        default=8.0,
        help=(
            "Reject a failed-camera repair edge when its training residual p90 "
            "exceeds this value, even if its aggregate inlier ratio passes."
        ),
    )
    parser.add_argument("--failed-pose-min-neighbors", type=int, default=2)
    parser.add_argument("--failed-focal-iterations", type=int, default=1000)
    parser.add_argument("--failed-focal-learning-rate", type=float, default=0.01)
    parser.add_argument("--failed-focal-max-scale", type=float, default=1.50)
    parser.add_argument("--failed-focal-log-sigma", type=float, default=0.15)
    parser.add_argument("--failed-focal-prior-weight", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--min-validation-inlier-ratio", type=float, default=0.50)
    parser.add_argument("--max-validation-p90-px", type=float, default=8.0)
    parser.add_argument("--min-validation-improvement", type=float, default=0.15)
    parser.add_argument("--min-camera-validation-observations", type=int, default=100)
    parser.add_argument(
        "--min-camera-validation-inlier-ratio", type=float, default=0.65,
        help=(
            "Minimum held-out fraction within 2.5 px for every observed camera. "
            "This is deliberately stricter than the old 0.25 false-pass floor."
        ),
    )
    parser.add_argument(
        "--max-camera-validation-p90-px", type=float, default=5.0,
        help="Maximum held-out p90 epipolar residual for every observed camera.",
    )
    parser.add_argument(
        "--min-explicit-pair-validation-inlier-ratio", type=float, default=0.35
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-reference", default="reference_export_dynamic_4dv.json")
    parser.add_argument("--output-report", default="dynamic_calibration_report.json")
    parser.add_argument(
        "--input-sparse",
        default=None,
        help="Existing COLMAP model whose points/tracks will be retained.",
    )
    parser.add_argument(
        "--output-sparse",
        default=None,
        help=(
            "Write a trainer-compatible COLMAP model with the solved cameras. "
            "Must be supplied together with --input-sparse."
        ),
    )
    parser.add_argument("--write-failed-candidate", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.camera_neighbors < 1 or args.wide_baseline_per_camera < 0:
        raise ValueError("camera pair counts must be non-negative, with >=1 neighbor")
    if not 0.0 <= args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in [0, 0.5)")
    if args.min_pair_inliers < 8 or args.max_pair_inliers < args.min_pair_inliers:
        raise ValueError("pair inlier limits are inconsistent")
    if args.min_relative_pose_inliers < 8:
        raise ValueError("--min-relative-pose-inliers must be at least 8")
    if not 0.0 <= args.min_pair_fit_ratio <= 1.0:
        raise ValueError("--min-pair-fit-ratio must be in [0,1]")
    if not 0.0 <= args.min_explicit_pair_validation_inlier_ratio <= 1.0:
        raise ValueError(
            "--min-explicit-pair-validation-inlier-ratio must be in [0,1]"
        )
    if not 0.0 <= args.min_camera_validation_inlier_ratio <= 1.0:
        raise ValueError("--min-camera-validation-inlier-ratio must be in [0,1]")
    if args.max_camera_validation_p90_px <= 0.0:
        raise ValueError("--max-camera-validation-p90-px must be positive")
    if args.max_rotation_delta_deg <= 0.0 or args.max_rotation_delta_deg >= 90.0:
        raise ValueError("--max-rotation-delta-deg must be in (0,90)")
    if args.failed_pose_iterations < 1 or args.failed_pose_learning_rate <= 0.0:
        raise ValueError("failed-camera pose optimizer settings must be positive")
    if not 0.0 < args.failed_pose_max_rotation_delta_deg < 90.0:
        raise ValueError("--failed-pose-max-rotation-delta-deg must be in (0,90)")
    if args.failed_pose_max_center_delta <= 0.0:
        raise ValueError("--failed-pose-max-center-delta must be positive")
    if args.failed_pose_rotation_sigma_deg <= 0.0:
        raise ValueError("--failed-pose-rotation-sigma-deg must be positive")
    if args.failed_pose_center_sigma <= 0.0 or args.failed_pose_prior_weight < 0.0:
        raise ValueError("failed-camera pose prior settings are invalid")
    if not 0.0 <= args.failed_pose_min_neighbor_inlier_ratio <= 1.0:
        raise ValueError("failed-camera neighbor inlier ratio must be in [0,1]")
    if args.failed_pose_max_neighbor_p90_px <= 0.0:
        raise ValueError("--failed-pose-max-neighbor-p90-px must be positive")
    if args.failed_pose_min_neighbors < 1:
        raise ValueError("--failed-pose-min-neighbors must be positive")
    if args.failed_focal_iterations < 1 or args.failed_focal_learning_rate <= 0.0:
        raise ValueError("failed-camera focal optimizer settings must be positive")
    if args.failed_focal_max_scale <= 1.0:
        raise ValueError("--failed-focal-max-scale must be greater than 1")
    if args.failed_focal_log_sigma <= 0.0 or args.failed_focal_prior_weight < 0.0:
        raise ValueError("failed-camera focal prior settings are invalid")
    if bool(args.input_sparse) != bool(args.output_sparse):
        raise ValueError("--input-sparse and --output-sparse must be supplied together")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    base = json.loads(Path(args.base_reference).read_text(encoding="utf-8"))
    capture = json.loads(Path(args.capture_json).read_text(encoding="utf-8"))
    snapshot = json.loads(Path(args.snapshot_record).read_text(encoding="utf-8"))
    prior, alignment = build_effective_prior(
        base, capture, snapshot, pose_prior_source=args.pose_prior_source
    )
    prep = _load_prep_module()
    with __import__("tempfile").TemporaryDirectory(prefix="trigs-4dv-prior-") as name:
        prior_path = Path(name) / "prior.json"
        prior_path.write_text(json.dumps(prior), encoding="utf-8")
        cameras, _ = prep.load_reference_export(prior_path)
    folders = [str(camera.folder) for camera in cameras]
    matrices = [camera._K.copy() for camera in cameras]
    prior_rotations = np.stack([camera.R for camera in cameras])
    centers = np.stack([camera.center_world for camera in cameras])
    pairs = choose_pairs(
        centers, args.camera_neighbors, args.wide_baseline_per_camera,
        args.camera_pairs, folders,
    )
    if args.max_pairs > 0:
        pairs = pairs[: int(args.max_pairs)]
    print(
        f"Effective 4DV prior: cameras={len(cameras)} pairs={len(pairs)} "
        f"frames={list(args.frames)}"
    )
    print(
        "Center alignment: "
        f"median={alignment['center_alignment_median']:.4f} "
        f"p90={alignment['center_alignment_p90']:.4f} "
        f"max={alignment['center_alignment_max']:.4f}"
    )
    blocks, match_rows = gather_matches(args, cameras, matrices, pairs)
    accepted_edges = sorted({(block.camera_a, block.camera_b) for block in blocks})
    folder_index = {folder: index for index, folder in enumerate(folders)}
    requested_explicit_edges = {
        tuple(sorted((folder_index[left], folder_index[right])))
        for left, right in args.camera_pairs
    }
    missing_explicit_edges = sorted(requested_explicit_edges - set(accepted_edges))
    if args.optional_camera_pairs and missing_explicit_edges:
        print(
            f"{len(missing_explicit_edges)} of {len(requested_explicit_edges)} explicit "
            "pairs did not pass pair-fit filtering; continuing because "
            "--optional-camera-pairs is set"
        )
        missing_explicit_edges = []
    components = connected_components(len(cameras), accepted_edges)
    component_folders = [[folders[index] for index in values] for values in components]
    if len(components) != 1 or missing_explicit_edges:
        report = {
            "status": (
                "failed_disconnected_camera_graph" if len(components) != 1
                else "failed_missing_explicit_pairs"
            ),
            "components": component_folders, "matches": match_rows,
            "missing_explicit_pairs": [
                [folders[left], folders[right]]
                for left, right in missing_explicit_edges
            ],
        }
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if len(components) != 1:
            print("Calibration graph is disconnected:")
            for values in component_folders:
                print(f"  {len(values)} cameras: {','.join(values)}")
        if missing_explicit_edges:
            print("Required explicit pairs did not pass pair-fit filtering:")
            for left, right in missing_explicit_edges:
                print(f"  {folders[left]}:{folders[right]}")
        return 2
    train = pack_observations(blocks, centers, matrices, train=True)
    validation = pack_observations(blocks, centers, matrices, train=False)
    print(
        f"Accepted blocks={len(blocks)} unique_edges={len(accepted_edges)} "
        f"train_obs={len(train['camera_a'])} validation_obs={len(validation['camera_a'])}"
    )
    initialized_rotations, initialization = initialize_rotations_from_pair_poses(
        blocks,
        centers,
        prior_rotations,
        min_pose_inliers=args.min_relative_pose_inliers,
    )
    initialization_residual = residuals_numpy(initialized_rotations, validation)
    initialization["heldout_residual"] = quantiles(initialization_residual)
    initialization["heldout_inlier_ratio_2_5px"] = float(
        np.mean(initialization_residual <= 2.5)
    )
    print(
        "Pair-pose initialization: "
        f"cameras={initialization['initialized_cameras']}/{len(cameras)} "
        f"blocks={initialization['relative_pose_blocks']} "
        f"inliers@2.5px={100*initialization['heldout_inlier_ratio_2_5px']:.1f}% "
        f"p90={format_pixels(initialization['heldout_residual']['p90'])}"
    )
    initialized_components = [
        row for row in initialization.get("components", [])
        if row.get("initialized")
    ]
    if initialized_components:
        largest = max(
            initialized_components, key=lambda row: len(row["camera_indices"])
        )
        sync = largest["synchronization"]
        gauge = largest["gauge"]
        print(
            "Rotation synchronization: "
            f"edges={sync.get('unique_edges', 0)} "
            f"relative-p90={sync.get('rotation_residual_p90_deg', float('nan')):.3f}deg "
            f"cycle-p90={sync.get('cycle_error_p90_deg', float('nan')):.3f}deg; "
            f"world-gauge-p90={gauge.get('angular_residual_p90_deg', float('nan')):.3f}deg"
        )
    solved_rotations, optimization = optimize_rotations(
        initialized_rotations, train, args
    )
    preliminary_validation = validation_report(
        prior_rotations, solved_rotations, validation, blocks, folders, args
    )
    pose_validation = preliminary_validation
    solved_centers = centers.copy()
    solved_focal_scales = np.ones((len(cameras), 2), dtype=np.float64)
    failed_pose_refinement = {"enabled": False, "reason": "not_needed"}
    neighbor_selection = {"enabled": False, "reason": "not_needed"}
    consistent_pose_refinement = {"enabled": False, "reason": "not_needed"}
    failed_focal_refinement = {"enabled": False, "reason": "not_needed"}
    active_train = train
    failed_camera_indices = [
        int(row["camera_index"])
        for row in preliminary_validation["failed_cameras"]
    ]
    if failed_camera_indices and not args.disable_failed_camera_pose_refinement:
        print(
            "Refining full pose for failed cameras while freezing the validated "
            "network: "
            + ",".join(folders[index] for index in failed_camera_indices)
        )
        solved_rotations, solved_centers, failed_pose_refinement = (
            optimize_failed_camera_poses(
                solved_rotations, centers, train, failed_camera_indices, args
            )
        )
        failed_pose_refinement["folders"] = [
            folders[index] for index in failed_camera_indices
        ]
        solved_validation = packed_with_centers(validation, solved_centers)
        validation_summary = validation_report(
            prior_rotations,
            solved_rotations,
            solved_validation,
            blocks,
            folders,
            args,
            before_validation=validation,
        )
        pose_validation = validation_summary
        remaining_failed_indices = [
            int(row["camera_index"])
            for row in validation_summary["failed_cameras"]
        ]
        if remaining_failed_indices:
            train_keep, neighbor_selection = (
                select_consistent_failed_camera_neighbors(
                    solved_rotations,
                    solved_centers,
                    train,
                    blocks,
                    folders,
                    remaining_failed_indices,
                    args,
                )
            )
            active_train = subset_packed(train, train_keep)
            selected_text = []
            for row in neighbor_selection["per_camera"]:
                selected_text.append(
                    f"{row['folder']}=>{','.join(row['selected_neighbors'])}"
                )
            print(
                "Re-solving failed cameras using consistent training neighbors: "
                + "; ".join(selected_text)
            )
            refinable_indices = [
                int(row["camera_index"])
                for row in neighbor_selection["per_camera"]
                if row["selected_pair_ids"]
            ]
            if refinable_indices:
                solved_rotations, solved_centers, consistent_pose_refinement = (
                    optimize_failed_camera_poses(
                        solved_rotations,
                        solved_centers,
                        active_train,
                        refinable_indices,
                        args,
                    )
                )
                consistent_pose_refinement["folders"] = [
                    folders[index] for index in refinable_indices
                ]
            else:
                consistent_pose_refinement = {
                    "enabled": False,
                    "reason": "no_consistent_training_neighbors",
                }
            # Neighbor selection is allowed to robustify the *training* set,
            # but validation must remain the original untouched holdout.  The
            # previous implementation pruned contradictory held-out edges and
            # could therefore publish a camera that only agreed with its hand-
            # picked neighbors.
            solved_validation = packed_with_centers(validation, solved_centers)
            validation_summary = validation_report(
                prior_rotations,
                solved_rotations,
                solved_validation,
                blocks,
                folders,
                args,
                before_validation=validation,
            )
            pose_validation = validation_summary
            remaining_failed_indices = [
                int(row["camera_index"])
                for row in validation_summary["failed_cameras"]
            ]
        if remaining_failed_indices:
            focal_refinable_indices = [
                index for index in remaining_failed_indices
                if np.any(active_train["camera_a"] == index)
                or np.any(active_train["camera_b"] == index)
            ]
            print(
                "Refining focal scale for cameras still failing after full pose: "
                + ",".join(folders[index] for index in focal_refinable_indices)
            )
            if focal_refinable_indices:
                solved_focal_scales, failed_focal_refinement = (
                    optimize_failed_camera_focals(
                        solved_rotations,
                        solved_centers,
                        active_train,
                        focal_refinable_indices,
                        args,
                    )
                )
                failed_focal_refinement["folders"] = [
                    folders[index] for index in focal_refinable_indices
                ]
            else:
                failed_focal_refinement = {
                    "enabled": False,
                    "reason": "no_consistent_training_neighbors",
                }
            solved_validation = packed_with_calibration(
                validation, solved_centers, solved_focal_scales
            )
            validation_summary = validation_report(
                prior_rotations,
                solved_rotations,
                solved_validation,
                blocks,
                folders,
                args,
                before_validation=validation,
            )
            remaining_failed_indices = [
                int(row["camera_index"])
                for row in validation_summary["failed_cameras"]
            ]
            post_focal_pose_indices = [
                index for index in remaining_failed_indices
                if index in focal_refinable_indices
            ]
            if post_focal_pose_indices:
                # Focal and pose are coupled.  After solving zoom, run one
                # final pose/centre refinement with the corrected rays instead
                # of leaving the pose at the value that compensated for the
                # old focal length.
                calibrated_train = packed_with_calibration(
                    active_train, solved_centers, solved_focal_scales
                )
                solved_rotations, solved_centers, final_pose_refinement = (
                    optimize_failed_camera_poses(
                        solved_rotations,
                        solved_centers,
                        calibrated_train,
                        post_focal_pose_indices,
                        args,
                    )
                )
                final_pose_refinement["folders"] = [
                    folders[index] for index in post_focal_pose_indices
                ]
                failed_focal_refinement["post_focal_pose_refinement"] = (
                    final_pose_refinement
                )
                solved_validation = packed_with_calibration(
                    validation, solved_centers, solved_focal_scales
                )
                validation_summary = validation_report(
                    prior_rotations,
                    solved_rotations,
                    solved_validation,
                    blocks,
                    folders,
                    args,
                    before_validation=validation,
                )
    else:
        if failed_camera_indices:
            failed_pose_refinement = {
                "enabled": False,
                "reason": "disabled",
                "folders": [folders[index] for index in failed_camera_indices],
            }
        validation_summary = preliminary_validation
    if failed_focal_refinement.get("enabled"):
        refined_indices = np.asarray(
            failed_focal_refinement["camera_indices"], dtype=np.int64
        )
        refined_scales = solved_focal_scales[refined_indices]
        limit = float(args.failed_focal_max_scale)
        lower = 1.0 / limit
        saturated_mask = np.any(
            (refined_scales <= lower * 1.01)
            | (refined_scales >= limit / 1.01),
            axis=1,
        )
        saturated_folders = [
            folders[int(index)]
            for index, saturated in zip(refined_indices, saturated_mask)
            if saturated
        ]
        failed_focal_refinement["saturated_folders"] = saturated_folders
        validation_summary["gates"]["focal_not_saturated"] = (
            len(saturated_folders) == 0
        )
        validation_summary["passed"] = bool(
            all(validation_summary["gates"].values())
        )
    combined_optimization = {
        "global_rotation": optimization,
        "failed_camera_pose": failed_pose_refinement,
        "failed_camera_neighbor_selection": neighbor_selection,
        "failed_camera_consistent_pose": consistent_pose_refinement,
        "failed_camera_focal": failed_focal_refinement,
    }
    report = {
        "status": "passed" if validation_summary["passed"] else "failed_validation",
        "inputs": {
            "base_reference": str(Path(args.base_reference)),
            "capture_json": str(Path(args.capture_json)),
            "snapshot_record": str(Path(args.snapshot_record)),
            "images_root": str(Path(args.images_root)), "frames": list(args.frames),
        },
        "validation_policy": {
            "holdout_is_never_neighbor_pruned": True,
            "min_camera_observations": int(
                args.min_camera_validation_observations
            ),
            "min_camera_inlier_ratio_2_5px": float(
                args.min_camera_validation_inlier_ratio
            ),
            "max_camera_p90_px": float(args.max_camera_validation_p90_px),
            "min_overall_inlier_ratio_2_5px": float(
                args.min_validation_inlier_ratio
            ),
            "max_overall_p90_px": float(args.max_validation_p90_px),
        },
        "alignment": alignment, "components": component_folders,
        "match_summary": {
            "requested_pair_frames": len(match_rows),
            "accepted_pair_frames": len(blocks),
            "accepted_edges": len(accepted_edges),
        },
        "matches": match_rows, "initialization": initialization,
        "optimization": combined_optimization,
        "preliminary_validation": preliminary_validation,
        "pose_validation": pose_validation,
        "validation": validation_summary,
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Held-out validation: "
        f"inliers@2.5px {100*validation_summary['before_inlier_ratio_2_5px']:.1f}% -> "
        f"{100*validation_summary['after_inlier_ratio_2_5px']:.1f}%, "
        f"p90 {format_pixels(validation_summary['before']['p90'])} -> "
        f"{format_pixels(validation_summary['after']['p90'])}"
    )
    print(f"Validation gates: {validation_summary['gates']}")
    if validation_summary["passed"] or args.write_failed_candidate:
        write_solved_reference(
            prior, solved_rotations, solved_centers, solved_focal_scales,
            combined_optimization, validation_summary, Path(args.output_reference),
        )
        print(f"Wrote calibrated reference: {args.output_reference}")
        if args.input_sparse:
            sparse_summary = write_solved_colmap(
                Path(args.input_sparse), Path(args.output_sparse), prior
            )
            report["output_colmap"] = sparse_summary
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote calibrated COLMAP model: {args.output_sparse}")
    else:
        print("Validation failed; refusing to write a calibrated reference.")
    print(f"Report: {report_path}")
    return 0 if validation_summary["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
