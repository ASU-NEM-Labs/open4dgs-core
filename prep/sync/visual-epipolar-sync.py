#!/usr/bin/env python3
"""
Estimate camera sync offsets visually from calibrated multi-view footage.

This does NOT run full RoMa initialization or training. It uses:
  1) RoMa cross-view matches at the nominal frame,
  2) adaptive multi-frame same-camera tracks to fit local 2D trajectories,
  3) spatially balanced epipolar phase estimates (with optional direct search),
  4) spatial block-bootstrap uncertainty and circular pair aggregation,
  5) a preliminary global solve (the robust graph script is the final solve).

The estimator is designed for your situation: timecode-aligned but non-genlocked
multi-camera footage where offsets are sub-frame. It avoids the crude RoMa init
nearest-frame problem because it estimates fractional-frame offsets from motion,
rather than rounding offsets to whole frames.

Output conventions:
  --output-json uses PLUS convention:
      actual_time_sec = nominal_time_sec + offset_sec[camera]
  --output-selfcap-json is the negated file for SelfCap-style loaders:
      actual_time_sec = nominal_time_sec - sync_json[camera]

Caveat:
  This is a sync estimator, not a final renderer. It needs moving foreground
  features. Static background carries no timing signal and is filtered out using
  optical-flow magnitude and, optionally, your 3D stage capsule.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import functools
import gc
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __name__ == "__main__":
    # Enable before native imports so SIGSEGV reports the child's Python stack.
    faulthandler.enable(all_threads=True)

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Camera geometry
# -----------------------------------------------------------------------------

@dataclass
class Camera:
    name: str
    folder: str
    R: np.ndarray
    t: np.ndarray
    camera_model_name: str
    width: int
    height: int
    params: np.ndarray

    @functools.cached_property
    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t

    @functools.cached_property
    def extrinsic(self) -> np.ndarray:
        return np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)

    @functools.cached_property
    def K(self) -> np.ndarray:
        m = self.camera_model_name.upper()
        p = self.params.astype(np.float64)
        if m == "SIMPLE_PINHOLE":
            f, cx, cy = p[:3]
            fx = fy = f
        elif m in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy, cx, cy = p[:4]
        elif m in ("SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = p[:3]
            fx = fy = f
        else:
            raise NotImplementedError(f"Unsupported camera model for K: {self.camera_model_name}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @functools.cached_property
    def D(self) -> np.ndarray:
        m = self.camera_model_name.upper()
        p = self.params.astype(np.float64)
        if m in ("SIMPLE_PINHOLE", "PINHOLE"):
            return np.zeros((4, 1), dtype=np.float64)
        if m == "SIMPLE_RADIAL":
            return np.array([[p[3]], [0.0], [0.0], [0.0]], dtype=np.float64)
        if m == "RADIAL":
            return np.array([[p[3]], [p[4]], [0.0], [0.0]], dtype=np.float64)
        if m == "OPENCV":
            return p[4:8].reshape(-1, 1).astype(np.float64)
        if m == "FULL_OPENCV":
            return p[4:].reshape(-1, 1).astype(np.float64)
        if m == "OPENCV_FISHEYE":
            return p[4:8].reshape(-1, 1).astype(np.float64)
        raise NotImplementedError(f"Unsupported distortion model: {self.camera_model_name}")

    def undistort_norm(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.camera_model_name.upper() == "OPENCV_FISHEYE":
            out = cv2.fisheye.undistortPoints(uv, self.K, self.D, P=None)
        else:
            out = cv2.undistortPoints(uv, self.K, self.D, P=None)
        return out.reshape(-1, 2)


def load_reference_export(path: Path) -> Tuple[List[Camera], Optional[dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams: List[Camera] = []
    for item in data["cameras"]:
        cams.append(Camera(
            name=str(item["name"]),
            folder=str(item["folder"]),
            R=np.asarray(item["R"], dtype=np.float64),
            t=np.asarray(item["t"], dtype=np.float64).reshape(3),
            camera_model_name=str(item["camera_model_name"]),
            width=int(item["width"]),
            height=int(item["height"]),
            params=np.asarray(item["params"], dtype=np.float64),
        ))
    return cams, data.get("scene_meta")


def essential_matrix(cam_a: Camera, cam_b: Camera) -> np.ndarray:
    # Transform A camera coordinates to B camera coordinates.
    R_ba = cam_b.R @ cam_a.R.T
    t_ba = cam_b.t - R_ba @ cam_a.t
    tx = np.array([
        [0.0, -t_ba[2], t_ba[1]],
        [t_ba[2], 0.0, -t_ba[0]],
        [-t_ba[1], t_ba[0], 0.0],
    ], dtype=np.float64)
    return tx @ R_ba


def triangulate_dlt(obs: Sequence[Tuple[Camera, np.ndarray]]) -> Optional[np.ndarray]:
    if len(obs) < 2:
        return None
    rows = []
    for cam, uv in obs:
        xy = cam.undistort_norm(np.asarray(uv).reshape(1, 2))[0]
        P = cam.extrinsic
        u, v = float(xy[0]), float(xy[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    return X if np.all(np.isfinite(X)) else None


def parse_vec3(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected x,y,z")
    return np.asarray(vals, dtype=np.float64)


def read_cached_observations(path: Path) -> List[dict]:
    """Load our raw CSV back into the numeric types used by aggregation."""
    if not path.exists():
        return []
    integer_fields = {"cam_a", "cam_b", "frame", "n_used", "n_dynamic_matches", "n_spatial_cells"}
    boolean_fields = {"direct_phase_boundary"}
    string_fields = {"error", "folder_a", "folder_b", "phase_estimator"}
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                if key in string_fields:
                    row[key] = value
                elif key in boolean_fields:
                    row[key] = value.strip().lower() in {"1", "true", "yes"}
                elif key in integer_fields:
                    row[key] = int(float(value))
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            if {"cam_a", "cam_b", "frame"} <= row.keys():
                rows.append(row)
    return rows


def write_observations_atomic(path: Path, rows: Sequence[dict]) -> None:
    """Keep the previous checkpoint intact until its replacement is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inside_capsule(X: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> bool:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(X - a)) <= radius
    tau = float(((X - a) @ ab) / denom)
    tau = max(0.0, min(1.0, tau))
    closest = a + tau * ab
    return float(np.linalg.norm(X - closest)) <= float(radius)


# -----------------------------------------------------------------------------
# IO + pair selection
# -----------------------------------------------------------------------------

def image_path(images_root: Path, cam: Camera, frame_idx: int, image_pattern: str) -> Path:
    return images_root / cam.folder / image_pattern.format(frame=int(frame_idx))


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    return cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2GRAY)


def choose_camera_pairs(cameras: Sequence[Camera], neighbors_per_camera: int, wide_baseline_per_camera: int) -> List[Tuple[int, int]]:
    n = len(cameras)
    if n < 2:
        return []
    if neighbors_per_camera <= 0 or neighbors_per_camera >= n - 1:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    centers = np.stack([c.center_world for c in cameras], axis=0)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    pairs = set()
    for i in range(n):
        order = np.argsort(d[i])
        for j in order[:min(neighbors_per_camera, n - 1)]:
            pairs.add(tuple(sorted((i, int(j)))))
        if wide_baseline_per_camera > 0:
            finite = np.where(np.isfinite(d[i]))[0]
            far = finite[np.argsort(-d[i, finite])]
            for j in far[:min(wide_baseline_per_camera, len(far))]:
                pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


def parse_pairs(s: Optional[str], cameras: Sequence[Camera]) -> Optional[List[Tuple[int, int]]]:
    if not s:
        return None
    by_folder = {c.folder: i for i, c in enumerate(cameras)}
    by_key = {f"{i:04d}": i for i in range(len(cameras))}
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = re_split_pair(token)
        ia = by_folder.get(a, by_key.get(a, int(a) if a.isdigit() and int(a) < len(cameras) else None))
        ib = by_folder.get(b, by_key.get(b, int(b) if b.isdigit() and int(b) < len(cameras) else None))
        if ia is None or ib is None:
            raise ValueError(f"Could not parse pair '{token}'")
        out.append(tuple(sorted((int(ia), int(ib)))))
    return sorted(set(out))


def re_split_pair(token: str) -> Tuple[str, str]:
    for sep in ["-", ":", "/"]:
        if sep in token:
            a, b = token.split(sep, 1)
            return a.strip(), b.strip()
    raise ValueError(f"Pair token must look like 01-02 or 0000-0001: {token}")


def parse_frames(args) -> List[int]:
    if args.frames:
        return sorted({int(x.strip()) for x in args.frames.split(",") if x.strip()})
    if args.frame_start is None or args.frame_end is None:
        raise ValueError("Use --frames or --frame-start/--frame-end")
    return list(range(int(args.frame_start), int(args.frame_end) + 1, int(args.frame_step)))


# -----------------------------------------------------------------------------
# Matching and flow
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_roma_backend_factory():
    """Load the shared RoMa v1/v2 adapter without requiring ``prep`` as a package."""
    path = Path(__file__).resolve().parents[1] / "init" / "roma_backends.py"
    spec = importlib.util.spec_from_file_location("trigs_roma_backends", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RoMa backends from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_roma_backend


def run_roma_matches(model, path_a: Path, path_b: Path, image_a: np.ndarray, image_b: np.ndarray,
                     device: str, num_matches: int, min_certainty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ``RomaBackend`` owns the model device and normalizes the differing v1
    # (romatch) and v2 (romav2/DINOv3) APIs to pixel-coordinate points.
    del device
    a, b, c = model.match_points(
        path_a, path_b, image_a, image_b, num=int(num_matches)
    )
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32).reshape(-1)
    keep = (c >= min_certainty) & np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1) & np.isfinite(c)
    return a[keep], b[keep], c[keep]


def sparse_multiframe_motion(
    images_root: Path,
    cam: Camera,
    frame: int,
    image_pattern: str,
    pts: np.ndarray,
    baselines: Sequence[int],
    min_track_displacement_px: float,
    max_track_displacement_px: float,
    lk_win: int = 21,
    fb_threshold_px: float = 1.5,
    debug: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a local quadratic point trajectory from sequential LK tracks.

    A one-frame derivative is dominated by tracking quantization on slow clips.
    This routine tracks out to +/- max(baselines), then selects the shortest
    symmetric baseline whose end-to-end displacement is informative.  It
    returns velocity (px/frame), acceleration (px/frame^2), validity,
    curvature/velocity, and the selected half-window in frames.

    ``debug`` (diagnostics only; None = unchanged behaviour) receives, per
    baseline k, whether the symmetric +/-k track survived LK + forward/backward
    checks ("track_ok_k<k>") and its end-to-end displacement ("disp_k<k>").
    """
    n = len(pts)
    empty2 = np.zeros((n, 2), np.float32)
    if n == 0:
        return empty2, empty2.copy(), np.zeros(0, bool), np.zeros(0, np.float32), np.zeros(0, np.int16)

    baselines = sorted({int(k) for k in baselines if int(k) > 0})
    if not baselines:
        raise ValueError("At least one positive temporal baseline is required")
    max_k = max(baselines)
    p0 = pts.astype(np.float32)
    gray0 = load_gray(image_path(images_root, cam, frame, image_pattern))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    lk_params = dict(winSize=(lk_win, lk_win), maxLevel=3, criteria=criteria)

    def in_image(points: np.ndarray, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        return (np.all(np.isfinite(points), axis=1)
                & (points[:, 0] >= 0) & (points[:, 0] < w)
                & (points[:, 1] >= 0) & (points[:, 1] < h))

    positions: Dict[int, np.ndarray] = {0: p0.copy()}
    validity: Dict[int, np.ndarray] = {0: in_image(p0, gray0)}

    for direction in (-1, 1):
        prev_gray = gray0
        prev_pts = p0.copy()
        prev_valid = validity[0].copy()
        for step in range(1, max_k + 1):
            active = np.flatnonzero(prev_valid)
            if len(active) == 0:
                break
            try:
                next_gray = load_gray(image_path(images_root, cam, frame + direction * step, image_pattern))
            except FileNotFoundError:
                break
            tracked, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_pts[active].reshape(-1, 1, 2), None, **lk_params)
            if tracked is None or st is None:
                break
            tracked = tracked.reshape(-1, 2)
            forward_ok = st.reshape(-1).astype(bool) & in_image(tracked, next_gray)
            survivors = active[forward_ok]
            nxt = prev_pts.copy()
            nxt[survivors] = tracked[forward_ok]
            ok = np.zeros(n, dtype=bool)

            # A forward/backward check prevents long-baseline drift from being
            # mistaken for slow, coherent scene motion.
            # Failed LK outputs must never become inputs to another native call.
            if len(survivors):
                back, bst, _ = cv2.calcOpticalFlowPyrLK(
                    next_gray, prev_gray, nxt[survivors].reshape(-1, 1, 2), None, **lk_params)
                if back is not None and bst is not None:
                    back = back.reshape(-1, 2)
                    fb = np.linalg.norm(back - prev_pts[survivors], axis=1)
                    ok[survivors] = (bst.reshape(-1).astype(bool) & np.isfinite(fb)
                                     & (fb <= float(fb_threshold_px)))

            key = direction * step
            positions[key] = nxt.copy()
            validity[key] = ok.copy()
            prev_pts = nxt
            prev_valid = ok
            prev_gray = next_gray

    velocity = np.zeros((n, 2), np.float32)
    acceleration = np.zeros((n, 2), np.float32)
    chosen = np.zeros(n, np.int16)
    fallback = np.zeros(n, np.int16)

    for k in baselines:
        if -k not in positions or k not in positions:
            continue
        ok = validity[-k] & validity[k]
        disp = np.linalg.norm(positions[k] - positions[-k], axis=1)
        if debug is not None:
            debug[f"track_ok_k{k}"] = ok.copy()
            debug[f"disp_k{k}"] = disp.astype(np.float32)
        within_max = disp <= float(max_track_displacement_px)
        fallback[ok & within_max] = int(k)
        take = (chosen == 0) & ok & within_max & (disp >= float(min_track_displacement_px))
        if np.any(take):
            pm, pp = positions[-k], positions[k]
            velocity[take] = (pp[take] - pm[take]) / float(2 * k)
            acceleration[take] = (pp[take] - 2.0 * p0[take] + pm[take]) / float(k * k)
            chosen[take] = int(k)

    # If even the longest valid track is below the requested displacement,
    # retain it so the later epipolar-information gate can make the final call.
    for k in reversed(baselines):
        take = (chosen == 0) & (fallback == k)
        if np.any(take):
            pm, pp = positions[-k], positions[k]
            velocity[take] = (pp[take] - pm[take]) / float(2 * k)
            acceleration[take] = (pp[take] - 2.0 * p0[take] + pm[take]) / float(k * k)
            chosen[take] = int(k)

    valid = chosen > 0
    speed = np.linalg.norm(velocity, axis=1)
    curvature = np.linalg.norm(acceleration, axis=1) / np.maximum(speed, 1e-3)
    return velocity, acceleration, valid, curvature.astype(np.float32), chosen


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order].astype(np.float64), 1e-12)
    cs = np.cumsum(w)
    cutoff = 0.5 * float(cs[-1])
    return float(v[int(np.searchsorted(cs, cutoff))])


def mad(values: np.ndarray, center: Optional[float] = None) -> float:
    if len(values) == 0:
        return float("nan")
    if center is None:
        center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def wrap_period(values, period: float = 1.0):
    values = np.asarray(values, dtype=np.float64)
    return (values + 0.5 * period) % period - 0.5 * period


def circular_weighted_median(values: np.ndarray, weights: np.ndarray, period: float = 1.0) -> float:
    """Weighted L1 center on a circle, returned in [-period/2, period/2)."""
    values = wrap_period(values, period)
    if len(values) == 0:
        return float("nan")
    costs = np.asarray([
        np.sum(np.maximum(weights, 1e-12) * np.abs(wrap_period(values - c, period)))
        for c in values
    ])
    center = float(values[int(np.argmin(costs))])
    unwrapped = center + wrap_period(values - center, period)
    return float(wrap_period(weighted_median(unwrapped, weights), period))


# -----------------------------------------------------------------------------
# Pairwise visual offset estimation
# -----------------------------------------------------------------------------

def estimate_pair_frame(
    model,
    cameras: Sequence[Camera],
    images_root: Path,
    image_pattern: str,
    frame: int,
    pair: Tuple[int, int],
    device: str,
    roma_num_matches: int,
    roma_min_certainty: float,
    min_flow_px: float,
    max_abs_shift_frames: float,
    min_epipolar_slope: float,
    min_epipolar_motion_px: float,
    max_flow_curvature: float,
    temporal_baselines: Sequence[int],
    min_track_displacement_px: float,
    max_track_displacement_px: float,
    phase_search_steps: int,
    phase_period_frames: float,
    spatial_grid: int,
    bootstrap_samples: int,
    phase_estimator: str,
    hybrid_max_disagreement_frames: float,
    rs_mode: str,
    capsule_a: Optional[np.ndarray],
    capsule_b: Optional[np.ndarray],
    capsule_radius: Optional[float],
    *,
    roi: Optional[Dict[int, Sequence[float]]] = None,
    debug_sink: Optional[dict] = None,
    rs_fixed_readout_frames: float = 0.0,
) -> Optional[dict]:
    """One frame/pair phase measurement.

    Diagnostics-only options (both None = unchanged production behaviour):
      roi         {camera index: (x0, y0, x1, y1)} raw-image pixel boxes for THIS frame. A RoMa match is
                  kept only if its point lies inside the box of each camera of the pair that has one;
                  the equal-weight spatial cells are then laid over camera B's box instead of the image.
      debug_sink  dict filled with stage counts, the per-point arrays of the accepted matches and, for
                  the direct estimator, the full phase cost curve (also on censored/unstable returns).
    rs_mode 'fixed' with rs_fixed_readout_frames = rho (frames, rows = readout axis, raw image rows): every
    point's phase is converted to the MIDPOINT reference before aggregation, delta_mid = delta_i -
    rho * (v_a/H_a - v_b/H_b) (trigs.capture_timing convention), so delta_b_frames is the difference of the two
    cameras' midpoint constants. Other modes are unchanged.
    """
    ia, ib = pair
    ca, cb = cameras[ia], cameras[ib]
    dbg = debug_sink
    if dbg is not None:
        dbg.clear()
        dbg["counts"] = {}
    pa = image_path(images_root, ca, frame, image_pattern)
    pb = image_path(images_root, cb, frame, image_pattern)
    try:
        im_a = load_rgb(pa)
        im_b = load_rgb(pb)
    except FileNotFoundError as e:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder, "error": f"image_missing:{e}"}

    try:
        uv_a, uv_b, cert = run_roma_matches(model, pa, pb, im_a, im_b, device, roma_num_matches, roma_min_certainty)
    except Exception as e:
        return {"frame": frame, "cam_a": ia, "cam_b": ib,
                "folder_a": ca.folder, "folder_b": cb.folder, "error": f"roma_failed: {e}"}

    if dbg is not None:
        dbg["counts"]["roma"] = int(len(uv_a))
    if roi is not None:
        in_roi = np.ones(len(uv_a), dtype=bool)
        for ci, uv in ((ia, uv_a), (ib, uv_b)):
            box = roi.get(ci)
            if box is not None:
                x0, y0, x1, y1 = (float(v) for v in box)
                in_roi &= (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        uv_a, uv_b, cert = uv_a[in_roi], uv_b[in_roi], cert[in_roi]
        if dbg is not None:
            dbg["counts"]["roi"] = int(len(uv_a))
    if dbg is not None:
        dbg["roi_uv_a"], dbg["roi_uv_b"], dbg["roi_cert"] = uv_a.copy(), uv_b.copy(), cert.copy()

    if len(uv_a) < 20:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder, "error": f"few_roma_matches:{len(uv_a)}"}

    motion_args = (temporal_baselines, min_track_displacement_px, max_track_displacement_px)
    mdbg_a: Optional[dict] = {} if dbg is not None else None
    mdbg_b: Optional[dict] = {} if dbg is not None else None
    try:
        flow_b, accel_b, valid_b, curv_b, baseline_b = sparse_multiframe_motion(
            images_root, cb, frame, image_pattern, uv_b, *motion_args, debug=mdbg_b)
        flow_a, accel_a, valid_a, curv_a, baseline_a = sparse_multiframe_motion(
            images_root, ca, frame, image_pattern, uv_a, *motion_args, debug=mdbg_a)
    except FileNotFoundError as e:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder, "error": f"track_image_missing:{e}"}

    flow_mag = np.maximum(np.linalg.norm(flow_a, axis=1), np.linalg.norm(flow_b, axis=1))
    keep = valid_b & valid_a & (flow_mag >= float(min_flow_px))
    curv_all = np.fmax(curv_a, curv_b)
    if dbg is not None:
        dbg["roi_valid_a"], dbg["roi_valid_b"] = valid_a.copy(), valid_b.copy()
        dbg["roi_flow_mag"], dbg["roi_curv"] = flow_mag.copy(), curv_all.copy()
        dbg["track_a"], dbg["track_b"] = mdbg_a, mdbg_b
        c = dbg["counts"]
        c["tracks_valid_both"] = int((valid_a & valid_b).sum())
        c["flow_ge_min"] = int(keep.sum())

    # Constant-velocity guard: drop points whose acceleration is large relative
    # to their velocity. curvature/velocity > threshold means the +/-1 frame
    # linearization is unreliable at this point (impacts, direction reversals).
    if max_flow_curvature is not None and max_flow_curvature > 0:
        curv = np.fmax(curv_a, curv_b)
        keep &= ~(np.isfinite(curv) & (curv > float(max_flow_curvature)))
    if dbg is not None:
        dbg["counts"]["curvature_ok"] = int(keep.sum())
    src_idx = np.nonzero(keep)[0]  # index into the (ROI-filtered) RoMa set, for diagnostics

    if keep.sum() < 20:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder,
                "error": f"few_flow_matches:{int(keep.sum())}"}

    uv_a = uv_a[keep]
    uv_b = uv_b[keep]
    flow_a, flow_b = flow_a[keep], flow_b[keep]
    accel_a, accel_b = accel_a[keep], accel_b[keep]
    baseline_a, baseline_b = baseline_a[keep], baseline_b[keep]
    flow_mag = flow_mag[keep]
    cert = cert[keep]

    if capsule_a is not None and capsule_b is not None and capsule_radius is not None:
        cap_keep = np.zeros((len(uv_a),), dtype=bool)
        # Limit triangulation cost if there are many matches: still deterministic.
        for i, (ua, ub) in enumerate(zip(uv_a, uv_b)):
            X = triangulate_dlt([(ca, ua), (cb, ub)])
            if X is not None and inside_capsule(X, capsule_a, capsule_b, capsule_radius):
                cap_keep[i] = True
        uv_a, uv_b = uv_a[cap_keep], uv_b[cap_keep]
        flow_a, flow_b = flow_a[cap_keep], flow_b[cap_keep]
        accel_a, accel_b = accel_a[cap_keep], accel_b[cap_keep]
        baseline_a, baseline_b = baseline_a[cap_keep], baseline_b[cap_keep]
        flow_mag, cert = flow_mag[cap_keep], cert[cap_keep]
        src_idx = src_idx[cap_keep]
        if dbg is not None:
            dbg["counts"]["capsule"] = int(len(uv_a))
        if len(uv_a) < 20:
            return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                    "folder_a": ca.folder, "folder_b": cb.folder,
                    "error": f"few_capsule_matches:{len(uv_a)}"}

    E = essential_matrix(ca, cb)

    f_scale = 0.5 * (
        0.5 * float(ca.K[0, 0] + ca.K[1, 1]) +
        0.5 * float(cb.K[0, 0] + cb.K[1, 1]))

    def trajectory(uv, velocity, acceleration, t):
        return uv + velocity * float(t) + 0.5 * acceleration * float(t * t)

    def signed_sampson_px(ua, ub):
        xa = ca.undistort_norm(ua)
        xb = cb.undistort_norm(ub)
        xa_h = np.concatenate([xa, np.ones((len(xa), 1))], axis=1)
        xb_h = np.concatenate([xb, np.ones((len(xb), 1))], axis=1)
        Exa = (E @ xa_h.T).T
        Etxb = (E.T @ xb_h.T).T
        e = np.sum(xb_h * Exa, axis=1)
        denom = np.sqrt(Exa[:, 0] ** 2 + Exa[:, 1] ** 2 +
                        Etxb[:, 0] ** 2 + Etxb[:, 1] ** 2 + 1e-12)
        return e / denom * f_scale

    r0 = signed_sampson_px(uv_a, uv_b)
    r_minus = signed_sampson_px(
        trajectory(uv_a, flow_a, accel_a, +0.25),
        trajectory(uv_b, flow_b, accel_b, -0.25))
    r_plus = signed_sampson_px(
        trajectory(uv_a, flow_a, accel_a, -0.25),
        trajectory(uv_b, flow_b, accel_b, +0.25))
    # r_minus and r_plus correspond to candidate phases -0.5 and +0.5,
    # respectively: their separation is one full frame, not half a frame.
    de_px = np.abs(r_plus - r_minus)
    de_signed_px = r_plus - r_minus
    sampson0 = (r0 / max(f_scale, 1e-9)) ** 2
    finite = np.isfinite(r0) & np.isfinite(de_px) & np.isfinite(sampson0)
    finite &= de_px >= float(min_epipolar_motion_px)
    if dbg is not None:
        dbg["counts"]["epipolar_motion_ok"] = int(finite.sum())
    if finite.sum() < 15:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder,
                "error": f"few_epipolar_candidates:{int(finite.sum())}"}

    uv_a, uv_b = uv_a[finite], uv_b[finite]
    flow_a, flow_b = flow_a[finite], flow_b[finite]
    accel_a, accel_b = accel_a[finite], accel_b[finite]
    baseline_a, baseline_b = baseline_a[finite], baseline_b[finite]
    flow_mag, cert = flow_mag[finite], cert[finite]
    r0, de_px, de_signed_px, sampson0 = r0[finite], de_px[finite], de_signed_px[finite], sampson0[finite]
    src_idx = src_idx[finite]

    period = float(phase_period_frames)
    search_half = min(float(max_abs_shift_frames), 0.5 * period)
    point_weights = np.maximum(cert.astype(np.float64) * np.clip(de_px, 0.0, 20.0), 1e-6)
    grid_n = max(int(spatial_grid), 2)
    box_b = roi.get(ib) if roi is not None else None
    if box_b is not None:  # diagnostics: equal-weight cells over camera B's ROI, not the whole image
        bx0, by0, bx1, by1 = (float(v) for v in box_b)
        gx = np.clip(((uv_b[:, 0] - bx0) / max(bx1 - bx0, 1.0) * grid_n).astype(int), 0, grid_n - 1)
        gy = np.clip(((uv_b[:, 1] - by0) / max(by1 - by0, 1.0) * grid_n).astype(int), 0, grid_n - 1)
    else:
        gx = np.clip((uv_b[:, 0] / max(float(im_b.shape[1]), 1.0) * grid_n).astype(int), 0, grid_n - 1)
        gy = np.clip((uv_b[:, 1] / max(float(im_b.shape[0]), 1.0) * grid_n).astype(int), 0, grid_n - 1)
    cell_ids = gy * grid_n + gx
    cells = np.unique(cell_ids)
    if dbg is not None:
        dbg["accepted"] = dict(
            src_idx=src_idx.copy(), uv_a=uv_a.copy(), uv_b=uv_b.copy(), flow_a=flow_a.copy(), flow_b=flow_b.copy(),
            accel_a=accel_a.copy(), accel_b=accel_b.copy(), baseline_a=baseline_a.copy(), baseline_b=baseline_b.copy(),
            flow_mag=flow_mag.copy(), cert=cert.copy(), weight=point_weights.copy(), cell=cell_ids.copy(),
            r0_px=r0.copy(), de_signed_px=de_signed_px.copy())
        dbg["period_frames"], dbg["search_half_frames"] = period, search_half
    if len(cells) < 3:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder,
                "error": f"few_spatial_cells:{len(cells)}"}

    # Primary estimator: form a phase for each spatial cell from the local
    # epipolar derivative, then combine cells with equal influence. This keeps
    # thousands of correlated matches on one limb from overwhelming the frame.
    delta_all = -r0 / np.where(np.abs(de_signed_px) < 1e-9, np.nan, de_signed_px)
    rs_fixed = float(rs_fixed_readout_frames) if rs_mode == "fixed" else 0.0
    # per-point readout term (frames) between the two cameras' raw rows; zero unless rs_mode == 'fixed'
    rs_pt = rs_fixed * ((uv_a[:, 1].astype(np.float64) / float(im_a.shape[0]))
                        - (uv_b[:, 1].astype(np.float64) / float(im_b.shape[0])))
    if rs_fixed != 0.0:
        delta_all = delta_all - rs_pt
    delta_all = wrap_period(delta_all, period)
    lin_ok = np.isfinite(delta_all) & (np.abs(delta_all) <= search_half + 1e-6)
    if lin_ok.sum() < 15:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder,
                "error": f"few_linear_phase_candidates:{int(lin_ok.sum())}"}
    delta = delta_all[lin_ok]
    weights = point_weights[lin_ok]
    linear_cell_ids = cell_ids[lin_ok]
    linear_cells = np.unique(linear_cell_ids)
    if len(linear_cells) < 3:
        return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                "folder_a": ca.folder, "folder_b": cb.folder,
                "error": f"few_linear_phase_cells:{len(linear_cells)}"}
    linear_cell_phase = np.asarray([
        circular_weighted_median(delta[linear_cell_ids == cell],
                                 weights[linear_cell_ids == cell], period)
        for cell in linear_cells
    ], dtype=np.float64)
    equal_cell_weights = np.ones(len(linear_cell_phase), dtype=np.float64)
    linear_med = circular_weighted_median(linear_cell_phase, equal_cell_weights, period)

    rng = np.random.default_rng((int(frame) + 1) * 1000003 + ia * 1009 + ib)
    linear_boot = []
    for _ in range(max(int(bootstrap_samples), 0)):
        sample = rng.integers(0, len(linear_cell_phase), size=len(linear_cell_phase))
        linear_boot.append(circular_weighted_median(
            linear_cell_phase[sample], np.ones(len(sample)), period))
    if linear_boot:
        linear_boot_delta = wrap_period(np.asarray(linear_boot) - linear_med, period)
        linear_se = max(1.4826 * mad(linear_boot_delta, 0.0), 0.005)
    else:
        linear_se = 0.03

    # Optional direct objective. A minimum on either edge is censored, not a
    # precise +/-0.5-frame measurement. Hybrid mode falls back to the linear
    # estimate whenever the direct result is censored or disagrees materially.
    direct_med = float("nan")
    direct_se = float("nan")
    direct_boundary = False
    direct_accepted = False
    if phase_estimator in ("direct", "hybrid"):
        steps = max(int(phase_search_steps), 21)
        phase_grid = np.linspace(-search_half, search_half, steps, endpoint=True)
        losses = np.empty((len(phase_grid), len(uv_a)), dtype=np.float32)
        huber_px = 1.0
        for gi, delta_candidate in enumerate(phase_grid):
            if rs_fixed != 0.0:  # per-point total shift = midpoint delta + readout term
                dpt = (delta_candidate + rs_pt)[:, None]
                ua = uv_a + flow_a * (-0.5 * dpt) + 0.5 * accel_a * (0.5 * dpt) ** 2
                ub = uv_b + flow_b * (0.5 * dpt) + 0.5 * accel_b * (0.5 * dpt) ** 2
            else:
                ua = trajectory(uv_a, flow_a, accel_a, -0.5 * delta_candidate)
                ub = trajectory(uv_b, flow_b, accel_b, +0.5 * delta_candidate)
            ar = np.abs(signed_sampson_px(ua, ub))
            losses[gi] = np.where(ar <= huber_px, 0.5 * ar * ar,
                                  huber_px * (ar - 0.5 * huber_px)).astype(np.float32)
        cell_cost = np.empty((len(cells), len(phase_grid)), dtype=np.float64)
        for ci, cell in enumerate(cells):
            use = cell_ids == cell
            ww = point_weights[use]
            cell_cost[ci] = np.sum(losses[:, use] * ww[None, :], axis=1) / np.sum(ww)
        total_cost = np.mean(cell_cost, axis=0)
        best_idx = int(np.argmin(total_cost))
        direct_boundary = best_idx <= 1 or best_idx >= len(phase_grid) - 2
        direct_med = float(phase_grid[best_idx])
        if dbg is not None:
            # Cost vs candidate delta_b (frames): shifting A by -delta/2 and B by +delta/2 along
            # their local trajectories; delta = (capture time of A) - (capture time of B).
            dbg["phase_grid"], dbg["total_cost"] = phase_grid.copy(), total_cost.copy()
            dbg["cell_cost"], dbg["best_idx"], dbg["direct_boundary"] = cell_cost.copy(), best_idx, bool(direct_boundary)
        if not direct_boundary and 0 < best_idx < len(phase_grid) - 1:
            ym, y0, yp = total_cost[best_idx - 1:best_idx + 2]
            denom_q = ym - 2.0 * y0 + yp
            if np.isfinite(denom_q) and denom_q > 1e-12:
                step = float(phase_grid[1] - phase_grid[0])
                direct_med += float(np.clip(0.5 * (ym - yp) / denom_q, -1.0, 1.0) * step)
        direct_med = float(wrap_period(direct_med, period))
        if not direct_boundary:
            direct_boot = []
            for _ in range(max(int(bootstrap_samples), 0)):
                sample = rng.integers(0, len(cells), size=len(cells))
                idx = int(np.argmin(np.mean(cell_cost[sample], axis=0)))
                if 1 < idx < len(phase_grid) - 2:
                    direct_boot.append(float(phase_grid[idx]))
            if len(direct_boot) >= max(5, int(bootstrap_samples) // 4):
                direct_boot_delta = wrap_period(np.asarray(direct_boot) - direct_med, period)
                direct_se = max(1.4826 * mad(direct_boot_delta, 0.0),
                                float(phase_grid[1] - phase_grid[0]) * 0.5)

    if phase_estimator == "direct":
        if direct_boundary or not np.isfinite(direct_se):
            return {"frame": int(frame), "cam_a": int(ia), "cam_b": int(ib),
                    "folder_a": ca.folder, "folder_b": cb.folder,
                    "error": "phase_search_boundary" if direct_boundary else "unstable_direct_phase"}
        med, phase_se, direct_accepted = direct_med, direct_se, True
    elif phase_estimator == "hybrid" and not direct_boundary and np.isfinite(direct_se) and \
            abs(float(wrap_period(direct_med - linear_med, period))) <= float(hybrid_max_disagreement_frames):
        med, phase_se, direct_accepted = direct_med, max(direct_se, linear_se), True
    else:
        med, phase_se = linear_med, linear_se

    scatter = mad(wrap_period(delta - med, period), 0.0)
    n_used = int(len(uv_a))
    med_raw = float(linear_med)

    # Row coordinate difference for the rolling-shutter model. With a shared
    # readout duration rho (in frames) and vertical readout:
    #   delta_i = delta_mid + rho * (v_a/H_a - v_b/H_b)
    # so the intercept at s=0 is the mid-sensor-referenced offset and the slope
    # is the readout duration. (The +/-0.5 center terms cancel for shared rho.)
    H_a = float(im_a.shape[0])
    H_b = float(im_b.shape[0])
    s_all = (uv_a[:, 1].astype(np.float64) / H_a) - (uv_b[:, 1].astype(np.float64) / H_b)
    s = s_all[lin_ok]

    # Rolling-shutter fit: weighted LSQ of delta ~ c0 + c1 * s, robustified by
    # one reweighting pass on residual inliers. Only attempted when the row
    # spread actually constrains the slope.
    rs_slope = float("nan")
    rs_row_spread = float(np.std(s)) if len(s) else float("nan")
    if rs_mode != "off" and len(delta) >= 20 and rs_row_spread > 0.05:
        def _wfit(dd, ss, ww):
            A2 = np.stack([np.ones_like(ss), ss], axis=1)
            sw = np.sqrt(np.maximum(ww, 1e-9))
            c, *_ = np.linalg.lstsq(A2 * sw[:, None], dd * sw, rcond=None)
            return c

        c = _wfit(delta, s, weights)
        resid = delta - (c[0] + c[1] * s)
        rmed = float(np.median(resid))
        rmad = mad(resid, rmed)
        fit_inlier = np.abs(resid - rmed) <= max(0.10, 3.0 * rmad)
        if fit_inlier.sum() >= 15:
            c = _wfit(delta[fit_inlier], s[fit_inlier], weights[fit_inlier])
            resid = delta - (c[0] + c[1] * s)
        rs_slope = float(c[1])
        if rs_mode == "detrend":
            med = float(wrap_period(c[0], period))
            use = fit_inlier if fit_inlier.sum() >= 15 else np.ones(len(delta), bool)
            scatter = mad(resid[use])

    return {
        "frame": int(frame),
        "cam_a": int(ia),
        "cam_b": int(ib),
        "folder_a": ca.folder,
        "folder_b": cb.folder,
        # delta_b_frames: shift camera B's frame coordinate by this many frames
        # to make it geometrically simultaneous with camera A at the same nominal frame.
        # In --rs-mode detrend this is referenced to the sensor mid-row of both cameras.
        "delta_b_frames": float(med),
        "delta_b_frames_raw_median": med_raw,
        "mad_frames": float(scatter),
        "phase_se_frames": float(phase_se),
        "phase_estimator": "direct" if direct_accepted else "linear",
        "linear_phase_frames": float(linear_med),
        "linear_phase_se_frames": float(linear_se),
        "direct_phase_frames": float(direct_med),
        "direct_phase_se_frames": float(direct_se),
        "direct_phase_boundary": bool(direct_boundary),
        "n_used": int(n_used),
        "n_dynamic_matches": int(len(delta)),
        "n_spatial_cells": int(len(cells)),
        "median_flow_px": float(np.median(flow_mag)),
        "median_epi_motion_px": float(np.median(de_px)),
        "median_sampson0": float(np.median(sampson0)),
        "median_track_baseline_frames": float(np.median(np.minimum(baseline_a, baseline_b))),
        # Rolling-shutter diagnostics: slope in frames of shared readout time.
        "rs_slope_frames": rs_slope,
        "rs_row_spread": rs_row_spread,
    }


# -----------------------------------------------------------------------------
# Global solve
# -----------------------------------------------------------------------------

def aggregate_pair_rows(
    rows: List[dict],
    fps: float,
    min_obs: int,
    min_frames: int = 3,
    max_mad_frames: float = 0.20,
    phase_period_frames: float = 1.0,
) -> List[dict]:
    grouped: Dict[Tuple[int, int], List[dict]] = {}
    for r in rows:
        if "delta_b_frames" not in r:
            continue
        grouped.setdefault((int(r["cam_a"]), int(r["cam_b"])), []).append(r)

    out = []
    for (a, b), rs in grouped.items():
        vals = np.asarray([r["delta_b_frames"] for r in rs], dtype=np.float64)
        ns = np.asarray([r.get("n_used", 0) for r in rs], dtype=np.float64)
        scat = np.asarray([max(float(r.get("mad_frames", 0.25)), 0.03) for r in rs], dtype=np.float64)
        frame_se = np.asarray([max(float(r.get("phase_se_frames", 0.10)), 0.01) for r in rs], dtype=np.float64)
        valid = np.isfinite(vals) & np.isfinite(frame_se) & (ns >= max(5, min_obs // 4))
        vals, ns, scat, frame_se = vals[valid], ns[valid], scat[valid], frame_se[valid]
        if len(vals) == 0:
            continue
        # One temporal window gets one uncertainty-derived vote. Dense matches
        # are spatially correlated, so n_used must not dominate aggregation.
        w = np.minimum(1.0 / np.maximum(frame_se, 0.02) ** 2, 2500.0)
        med = circular_weighted_median(vals, w, phase_period_frames)
        total_n = int(np.sum(ns))
        if total_n < min_obs or len(vals) < int(min_frames):
            continue
        # Cross-frame MAD is only meaningful with >= 2 frames. A single frame
        # has MAD = 0 by construction, which previously earned it the *maximum*
        # scatter-based weight. Fall back to the (pessimistic) in-frame scatter.
        if len(vals) >= 2:
            deviations = np.abs(wrap_period(vals - med, phase_period_frames))
            m = mad(deviations, 0.0)
            p80 = float(np.quantile(deviations, 0.80))
        else:
            m = max(float(np.median(scat)), 0.15)
            p80 = m
        # MAD alone can hide one contradictory observation when an edge has
        # only 3-5 windows. The upper robust deviation catches that multimodality.
        if max_mad_frames > 0 and max(m, p80) > float(max_mad_frames):
            continue
        # Combine block-bootstrap precision with observed window-to-window
        # heterogeneity. This is deliberately conservative for sparse edges.
        sigma_from_frames = 1.0 / math.sqrt(float(np.sum(w)))
        sigma_from_scatter = 1.4826 * m / math.sqrt(max(len(vals), 1))
        pair_sigma_frames = max(sigma_from_frames, sigma_from_scatter, 0.005)
        slopes = np.asarray([
            r.get("rs_slope_frames") for r in rs
            if r.get("rs_slope_frames") is not None and np.isfinite(r.get("rs_slope_frames", float("nan")))
        ], dtype=np.float64)
        # plus convention equation: offset_a - offset_b = delta_b_frames / fps
        out.append({
            "cam_a": a,
            "cam_b": b,
            "folder_a": rs[0].get("folder_a", ""),
            "folder_b": rs[0].get("folder_b", ""),
            "delta_b_frames": float(med),
            "delta_sec_a_minus_b": float(med / fps),
            "mad_frames_across_frames": float(m),
            "p80_deviation_frames": float(p80),
            "pair_sigma_frames": float(pair_sigma_frames),
            "pair_sigma_ms": float(pair_sigma_frames / fps * 1000.0),
            "n_total": total_n,
            "n_frames": int(len(vals)),
            "rs_slope_frames_median": float(np.median(slopes)) if len(slopes) else float("nan"),
            "weight": float(1.0 / max(pair_sigma_frames, 1e-6) ** 2),
        })
    return out


def solve_offsets(pair_estimates: List[dict], n_cams: int, ref_cam: int) -> Tuple[np.ndarray, dict]:
    rows = []
    rhs = []
    weights = []
    for e in pair_estimates:
        a = int(e["cam_a"]); b = int(e["cam_b"])
        d = float(e["delta_sec_a_minus_b"])
        w = float(e.get("weight", 1.0))
        row = np.zeros((n_cams,), dtype=np.float64)
        row[a] = 1.0
        row[b] = -1.0
        rows.append(row)
        rhs.append(d)
        weights.append(math.sqrt(max(w, 1e-6)))

    # Gauge fix: ref_cam offset = 0 with high weight.
    row = np.zeros((n_cams,), dtype=np.float64)
    row[int(ref_cam)] = 1.0
    rows.append(row)
    rhs.append(0.0)
    weights.append(1e6)

    A = np.stack(rows, axis=0)
    y = np.asarray(rhs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    Aw = A * w[:, None]
    yw = y * w
    x, residuals, rank, s = np.linalg.lstsq(Aw, yw, rcond=None)

    # Median-center for compatibility with existing sync.json style.
    x = x - np.median(x)

    # Diagnostics.
    resids = []
    for e in pair_estimates:
        a = int(e["cam_a"]); b = int(e["cam_b"])
        pred = x[a] - x[b]
        obs = float(e["delta_sec_a_minus_b"])
        resids.append(pred - obs)
    resids_np = np.asarray(resids, dtype=np.float64) if resids else np.zeros((0,), dtype=np.float64)
    diag = {
        "rank": int(rank),
        "pair_count": int(len(pair_estimates)),
        "residual_median_ms": float(np.median(resids_np) * 1000.0) if len(resids_np) else 0.0,
        "residual_mad_ms": float(mad(resids_np) * 1000.0) if len(resids_np) else 0.0,
        "residual_rms_ms": float(np.sqrt(np.mean(resids_np ** 2)) * 1000.0) if len(resids_np) else 0.0,
    }
    return x, diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual epipolar sync estimator for calibrated multicam footage.")
    ap.add_argument("--reference-json", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--image-pattern", default="{frame:06d}.jpg")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--frames", default=None, help="Comma-separated frame list, e.g. 150,220,300")
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roma-backend", choices=("v1", "v2"), default="v2",
                    help="RoMa implementation: v1 uses romatch; v2 uses romav2/DINOv3.")
    ap.add_argument("--roma-model", choices=("outdoor", "indoor", "tiny-outdoor"), default="outdoor",
                    help="RoMa v1 checkpoint (ignored by v2).")
    ap.add_argument("--roma-v2-setting", choices=("turbo", "fast", "base", "precise"), default="precise",
                    help="RoMa v2 inference setting; precise is highest quality and slowest.")
    ap.add_argument("--half", action="store_true",
                    help="Use fp16 weights with RoMa v1 only; ignored by RoMa v2.")
    ap.add_argument("--camera-neighbors", type=int, default=6)
    ap.add_argument("--wide-baseline-per-camera", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=120, help="Limit pair count for fast validation. 0 = all pairs.")
    ap.add_argument("--pairs", default=None, help="Explicit pair list like '01-02,01-03' or '0000-0001'.")
    ap.add_argument("--roma-num-matches", type=int, default=4000)
    ap.add_argument("--roma-min-certainty", type=float, default=0.10)
    ap.add_argument("--min-flow-px", type=float, default=1.0, help="Keep only moving features.")
    ap.add_argument("--max-abs-shift-frames", type=float, default=0.5,
                    help="Half-width of phase search. Capped at half --phase-period-frames.")
    ap.add_argument("--min-epipolar-slope", type=float, default=0.0,
                    help="DEPRECATED (scene-scale dependent). Use --min-epipolar-motion-px. 0 disables.")
    ap.add_argument("--min-epipolar-motion-px", type=float, default=0.25,
                    help="Min epipolar-normal motion in camera-B pixels per frame of shift. Scale-invariant signal-strength gate.")
    ap.add_argument("--max-flow-curvature", type=float, default=1.0,
                    help="Reject points with |accel|/|velocity| above this (constant-velocity guard). 0 disables.")
    ap.add_argument("--temporal-baselines", default="1,2,4,8",
                    help="Symmetric tracking half-windows used for adaptive slow-motion trajectories.")
    ap.add_argument("--min-track-displacement-px", type=float, default=3.0,
                    help="Prefer the shortest baseline with at least this end-to-end displacement.")
    ap.add_argument("--max-track-displacement-px", type=float, default=40.0)
    ap.add_argument("--phase-search-steps", type=int, default=81)
    ap.add_argument("--phase-period-frames", type=float, default=1.0,
                    help="Frame phase period; visual phase differences are solved modulo this value.")
    ap.add_argument("--phase-estimator", choices=["linear", "hybrid", "direct"], default="linear",
                    help="Production default is spatially bootstrapped linear phase. Direct search is optional and boundary-censored.")
    ap.add_argument("--hybrid-max-disagreement-frames", type=float, default=0.15)
    ap.add_argument("--spatial-grid", type=int, default=6,
                    help="Grid width/height for equal-cell robust costs and block bootstrap.")
    ap.add_argument("--bootstrap-samples", type=int, default=48)
    ap.add_argument("--rs-readout-ms", type=float, default=0.0,
                    help="With --rs-mode fixed: known row readout (ms, full height, >0 = lower rows later); phases "
                         "become midpoint-referenced (trigs.capture_timing convention).")
    ap.add_argument("--rs-mode", choices=["off", "report", "detrend", "fixed"], default="report",
                    help="Rolling shutter: 'report' fits the row-dependent term per pair for diagnostics only; 'detrend' also removes it and references offsets to the sensor mid-row.")
    ap.add_argument("--min-pair-observations", type=int, default=60)
    ap.add_argument("--min-pair-frames", type=int, default=4)
    ap.add_argument("--max-pair-mad-frames", type=float, default=0.30)
    ap.add_argument("--ref-cam", type=int, default=0)
    ap.add_argument("--stage-capsule-a", default=None)
    ap.add_argument("--stage-capsule-b", default=None)
    ap.add_argument("--stage-capsule-radius", type=float, default=None)
    ap.add_argument("--output-json", default="sync_visual_plus.json")
    ap.add_argument("--output-selfcap-json", default="sync_visual_selfcap.json")
    ap.add_argument("--pair-csv", default="visual_sync_pair_observations.csv")
    ap.add_argument("--cache-csv", default=None,
                    help="Persistent superset cache used by --resume. When omitted, --pair-csv is used directly.")
    ap.add_argument("--pair-summary-csv", default="visual_sync_pair_summary.csv")
    ap.add_argument("--report-json", default="visual_sync_report.json")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed frame/pair rows from --cache-csv (or --pair-csv). Use only when estimator settings and inputs are unchanged.")
    ap.add_argument("--max-new-measurements", type=int, default=0,
                    help="Abort before loading RoMa if more than this many uncached frame/pair jobs are planned. 0 disables the cap.")
    argv = sys.argv[1:]
    for option in ("--stage-capsule-a", "--stage-capsule-b"):
        if option in argv:
            i = argv.index(option)
            if i + 1 < len(argv):
                argv[i:i + 2] = [f"{option}={argv[i + 1]}"]
    args = ap.parse_args(argv)

    cameras, scene_meta = load_reference_export(args.reference_json)
    frames = parse_frames(args)
    if not frames:
        raise SystemExit("No frames selected.")

    pairs = parse_pairs(args.pairs, cameras)
    if pairs is None:
        pairs = choose_camera_pairs(cameras, args.camera_neighbors, args.wide_baseline_per_camera)
    if args.max_pairs and args.max_pairs > 0 and len(pairs) > args.max_pairs:
        # Spread pair selection across the ring rather than taking only the first block.
        idx = np.linspace(0, len(pairs) - 1, args.max_pairs).round().astype(int)
        pairs = [pairs[int(i)] for i in idx]

    cap_a = parse_vec3(args.stage_capsule_a)
    cap_b = parse_vec3(args.stage_capsule_b)
    cap_r = args.stage_capsule_radius
    if (cap_a is None) != (cap_b is None):
        raise ValueError("Provide both --stage-capsule-a and --stage-capsule-b")

    if args.rs_mode == "fixed" and float(args.rs_readout_ms) == 0.0:
        raise SystemExit("--rs-mode fixed needs a nonzero --rs-readout-ms")
    temporal_baselines = [int(v.strip()) for v in args.temporal_baselines.split(",") if v.strip()]
    if not temporal_baselines or min(temporal_baselines) <= 0:
        raise ValueError("--temporal-baselines must contain positive integers")

    requested = {(int(frame), int(pair[0]), int(pair[1])) for frame in frames for pair in pairs}
    cache_path = Path(args.cache_csv) if args.cache_csv else Path(args.pair_csv)
    cached_rows = read_cached_observations(cache_path) if args.resume else []
    cached_by_key = {
        (int(row["frame"]), int(row["cam_a"]), int(row["cam_b"])): row
        for row in cached_rows
        if (int(row["frame"]), int(row["cam_a"]), int(row["cam_b"])) in requested
    }
    obs_rows: List[dict] = list(cached_by_key.values())
    missing = [(frame, pair) for frame in frames for pair in pairs
               if (int(frame), int(pair[0]), int(pair[1])) not in cached_by_key]
    print(f"Cameras={len(cameras)} pairs={len(pairs)} frames={frames[:8]}{'...' if len(frames)>8 else ''}")
    print(f"Frame/pair work units: {len(requested)} total, {len(missing)} to compute")
    if cached_by_key:
        print(f"Resuming {len(cached_by_key)}/{len(requested)} cached frame/pair measurements; {len(missing)} remain")
    if args.max_new_measurements > 0 and len(missing) > args.max_new_measurements:
        raise SystemExit(
            f"Refusing {len(missing)} uncached frame/pair measurements (cap "
            f"{args.max_new_measurements}). Reduce frames/pairs, resume a compatible "
            "cache, or explicitly raise --max-new-measurements.")

    # Preserve the full ledger on subset/expansion runs, including error rows.
    checkpoint_by_key = {
        (int(row["frame"]), int(row["cam_a"]), int(row["cam_b"])): row
        for row in cached_rows
    }
    progress_path = cache_path.with_suffix(".progress.json")
    if missing:
        # A fresh run must not leave an older cache eligible for resume if it
        # fails before its first completed pair.
        write_observations_atomic(cache_path, list(checkpoint_by_key.values()))
        progress_path.write_text(json.dumps({"status": "loading_model"}, indent=2), encoding="utf-8")
        print(f"Checkpoint: {cache_path}; current work: {progress_path}", flush=True)
        print(f"Runtime: Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, "
              f"NumPy {np.__version__}, PyTorch {torch.__version__}, "
              f"CUDA {torch.version.cuda}, OpenCV threads={cv2.getNumThreads()}", flush=True)

    model = None
    if missing:
        print("Loading RoMa...")
        try:
            create_roma_backend = load_roma_backend_factory()
            model = create_roma_backend(
                backend=args.roma_backend,
                device=args.device,
                model=args.roma_model,
                v2_setting=args.roma_v2_setting,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            package = "romav2" if args.roma_backend == "v2" else "romatch"
            raise SystemExit(
                "RoMa is required for cross-camera correspondence. Run this command in the "
                f"project environment where `import {package}` succeeds."
            ) from exc
        if args.half and args.roma_backend == "v1" and args.device.startswith("cuda") and torch.cuda.is_available():
            model._model = model._model.half()
        elif args.half:
            print("[warn] --half is ignored by RoMa v2.")
        print(f"RoMa backend: {model.name}")

    by_frame: Dict[int, List[Tuple[int, int]]] = {}
    for frame, pair in missing:
        by_frame.setdefault(int(frame), []).append(pair)
    for frame in tqdm(sorted(by_frame), desc="frames"):
        for pair in tqdm(by_frame[frame], desc=f"pairs f{frame}", leave=False):
            progress_path.write_text(json.dumps({
                "status": "measuring", "frame": int(frame),
                "cam_a": int(pair[0]), "cam_b": int(pair[1]),
                "folder_a": cameras[pair[0]].folder, "folder_b": cameras[pair[1]].folder,
                "completed_requested": len(obs_rows), "total_requested": len(requested),
            }, indent=2), encoding="utf-8")
            r = estimate_pair_frame(
                model, cameras, args.images_root, args.image_pattern, frame, pair, args.device,
                args.roma_num_matches, args.roma_min_certainty, args.min_flow_px,
                args.max_abs_shift_frames, args.min_epipolar_slope,
                args.min_epipolar_motion_px, args.max_flow_curvature,
                temporal_baselines, args.min_track_displacement_px,
                args.max_track_displacement_px, args.phase_search_steps,
                args.phase_period_frames, args.spatial_grid, args.bootstrap_samples,
                args.phase_estimator, args.hybrid_max_disagreement_frames,
                args.rs_mode, cap_a, cap_b, cap_r,
                rs_fixed_readout_frames=(float(args.rs_readout_ms) * float(args.fps) / 1000.0
                                         if args.rs_mode == "fixed" else 0.0),
            )
            if r is not None:
                obs_rows.append(r)
                checkpoint_by_key[(int(frame), int(pair[0]), int(pair[1]))] = r
                write_observations_atomic(cache_path, list(checkpoint_by_key.values()))
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if missing:
        progress_path.write_text(json.dumps({
            "status": "measurements_complete", "completed_requested": len(obs_rows),
            "total_requested": len(requested),
        }, indent=2), encoding="utf-8")

    # Write only the active request to the observation CSV.
    write_observations_atomic(Path(args.pair_csv), obs_rows)

    # The active CSV intentionally contains only this invocation's requested
    # rows. Keep a separate superset ledger so a cheap subset run cannot erase
    # measurements that a later expansion can reuse.
    if args.cache_csv:
        write_observations_atomic(cache_path,
                                  [checkpoint_by_key[key] for key in sorted(checkpoint_by_key)])

    pair_est = aggregate_pair_rows(
        obs_rows, args.fps, args.min_pair_observations,
        min_frames=args.min_pair_frames,
        max_mad_frames=args.max_pair_mad_frames,
        phase_period_frames=args.phase_period_frames,
    )
    with open(args.pair_summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["cam_a", "cam_b", "folder_a", "folder_b", "delta_b_frames", "delta_sec_a_minus_b",
                  "mad_frames_across_frames", "p80_deviation_frames", "pair_sigma_frames", "pair_sigma_ms",
                  "n_total", "n_frames", "rs_slope_frames_median", "weight"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_est)

    if len(pair_est) < max(3, len(cameras) // 4):
        print(f"[warn] Only {len(pair_est)} usable pair estimates. Add frames/pairs or lower --min-pair-observations.")

    offsets, diag = solve_offsets(pair_est, len(cameras), args.ref_cam)

    # Warn about cameras with no constraining pair estimate: their solved offset
    # is gauge-arbitrary, not measured.
    constrained = set()
    for e in pair_est:
        constrained.add(int(e["cam_a"]))
        constrained.add(int(e["cam_b"]))
    unconstrained = [cameras[i].folder for i in range(len(cameras)) if i not in constrained]
    if unconstrained:
        print(f"[warn] {len(unconstrained)} camera(s) have NO pair estimates; their offsets are meaningless: {unconstrained}")

    plus = {}
    for i in range(len(cameras)):
        folder = str(cameras[i].folder)
        key = f"{int(folder):04d}" if folder.isdigit() else folder
        plus[key] = float(offsets[i])

    # Rolling-shutter diagnostic across pairs.
    rs_slopes = np.asarray([e["rs_slope_frames_median"] for e in pair_est
                            if np.isfinite(e.get("rs_slope_frames_median", float("nan")))], dtype=np.float64)
    rs_summary = {}
    if len(rs_slopes) >= 5:
        rs_med = float(np.median(rs_slopes))
        rs_mad = float(mad(rs_slopes))
        readout_ms = rs_med / args.fps * 1000.0
        rs_summary = {
            "pairs_with_slope": int(len(rs_slopes)),
            "slope_median_frames": rs_med,
            "slope_mad_frames": rs_mad,
            "implied_shared_readout_ms": readout_ms,
            "mode": args.rs_mode,
        }
        significant = abs(rs_med) > 0.05 and abs(rs_med) > 2.0 * rs_mad
        rs_summary["significant"] = bool(significant)
        print(f"[rs] row-slope median={rs_med:+.3f} frames (MAD {rs_mad:.3f}) across {len(rs_slopes)} pairs "
              f"=> implied shared readout ~{readout_ms:+.2f} ms")
        if significant and args.rs_mode not in ("detrend", "fixed"):
            print("[rs] CONSISTENT rolling-shutter signature detected. Re-run with --rs-mode detrend "
                  "to reference offsets to the sensor mid-row.")
        elif significant:
            print("[rs] rolling-shutter term detected and removed (detrend); offsets are mid-row referenced.")
        else:
            print("[rs] no consistent rolling-shutter signature (slopes are small or incoherent); "
                  "cameras behave as effectively global shutter for sync purposes.")

    selfcap = {k: -v for k, v in plus.items()}
    Path(args.output_json).write_text(json.dumps(plus, indent=4), encoding="utf-8")
    Path(args.output_selfcap_json).write_text(json.dumps(selfcap, indent=4), encoding="utf-8")

    error_counts = {}
    for r in obs_rows:
        if "error" in r:
            key = str(r["error"]).split(":", 1)[0]
            error_counts[key] = error_counts.get(key, 0) + 1

    successful_rows = [r for r in obs_rows if "delta_b_frames" in r]
    direct_boundary_count = sum(bool(r.get("direct_phase_boundary", False)) for r in successful_rows)
    direct_boundary_fraction = direct_boundary_count / max(len(successful_rows), 1)
    output_boundary_count = sum(
        abs(abs(float(r["delta_b_frames"])) - min(args.max_abs_shift_frames, 0.5 * args.phase_period_frames)) < 1e-8
        for r in successful_rows)
    output_boundary_fraction = output_boundary_count / max(len(successful_rows), 1)
    censored_count = error_counts.get("phase_search_boundary", 0) + direct_boundary_count
    censored_fraction = censored_count / max(len(obs_rows), 1)
    estimator_counts = {}
    for r in successful_rows:
        key = str(r.get("phase_estimator", "unknown"))
        estimator_counts[key] = estimator_counts.get(key, 0) + 1

    report = {
        "fps": args.fps,
        "frames": frames,
        "temporal_baselines": temporal_baselines,
        "phase_period_frames": args.phase_period_frames,
        "phase_estimator_requested": args.phase_estimator,
        "phase_estimator_counts": estimator_counts,
        "direct_boundary_diagnostic_count": direct_boundary_count,
        "direct_boundary_diagnostic_fraction": direct_boundary_fraction,
        "censored_phase_count": censored_count,
        "censored_phase_fraction": censored_fraction,
        "output_boundary_count": output_boundary_count,
        "output_boundary_fraction": output_boundary_fraction,
        "stage_capsule": {
            "a": cap_a.tolist() if cap_a is not None else None,
            "b": cap_b.tolist() if cap_b is not None else None,
            "radius": cap_r,
            "enabled": bool(cap_a is not None and cap_b is not None and cap_r is not None),
        },
        "camera_count": len(cameras),
        "attempted_pairs": len(pairs),
        "requested_frame_pair_measurements": len(requested),
        "reused_frame_pair_measurements": len(cached_by_key),
        "computed_frame_pair_measurements": len(missing),
        "cached_frame_pair_measurements": len(cached_rows) + len(missing),
        "raw_observation_rows": len(obs_rows),
        "observation_error_counts": error_counts,
        "usable_pair_estimates": len(pair_est),
        "offset_convention": {
            "output_json": "actual_time = nominal_time + offset_sec[camera]",
            "output_selfcap_json": "actual_time = nominal_time - sync_json[camera]",
        },
        "solve_diagnostics": diag,
        "rolling_shutter": rs_summary,
        "unconstrained_cameras": unconstrained,
        "offset_range_ms": [float(np.min(offsets) * 1000.0), float(np.max(offsets) * 1000.0)],
        "offsets_plus_sec": plus,
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output_json}, {args.output_selfcap_json}")
    print(f"Pair estimates: {len(pair_est)} | residual MAD={diag['residual_mad_ms']:.3f}ms RMS={diag['residual_rms_ms']:.3f}ms")
    print(f"Offset range: {np.min(offsets)*1000:.2f}..{np.max(offsets)*1000:.2f} ms")


if __name__ == "__main__":
    main()
