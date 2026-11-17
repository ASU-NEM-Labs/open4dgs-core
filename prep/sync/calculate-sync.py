#!/usr/bin/env python3
"""One-command calibrated, timecode-anchored multicamera phase synchronization.

Pipeline:
  1. Scan every camera with adaptive multi-frame dense flow.
  2. Select time-distributed frames while guaranteeing per-camera coverage.
  3. Run multi-frame, spatially balanced epipolar phase estimation.
  4. Retry with targeted frames and more camera neighbors when coverage is weak.
  5. Run circular robust graph synchronization and publish only validated output.

The final JSON uses PLUS convention:
    actual_time_sec = nominal_time_sec + offset_sec[camera]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class MotionSample:
    frame: int
    camera: str
    speed_px_per_frame: float
    curvature_ratio: float
    baseline_frames: int
    score: float


def parse_ints(text: str) -> List[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def camera_key(folder: str) -> str:
    folder = str(folder)
    return f"{int(folder):04d}" if folder.isdigit() else folder


def load_cameras(reference_json: Path) -> List[dict]:
    data = json.loads(reference_json.read_text(encoding="utf-8"))
    return list(data["cameras"])


def write_zero_sync(path: Path, cameras: Sequence[dict]) -> Dict[str, float]:
    """Write a complete timecode-only PLUS-convention baseline."""
    zero = {camera_key(camera["folder"]): 0.0 for camera in cameras}
    zero = {key: zero[key] for key in sorted(zero)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zero, indent=4) + "\n", encoding="utf-8")
    return zero


def image_path(images_root: Path, folder: str, pattern: str, frame: int) -> Path:
    return images_root / str(folder) / pattern.format(frame=int(frame))


def flow_field(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        g0, g1, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)


def score_camera_frame(
    load_gray,
    frame: int,
    baselines: Sequence[int],
    downscale: int,
    min_track_displacement_px: float,
) -> Optional[Tuple[float, float, int, float]]:
    g0 = load_gray(frame)
    if g0 is None:
        return None
    fallback = None
    for k in baselines:
        gm, gp = load_gray(frame - k), load_gray(frame + k)
        if gm is None or gp is None:
            continue
        vf = flow_field(g0, gp) / float(k)
        vb = -flow_field(g0, gm) / float(k)
        velocity = 0.5 * (vf + vb)
        speed_small = np.linalg.norm(velocity, axis=2).ravel()
        if speed_small.size == 0:
            continue
        count = max(int(0.02 * speed_small.size), 50)
        idx = np.argpartition(speed_small, speed_small.size - count)[-count:]
        speed = float(np.mean(speed_small[idx])) * downscale
        accel = np.linalg.norm(vf - vb, axis=2).ravel() / float(k)
        curvature = float(np.median(accel[idx] / np.maximum(speed_small[idx], 1e-3)))
        displacement = 2.0 * k * speed
        candidate = (speed, curvature, int(k), speed / (1.0 + max(curvature - 0.25, 0.0) * 2.0))
        fallback = candidate
        if displacement >= min_track_displacement_px:
            return candidate
    return fallback


def scan_motion(
    cameras: Sequence[dict],
    images_root: Path,
    image_pattern: str,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    baselines: Sequence[int],
    downscale: int,
    min_track_displacement_px: float,
) -> List[MotionSample]:
    margin = max(baselines)
    frames = list(range(frame_start + margin, frame_end - margin + 1, frame_step))
    if not frames:
        raise ValueError("scan range is shorter than twice the largest temporal baseline")
    samples: List[MotionSample] = []
    print(f"Scanning {len(frames)} candidate frames across {len(cameras)} cameras...")
    for camera_index, camera in enumerate(cameras):
        folder = str(camera["folder"])

        @lru_cache(maxsize=96)
        def load_gray(frame: int):
            img = cv2.imread(str(image_path(images_root, folder, image_pattern, frame)), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            if downscale > 1:
                h, w = img.shape[:2]
                img = cv2.resize(img, (max(w // downscale, 1), max(h // downscale, 1)),
                                 interpolation=cv2.INTER_AREA)
            return img

        valid_count = 0
        for frame in frames:
            result = score_camera_frame(
                load_gray, frame, baselines, downscale, min_track_displacement_px)
            if result is None:
                continue
            speed, curvature, baseline, score = result
            samples.append(MotionSample(frame, folder, speed, curvature, baseline, score))
            valid_count += 1
        load_gray.cache_clear()
        print(f"  camera {camera_key(folder)}: {valid_count}/{len(frames)} candidates scored "
              f"({camera_index + 1}/{len(cameras)})")
    return samples


def write_scan_csv(path: Path, samples: Sequence[MotionSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera", "speed_px_per_frame", "curvature_ratio",
                         "baseline_frames", "information_proxy"])
        for s in samples:
            writer.writerow([s.frame, camera_key(s.camera), f"{s.speed_px_per_frame:.6f}",
                             f"{s.curvature_ratio:.6f}", s.baseline_frames, f"{s.score:.6f}"])


def read_scan_csv(path: Path) -> List[MotionSample]:
    samples = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(MotionSample(
                frame=int(row["frame"]), camera=str(row["camera"]),
                speed_px_per_frame=float(row["speed_px_per_frame"]),
                curvature_ratio=float(row["curvature_ratio"]),
                baseline_frames=int(row["baseline_frames"]),
                score=float(row["information_proxy"]),
            ))
    return samples


def select_frames(
    samples: Sequence[MotionSample],
    camera_folders: Sequence[str],
    time_bins: int,
    frames_per_bin: int,
    min_gap: int,
    min_camera_speed: float,
    max_curvature: float,
    min_camera_coverage: int,
    min_frame_camera_fraction: float,
    max_frames: int,
) -> Tuple[List[int], Dict[str, int]]:
    by_frame: Dict[int, Dict[str, MotionSample]] = defaultdict(dict)
    for sample in samples:
        by_frame[sample.frame][camera_key(sample.camera)] = sample
    cameras = [camera_key(c) for c in camera_folders]
    frames = sorted(by_frame)
    if not frames:
        raise ValueError("no frames were scored")

    def informative(sample: Optional[MotionSample]) -> bool:
        return bool(sample is not None and sample.speed_px_per_frame >= min_camera_speed and
                    sample.curvature_ratio <= max_curvature)

    def utility(frame: int, targets: Optional[set] = None) -> float:
        rows = by_frame[frame]
        cams = targets if targets is not None else set(cameras)
        useful = [rows.get(c) for c in cams if informative(rows.get(c))]
        if not useful:
            return -math.inf
        coverage = len(useful) / max(len(cams), 1)
        speed_term = float(np.median([min(s.score / max(min_camera_speed, 1e-6), 5.0) for s in useful]))
        return 100.0 * coverage + speed_term

    selected: List[int] = []
    edges = np.linspace(frames[0], frames[-1] + 1, max(time_bins, 1) + 1)
    for bi in range(max(time_bins, 1)):
        candidates = [f for f in frames if edges[bi] <= f < edges[bi + 1]]
        candidates.sort(key=utility, reverse=True)
        taken = 0
        for frame in candidates:
            useful_fraction = sum(informative(by_frame[frame].get(c)) for c in cameras) / max(len(cameras), 1)
            if useful_fraction < min_frame_camera_fraction:
                continue
            if any(abs(frame - other) < min_gap for other in selected):
                continue
            selected.append(frame)
            taken += 1
            if taken >= frames_per_bin or len(selected) >= max_frames:
                break

    def coverage_counts() -> Dict[str, int]:
        return {c: sum(informative(by_frame[f].get(c)) for f in selected) for c in cameras}

    # Greedy set cover for cameras that a global median would hide.
    while len(selected) < max_frames:
        counts = coverage_counts()
        deficient = {c for c, count in counts.items() if count < min_camera_coverage}
        if not deficient:
            break
        candidates = [f for f in frames if f not in selected and
                      all(abs(f - other) >= min_gap for other in selected)]
        if not candidates:
            break
        best = max(candidates, key=lambda f: utility(f, deficient))
        if not np.isfinite(utility(best, deficient)):
            break
        selected.append(best)

    selected.sort()
    return selected, coverage_counts()


def run_command(command: Sequence[str], log_path: Optional[Path] = None) -> None:
    print("\n+ " + " ".join(str(v) for v in command), flush=True)
    if log_path is None:
        subprocess.run([str(v) for v in command], check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(v) for v in command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, [str(v) for v in command])


def read_pair_degrees(summary_csv: Path, cameras: Sequence[str]) -> Dict[str, int]:
    degree = {camera_key(c): 0 for c in cameras}
    if not summary_csv.exists():
        return degree
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = camera_key(row["folder_a"]), camera_key(row["folder_b"])
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    return degree


def targeted_extra_frames(
    samples: Sequence[MotionSample],
    weak_cameras: Sequence[str],
    selected: Sequence[int],
    min_gap: int,
    min_speed: float,
    max_curvature: float,
    count: int,
) -> List[int]:
    weak = {camera_key(c) for c in weak_cameras}
    by_frame: Dict[int, List[MotionSample]] = defaultdict(list)
    for sample in samples:
        if camera_key(sample.camera) in weak and sample.speed_px_per_frame >= min_speed and sample.curvature_ratio <= max_curvature:
            by_frame[sample.frame].append(sample)
    ranked = sorted(by_frame, key=lambda f: (
        len(by_frame[f]), np.median([s.score for s in by_frame[f]])), reverse=True)
    out: List[int] = []
    occupied = list(selected)
    for frame in ranked:
        if all(abs(frame - other) >= min_gap for other in occupied):
            out.append(frame)
            occupied.append(frame)
        if len(out) >= count:
            break
    return sorted(out)


def build_estimator_command(
    args, script: Path, out_dir: Path, frames: Sequence[int], neighbors: int,
    resume: bool = False,
) -> List[str]:
    command = [
        sys.executable, str(script),
        "--reference-json", str(args.reference_json),
        "--images-root", str(args.images_root),
        "--image-pattern", args.image_pattern,
        "--fps", str(args.fps),
        "--frames", ",".join(map(str, frames)),
        "--device", args.device,
        "--camera-neighbors", str(neighbors),
        "--wide-baseline-per-camera", str(args.wide_baseline_per_camera),
        "--max-pairs", "0",
        "--roma-num-matches", str(args.roma_num_matches),
        "--roma-min-certainty", str(args.roma_min_certainty),
        "--min-flow-px", str(args.min_flow_px),
        "--max-abs-shift-frames", str(args.max_abs_shift_frames),
        "--min-epipolar-motion-px", str(args.min_epipolar_motion_px),
        "--max-flow-curvature", str(args.max_flow_curvature),
        "--temporal-baselines", ",".join(map(str, args.temporal_baselines)),
        "--min-track-displacement-px", str(args.min_track_displacement_px),
        "--max-track-displacement-px", str(args.max_track_displacement_px),
        "--phase-search-steps", str(args.phase_search_steps),
        "--phase-period-frames", str(args.phase_period_frames),
        "--phase-estimator", args.phase_estimator,
        "--hybrid-max-disagreement-frames", str(args.hybrid_max_disagreement_frames),
        "--spatial-grid", str(args.spatial_grid),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--max-new-measurements", str(args.max_new_measurements),
        "--rs-mode", args.rs_mode,
        "--rs-readout-ms", str(args.rs_readout_ms),
        "--min-pair-observations", str(args.min_pair_observations),
        "--min-pair-frames", str(args.min_pair_frames),
        "--max-pair-mad-frames", str(args.max_pair_mad_frames),
        "--output-json", str(out_dir / "sync_visual_plus.json"),
        "--output-selfcap-json", str(out_dir / "sync_visual_selfcap.json"),
        "--pair-csv", str(out_dir / "visual_sync_pair_observations.csv"),
        "--cache-csv", str(out_dir / "visual_sync_measurement_cache.csv"),
        "--pair-summary-csv", str(out_dir / "visual_sync_pair_summary.csv"),
        "--report-json", str(out_dir / "visual_sync_report.json"),
    ]
    if args.half:
        command.append("--half")
    if resume:
        command.append("--resume")
    capsule_values = (args.stage_capsule_a, args.stage_capsule_b, args.stage_capsule_radius)
    if not args.disable_stage_capsule and any(v is not None for v in capsule_values):
        if not all(v is not None for v in capsule_values):
            raise ValueError("stage capsule requires --stage-capsule-a, --stage-capsule-b, and --stage-capsule-radius")
        command.extend([
            f"--stage-capsule-a={args.stage_capsule_a}",
            f"--stage-capsule-b={args.stage_capsule_b}",
            "--stage-capsule-radius", str(args.stage_capsule_radius),
        ])
    return command


def grade_at_least(actual: str, required: str) -> bool:
    rank = {"POOR": 0, "UNKNOWN": 0, "MARGINAL": 1, "GOOD": 2, "EXCELLENT": 3}
    return rank.get(actual, 0) >= rank.get(required, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-json", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--image-pattern", default="{frame:06d}.jpg")
    ap.add_argument("--fps", required=True, type=float)
    ap.add_argument("--frame-start", required=True, type=int)
    ap.add_argument("--frame-end", required=True, type=int)
    ap.add_argument("--frame-step", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("sync_run"))
    ap.add_argument("--output-sync", type=Path, default=Path("sync.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action="store_true")

    # Motion scan and coverage selection.
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--temporal-baselines", type=parse_ints, default=parse_ints("1,2,4,8"))
    ap.add_argument("--min-track-displacement-px", type=float, default=3.0)
    ap.add_argument("--max-track-displacement-px", type=float, default=40.0)
    ap.add_argument("--min-camera-speed", type=float, default=0.35)
    ap.add_argument("--scan-max-curvature", type=float, default=1.0)
    ap.add_argument("--time-bins", type=int, default=8)
    ap.add_argument("--frames-per-bin", type=int, default=1)
    ap.add_argument("--min-frame-gap", type=int, default=12)
    ap.add_argument("--min-camera-coverage", type=int, default=3)
    ap.add_argument("--min-frame-camera-fraction", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--scan-only", action="store_true")

    # Visual estimator.
    ap.add_argument("--camera-neighbors", type=int, default=8)
    ap.add_argument("--wide-baseline-per-camera", type=int, default=2)
    ap.add_argument("--roma-num-matches", type=int, default=6000)
    ap.add_argument("--roma-min-certainty", type=float, default=0.10)
    ap.add_argument("--min-flow-px", type=float, default=0.20,
                    help="Velocity threshold after adaptive long-baseline tracking; do not lower casually.")
    ap.add_argument("--min-epipolar-motion-px", type=float, default=0.20)
    ap.add_argument("--max-flow-curvature", type=float, default=1.0)
    ap.add_argument("--max-abs-shift-frames", type=float, default=0.5)
    ap.add_argument("--phase-period-frames", type=float, default=1.0)
    ap.add_argument("--phase-search-steps", type=int, default=81)
    ap.add_argument("--phase-estimator", choices=["linear", "hybrid", "direct"], default="linear")
    ap.add_argument("--hybrid-max-disagreement-frames", type=float, default=0.15)
    ap.add_argument("--max-output-boundary-fraction", type=float, default=0.05,
                    help="Refuse publication if frame estimates pile up exactly on the allowed phase boundary.")
    ap.add_argument("--max-censored-phase-fraction", type=float, default=0.05,
                    help="Refuse publication when too many optional direct searches are boundary-censored.")
    ap.add_argument("--spatial-grid", type=int, default=6)
    ap.add_argument("--bootstrap-samples", type=int, default=48)
    ap.add_argument("--max-new-measurements", type=int, default=3000,
                    help="Safety cap on uncached RoMa frame/pair jobs per round. Raise explicitly for a deliberately large run; 0 disables.")
    ap.add_argument("--rs-mode", choices=["off", "report", "detrend", "fixed"], default="report")
    ap.add_argument("--rs-readout-ms", type=float, default=0.0,
                    help="Known row readout for --rs-mode fixed (ms, full height, >0 = lower rows later).")
    ap.add_argument("--min-pair-observations", type=int, default=40)
    ap.add_argument("--min-pair-frames", type=int, default=5)
    ap.add_argument("--max-pair-mad-frames", type=float, default=0.30)
    ap.add_argument("--max-pair-sigma-frames", type=float, default=0.09)
    ap.add_argument("--max-pair-sigma-ms", type=float, default=None,
                    help="Absolute override for --max-pair-sigma-frames.")
    ap.add_argument("--min-camera-degree", type=int, default=3)
    ap.add_argument("--drop-frames", type=float, default=0.12)
    ap.add_argument("--drop-ms", type=float, default=None,
                    help="Absolute override for --drop-frames.")

    # Stage foreground capsule. All three are passed through every round.
    ap.add_argument("--stage-capsule-a", help="World-space x,y,z endpoint")
    ap.add_argument("--stage-capsule-b", help="World-space x,y,z endpoint")
    ap.add_argument("--stage-capsule-radius", type=float)
    ap.add_argument("--disable-stage-capsule", action="store_true",
                    help="Ignore supplied capsule values for a diagnostic run.")

    # Adaptive retry and publication policy.
    ap.add_argument("--adaptive-rounds", type=int, default=0,
                    help="Expensive coverage expansions after the initial solve. Disabled by default; cached measurements are reused when enabled.")
    ap.add_argument("--adaptive-extra-frames", type=int, default=6)
    ap.add_argument("--adaptive-neighbor-step", type=int, default=4)
    ap.add_argument("--resume-observations", action="store_true",
                    help="Reuse compatible raw frame/pair rows already in --out-dir on the first round. Later adaptive rounds always reuse the current run.")
    ap.add_argument("--min-publish-grade", choices=["MARGINAL", "GOOD", "EXCELLENT"], default="GOOD")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--publish-on-failure", action="store_true",
                    help="Copy the candidate despite validation failure; report still marks it untrusted.")
    ap.add_argument("--disable-zero-fallback", action="store_true",
                    help="Leave --output-sync untouched on validation failure instead of publishing a complete all-zero timecode-only baseline.")
    # argparse treats a comma-separated negative vector as another option.
    # Normalize both common spellings so `--stage-capsule-a -1,2,3` works as
    # well as the unambiguous `--stage-capsule-a=-1,2,3` form.
    argv = sys.argv[1:]
    for option in ("--stage-capsule-a", "--stage-capsule-b"):
        if option in argv:
            i = argv.index(option)
            if i + 1 < len(argv):
                argv[i:i + 2] = [f"{option}={argv[i + 1]}"]
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_cameras(args.reference_json)
    folders = [str(c["folder"]) for c in cameras]
    scan_csv = args.out_dir / "frame_motion_scan_by_camera.csv"
    if args.resume_observations and scan_csv.exists():
        samples = read_scan_csv(scan_csv)
        print(f"Reusing motion scan with {len(samples)} camera/frame rows: {scan_csv}")
    else:
        samples = scan_motion(
            cameras, args.images_root, args.image_pattern, args.frame_start, args.frame_end,
            args.frame_step, args.temporal_baselines, args.downscale,
            args.min_track_displacement_px)
        write_scan_csv(scan_csv, samples)
    frames, coverage = select_frames(
        samples, folders, args.time_bins, args.frames_per_bin, args.min_frame_gap,
        args.min_camera_speed, args.scan_max_curvature, args.min_camera_coverage,
        args.min_frame_camera_fraction, args.max_frames)
    if not frames:
        raise SystemExit("No informative frames met the selection thresholds.")
    print(f"Selected {len(frames)} frames: {','.join(map(str, frames))}")
    weak_scan = sorted(c for c, count in coverage.items() if count < args.min_camera_coverage)
    if weak_scan:
        print(f"[warn] scan could not provide requested coverage for: {weak_scan}")
    (args.out_dir / "selected_frames.json").write_text(json.dumps({
        "frames": frames,
        "per_camera_informative_frame_count": coverage,
        "weak_cameras": weak_scan,
    }, indent=2), encoding="utf-8")
    if args.scan_only:
        return

    here = Path(__file__).resolve().parent
    estimator = here / "visual-epipolar-sync.py"
    solver = here / "robust-sync-graph.py"
    measurement_cache = args.out_dir / "visual_sync_measurement_cache.csv"
    active_observations = args.out_dir / "visual_sync_pair_observations.csv"
    if args.resume_observations and not measurement_cache.exists() and active_observations.exists():
        shutil.copy2(active_observations, measurement_cache)
        print(f"Seeded persistent cache from existing observations: {measurement_cache}")
    rounds = []
    neighbors = args.camera_neighbors
    for round_index in range(args.adaptive_rounds + 1):
        resume = args.resume_observations or round_index > 0
        run_command(build_estimator_command(
            args, estimator, args.out_dir, frames, neighbors, resume=resume))
        visual_report = json.loads((args.out_dir / "visual_sync_report.json").read_text(encoding="utf-8"))
        degree = read_pair_degrees(args.out_dir / "visual_sync_pair_summary.csv", folders)
        weak = sorted(c for c, value in degree.items() if value < args.min_camera_degree)
        rounds.append({"round": round_index, "frames": list(frames), "camera_neighbors": neighbors,
                       "degree": degree, "weak_cameras": weak,
                       "phase_estimator_counts": visual_report.get("phase_estimator_counts", {}),
                       "censored_phase_fraction": visual_report.get("censored_phase_fraction", 0.0),
                       "output_boundary_fraction": visual_report.get("output_boundary_fraction", 0.0)})
        if not weak or round_index >= args.adaptive_rounds:
            break
        extras = targeted_extra_frames(
            samples, weak, frames, args.min_frame_gap, args.min_camera_speed,
            args.scan_max_curvature, args.adaptive_extra_frames)
        neighbors = min(len(cameras) - 1, neighbors + args.adaptive_neighbor_step)
        if not extras and neighbors == rounds[-1]["camera_neighbors"]:
            break
        frames = sorted(set(frames) | set(extras))[:args.max_frames]
        print(f"Adaptive retry for {weak}: added frames {extras}, camera-neighbors={neighbors}")

    candidate = args.out_dir / "sync_candidate.json"
    robust_report = args.out_dir / "sync_robust_report.json"
    command = [
        sys.executable, str(solver),
        "--pair-summary", str(args.out_dir / "visual_sync_pair_summary.csv"),
        "--pair-obs-csv", str(args.out_dir / "visual_sync_pair_observations.csv"),
        "--reference-json", str(args.reference_json),
        "--fps", str(args.fps),
        "--out-json", str(candidate),
        "--report-json", str(robust_report),
        "--drop-frames", str(args.drop_frames),
        "--min-pair-frames", str(args.min_pair_frames),
        "--min-total-observations", str(args.min_pair_observations),
        "--max-pair-mad-frames", str(args.max_pair_mad_frames),
        "--max-pair-sigma-frames", str(args.max_pair_sigma_frames),
        "--min-camera-degree", str(args.min_camera_degree),
        "--phase-period-frames", str(args.phase_period_frames),
        "--omit-unconstrained",
    ]
    if args.drop_ms is not None:
        command += ["--drop-ms", str(args.drop_ms)]
    if args.max_pair_sigma_ms is not None:
        command += ["--max-pair-sigma-ms", str(args.max_pair_sigma_ms)]
    run_command(command, args.out_dir / "robust.log")
    # Keep the historical filename current as well as the explicitly named
    # candidate. A stale sync_robust.json is dangerously easy to mistake for
    # the result of the latest run.
    shutil.copy2(candidate, args.out_dir / "sync_robust.json")
    robust = json.loads(robust_report.read_text(encoding="utf-8"))
    visual_report = json.loads((args.out_dir / "visual_sync_report.json").read_text(encoding="utf-8"))
    complete = robust["cameras_solved"] == robust["cameras_total"]
    grade_ok = grade_at_least(robust["verdict"], args.min_publish_grade)
    boundary_fraction = float(visual_report.get("output_boundary_fraction", 0.0))
    boundary_ok = boundary_fraction <= args.max_output_boundary_fraction
    censored_fraction = float(visual_report.get("censored_phase_fraction", 0.0))
    censored_ok = censored_fraction <= args.max_censored_phase_fraction
    publish = (complete or args.allow_partial) and grade_ok and boundary_ok and censored_ok
    trusted = publish
    fallback_used = False
    output_mode = "validated_visual" if trusted else "none"
    if args.publish_on_failure:
        publish = True
        output_mode = "untrusted_visual_candidate"
    if publish:
        args.output_sync.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, args.output_sync)
        print(f"Published {'validated' if trusted else 'UNTRUSTED'} sync to {args.output_sync}")
    else:
        print(f"NOT publishing sync.json: grade={robust['verdict']} (required {args.min_publish_grade}), "
              f"cameras={robust['cameras_solved']}/{robust['cameras_total']}, "
              f"output_boundary_fraction={boundary_fraction:.1%} "
              f"(maximum {args.max_output_boundary_fraction:.1%}), "
              f"censored_phase_fraction={censored_fraction:.1%} "
              f"(maximum {args.max_censored_phase_fraction:.1%})")
        print(f"Candidate and diagnostics remain in {args.out_dir}")
        if not args.disable_zero_fallback:
            zero_candidate = args.out_dir / "sync_zero.json"
            write_zero_sync(zero_candidate, cameras)
            args.output_sync.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zero_candidate, args.output_sync)
            publish = True
            fallback_used = True
            output_mode = "timecode_only_zero"
            print(f"Published complete all-zero timecode-only fallback to {args.output_sync}")

    capsule_enabled = not args.disable_stage_capsule and args.stage_capsule_a is not None
    pipeline_report = {
        "published": publish,
        "trusted": trusted,
        "fallback_used": fallback_used,
        "output_mode": output_mode,
        "output_sync": str(args.output_sync),
        "selected_frames": frames,
        "scan_csv": str(scan_csv),
        "adaptive_rounds": rounds,
        "stage_capsule": {
            "enabled": capsule_enabled,
            "a": args.stage_capsule_a if capsule_enabled else None,
            "b": args.stage_capsule_b if capsule_enabled else None,
            "radius": args.stage_capsule_radius if capsule_enabled else None,
        },
        "phase_measurement": {
            "requested_estimator": args.phase_estimator,
            "output_boundary_fraction": boundary_fraction,
            "max_output_boundary_fraction": args.max_output_boundary_fraction,
            "boundary_check_passed": boundary_ok,
            "censored_phase_fraction": censored_fraction,
            "max_censored_phase_fraction": args.max_censored_phase_fraction,
            "censoring_check_passed": censored_ok,
            "visual_report": visual_report,
        },
        "validation": robust,
    }
    (args.out_dir / "sync_pipeline_report.json").write_text(
        json.dumps(pipeline_report, indent=2), encoding="utf-8")
    if not trusted and not fallback_used:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
