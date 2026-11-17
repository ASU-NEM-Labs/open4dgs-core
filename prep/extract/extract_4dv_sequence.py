#!/usr/bin/env python3
"""Import a 4DV.ai JCam capture into the TriGS image/sync contract.

The source capture remains untouched.  This adapter pairs JCam videos, record
logs, and reference snapshots; builds a common logical timeline from the
per-frame log timestamps; extracts one contiguous sequence per camera; and
writes the constant sub-frame offsets consumed by RoMa and the TriGS trainer.

sync.json uses the repository's PLUS convention::

    actual_capture_time = logical_frame_time + camera_offset_seconds

Exact source timestamps are retained in frame_timestamps.json so timestamp
quality and any clock drift are never hidden by the constant-offset summary.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CAMERA_RE = re.compile(r"^(JCam(?P<index>\d{3}))_")
LOG_LINE_RE = re.compile(
    r"^Frame\s+(?P<frame>\d+)\s+saved\s+at\s+"
    r"(?P<stamp>\d{8}_\d{6}_\d{9})\s+type\s+(?P<type>\d+)\s*$"
)
STAMP_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<ns>\d{9})$"
)
ROTATION_ALIASES = {
    "n": "n", "none": "n",
    "l": "l", "left": "l",
    "r": "r", "right": "r",
    "u": "u", "upsidedown": "u", "180": "u",
}


@dataclass(frozen=True)
class LogFrame:
    frame_number: int
    timestamp_ns: int
    frame_type: int


@dataclass(frozen=True)
class CameraSource:
    camera_id: str
    camera_index: int
    folder: str
    video_path: Path
    log_path: Path
    snapshot_path: Path
    frames: Tuple[LogFrame, ...]

    @property
    def timestamps_ns(self) -> Tuple[int, ...]:
        return tuple(row.timestamp_ns for row in self.frames)


@dataclass(frozen=True)
class CameraLadder:
    """Per-camera capture-time model fit to record_logs: ``time_ns(n) = t1_ns
    + (n - 1) * period_ns`` for video frame_number ``n`` (1-indexed).

    Fit by min-latency (lower-envelope) regression on encoder save times,
    which are one-sided: a frame can be delayed by encode/write contention
    but never saved before it was captured. IDR (``type 5``) frames are
    dropped before fitting -- they save a measured +1.87ms later than P
    frames on this rig, a NAL-type-dependent encode-latency bias that has
    nothing to do with capture time and would bias a naive fit.

    ``phase_se_ns``/``rate_se_ns`` come from a block bootstrap (block=10):
    residuals are autocorrelated (measured lag-1 ~= +0.72, a queueing
    signature), so an i.i.d. bootstrap would understate uncertainty by
    roughly 2x.
    """
    t1_ns: float
    period_ns: float
    phase_se_ns: float
    rate_se_ns: float
    fit_rms_ms: float
    fit_max_abs_ms: float
    n_log_frames: int
    n_fit_frames: int
    idr_bias_ms: Optional[float]

    def time_ns(self, frame_number: float) -> float:
        return self.t1_ns + (frame_number - 1.0) * self.period_ns

    def sigma_ns(self, frame_number: float) -> float:
        """1-sigma uncertainty on the modeled time at ``frame_number``,
        including extrapolation past the logged prefix."""
        return self.phase_se_ns + self.rate_se_ns * abs(frame_number - 1.0)


def fit_camera_ladder(
    frames: Sequence[LogFrame],
    *,
    drop_frame_types: Sequence[int] = (5,),
    trim_quantile: float = 0.35,
    trim_iters: int = 8,
    bootstrap_samples: int = 120,
    bootstrap_block: int = 10,
    seed: int = 0,
) -> CameraLadder:
    if len(frames) < 8:
        raise ValueError(
            f"Need at least 8 record_log frames to fit a camera ladder, got {len(frames)}"
        )
    all_n = np.array([row.frame_number for row in frames], dtype=np.float64)
    all_t = np.array([row.timestamp_ns for row in frames], dtype=np.float64)
    all_type = np.array([row.frame_type for row in frames], dtype=np.int64)

    keep = ~np.isin(all_type, np.asarray(list(drop_frame_types)))
    if int(keep.sum()) < 8:
        keep = np.ones_like(keep, dtype=bool)
    n = all_n[keep]
    t = all_t[keep]

    def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(a, y, rcond=None)
        return sol  # [period_ns, intercept_at_n=0]

    def _trimmed_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sol = _fit(x, y)
        for _ in range(trim_iters):
            resid = y - (sol[0] * x + sol[1])
            threshold = np.quantile(resid, trim_quantile)
            mask = resid <= threshold
            if int(mask.sum()) < 6:
                break
            sol = _fit(x[mask], y[mask])
        return sol

    sol = _trimmed_fit(n, t)
    period_ns, intercept_ns = float(sol[0]), float(sol[1])
    t1_ns = period_ns * 1.0 + intercept_ns

    final_resid = t - (sol[0] * n + sol[1])
    fit_rms_ms = float(np.sqrt(np.mean(final_resid ** 2)) / 1e6)
    fit_max_abs_ms = float(np.max(np.abs(final_resid)) / 1e6)

    idr_bias_ms: Optional[float] = None
    if int((~keep).sum()) >= 3:
        idr_resid = all_t[~keep] - (sol[0] * all_n[~keep] + sol[1])
        idr_bias_ms = float((np.mean(idr_resid) - np.mean(final_resid)) / 1e6)

    rng = np.random.default_rng(seed)
    n_pts = len(n)
    n_blocks = max(1, n_pts // bootstrap_block)
    boot_period = np.empty(bootstrap_samples)
    boot_t1 = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        picks = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([
            np.arange(p * bootstrap_block, min((p + 1) * bootstrap_block, n_pts))
            for p in picks
        ])
        bsol = _trimmed_fit(n[idx], t[idx])
        boot_period[i] = bsol[0]
        boot_t1[i] = bsol[0] * 1.0 + bsol[1]

    return CameraLadder(
        t1_ns=t1_ns,
        period_ns=period_ns,
        phase_se_ns=float(np.std(boot_t1)),
        rate_se_ns=float(np.std(boot_period)),
        fit_rms_ms=fit_rms_ms,
        fit_max_abs_ms=fit_max_abs_ms,
        n_log_frames=len(frames),
        n_fit_frames=int(keep.sum()),
        idr_bias_ms=idr_bias_ms,
    )


def ladder_nearest_frame_number(ladder: CameraLadder, target_ns: float) -> int:
    """Frame number (>= 1) whose modeled time is nearest ``target_ns``.

    Ties go to the earlier frame, matching the previous raw-timestamp
    bisect-based selection this replaces.
    """
    if ladder.period_ns <= 0:
        raise ValueError("Camera ladder period must be positive")
    n_below = math.floor((target_ns - ladder.t1_ns) / ladder.period_ns) + 1
    n_below = max(1, int(n_below))
    t_below = ladder.time_ns(n_below)
    t_above = ladder.time_ns(n_below + 1)
    if (target_ns - t_below) <= (t_above - target_ns):
        return n_below
    return n_below + 1


@dataclass(frozen=True)
class CameraSelection:
    source: CameraSource
    ladder: CameraLadder
    source_start_position: int
    source_start_frame: int
    timestamps_ns: Tuple[int, ...]
    raw_offset_sec: float
    sync_offset_sec: float
    residual_rms_ms: float
    residual_max_abs_ms: float
    drift_ms: float
    sigma_last_ms: float
    extrapolated_frames: int


@dataclass(frozen=True)
class Timeline:
    fps: float
    logical_origin_ns: int
    common_start_ns: int
    common_end_ns: int
    num_frames: int
    selections: Tuple[CameraSelection, ...]


@dataclass(frozen=True)
class MediaProbe:
    width: int
    height: int
    avg_fps: Optional[float]
    frame_count: Optional[int]


def parse_timestamp_ns(text: str) -> int:
    match = STAMP_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid 4DV timestamp: {text!r}")
    dt = datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    return calendar.timegm(dt.utctimetuple()) * 1_000_000_000 + int(match.group("ns"))


def format_timestamp_ns(value: int) -> str:
    seconds, nanos = divmod(int(value), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S_") + f"{nanos:09d}"


def parse_record_log(path: Path) -> Tuple[LogFrame, ...]:
    rows: List[LogFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = LOG_LINE_RE.match(line)
            if match is None:
                raise ValueError(f"{path}:{line_number}: unrecognized log line: {line.rstrip()!r}")
            rows.append(LogFrame(
                frame_number=int(match.group("frame")),
                timestamp_ns=parse_timestamp_ns(match.group("stamp")),
                frame_type=int(match.group("type")),
            ))

    if not rows:
        raise ValueError(f"No frame records found in {path}")
    expected = list(range(1, len(rows) + 1))
    actual = [row.frame_number for row in rows]
    if actual != expected:
        for position, (got, want) in enumerate(zip(actual, expected), start=1):
            if got != want:
                raise ValueError(
                    f"{path}: non-contiguous frame numbering at log row {position}: "
                    f"got {got}, expected {want}"
                )
        raise ValueError(f"{path}: frame numbering must be contiguous and start at 1")
    for left, right in zip(rows, rows[1:]):
        if right.timestamp_ns <= left.timestamp_ns:
            raise ValueError(
                f"{path}: timestamp is not increasing at frames "
                f"{left.frame_number}->{right.frame_number}"
            )
    return tuple(rows)


def _camera_id(path: Path) -> Optional[Tuple[str, int]]:
    match = CAMERA_RE.match(path.name)
    if match is None:
        return None
    camera_id = match.group(1)
    return camera_id, int(match.group("index"))


def _index_camera_files(paths: Iterable[Path], label: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in paths:
        parsed = _camera_id(path)
        if parsed is None:
            continue
        camera_id, _ = parsed
        if camera_id in result:
            raise ValueError(f"Duplicate {label} for {camera_id}: {result[camera_id]} and {path}")
        result[camera_id] = path
    return result


def find_manifest(capture_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path
    candidates = [
        path for path in sorted(capture_dir.glob("*.json"))
        if path.name not in {"sync.json", "frame_timestamps.json", "camera_map.json"}
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one capture manifest under {capture_dir}; "
            f"found {[path.name for path in candidates]}. Pass --manifest explicitly."
        )
    return candidates[0]


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("controller_type") != "jcam":
        raise ValueError(
            f"{path}: expected controller_type='jcam', got {payload.get('controller_type')!r}"
        )
    cameras = payload.get("data", {}).get("cameras", [])
    if not cameras:
        raise ValueError(f"{path}: no data.cameras entries")
    failed = [row.get("camera_id") for row in cameras if row.get("status") != "success"]
    if failed:
        raise ValueError(f"Manifest reports unsuccessful cameras: {failed}")
    return payload


def discover_sources(capture_dir: Path, manifest: dict) -> Tuple[CameraSource, ...]:
    videos = _index_camera_files((capture_dir / "videos").glob("JCam*.mp4"), "video")
    logs = _index_camera_files((capture_dir / "record_logs").glob("JCam*.log"), "record log")
    snapshots = _index_camera_files(
        (capture_dir / "snapshot" / "reference" / "images").glob("JCam*.*"),
        "reference snapshot",
    )

    sources: List[CameraSource] = []
    manifest_ids = [str(row["camera_id"]) for row in manifest["data"]["cameras"]]
    for camera_id in sorted(manifest_ids, key=lambda value: int(value[4:])):
        index = int(camera_id[4:])
        missing = [
            label for label, table in (
                ("video", videos), ("record log", logs), ("reference snapshot", snapshots)
            ) if camera_id not in table
        ]
        if missing:
            raise FileNotFoundError(f"{camera_id}: missing {', '.join(missing)}")
        sources.append(CameraSource(
            camera_id=camera_id,
            camera_index=index,
            folder=f"{index:04d}",
            video_path=videos[camera_id],
            log_path=logs[camera_id],
            snapshot_path=snapshots[camera_id],
            frames=parse_record_log(logs[camera_id]),
        ))

    unexpected = (set(videos) | set(logs) | set(snapshots)) - set(manifest_ids)
    if unexpected:
        raise ValueError(f"Capture contains cameras absent from manifest: {sorted(unexpected)}")
    return tuple(sources)


def probe_media(path: Path) -> MediaProbe:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for 4DV media validation") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video/image stream found in {path}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate")
    avg_fps = None
    if rate and rate != "0/0":
        numerator, denominator = rate.split("/", 1)
        if float(denominator) != 0:
            avg_fps = float(numerator) / float(denominator)
    count = stream.get("nb_frames")
    return MediaProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        avg_fps=avg_fps,
        frame_count=int(count) if count not in (None, "N/A") else None,
    )


def unlogged_trailing_frame_count(source: CameraSource, video: MediaProbe) -> Optional[int]:
    """Return the number of MP4 frames after the timestamped log prefix.

    Some 4DV exports retain a longer MP4 while record_logs contains only the
    first N timestamp mappings.  That is valid for an import wholly contained
    in the logged prefix.  A video shorter than the last logged frame is never
    valid because the log would reference media that does not exist.
    """
    if video.frame_count is None:
        return None
    last_logged_frame = source.frames[-1].frame_number
    if video.frame_count < last_logged_frame:
        raise ValueError(
            f"{source.camera_id}: MP4 reports {video.frame_count} frames but the "
            f"record log references frame {last_logged_frame}"
        )
    return int(video.frame_count - last_logged_frame)


def validate_source_media(
    sources: Sequence[CameraSource], manifest: dict, *, fps: float
) -> Dict[str, dict]:
    intrinsics = {row["camera_id"]: row["intrinsic"] for row in manifest["intri"]["data"]}

    def inspect(source: CameraSource) -> Tuple[str, MediaProbe, MediaProbe]:
        return source.camera_id, probe_media(source.video_path), probe_media(source.snapshot_path)

    with ThreadPoolExecutor(max_workers=min(8, max(len(sources), 1))) as executor:
        rows = list(executor.map(inspect, sources))

    result: Dict[str, dict] = {}
    rates = []
    cameras_with_unlogged_tail = []
    for source, (camera_id, video, snapshot) in zip(sources, rows):
        if camera_id != source.camera_id:
            raise RuntimeError("Internal camera/media probe ordering error")
        k = intrinsics[camera_id]
        expected_width = int(round(2.0 * float(k[0][2]) + 1.0))
        expected_height = int(round(2.0 * float(k[1][2]) + 1.0))
        expected_size = (expected_width, expected_height)
        if (video.width, video.height) != expected_size:
            raise ValueError(
                f"{camera_id}: video size {(video.width, video.height)} does not "
                f"match manifest calibration {expected_size}"
            )
        if (snapshot.width, snapshot.height) != expected_size:
            raise ValueError(
                f"{camera_id}: snapshot size {(snapshot.width, snapshot.height)} does not "
                f"match manifest calibration {expected_size}"
            )
        trailing_frames = unlogged_trailing_frame_count(source, video)
        if trailing_frames:
            cameras_with_unlogged_tail.append((camera_id, trailing_frames))
        if video.avg_fps is not None:
            rates.append(video.avg_fps)
        result[camera_id] = {
            "video_width": video.width,
            "video_height": video.height,
            "video_avg_fps": video.avg_fps,
            "video_frame_count": video.frame_count,
            "log_frame_count": len(source.frames),
            "unlogged_trailing_frames": trailing_frames,
        }
    if rates:
        print(
            f"Container fps range: {min(rates):.6f} .. {max(rates):.6f}; "
            f"requested logical fps={fps:.6f}"
        )
    print(f"Media/calibration validation passed for {len(sources)} cameras")
    if cameras_with_unlogged_tail:
        preview = ", ".join(
            f"{camera_id}=+{count}" for camera_id, count in cameras_with_unlogged_tail[:8]
        )
        print(
            f"[info] {len(cameras_with_unlogged_tail)} cameras have MP4 frames "
            f"after the timestamped log prefix ({preview}). These are only eligible "
            f"via camera-ladder extrapolation (see --max-ladder-sigma-ms / "
            f"--allow-ladder-extrapolation), not raw log lookup."
        )
    return result


def _median_ns(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def build_timeline(
    sources: Sequence[CameraSource],
    *,
    fps: float,
    start_offset_frames: int = 0,
    num_frames: Optional[int] = None,
    video_frame_counts: Optional[Mapping[str, int]] = None,
    max_ladder_sigma_ms: float = 4.0,
    allow_ladder_extrapolation: bool = False,
) -> Timeline:
    """Select a contiguous per-camera frame window from independently-fit
    camera clock ladders.

    Each camera free-runs its own clock (measured spread ~60.13-60.23fps
    across a 96-camera rig, uniformly random start phase spanning a full
    frame period). `fit_camera_ladder` turns each camera's logged frames
    into a robust ``time_ns(frame_number)`` model; this function finds, for
    a shared logical origin, the nearest frame_number per camera and
    median-centers the origin against the resulting phases -- unchanged in
    spirit from the previous raw-timestamp version, just evaluated against
    the model instead of a lookup table.

    The hard bound on ``num_frames`` is the shortest of: the actual video's
    frame count (if ``video_frame_counts`` is given), or -- when a camera has
    no known video length -- its logged frame count (this is *not* lifted by
    ``allow_ladder_extrapolation``; that flag only widens the *confidence*
    gate below, never fabricates frames past a known bound).

    A separate ladder-confidence gate applies whenever the requested window
    extends past a camera's logged prefix: the modeled time's 1-sigma
    uncertainty at the last selected frame (extrapolation compounds at the
    camera's own rate uncertainty) must stay under `max_ladder_sigma_ms`, or
    the call raises unless `allow_ladder_extrapolation` is set.
    """
    if not sources:
        raise ValueError("No camera sources")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if start_offset_frames < 0:
        raise ValueError("start_offset_frames must be >= 0")

    common_start_ns = max(source.frames[0].timestamp_ns for source in sources)
    common_end_ns = min(source.frames[-1].timestamp_ns for source in sources)
    if common_end_ns <= common_start_ns:
        raise ValueError("Camera logs have no overlapping capture interval")

    ladders: Dict[str, CameraLadder] = {
        source.camera_id: fit_camera_ladder(source.frames) for source in sources
    }

    period_ns = 1_000_000_000.0 / float(fps)
    requested_origin = int(round(common_start_ns + start_offset_frames * period_ns))
    if requested_origin > common_end_ns:
        raise ValueError("Requested start offset is beyond the common capture interval")

    # Median-center the residual clock phase, then reselect starts.  The common
    # absolute time origin is arbitrary; centering keeps sync offsets balanced.
    logical_origin_ns = requested_origin
    start_frames: Dict[str, int] = {}
    for _ in range(4):
        start_frames = {
            source.camera_id: ladder_nearest_frame_number(
                ladders[source.camera_id], logical_origin_ns
            )
            for source in sources
        }
        phases = [
            ladders[source.camera_id].time_ns(start_frames[source.camera_id]) - logical_origin_ns
            for source in sources
        ]
        centered_origin = int(round(logical_origin_ns + _median_ns(phases)))
        if centered_origin == logical_origin_ns:
            break
        logical_origin_ns = centered_origin

    # Hard bound: never select a frame number past a confirmed video length,
    # and never past the logged prefix when the video length is unknown.
    per_camera_cap: Dict[str, int] = {}
    for source in sources:
        n_c = start_frames[source.camera_id]
        hard_bound = None
        if video_frame_counts is not None:
            hard_bound = video_frame_counts.get(source.camera_id)
        if hard_bound is None:
            hard_bound = source.frames[-1].frame_number
        per_camera_cap[source.camera_id] = max(int(hard_bound) - n_c + 1, 0)

    maximum = min(per_camera_cap.values())
    chosen_count = maximum if num_frames is None else int(num_frames)
    if chosen_count <= 0:
        raise ValueError("num_frames must be positive")
    if chosen_count > maximum:
        worst = min(per_camera_cap, key=per_camera_cap.get)
        raise ValueError(
            f"Requested {chosen_count} frames, but the available window supports "
            f"at most {maximum} (bound by {worst}: {per_camera_cap[worst]})"
        )

    # Ladder-confidence gate: only bites once the window runs past a
    # camera's logged prefix, since sigma_ns() is ~0 (sub-frame-fit noise)
    # inside it and grows with distance from the fit past it.
    sigma_violations = []
    for source in sources:
        ladder = ladders[source.camera_id]
        n_last = start_frames[source.camera_id] + chosen_count - 1
        sigma_ms = ladder.sigma_ns(n_last) / 1e6
        if sigma_ms > max_ladder_sigma_ms:
            sigma_violations.append((source.camera_id, n_last, sigma_ms))
    if sigma_violations:
        detail = ", ".join(
            f"{camera_id}(n={n_last},sigma={sigma_ms:.2f}ms)"
            for camera_id, n_last, sigma_ms in sigma_violations[:8]
        )
        message = (
            f"{len(sigma_violations)} camera(s) exceed max_ladder_sigma_ms="
            f"{max_ladder_sigma_ms:.2f} at the requested window end: {detail}"
        )
        if not allow_ladder_extrapolation:
            raise ValueError(
                message + ". Reduce --num-frames or pass --allow-ladder-extrapolation."
            )
        print(f"[warning] {message}")

    raw_offsets: List[float] = []
    per_camera_ts: Dict[str, Tuple[int, ...]] = {}
    for source in sources:
        ladder = ladders[source.camera_id]
        n_c = start_frames[source.camera_id]
        modeled_ns = tuple(
            int(round(ladder.time_ns(n))) for n in range(n_c, n_c + chosen_count)
        )
        per_camera_ts[source.camera_id] = modeled_ns
        residuals_sec = [
            (timestamp - logical_origin_ns) / 1e9 - frame_index / float(fps)
            for frame_index, timestamp in enumerate(modeled_ns)
        ]
        raw_offsets.append(float(statistics.median(residuals_sec)))

    center_sec = float(statistics.median(raw_offsets))
    logical_origin_ns += int(round(center_sec * 1e9))

    selections: List[CameraSelection] = []
    for source in sources:
        ladder = ladders[source.camera_id]
        n_c = start_frames[source.camera_id]
        modeled_ns = per_camera_ts[source.camera_id]
        # Evaluate the scalar (legacy RoMa/import_report) offset at the
        # window midpoint rather than frame 0: halves worst-case error for
        # any consumer that only reads the constant, since the per-camera
        # rate error is smallest near the center of the fitted extrapolation.
        mid_index = (chosen_count - 1) / 2.0
        mid_actual_ns = ladder.time_ns(n_c + mid_index)
        mid_logical_ns = logical_origin_ns + mid_index * period_ns
        sync_offset_sec = (mid_actual_ns - mid_logical_ns) / 1e9
        raw_offset_sec = (modeled_ns[0] - logical_origin_ns) / 1e9
        modeled_drift_ms = (ladder.period_ns - period_ns) * (chosen_count - 1) / 1e6
        n_last = n_c + chosen_count - 1
        sigma_last_ms = ladder.sigma_ns(n_last) / 1e6
        extrapolated_frames = max(0, n_last - source.frames[-1].frame_number)
        selections.append(CameraSelection(
            source=source,
            ladder=ladder,
            source_start_position=n_c - 1,
            source_start_frame=n_c,
            timestamps_ns=modeled_ns,
            raw_offset_sec=raw_offset_sec,
            sync_offset_sec=sync_offset_sec,
            residual_rms_ms=ladder.fit_rms_ms,
            residual_max_abs_ms=ladder.fit_max_abs_ms,
            drift_ms=modeled_drift_ms,
            sigma_last_ms=sigma_last_ms,
            extrapolated_frames=extrapolated_frames,
        ))

    return Timeline(
        fps=float(fps),
        logical_origin_ns=logical_origin_ns,
        common_start_ns=common_start_ns,
        common_end_ns=common_end_ns,
        num_frames=chosen_count,
        selections=tuple(selections),
    )


def normalize_rotation(value: str) -> str:
    key = str(value).strip().lower()
    if key not in ROTATION_ALIASES:
        raise ValueError(f"Unknown rotation {value!r}; expected l/r/u/n")
    return ROTATION_ALIASES[key]


def load_rotation_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Rotation JSON must be an object")
    return {str(key): normalize_rotation(value) for key, value in raw.items()}


def rotation_for_source(
    source: CameraSource, rotation_map: Mapping[str, str], default: str
) -> str:
    candidates = (
        source.camera_id,
        source.folder,
        source.video_path.name,
        source.snapshot_path.name,
    )
    for key in candidates:
        if key in rotation_map:
            return normalize_rotation(rotation_map[key])
    return normalize_rotation(default)


def rotation_filter(rotation: str) -> Optional[str]:
    return {"n": None, "l": "transpose=2", "r": "transpose=1", "u": "hflip,vflip"}[
        normalize_rotation(rotation)
    ]


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def _prepare_camera_dir(path: Path, *, skip_existing: bool, overwrite: bool) -> bool:
    if path.exists() and any(path.iterdir()):
        if skip_existing:
            return False
        if not overwrite:
            raise FileExistsError(
                f"Output camera directory is not empty: {path}. "
                "Use --skip-existing or --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return True


def _sequence_is_complete(path: Path, *, count: int, extension: str) -> bool:
    expected = {f"{index:06d}.{extension}" for index in range(1, count + 1)}
    actual = {entry.name for entry in path.glob(f"*.{extension}")}
    return actual == expected


def extract_camera(
    selection: CameraSelection,
    *,
    output_dir: Path,
    image_format: str,
    jpeg_quality: int,
    rotation: str,
    skip_existing: bool,
    overwrite: bool,
) -> None:
    camera_dir = output_dir / "images" / selection.source.folder
    if skip_existing and camera_dir.exists() and any(camera_dir.iterdir()):
        if not _sequence_is_complete(
            camera_dir, count=len(selection.timestamps_ns), extension=image_format
        ):
            raise RuntimeError(
                f"Cannot skip incomplete sequence {camera_dir}; expected "
                f"{len(selection.timestamps_ns)} numbered .{image_format} images"
            )
        print(f"[skip] complete sequence {camera_dir}")
        return
    if not _prepare_camera_dir(camera_dir, skip_existing=skip_existing, overwrite=overwrite):
        print(f"[skip] {camera_dir}")
        return

    first_zero = selection.source_start_frame - 1
    last_zero = first_zero + len(selection.timestamps_ns) - 1
    filters = [f"select=between(n\\,{first_zero}\\,{last_zero})"]
    rotate = rotation_filter(rotation)
    if rotate:
        filters.append(rotate)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(selection.source.video_path),
        "-vf", ",".join(filters),
        "-fps_mode", "passthrough",
        "-start_number", "1",
    ]
    if image_format == "jpg":
        command += ["-q:v", str(jpeg_quality)]
    command.append(str(camera_dir / f"%06d.{image_format}"))
    run_command(command)

    extracted = len(list(camera_dir.glob(f"*.{image_format}")))
    if extracted != len(selection.timestamps_ns):
        raise RuntimeError(
            f"{selection.source.camera_id}: extracted {extracted} frames, "
            f"expected {len(selection.timestamps_ns)}"
        )


def normalize_reference_image(
    selection: CameraSelection,
    *,
    output_dir: Path,
    image_format: str,
    jpeg_quality: int,
    rotation: str,
    skip_existing: bool,
    overwrite: bool,
) -> None:
    camera_dir = output_dir / "reference_images" / selection.source.folder
    destination = camera_dir / f"000001.{image_format}"
    if skip_existing and camera_dir.exists() and any(camera_dir.iterdir()):
        if not destination.exists():
            raise RuntimeError(
                f"Cannot skip incomplete reference directory {camera_dir}; "
                f"missing {destination.name}"
            )
        print(f"[skip] complete reference {camera_dir}")
        return
    if not _prepare_camera_dir(camera_dir, skip_existing=skip_existing, overwrite=overwrite):
        print(f"[skip] {camera_dir}")
        return
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(selection.source.snapshot_path),
    ]
    rotate = rotation_filter(rotation)
    if rotate:
        command += ["-vf", rotate]
    command += ["-frames:v", "1"]
    if image_format == "jpg":
        command += ["-q:v", str(jpeg_quality)]
    command.append(str(destination))
    run_command(command)
    if not destination.exists():
        raise RuntimeError(f"Failed to create reference image: {destination}")


def matmul3(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(float(left[i][k]) * float(right[k][j]) for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def matvec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(float(matrix[i][k]) * float(vector[k]) for k in range(3)) for i in range(3)]


def camera_center(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> List[float]:
    return [
        -sum(float(rotation[row][column]) * float(translation[row]) for row in range(3))
        for column in range(3)
    ]


def determinant3(matrix: Sequence[Sequence[float]]) -> float:
    a = matrix
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def rotation_matrix_for_image(rotation: str) -> List[List[float]]:
    return {
        "n": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "l": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "r": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "u": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    }[normalize_rotation(rotation)]


def rotate_calibration(
    intrinsic: Sequence[Sequence[float]],
    extrinsic: Sequence[Sequence[float]],
    rotation: str,
) -> Tuple[List[List[float]], List[List[float]], int, int]:
    k = [[float(value) for value in row] for row in intrinsic]
    e = [[float(value) for value in row] for row in extrinsic]
    if len(k) != 3 or any(len(row) != 3 for row in k):
        raise ValueError("Intrinsic matrix must be 3x3")
    if len(e) != 3 or any(len(row) != 4 for row in e):
        raise ValueError("Extrinsic matrix must be 3x4")

    old_width = int(round(2.0 * k[0][2] + 1.0))
    old_height = int(round(2.0 * k[1][2] + 1.0))
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    rotation = normalize_rotation(rotation)
    if rotation == "n":
        new_width, new_height = old_width, old_height
        new_k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    elif rotation == "l":
        new_width, new_height = old_height, old_width
        new_k = [[fy, 0.0, cy], [0.0, fx, old_width - 1.0 - cx], [0.0, 0.0, 1.0]]
    elif rotation == "r":
        new_width, new_height = old_height, old_width
        new_k = [[fy, 0.0, old_height - 1.0 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]]
    else:
        new_width, new_height = old_width, old_height
        new_k = [[fx, 0.0, old_width - 1.0 - cx], [0.0, fy, old_height - 1.0 - cy], [0.0, 0.0, 1.0]]

    q = rotation_matrix_for_image(rotation)
    old_r = [row[:3] for row in e]
    old_t = [row[3] for row in e]
    new_r = matmul3(q, old_r)
    new_t = matvec3(q, old_t)
    new_e = [new_r[row] + [new_t[row]] for row in range(3)]
    return new_k, new_e, new_width, new_height


def rotation_matrix_to_qvec(matrix: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    # Scalar-first quaternion, matching COLMAP's images.txt convention.
    r = matrix
    trace = float(r[0][0] + r[1][1] + r[2][2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    values = (qw / norm, qx / norm, qy / norm, qz / norm)
    if values[0] < 0:
        values = tuple(-value for value in values)
    return values


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def write_manifest_calibration(
    manifest: dict,
    timeline: Timeline,
    rotations: Mapping[str, str],
    output_dir: Path,
    image_format: str,
    *,
    overwrite: bool = False,
) -> None:
    intrinsics = {row["camera_id"]: row["intrinsic"] for row in manifest["intri"]["data"]}
    extrinsics = {row["camera_id"]: row["extrinsic"] for row in manifest["extri"]["data"]}
    expected = {selection.source.camera_id for selection in timeline.selections}
    if set(intrinsics) != expected or set(extrinsics) != expected:
        raise ValueError(
            "Manifest calibration camera IDs do not exactly match imported cameras: "
            f"intri_missing={sorted(expected - set(intrinsics))}, "
            f"extri_missing={sorted(expected - set(extrinsics))}"
        )

    cameras = []
    centers = []
    sparse_dir = output_dir / "sparse" / "0"
    if sparse_dir.exists() and any(sparse_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Calibration model already exists at {sparse_dir}; use --overwrite "
                "or --no-manifest-calibration"
            )
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# Camera list with one line of data per camera:",
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = ["# Image list with two lines of data per image:",
                   "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                   "# POINTS2D[] as (X, Y, POINT3D_ID)"]

    for image_id, selection in enumerate(timeline.selections, start=1):
        source = selection.source
        rotation = rotations[source.camera_id]
        k, e, width, height = rotate_calibration(
            intrinsics[source.camera_id], extrinsics[source.camera_id], rotation
        )
        r = [row[:3] for row in e]
        t = [row[3] for row in e]
        det = determinant3(r)
        if not 0.999 <= det <= 1.001:
            raise ValueError(f"{source.camera_id}: invalid rotation determinant {det}")
        q = rotation_matrix_to_qvec(r)
        centers.append(camera_center(r, t))
        name = f"{source.folder}/000001.{image_format}"
        params = [k[0][0], k[1][1], k[0][2], k[1][2]]
        camera_lines.append(
            f"{image_id} PINHOLE {width} {height} " + " ".join(f"{value:.17g}" for value in params)
        )
        image_lines.append(
            f"{image_id} " + " ".join(f"{value:.17g}" for value in (*q, *t))
            + f" {image_id} {name}"
        )
        image_lines.append("")
        cameras.append({
            "name": name,
            "folder": source.folder,
            "image_name": f"000001.{image_format}",
            "source_camera_id": source.camera_id,
            "R": r,
            "t": t,
            "camera_model_name": "PINHOLE",
            "width": width,
            "height": height,
            "params": params,
        })

    radii = [math.sqrt(sum(value * value for value in center)) for center in centers]
    scene_meta = {
        "scene_center": [0.0, 0.0, 0.0],
        "scene_radius_p90": percentile(radii, 0.90),
        "scene_radius_p95": percentile(radii, 0.95),
        "scene_radius_max": max(radii),
        "n_points": 0,
        "bounds_source": "4dv_manifest_camera_ring",
        "calibration_scale": manifest.get("intri", {}).get("scale"),
    }
    reference = {
        "cameras": cameras,
        "scene_meta": scene_meta,
        "calibration_source": "4dv_manifest",
        "calibration_session_id": manifest.get("extri", {}).get("session_id"),
    }

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n",
        encoding="utf-8",
    )
    (output_dir / "reference_export.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sparse" / "scene_meta.json").write_text(
        json.dumps(scene_meta, indent=2) + "\n", encoding="utf-8"
    )


def log_interval_stats(frames: Sequence[LogFrame], fps: float) -> dict:
    deltas_ms = [
        (right.timestamp_ns - left.timestamp_ns) / 1e6
        for left, right in zip(frames, frames[1:])
    ]
    frame_types: Dict[str, int] = {}
    for frame in frames:
        key = str(frame.frame_type)
        frame_types[key] = frame_types.get(key, 0) + 1
    if not deltas_ms:
        return {"median_delta_ms": None, "max_delta_ms": None, "frame_types": frame_types}
    expected_ms = 1000.0 / float(fps)
    return {
        "median_delta_ms": statistics.median(deltas_ms),
        "min_delta_ms": min(deltas_ms),
        "max_delta_ms": max(deltas_ms),
        "large_interval_count": sum(value > 1.5 * expected_ms for value in deltas_ms),
        "frame_types": frame_types,
    }


def write_metadata(
    capture_dir: Path,
    manifest_path: Path,
    timeline: Timeline,
    rotations: Mapping[str, str],
    output_dir: Path,
    *,
    max_sync_rms_ms: float,
    max_sync_drift_ms: float,
    max_ladder_sigma_ms: float,
) -> None:
    sync = {
        selection.source.folder: selection.sync_offset_sec
        for selection in timeline.selections
    }
    camera_map = {
        selection.source.folder: {
            "camera_id": selection.source.camera_id,
            "camera_index": selection.source.camera_index,
            "video": str(selection.source.video_path),
            "record_log": str(selection.source.log_path),
            "reference_snapshot": str(selection.source.snapshot_path),
            "rotation": rotations[selection.source.camera_id],
        }
        for selection in timeline.selections
    }
    any_extrapolated = any(
        selection.extrapolated_frames > 0 for selection in timeline.selections
    )
    timestamps = {
        "schema_version": 1,
        "convention": "actual_capture_time = logical_time + camera_offset",
        "fps": timeline.fps,
        "logical_origin_timestamp": format_timestamp_ns(timeline.logical_origin_ns),
        "logical_origin_timestamp_ns": timeline.logical_origin_ns,
        "num_frames": timeline.num_frames,
        "extrapolated_beyond_log": any_extrapolated,
        "cameras": {
            selection.source.folder: {
                "camera_id": selection.source.camera_id,
                "source_start_frame": selection.source_start_frame,
                "sync_offset_sec": selection.sync_offset_sec,
                "timestamp_ns": list(selection.timestamps_ns),
                "ladder": {
                    "t1_ns": selection.ladder.t1_ns,
                    "period_ns": selection.ladder.period_ns,
                    "phase_se_ms": selection.ladder.phase_se_ns / 1e6,
                    "rate_se_us_per_frame": selection.ladder.rate_se_ns / 1e3,
                    "fit_rms_ms": selection.ladder.fit_rms_ms,
                    "fit_max_abs_ms": selection.ladder.fit_max_abs_ms,
                    "idr_bias_ms": selection.ladder.idr_bias_ms,
                    "n_log_frames": selection.ladder.n_log_frames,
                    "n_fit_frames": selection.ladder.n_fit_frames,
                    "extrapolated_frames": selection.extrapolated_frames,
                    "sigma_at_last_frame_ms": selection.sigma_last_ms,
                },
            }
            for selection in timeline.selections
        },
    }
    camera_reports = []
    unstable = []
    for selection in timeline.selections:
        row = {
            "camera_id": selection.source.camera_id,
            "folder": selection.source.folder,
            "log_frame_count": len(selection.source.frames),
            "source_start_frame": selection.source_start_frame,
            "source_end_frame": selection.source_start_frame + timeline.num_frames - 1,
            "sync_offset_ms": selection.sync_offset_sec * 1000.0,
            "ladder_fit_rms_ms": selection.residual_rms_ms,
            "ladder_fit_max_abs_ms": selection.residual_max_abs_ms,
            "ladder_modeled_drift_ms_over_window": selection.drift_ms,
            "ladder_phase_se_ms": selection.ladder.phase_se_ns / 1e6,
            "ladder_rate_se_us_per_frame": selection.ladder.rate_se_ns / 1e3,
            "ladder_idr_bias_ms": selection.ladder.idr_bias_ms,
            "ladder_sigma_at_last_frame_ms": selection.sigma_last_ms,
            "extrapolated_frames": selection.extrapolated_frames,
            "log_intervals": log_interval_stats(selection.source.frames, timeline.fps),
        }
        # NOTE: drift_ms (modeled rate vs *nominal* fps, accumulated over the
        # window) is reported but deliberately not gated on. Every real
        # camera runs a hair off nominal 60fps and this grows with window
        # length by construction; it is not a defect; frame_timestamps.json
        # already carries the correct per-frame modeled time regardless.
        # sigma_last_ms is the calibrated successor (uncertainty *on* that
        # model, not deviation from an arbitrary nominal grid) and is what
        # should gate trust in an extrapolated window.
        if (
            selection.residual_rms_ms > max_sync_rms_ms
            or selection.sigma_last_ms > max_ladder_sigma_ms
        ):
            unstable.append(selection.source.camera_id)
        camera_reports.append(row)

    report = {
        "schema_version": 2,
        "capture_dir": str(capture_dir),
        "manifest": str(manifest_path),
        "camera_count": len(timeline.selections),
        "fps": timeline.fps,
        "num_frames": timeline.num_frames,
        "common_start_timestamp": format_timestamp_ns(timeline.common_start_ns),
        "common_end_timestamp": format_timestamp_ns(timeline.common_end_ns),
        "logical_origin_timestamp": format_timestamp_ns(timeline.logical_origin_ns),
        "sync_convention": "PLUS",
        "extrapolated_beyond_log": any_extrapolated,
        "thresholds": {
            "max_sync_rms_ms": max_sync_rms_ms,
            "max_sync_drift_ms": max_sync_drift_ms,
            "max_ladder_sigma_ms": max_ladder_sigma_ms,
        },
        "unstable_cameras": unstable,
        "cameras": camera_reports,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sync.json").write_text(json.dumps(sync, indent=2) + "\n", encoding="utf-8")
    (output_dir / "camera_map.json").write_text(
        json.dumps(camera_map, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "frame_timestamps.json").write_text(
        json.dumps(timestamps, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "import_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def print_audit(timeline: Timeline) -> None:
    offsets_ms = [selection.sync_offset_sec * 1000.0 for selection in timeline.selections]
    rms_ms = [selection.residual_rms_ms for selection in timeline.selections]
    drift_ms = [abs(selection.drift_ms) for selection in timeline.selections]
    sigma_ms = [selection.sigma_last_ms for selection in timeline.selections]
    starts = [selection.source_start_frame for selection in timeline.selections]
    extrapolated = [
        selection for selection in timeline.selections if selection.extrapolated_frames > 0
    ]
    print(f"Cameras: {len(timeline.selections)}")
    print(f"Common interval: {format_timestamp_ns(timeline.common_start_ns)} .. "
          f"{format_timestamp_ns(timeline.common_end_ns)}")
    print(f"Logical frame 1: {format_timestamp_ns(timeline.logical_origin_ns)}")
    print(f"Output frames: {timeline.num_frames} at {timeline.fps:g} fps")
    print(f"Source start-frame range: {min(starts)} .. {max(starts)}")
    print(f"sync offset range (window midpoint): {min(offsets_ms):+.3f} .. {max(offsets_ms):+.3f} ms")
    print(f"ladder fit RMS max: {max(rms_ms):.3f} ms")
    print(f"ladder modeled drift over window, max: {max(drift_ms):.3f} ms")
    print(f"ladder sigma at last frame, max: {max(sigma_ms):.3f} ms")
    if extrapolated:
        worst = max(extrapolated, key=lambda s: s.extrapolated_frames)
        print(
            f"[info] {len(extrapolated)} camera(s) extend past their logged prefix "
            f"(worst: {worst.source.camera_id} by {worst.extrapolated_frames} frames, "
            f"sigma={worst.sigma_last_ms:.3f}ms)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a timestamped 4DV.ai JCam capture for RoMa/TriGS"
    )
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--start-offset-frames", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=1,
                        help="FFmpeg JPEG qscale (1 is highest quality)")
    parser.add_argument("--rotation-file", type=Path, default=None)
    parser.add_argument("--default-rot", choices=("l", "r", "u", "n"), default="n")
    parser.add_argument("--max-sync-rms-ms", type=float, default=4.0)
    parser.add_argument("--max-sync-drift-ms", type=float, default=4.0)
    parser.add_argument("--max-ladder-sigma-ms", type=float, default=4.0,
                        help="1-sigma modeled-time uncertainty budget at the last "
                             "selected frame; the ladder is extrapolated past the "
                             "logged prefix, so this grows with window length")
    parser.add_argument("--allow-ladder-extrapolation", action="store_true",
                        help="Proceed (with a warning) when a camera's ladder "
                             "extrapolation exceeds --max-ladder-sigma-ms, instead "
                             "of raising")
    parser.add_argument("--allow-unstable-timestamps", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit discovery/timestamps without writing or decoding")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-manifest-calibration", action="store_true",
                        help="Do not emit reference_export.json and sparse/0 from manifest calibration")
    parser.add_argument("--skip-media-probe", action="store_true",
                        help="Skip ffprobe frame-count/dimension validation")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    capture_dir = args.capture_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not capture_dir.exists():
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")
    if args.skip_existing and args.overwrite:
        raise ValueError("Use only one of --skip-existing or --overwrite")

    manifest_path = find_manifest(capture_dir, args.manifest)
    manifest = load_manifest(manifest_path)
    sources = discover_sources(capture_dir, manifest)
    video_frame_counts: Optional[Dict[str, int]] = None
    if not args.skip_media_probe:
        media_report = validate_source_media(sources, manifest, fps=args.fps)
        video_frame_counts = {
            camera_id: info["video_frame_count"]
            for camera_id, info in media_report.items()
            if info["video_frame_count"] is not None
        }
    timeline = build_timeline(
        sources,
        fps=args.fps,
        start_offset_frames=args.start_offset_frames,
        num_frames=args.num_frames,
        video_frame_counts=video_frame_counts,
        max_ladder_sigma_ms=args.max_ladder_sigma_ms,
        allow_ladder_extrapolation=args.allow_ladder_extrapolation,
    )
    print_audit(timeline)

    # See the matching NOTE in write_metadata: drift_ms is not gated here,
    # only reported -- it tracks a camera's honest rate vs *nominal* fps,
    # which every real camera has some of, and which the per-frame ladder
    # already corrects for regardless of its size.
    unstable = [
        selection.source.camera_id for selection in timeline.selections
        if selection.residual_rms_ms > args.max_sync_rms_ms
        or selection.sigma_last_ms > args.max_ladder_sigma_ms
    ]
    if unstable:
        message = (
            f"Ladder fit quality or extrapolation confidence exceeds thresholds "
            f"for {len(unstable)} cameras: {unstable[:12]}"
        )
        if not args.allow_unstable_timestamps:
            raise RuntimeError(message + ". Review the audit or pass --allow-unstable-timestamps.")
        print("[warning] " + message)

    if args.dry_run:
        return 0

    rotation_map = load_rotation_map(args.rotation_file)
    rotations = {
        source.camera_id: rotation_for_source(source, rotation_map, args.default_rot)
        for source in sources
    }
    for selection in timeline.selections:
        rotation = rotations[selection.source.camera_id]
        extract_camera(
            selection,
            output_dir=output_dir,
            image_format=args.format,
            jpeg_quality=args.jpeg_quality,
            rotation=rotation,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )
        normalize_reference_image(
            selection,
            output_dir=output_dir,
            image_format=args.format,
            jpeg_quality=args.jpeg_quality,
            rotation=rotation,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )

    write_metadata(
        capture_dir,
        manifest_path,
        timeline,
        rotations,
        output_dir,
        max_sync_rms_ms=args.max_sync_rms_ms,
        max_sync_drift_ms=args.max_sync_drift_ms,
        max_ladder_sigma_ms=args.max_ladder_sigma_ms,
    )
    if not args.no_manifest_calibration:
        write_manifest_calibration(
            manifest, timeline, rotations, output_dir, args.format,
            overwrite=args.overwrite,
        )
    print(f"4DV import complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
