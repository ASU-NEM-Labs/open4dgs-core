#!/usr/bin/env python3
"""Extract already-synchronized multiview videos into camera image folders.

The first decoded frame of every video is written as ``000001`` by default.
No timecode or synchronization adjustment is performed: frame N in every
output folder comes from frame N in its source video.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    camera_name: str
    width: int
    height: int
    frame_rate: str


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def discover_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        (
            path
            for path in videos_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: _natural_key(path.stem),
    )
    if not videos:
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"No videos found in {videos_dir}; expected one of: {extensions}")

    camera_names = [path.stem for path in videos]
    duplicates = sorted({name for name in camera_names if camera_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Video filename stems must be unique because they become camera folders; "
            f"duplicates: {duplicates}"
        )
    return videos


def probe_video(path: Path, ffprobe: str = "ffprobe") -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{ffprobe!r} was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}:\n{exc.stderr.strip()}") from exc

    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return VideoInfo(
        path=path,
        camera_name=path.stem,
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_rate=str(stream.get("avg_frame_rate", "0/0")),
    )


def build_ffmpeg_command(
    *,
    info: VideoInfo,
    camera_dir: Path,
    image_format: str,
    start_number: int,
    digits: int,
    jpeg_quality: int,
    png_compression: int,
    max_frames: int | None,
    ffmpeg: str,
) -> tuple[list[str], str]:
    extension = "png" if image_format == "png" else "jpg"
    output_pattern = camera_dir / f"%0{digits}d.{extension}"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    if image_format == "png":
        command.extend(["-c:v", "png", "-compression_level", str(png_compression)])
    else:
        command.extend(["-c:v", "mjpeg", "-q:v", str(jpeg_quality)])
    command.extend(["-start_number", str(start_number), str(output_pattern)])
    return command, extension


def extract_one(
    info: VideoInfo,
    staging_root: Path,
    args: argparse.Namespace,
) -> tuple[str, int]:
    camera_dir = staging_root / info.camera_name
    camera_dir.mkdir(parents=True, exist_ok=False)
    command, extension = build_ffmpeg_command(
        info=info,
        camera_dir=camera_dir,
        image_format=args.format,
        start_number=args.start_number,
        digits=args.digits,
        jpeg_quality=args.jpeg_quality,
        png_compression=args.png_compression,
        max_frames=args.max_frames,
        ffmpeg=args.ffmpeg,
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{args.ffmpeg!r} was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {info.path}:\n{exc.stderr.strip()}") from exc

    frames = sorted(camera_dir.glob(f"*.{extension}"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {info.path}")

    expected_first = camera_dir / f"{args.start_number:0{args.digits}d}.{extension}"
    if frames[0] != expected_first:
        raise RuntimeError(
            f"Unexpected first output for {info.path}: {frames[0].name}; "
            f"expected {expected_first.name}"
        )
    return info.camera_name, len(frames)


def _validate_destination(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new directory so an old partial sequence cannot be mixed in."
            )


def extract_sequence(args: argparse.Namespace) -> dict[str, int]:
    videos_dir = args.videos_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not videos_dir.is_dir():
        raise ValueError(f"Videos directory does not exist: {videos_dir}")
    _validate_destination(output_dir)

    videos = discover_videos(videos_dir)
    infos = [probe_video(path, ffprobe=args.ffprobe) for path in videos]

    dimensions = {(info.width, info.height) for info in infos}
    rates = {info.frame_rate for info in infos}
    if len(dimensions) != 1:
        raise ValueError(f"Videos do not share one resolution: {sorted(dimensions)}")
    if len(rates) != 1:
        raise ValueError(f"Videos do not share one frame rate: {sorted(rates)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-extract-",
        dir=output_dir.parent,
    ) as staging_name:
        staging_root = Path(staging_name)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(extract_one, info, staging_root, args): info
                for info in infos
            }
            for future in as_completed(futures):
                camera_name, count = future.result()
                counts[camera_name] = count
                print(f"[{camera_name}] extracted {count} frames", flush=True)

        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            details = ", ".join(f"{name}={counts[name]}" for name in sorted(counts, key=_natural_key))
            raise RuntimeError(
                "The synchronized videos produced different frame counts; "
                f"nothing was published. Counts: {details}"
            )

        # Publish only after every camera completed and passed validation.
        if output_dir.exists():
            output_dir.rmdir()  # validated empty above
        shutil.move(str(staging_root), str(output_dir))

    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract synchronized synthetic camera videos into images/<camera>/NNNNNN. "
            "The first decoded video frame becomes 000001 by default."
        )
    )
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("png", "jpeg", "jpg"), default="png")
    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Filename number for the first decoded frame (default: 1).",
    )
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional number of frames per video, useful for a short validation run.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="FFmpeg MJPEG qscale; 2 is high quality, larger values reduce quality.",
    )
    parser.add_argument("--png-compression", type=int, choices=range(0, 10), default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)

    args.format = "png" if args.format == "png" else "jpeg"
    if args.start_number < 0:
        parser.error("--start-number must be non-negative")
    if args.digits < 1:
        parser.error("--digits must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not 1 <= args.jpeg_quality <= 31:
        parser.error("--jpeg-quality must be between 1 and 31")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    counts = extract_sequence(args)
    count = next(iter(counts.values()))
    extension = "png" if args.format == "png" else "jpg"
    width = args.digits
    last_number = args.start_number + count - 1
    print(
        f"Done: {len(counts)} cameras x {count} frames -> {args.output_dir}\n"
        f"Files: {args.start_number:0{width}d}.{extension} .. "
        f"{last_number:0{width}d}.{extension}"
    )


if __name__ == "__main__":
    main()
