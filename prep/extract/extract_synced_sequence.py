#!/usr/bin/env python3
import os
import sys
import glob
import cv2
import argparse
import json
import re
import subprocess
from fractions import Fraction
from tqdm import tqdm


TAKE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])C(\d{4})(?!\d)", re.IGNORECASE)


def filename_take_number(file_path):
    """Return the C#### take number embedded in a filename, if present."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = TAKE_TOKEN_RE.search(stem)
    return int(match.group(1)) if match else None


def discover_video_tasks(input_dir, take):
    """Find videos in either a flat capture folder or per-camera folders."""
    root_mp4s = sorted(glob.glob(os.path.join(input_dir, "*.[mM][pP]4")))
    if root_mp4s:
        tagged_mp4s = [path for path in root_mp4s if filename_take_number(path) is not None]
        if tagged_mp4s:
            selected = [path for path in tagged_mp4s if filename_take_number(path) == take]
            ignored_untagged = len(root_mp4s) - len(tagged_mp4s)
            if ignored_untagged:
                print(
                    f"⚠️ Ignoring {ignored_untagged} root MP4(s) without a C#### take token."
                )
            return [
                (idx, os.path.basename(path), path)
                for idx, path in enumerate(selected, start=1)
            ]

        print(
            "ℹ️ Root MP4 filenames do not contain C#### take tokens; "
            "processing all root MP4s."
        )
        return [
            (idx, os.path.basename(path), path)
            for idx, path in enumerate(root_mp4s, start=1)
        ]

    subdirs = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    ]

    def natural_sort_key(value):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", value)
        ]

    subdirs.sort(key=natural_sort_key)
    expected_take_name = f"C{take:04d}"
    tasks = []
    for idx, subdir in enumerate(subdirs, start=1):
        vid_files = sorted(glob.glob(os.path.join(subdir, "*.[mM][pP]4")))
        if not vid_files:
            continue

        matched_file = next(
            (
                path
                for path in vid_files
                if filename_take_number(path) == take
            ),
            None,
        )
        folder_name = os.path.basename(subdir)
        if matched_file:
            tasks.append((idx, folder_name, matched_file))
        else:
            print(
                f"⚠️ Warning: Take {take} (expected '{expected_take_name}') "
                f"not found in '{folder_name}'."
            )
    return tasks


def ffprobe_json(file_path):
    """Return ffprobe JSON containing streams, frame rate, and timecode tags."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format_tags=timecode:stream=index,codec_type,r_frame_rate,avg_frame_rate:stream_tags=timecode",
        "-of", "json",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("❌ ffprobe not found. Install FFmpeg and make sure ffprobe is on your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"❌ ffprobe failed for {file_path}:\n{exc.stderr}")
        sys.exit(1)

    return json.loads(result.stdout)


def parse_rate(rate_str):
    """Parse ffprobe frame-rate strings such as '30000/1001' or '24/1'."""
    if not rate_str or rate_str == "0/0":
        return None
    return Fraction(rate_str)


def nominal_tc_fps(rate):
    """
    SMPTE timecode counts use nominal integer frame numbers:
      23.976 -> 24, 29.97 -> 30, 59.94 -> 60, etc.
    """
    fps_float = float(rate)
    common = [24, 25, 30, 48, 50, 60, 100, 120]
    return min(common, key=lambda x: abs(x - fps_float))


def get_video_metadata(file_path):
    """Return (start_timecode, actual_fps_fraction, nominal_fps)."""
    data = ffprobe_json(file_path)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {file_path}")

    rate = parse_rate(video_stream.get("avg_frame_rate")) or parse_rate(video_stream.get("r_frame_rate"))
    if rate is None:
        raise RuntimeError(f"Could not determine frame rate for {file_path}")

    # Prefer the actual video stream's timecode.
    tags = video_stream.get("tags", {}) or {}
    timecode = tags.get("timecode")

    # Then look for a separate tmcd/data stream or any stream-level timecode.
    if timecode is None:
        for stream in data.get("streams", []):
            tags = stream.get("tags", {}) or {}
            if "timecode" in tags:
                timecode = tags["timecode"]
                break

    # Then fall back to container/format tag.
    if timecode is None:
        tags = data.get("format", {}).get("tags", {}) or {}
        timecode = tags.get("timecode")

    if timecode is None:
        raise RuntimeError(
            f"No embedded timecode found in {os.path.basename(file_path)}. "
            "Check with: ffprobe -show_streams -show_format yourfile.mp4"
        )

    return timecode, rate, nominal_tc_fps(rate)


def parse_timecode(tc):
    """Parse HH:MM:SS:FF or drop-frame HH:MM:SS;FF / HH:MM:SS.FF."""
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2})([:;.])(\d{2})$", tc.strip())
    if not m:
        raise ValueError(f"Invalid timecode '{tc}'. Use HH:MM:SS:FF or HH:MM:SS;FF.")

    hh, mm, ss, sep, ff = m.groups()
    return int(hh), int(mm), int(ss), int(ff), sep in [";", "."]


def tc_to_frames(tc, nominal_fps):
    """Convert SMPTE timecode to an absolute frame number from 00:00:00:00."""
    hh, mm, ss, ff, drop = parse_timecode(tc)

    if ff >= nominal_fps:
        raise ValueError(f"Frame number {ff} is not valid for {nominal_fps} fps timecode: {tc}")

    total_minutes = hh * 60 + mm
    frames = ((hh * 3600) + (mm * 60) + ss) * nominal_fps + ff

    if drop:
        if nominal_fps not in (30, 60):
            raise ValueError(f"Drop-frame timecode is only expected for 29.97/59.94 style rates, got {nominal_fps}.")
        drop_frames = 2 if nominal_fps == 30 else 4
        frames -= drop_frames * (total_minutes - total_minutes // 10)

    return frames


def frames_per_24h(nominal_fps, drop=False):
    """Number of frames represented by one 24-hour timecode cycle."""
    frames = 24 * 60 * 60 * nominal_fps
    if drop:
        drop_frames = 2 if nominal_fps == 30 else 4
        total_minutes = 24 * 60
        frames -= drop_frames * (total_minutes - total_minutes // 10)
    return frames


def tc_delta_frames(clip_start_tc, target_tc, nominal_fps, allow_24h_rollover=False):
    """
    Return local frame offset from clip_start_tc to target_tc.
    """
    clip_drop = parse_timecode(clip_start_tc)[4]
    target_drop = parse_timecode(target_tc)[4]
    if clip_drop != target_drop:
        raise ValueError("Clip start timecode and requested timecode must both be drop-frame or both be non-drop-frame.")

    clip_abs = tc_to_frames(clip_start_tc, nominal_fps)
    target_abs = tc_to_frames(target_tc, nominal_fps)
    delta = target_abs - clip_abs

    if delta >= 0:
        return delta

    if allow_24h_rollover:
        clip_hh = parse_timecode(clip_start_tc)[0]
        target_hh = parse_timecode(target_tc)[0]
        if clip_hh >= 20 and target_hh <= 4:
            return delta + frames_per_24h(nominal_fps, clip_drop)

    raise ValueError(
        f"Requested TC {target_tc} is {-delta} timecode frame(s) before this clip starts at {clip_start_tc}. "
        "Pick a start timecode at or after the latest Start TC across all cameras."
    )


def frames_between_timecodes(start_tc, end_tc, nominal_fps, allow_24h_rollover=False):
    """Inclusive frame count between two timecodes."""
    start_drop = parse_timecode(start_tc)[4]
    end_drop = parse_timecode(end_tc)[4]
    if start_drop != end_drop:
        raise ValueError("Start and end timecodes must both be drop-frame or both be non-drop-frame.")

    start = tc_to_frames(start_tc, nominal_fps)
    end = tc_to_frames(end_tc, nominal_fps)

    if end < start:
        if not allow_24h_rollover:
            raise ValueError(
                f"End TC {end_tc} is before start TC {start_tc}. "
                "Use --allow-24h-rollover only if this intentionally crosses midnight."
            )
        end += frames_per_24h(nominal_fps, start_drop)

    return (end - start) + 1


def main():
    parser = argparse.ArgumentParser(description="Extract synced frame sequences by embedded timecode.")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing mp4s or camera folders")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")

    parser.add_argument(
        "--take", 
        type=int, 
        default=1, 
        help="Specify the take number to extract. Defaults to 1."
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction if the destination folder already exists and is not empty."
    )

    parser.add_argument(
        "--compact-sequence",
        action="store_true",
        help="Renumber output folders continuously (e.g., 01, 02, 03) if intermediate cameras are skipped due to a missing take."
    )

    parser.add_argument("--start-frame", "-s", type=int, help="Legacy mode: start frame inside every file, 0-indexed")
    parser.add_argument("--end-frame", "-e", type=int, help="Legacy mode: end frame inside every file")

    parser.add_argument("--start-timecode", help="Shared source timecode to start extracting")
    parser.add_argument("--offset-start", type=int, default=0, help="Frames to offset from start-timecode")
    parser.add_argument("--end-timecode", help="Shared source timecode to stop extracting")
    parser.add_argument("--num-frames", type=int, help="Number of frames to extract")
    parser.add_argument("--allow-24h-rollover", action="store_true")

    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg")

    # --- ROTATION ARGUMENTS ---
    parser.add_argument(
        "--rotation-file", 
        type=str, 
        help="Path to a JSON file mapping original folder names (or filenames) to rotations (l, r, u, n). Highly recommended for reliability."
    )
    parser.add_argument("--rotations", type=str, help="Legacy: Batch rotation map (e.g., '1-r 4-l'). Brittle if used with --compact-sequence.")
    parser.add_argument("--default-rot", type=str, default="n", help="Default rotation (l/r/u/n). Defaults to 'n' (none).")
    parser.add_argument(
        "--force-rotation",
        type=str,
        choices=["l", "r", "u", "n", "left", "right", "upsidedown", "none"],
        help="Force the same rotation for every video, overriding all other maps."
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    img_ext = args.format.lower()

    timecode_mode = args.start_timecode is not None
    frame_mode = args.start_frame is not None or args.end_frame is not None

    if timecode_mode and frame_mode:
        print("❌ Use either timecode mode OR frame mode, not both.")
        sys.exit(1)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --- FILE DISCOVERY LOGIC ---
    video_tasks = discover_video_tasks(input_dir, args.take)

    if len(video_tasks) == 0:
        print(
            f"No MP4 files for take C{args.take:04d} were found in "
            f"{input_dir} or its subdirectories."
        )
        sys.exit(0)

    if args.compact_sequence:
        video_tasks = [(i, name, path) for i, (_, name, path) in enumerate(video_tasks, start=1)]

    # --- SETUP ROTATIONS ---
    json_rot_map = {}
    if args.rotation_file:
        try:
            with open(args.rotation_file, 'r') as f:
                json_rot_map = json.load(f)
            print(f"📄 Loaded rotation configurations from {args.rotation_file}")
        except Exception as e:
            print(f"❌ Error loading --rotation-file: {e}")
            sys.exit(1)

    legacy_batch_mode = args.rotations is not None
    legacy_rot_map = {}
    if legacy_batch_mode:
        try:
            pairs = args.rotations.split()
            for p in pairs:
                idx_str, rot_char = p.split('-')
                legacy_rot_map[int(idx_str)] = rot_char.strip().lower()
        except Exception:
            print("❌ Error parsing --rotations. Please use the format: '1-r 4-l 12-u'")
            sys.exit(1)

    force_rotation = args.force_rotation.lower() if args.force_rotation else None

    print(f"Found {len(video_tasks)} videos to process.")

    last_choice = 'l'

    for count, source_name, file_path in video_tasks:
        filename = os.path.basename(file_path)
        cam_name = f"{count:02d}"
        cam_out_dir = os.path.join(output_dir, cam_name)

        print("\n-------------------------------------------------")
        print(f"🎥 Next Video: {filename} (Source: {source_name}) -> Camera {cam_name}")
        print(f"📁 Saving Sequence To: {cam_out_dir}/")

        if args.skip_existing and os.path.exists(cam_out_dir) and any(os.scandir(cam_out_dir)):
            print(f"⏭️ Skipping {cam_out_dir} because it already exists and contains files.")
            continue

        if timecode_mode:
            try:
                clip_start_tc, actual_rate, nominal_fps = get_video_metadata(file_path)
                start_frame = tc_delta_frames(
                    clip_start_tc,
                    args.start_timecode,
                    nominal_fps,
                    allow_24h_rollover=args.allow_24h_rollover,
                )

                start_frame += args.offset_start

                if start_frame < 0:
                    print(f"❌ Error: Offset pushes start frame before clip begins.")
                    continue

                if args.end_timecode:
                    total_frames_to_extract = frames_between_timecodes(
                        args.start_timecode, args.end_timecode, nominal_fps, allow_24h_rollover=args.allow_24h_rollover
                    )
                    total_frames_to_extract -= args.offset_start
                    if total_frames_to_extract <= 0:
                        print(f"❌ Error: Start offset pushes start frame past end timecode.")
                        continue
                else:
                    total_frames_to_extract = args.num_frames

                end_frame = start_frame + total_frames_to_extract - 1

            except Exception as exc:
                print(f"❌ Timecode error for {filename}: {exc}")
                continue
        else:
            start_frame = args.start_frame
            end_frame = args.end_frame
            total_frames_to_extract = (end_frame - start_frame) + 1

        # --- DETERMINE ROTATION ---
        if force_rotation is not None:
            user_choice = force_rotation
            print(f"🔒 Forced rotation: '{user_choice}'")
        elif source_name in json_rot_map:
            user_choice = json_rot_map[source_name].lower()
            print(f"📄 JSON map applying rotation for '{source_name}': '{user_choice}'")
        elif legacy_batch_mode:
            user_choice = legacy_rot_map.get(count, args.default_rot.lower())
            print(f"🔄 Legacy map applying rotation for Index {count}: '{user_choice}'")
        else:
            print("Options: [l]eft, [r]ight, [u]psidedown, [n]one")
            user_choice = input(f"Rotation? (Press Enter to use '{last_choice}'): ").strip().lower()
            if not user_choice:
                user_choice = last_choice
            last_choice = user_choice

        rot_flag = None
        if user_choice in ['l', 'left']:
            rot_flag = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif user_choice in ['r', 'right']:
            rot_flag = cv2.ROTATE_90_CLOCKWISE
        elif user_choice in ['u', 'upsidedown']:
            rot_flag = cv2.ROTATE_180
        elif user_choice in ['n', 'none']:
            rot_flag = None
        else:
            print(f"⚠️ Invalid choice '{user_choice}'. Skipping {filename}.")
            continue

        os.makedirs(cam_out_dir, exist_ok=True)
        print("-------------------------------------------------")

        cap = cv2.VideoCapture(file_path)

        current_frame = 0
        while current_frame < start_frame:
            ret = cap.grab()
            if not ret:
                break
            current_frame += 1

        if current_frame < start_frame:
            print(f"❌ Error: {filename} is shorter than the requested synced start frame ({start_frame}).")
            cap.release()
            continue

        success = True
        out_idx = 1

        for _ in tqdm(range(total_frames_to_extract), desc="Extracting", leave=False):
            ret, frame = cap.read()
            if not ret:
                print(f"\n⚠️ Reached end of video unexpectedly.")
                success = False
                break

            if rot_flag is not None:
                frame = cv2.rotate(frame, rot_flag)

            out_path = os.path.join(cam_out_dir, f"{out_idx:06d}.{img_ext}")

            if img_ext == "png":
                cv2.imwrite(out_path, frame)
            else:
                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            out_idx += 1

        cap.release()

        if success:
            print("✅ Success!")
        else:
            print(f"❌ Error processing {filename}")

    print(f"\n🎉 All done! Timecode-synced and rotated image sequences are in {output_dir}")


if __name__ == '__main__':
    main()
