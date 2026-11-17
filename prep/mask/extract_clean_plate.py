import os
import csv
import cv2
import argparse
from tqdm import tqdm

VIDEO_EXTS = ('.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.ts')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def build_pairing(source_files, images_dir):
    """
    Pair sorted source files (videos or images) with sorted camera folder
    names from images_dir, positionally
    (JCam000_... -> images/0000, JCam001_... -> images/0001, ...).
    """
    try:
        cam_names = sorted([
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ])
    except FileNotFoundError:
        print(f"Error: The images directory '{images_dir}' does not exist.")
        return None

    if len(source_files) != len(cam_names):
        print(
            f"Error: Found {len(source_files)} source file(s) in source-root but "
            f"{len(cam_names)} camera folder(s) in '{images_dir}'. "
            f"Counts must match to pair them positionally."
        )
        print("Source files (sorted):")
        for v in source_files:
            print(f"  {v}")
        print("Camera folders (sorted):")
        for c in cam_names:
            print(f"  {c}")
        return None

    return list(zip(source_files, cam_names))


def print_pairing(pairing):
    print("\nSource file -> Camera folder pairing:")
    print(f"{'Source file':<55} {'Camera':<10} {'Output file'}")
    print("-" * 85)
    for source_name, cam_name in pairing:
        print(f"{source_name:<55} {cam_name:<10} {cam_name}.jpg")
    print()


def write_pairing_manifest(pairing, output_dir):
    manifest_path = os.path.join(output_dir, "clean_plate_pairing.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "camera_folder", "output_file"])
        for source_name, cam_name in pairing:
            writer.writerow([source_name, cam_name, f"{cam_name}.jpg"])
    print(f"Pairing manifest written to: {manifest_path}")


def load_source_frame(source_path, frame_number):
    """Return a BGR frame from either a video (seek to frame_number) or an image file."""
    ext = os.path.splitext(source_path)[1].lower()

    if ext in IMAGE_EXTS:
        frame = cv2.imread(source_path)
        if frame is None:
            print(f"\n[Error] Failed to read image: {os.path.basename(source_path)}")
        return frame

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"\n[Error] Could not open video file: {os.path.basename(source_path)}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"\n[Error] {os.path.basename(source_path)} only has {total_frames} frames. "
              f"Index {frame_number} is out of range.")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"\n[Error] Failed to read frame {frame_number} from {os.path.basename(source_path)}")
        return None

    return frame


def extract_frames_direct(source_root, output_dir, frame_number=0, images_dir=None, assume_yes=False):
    """
    Scans source_root for clean-plate source files (either single-frame
    images or short videos) and produces one clean-plate JPEG per camera.

    If images_dir is given, output files are named after the matching
    camera folder (sorted source list <-> sorted camera folder list,
    paired positionally) instead of the source file's own basename, so
    the result lines up with what generate_masks.py expects
    (background_dir/{cam_name}.jpg).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        all_files = sorted(os.listdir(source_root))
    except FileNotFoundError:
        print(f"Error: The directory '{source_root}' does not exist.")
        return

    source_files = [f for f in all_files if f.lower().endswith(VIDEO_EXTS + IMAGE_EXTS)]

    if not source_files:
        print(f"No video or image files found in {source_root}")
        return

    n_images = sum(1 for f in source_files if f.lower().endswith(IMAGE_EXTS))
    n_videos = len(source_files) - n_images
    print(f"Found {len(source_files)} source file(s): {n_images} image(s), {n_videos} video(s).")

    # Determine output naming
    pairing = None
    if images_dir:
        pairing = build_pairing(source_files, images_dir)
        if pairing is None:
            return
        print_pairing(pairing)
        if not assume_yes:
            resp = input("Does this pairing look correct? [y/N]: ").strip().lower()
            if resp != "y":
                print("Aborted. Re-check --source-root / --images-dir naming and try again.")
                return
        write_pairing_manifest(pairing, output_dir)
        name_map = {source: cam for source, cam in pairing}

    for source_name in tqdm(source_files, desc="Extracting Clean Plates"):
        source_path = os.path.join(source_root, source_name)

        frame = load_source_frame(source_path, frame_number)
        if frame is None:
            continue

        if pairing:
            base_name = name_map[source_name]
        else:
            base_name = os.path.splitext(source_name)[0]
        out_path = os.path.join(output_dir, f"{base_name}.jpg")

        # Quality 98 is nearly lossless but keeps file size reasonable
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    print(f"\nSuccess! Extracted plates are located in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract/copy a synced clean-plate frame per camera from a folder of videos or images."
    )

    parser.add_argument("--source-root", dest="source_root",
                         help="Directory containing the clean-plate source files (videos and/or images)")
    parser.add_argument("--video-root", dest="source_root",
                         help=argparse.SUPPRESS)  # deprecated alias
    parser.add_argument("--output-dir", required=True, help="Directory to save the clean plates")
    parser.add_argument("--frame-number", type=int, default=0,
                         help="0-based index of the frame to extract (ignored for image sources)")
    parser.add_argument("--images-dir", default=None,
                         help="Path to data_dir/images. If given, output files are named after the "
                              "matching camera folder (sorted source list paired positionally with "
                              "sorted camera folder list) instead of the source file's own filename, "
                              "so generate_masks.py can find them as background_dir/{cam_name}.jpg")
    parser.add_argument("--yes", action="store_true", dest="assume_yes",
                         help="Skip the pairing confirmation prompt (for scripted/CI use)")

    args = parser.parse_args()

    if args.source_root is None:
        parser.error("the following arguments are required: --source-root")

    extract_frames_direct(
        args.source_root, args.output_dir, args.frame_number,
        images_dir=args.images_dir, assume_yes=args.assume_yes,
    )
