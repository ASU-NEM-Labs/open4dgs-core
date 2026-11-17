#!/usr/bin/env python3
"""Export 4DAnyone cameras to trainer JSON files and a COLMAP sparse model."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


PINHOLE_MODEL_ID = 1


def _camera_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cameras"), list):
        records = payload["cameras"]
    else:
        raise ValueError("Expected cameras.json to contain a list or a {'cameras': [...]} object.")

    if not records:
        raise ValueError("cameras.json contains no cameras.")

    records = sorted(records, key=lambda camera: int(camera.get("camera_id", 0)))
    source_ids = [int(record.get("camera_id", index)) for index, record in enumerate(records)]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"camera_id values must be unique, got {source_ids}.")

    for index, record in enumerate(records):
        source_id = int(record.get("camera_id", index))
        video = record.get("video")
        if video is not None:
            video_stem = Path(str(video)).stem
            if not video_stem.isdigit() or int(video_stem) != source_id:
                raise ValueError(
                    f"Camera {source_id}: video {video!r} does not match camera_id. "
                    "Camera identity must be preserved through extraction and conversion."
                )
    return records


def _detect_reference_image(
    records: list[dict[str, Any]], images_root: Path
) -> tuple[int, str]:
    """Detect one shared first frame and extension from camera image folders."""
    detected: list[tuple[int, str, str]] = []
    for sequence_index, record in enumerate(records):
        source_id = int(record.get("camera_id", sequence_index))
        folder = f"{source_id:02d}"
        camera_dir = images_root / folder
        if not camera_dir.is_dir():
            raise ValueError(f"Camera {source_id}: image folder does not exist: {camera_dir}")

        candidates = sorted(
            (
                path
                for path in camera_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.stem.isdigit()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=lambda path: (int(path.stem), path.name),
        )
        if not candidates:
            raise ValueError(f"Camera {source_id}: no numeric JPEG/PNG frames in {camera_dir}")
        first = candidates[0]
        detected.append((int(first.stem), first.suffix.lower().lstrip("."), folder))

    frame_extensions = {(frame, extension) for frame, extension, _ in detected}
    if len(frame_extensions) != 1:
        details = ", ".join(
            f"{folder}={frame:06d}.{extension}"
            for frame, extension, folder in detected
        )
        raise ValueError(
            "Camera folders do not share one first frame and extension: " + details
        )
    frame_index, extension = next(iter(frame_extensions))
    return int(frame_index), str(extension)


def _as_homogeneous(matrix: Any) -> np.ndarray:
    c2w = np.asarray(matrix, dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 3x4 or 4x4, got {c2w.shape}.")
    return c2w


def _rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP's [qw, qx, qy, qz]."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {r.shape}.")

    # Use the direct scalar-first conversion so the returned quaternion
    # round-trips through COLMAP's qvec2rotmat convention to *r*, not r.T.
    # The previous eigen-based implementation had the antisymmetric terms'
    # signs reversed and therefore encoded the transpose of every nonsymmetric
    # world-to-camera rotation while leaving its translation unchanged.
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qvec = np.asarray(
            [
                0.25 * scale,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(r)))
        if diagonal_index == 0:
            scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qvec = np.asarray(
                [
                    (r[2, 1] - r[1, 2]) / scale,
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qvec = np.asarray(
                [
                    (r[0, 2] - r[2, 0]) / scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qvec = np.asarray(
                [
                    (r[1, 0] - r[0, 1]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation produced an invalid zero/non-finite quaternion.")
    qvec /= norm
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _build_entries(
    records: list[dict[str, Any]], frame_index: int, extension: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cameras: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []

    for sequence_index, record in enumerate(records):
        source_id = int(record.get("camera_id", sequence_index))
        camera_id = sequence_index + 1
        image_id = sequence_index + 1
        folder = f"{source_id:02d}"
        image_name = f"{frame_index:06d}.{extension}"
        relative_name = f"{folder}/{image_name}"

        intrinsic = np.asarray(record["K"], dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"Camera {source_id}: K must be 3x3, got {intrinsic.shape}.")
        params = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
            dtype=np.float64,
        )

        c2w = _as_homogeneous(record["camera_to_world"])
        w2c = np.linalg.inv(c2w)
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]

        # Remove tiny numerical drift while preserving the represented rotation.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        width = int(record["image_width"])
        height = int(record["image_height"])
        cameras.append(
            {
                "camera_id": camera_id,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": params,
            }
        )
        images.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "source_camera_id": source_id,
                "name": relative_name,
                "folder": folder,
                "image_name": image_name,
                "R": rotation,
                "t": translation,
                "qvec": _rotation_to_qvec(rotation),
            }
        )

    return cameras, images


def _write_colmap_text(
    sparse_dir: Path, cameras: list[dict[str, Any]], images: list[dict[str, Any]]
) -> None:
    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras:
            params = " ".join(f"{value:.17g}" for value in camera["params"])
            handle.write(
                f"{camera['camera_id']} PINHOLE {camera['width']} {camera['height']} {params}\n"
            )

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for image in images:
            pose = np.concatenate([image["qvec"], image["t"]])
            pose_text = " ".join(f"{value:.17g}" for value in pose)
            handle.write(
                f"{image['image_id']} {pose_text} {image['camera_id']} {image['name']}\n\n"
            )

    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        handle.write("# Number of points: 0, mean track length: 0\n")


def _write_colmap_binary(
    sparse_dir: Path, cameras: list[dict[str, Any]], images: list[dict[str, Any]]
) -> None:
    with (sparse_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            handle.write(
                struct.pack(
                    "<iiQQ",
                    camera["camera_id"],
                    PINHOLE_MODEL_ID,
                    camera["width"],
                    camera["height"],
                )
            )
            handle.write(struct.pack("<4d", *camera["params"]))

    with (sparse_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            handle.write(struct.pack("<i", image["image_id"]))
            handle.write(struct.pack("<4d", *image["qvec"]))
            handle.write(struct.pack("<3d", *image["t"]))
            handle.write(struct.pack("<i", image["camera_id"]))
            handle.write(image["name"].encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))

    with (sparse_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 0))


def export(cameras_json: Path, output_root: Path, frame_index: int, extension: str) -> None:
    with cameras_json.open("r", encoding="utf-8") as handle:
        records = _camera_records(json.load(handle))

    cameras, images = _build_entries(records, frame_index, extension)
    sparse_dir = output_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scene_meta = {
        "scene_center": [0.0, 0.0, 0.0],
        "scene_radius_p90": 1.0,
        "scene_radius_p95": 1.0,
        "scene_radius_max": 1.0,
        "n_points": 0,
    }
    reference_cameras = []
    for camera, image in zip(cameras, images):
        reference_cameras.append(
            {
                "name": image["name"],
                "folder": image["folder"],
                "image_name": image["image_name"],
                "R": image["R"].tolist(),
                "t": image["t"].reshape(3).tolist(),
                "camera_model_name": camera["model"],
                "width": camera["width"],
                "height": camera["height"],
                "params": camera["params"].tolist(),
            }
        )

    with (output_root / "reference_export.json").open("w", encoding="utf-8") as handle:
        json.dump({"cameras": reference_cameras, "scene_meta": scene_meta}, handle, indent=2)
    with (output_root / "scene_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(scene_meta, handle, indent=2)

    _write_colmap_text(sparse_dir, cameras, images)
    _write_colmap_binary(sparse_dir, cameras, images)

    print(f"Exported {len(cameras)} registered cameras")
    print(f"Reference JSON: {output_root / 'reference_export.json'}")
    print(f"Scene metadata: {output_root / 'scene_meta.json'}")
    print(f"COLMAP model:   {sparse_dir}")
    print("COLMAP points3D is intentionally empty; poses came directly from 4DAnyone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create trainer JSON and sparse/0 directly from 4DAnyone cameras.json."
    )
    parser.add_argument("--cameras-json", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Dataset root where reference_export.json, scene_meta.json, and sparse/0 are written.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help=(
            "Synchronized reference frame used in COLMAP image names. When "
            "--images-root is supplied, defaults to the first numeric frame on disk; "
            "otherwise defaults to 0."
        ),
    )
    parser.add_argument(
        "--image-extension",
        default=None,
        help=(
            "Image extension without a dot. When --images-root is supplied, defaults "
            "to the detected extension; otherwise defaults to png."
        ),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help=(
            "Optional images directory used to validate camera folders and auto-detect "
            "the first frame/extension, e.g. <dataset>/images."
        ),
    )
    args = parser.parse_args()

    cameras_json = args.cameras_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    with cameras_json.open("r", encoding="utf-8") as handle:
        records = _camera_records(json.load(handle))

    detected_frame: int | None = None
    detected_extension: str | None = None
    if args.images_root is not None:
        images_root = args.images_root.expanduser().resolve()
        detected_frame, detected_extension = _detect_reference_image(records, images_root)

    frame_index = detected_frame if args.frame_index is None else int(args.frame_index)
    if frame_index is None:
        frame_index = 0
    if frame_index < 0:
        parser.error("--frame-index must be non-negative")

    extension = (
        detected_extension if args.image_extension is None else args.image_extension
    )
    if extension is None:
        extension = "png"
    extension = extension.lower().lstrip(".")
    if not extension or "/" in extension or "\\" in extension:
        parser.error("--image-extension must be a simple extension such as png or jpg")
    if detected_frame is not None and frame_index != detected_frame:
        parser.error(
            f"--frame-index={frame_index} disagrees with first frame on disk: "
            f"{detected_frame:06d}"
        )
    if detected_extension is not None and extension != detected_extension:
        parser.error(
            f"--image-extension={extension!r} disagrees with images on disk: "
            f"{detected_extension!r}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    export(cameras_json, output_root, frame_index, extension)


if __name__ == "__main__":
    main()
