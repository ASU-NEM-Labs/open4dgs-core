#!/usr/bin/env python3
"""Build a one-instant COLMAP dataset for a static BRUSH training test.

The reference COLMAP model contains one registered image/pose per physical
camera, while ``images/<camera>/<frame>`` contains the video frames.  This
tool copies one synchronized frame into the exact image names recorded by
COLMAP.  BRUSH can therefore consume the resulting directory directly.

If ``--npz-path`` is supplied, points belonging to the selected NPZ instant
(plus global static points) are exported as ``init.ply``.  BRUSH gives that
file precedence over COLMAP's ``points3D`` initialization.  Omit the option to
make a control dataset initialized from COLMAP points instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file

from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_colmap_io():
    """Load the repository's lightweight COLMAP reader without trigs.io imports."""
    source = trigs_file("io/colmap.py")
    if not source.exists():
        raise RuntimeError(f"Repository COLMAP reader not found: {source}")
    spec = importlib.util.spec_from_file_location("_trigs_colmap_io", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load COLMAP reader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_extension(model_dir: Path) -> str:
    for ext in (".bin", ".txt"):
        if all((model_dir / f"{stem}{ext}").is_file() for stem in ("cameras", "images", "points3D")):
            return ext
    raise FileNotFoundError(
        f"{model_dir} is not a complete COLMAP model; expected cameras, images, "
        "and points3D files in either .bin or .txt format"
    )


def _read_model_metadata(model_dir: Path, ext: str):
    colmap_io = _load_colmap_io()
    if ext == ".bin":
        cameras = colmap_io.read_cameras_binary(str(model_dir / "cameras.bin"))
        images = colmap_io.read_images_binary(str(model_dir / "images.bin"))
    else:
        cameras = colmap_io.read_cameras_text(str(model_dir / "cameras.txt"))
        images = colmap_io.read_images_text(str(model_dir / "images.txt"))
    return cameras, images


def _safe_colmap_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe COLMAP image name: {name!r}")
    return path


def _camera_folder(image_name: str) -> PurePosixPath:
    path = _safe_colmap_name(image_name)
    return path.parent if path.parent != PurePosixPath(".") else PurePosixPath(path.stem)


def _find_frame_file(root: Path, camera_folder: PurePosixPath, frame: int, digits: int) -> Path:
    stem = f"{frame:0{digits}d}"
    base = root.joinpath(*camera_folder.parts)
    found = [
        path
        for path in base.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS
    ] if base.is_dir() else []
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for camera {camera_folder.as_posix()!r}, "
            f"frame {frame} under {base}; found {[str(p) for p in found]}"
        )
    return found[0]


def _copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unknown copy mode: {mode}")


def _scalar(npz: np.lib.npyio.NpzFile, name: str, default: Any = None) -> Any:
    if name not in npz:
        return default
    value = np.asarray(npz[name])
    return value.reshape(()).item()


def _npz_rows_for_static_frame(
    npz: np.lib.npyio.NpzFile,
    frame: int,
    *,
    allow_nearest: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    for name in ("positions", "colors"):
        if name not in npz:
            raise KeyError(f"NPZ is missing required array {name!r}")

    positions = np.asarray(npz["positions"])
    colors = np.asarray(npz["colors"])
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"NPZ positions must have shape [N,3], got {positions.shape}")
    if colors.ndim != 2 or colors.shape[1] < 3 or len(colors) != len(positions):
        raise ValueError(f"NPZ colors must have shape [N,>=3], got {colors.shape}")

    count = len(positions)
    is_static = (
        np.asarray(npz["is_static"]).reshape(-1).astype(bool)
        if "is_static" in npz
        else np.zeros(count, dtype=bool)
    )
    if len(is_static) != count:
        raise ValueError("NPZ is_static length does not match positions")

    selection_kind: str
    selected_instant: float | int
    if "source_frames" in npz:
        source_frames = np.asarray(npz["source_frames"]).reshape(-1).astype(np.int64)
        if len(source_frames) != count:
            raise ValueError("NPZ source_frames length does not match positions")
        available = np.unique(source_frames[~is_static] if (~is_static).any() else source_frames)
        if available.size == 0:
            dynamic = np.zeros(count, dtype=bool)
            selected_instant = int(frame)
        else:
            nearest = int(available[np.argmin(np.abs(available - int(frame)))])
            if nearest != int(frame) and not allow_nearest:
                preview = ", ".join(map(str, available[:20]))
                raise ValueError(
                    f"NPZ has no dynamic rows for source frame {frame}. Available frames: "
                    f"{preview}{' ...' if len(available) > 20 else ''}. Choose a keyframe or "
                    "pass --npz-nearest-frame explicitly."
                )
            selected_instant = nearest
            dynamic = (~is_static) & (source_frames == nearest)
        selection_kind = "source_frames"
    elif "times" in npz:
        times = np.asarray(npz["times"]).reshape(-1).astype(np.float64)
        if len(times) != count:
            raise ValueError("NPZ times length does not match positions")
        frame_start = int(_scalar(npz, "frame_start", frame))
        frame_end = int(_scalar(npz, "frame_end", frame_start))
        denom = max(frame_end - frame_start, 1)
        target_time = (int(frame) - frame_start) / denom
        dynamic_times = np.unique(times[~is_static] if (~is_static).any() else times)
        if dynamic_times.size == 0:
            dynamic = np.zeros(count, dtype=bool)
            selected_instant = float(target_time)
        else:
            selected_instant = float(dynamic_times[np.argmin(np.abs(dynamic_times - target_time))])
            dynamic = (~is_static) & np.isclose(times, selected_instant, rtol=0.0, atol=1e-7)
        selection_kind = "normalized_times_nearest"
    else:
        raise KeyError("NPZ needs source_frames or times to select one static instant")

    keep = is_static | dynamic
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(colors[:, :3]).all(axis=1)
    selected = keep & finite
    if not selected.any():
        raise ValueError(f"No finite NPZ points remain for frame {frame}")

    details = {
        "input_rows": int(count),
        "selection_kind": selection_kind,
        "requested_frame": int(frame),
        "selected_instant": selected_instant,
        "selected_rows": int(selected.sum()),
        "selected_static_rows": int((selected & is_static).sum()),
        "selected_dynamic_rows": int((selected & ~is_static).sum()),
        "discarded_nonfinite_rows": int((keep & ~finite).sum()),
    }
    return selected, details


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors[:, :3], dtype=np.float32)
    if colors.size and float(np.nanmax(colors)) > 1.5:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _write_brush_init_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    """Write binary XYZ + float RGB; BRUSH fills scale, opacity, and rotation."""
    rows = np.empty(
        len(positions),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "<f4"), ("green", "<f4"), ("blue", "<f4"),
        ],
    )
    for index, name in enumerate(("x", "y", "z")):
        rows[name] = positions[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        rows[name] = colors[:, index]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by build-brush-static-dataset.py\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float red\nproperty float green\nproperty float blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        rows.tofile(handle)


def _visibility_summary(positions: np.ndarray, images: Iterable[Any]) -> dict[str, Any]:
    """Cheap coordinate-frame check: fraction of points with positive camera Z."""
    sample = np.asarray(positions, dtype=np.float64)
    if len(sample) > 200_000:
        sample = sample[np.linspace(0, len(sample) - 1, 200_000, dtype=np.int64)]
    fractions = []
    for image in images:
        rotation = image.qvec2rotmat()
        camera_z = sample @ rotation[2, :] + float(image.tvec[2])
        fractions.append(float(np.mean(camera_z > 0.0)))
    return {
        "sampled_points": int(len(sample)),
        "positive_depth_fraction_min": float(min(fractions)),
        "positive_depth_fraction_median": float(np.median(fractions)),
        "positive_depth_fraction_max": float(max(fractions)),
    }


def _readme_text(dataset_dir: Path, mask_mode: str, has_npz: bool) -> str:
    init_note = "NPZ-derived init.ply" if has_npz else "COLMAP points3D"
    if mask_mode == "transparent":
        mask_note = (
            "Masks: embedded as soft image alpha. BRUSH matches output transparency, "
            "including penalizing opacity outside the subject."
        )
        main_alpha_arg = " --alpha-mode transparent"
    elif mask_mode == "ignore":
        mask_note = (
            "Masks: separate masks/ tree (ignored pixels do not contribute to loss). "
            "Use current BRUSH main; v0.3.0 cannot pair this nested mask layout."
        )
        main_alpha_arg = " --alpha-mode masked"
    else:
        mask_note = "Masks: disabled."
        main_alpha_arg = ""
    return f"""BRUSH static diagnostic dataset
================================

Initialization: {init_note}
{mask_note}

BRUSH v0.3.0 release, with viewer:
  brush {dataset_dir.resolve()} --with-viewer --total-steps 30000 --max-resolution 1920 --export-path {dataset_dir.resolve() / 'brush_exports'}

BRUSH v0.3.0 release, headless:
  brush {dataset_dir.resolve()} --total-steps 30000 --max-resolution 1920 --export-path {dataset_dir.resolve() / 'brush_exports'}

Current BRUSH main branch:
  brush {dataset_dir.resolve()} --with-viewer --total-train-iters 30000 --max-resolution 1920{main_alpha_arg} --export-path {dataset_dir.resolve() / 'brush_exports'}

BRUSH automatically uses init.ply when present. In masked mode, white/255 mask
pixels contribute to RGB loss and black/0 pixels are ignored.
"""


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    model_dir = args.reference_colmap.resolve()
    images_root = (args.images_root or data_dir / "images").resolve()
    masks_root = None if args.skip_masks else (args.masks_root or data_dir / "masks").resolve()
    if masks_root is not None and not masks_root.exists():
        if args.masks_root is not None:
            raise FileNotFoundError(f"Masks root does not exist: {masks_root}")
        masks_root = None
    mask_mode = "none" if masks_root is None else args.mask_mode

    ext = _model_extension(model_dir)
    cameras, registered = _read_model_metadata(model_dir, ext)
    images = sorted(registered.values(), key=lambda image: image.name)
    if not images:
        raise ValueError(f"COLMAP model has no registered images: {model_dir}")

    camera_folders = [_camera_folder(image.name) for image in images]
    duplicates = sorted({p.as_posix() for p in camera_folders if camera_folders.count(p) > 1})
    if duplicates:
        raise ValueError(
            "Reference model must contain one registered image per physical camera; "
            f"duplicate camera folders: {duplicates}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    manifest: dict[str, Any] = {
        "format": "brush-static-colmap-v1",
        "source_data_dir": str(data_dir),
        "source_colmap": str(model_dir),
        "image_frame": int(args.frame),
        "mask_frame": int(args.frame + args.mask_frame_offset),
        "copy_mode": args.copy_mode,
        "registered_views": len(images),
        "views": [],
    }

    try:
        sparse_out = temp_dir / "sparse/0"
        sparse_out.mkdir(parents=True)
        for stem in ("cameras", "images", "points3D"):
            _copy_file(model_dir / f"{stem}{ext}", sparse_out / f"{stem}{ext}", args.copy_mode)

        missing_masks: list[str] = []
        for image, camera_folder in zip(images, camera_folders):
            colmap_name = _safe_colmap_name(image.name)
            source_image = _find_frame_file(images_root, camera_folder, args.frame, args.frame_digits)
            with Image.open(source_image) as decoded:
                actual_size = tuple(decoded.size)
            camera = cameras[image.camera_id]
            expected_size = (int(camera.width), int(camera.height))
            if actual_size != expected_size:
                raise ValueError(
                    f"Image/COLMAP resolution mismatch for {camera_folder.as_posix()}: "
                    f"{source_image} is {actual_size[0]}x{actual_size[1]}, camera "
                    f"{image.camera_id} expects {expected_size[0]}x{expected_size[1]}"
                )
            source_mask = None
            destination_mask = None
            if masks_root is not None:
                try:
                    source_mask = _find_frame_file(
                        masks_root,
                        camera_folder,
                        args.frame + args.mask_frame_offset,
                        args.frame_digits,
                    )
                except FileNotFoundError:
                    if not args.allow_missing_masks:
                        raise
                    missing_masks.append(camera_folder.as_posix())
                if source_mask is not None:
                    if mask_mode == "ignore":
                        mask_rel = colmap_name.parent / f"{colmap_name.stem}{source_mask.suffix.lower()}"
                        destination_mask = temp_dir / "masks" / Path(*mask_rel.parts)
                        _copy_file(source_mask, destination_mask, args.copy_mode)

            destination_image = temp_dir / "images" / Path(*colmap_name.parts)
            if mask_mode == "transparent":
                if source_mask is None:
                    raise FileNotFoundError(
                        f"Transparent mask mode requires a mask for {camera_folder.as_posix()}"
                    )
                with Image.open(source_image) as source_decoded, Image.open(source_mask) as mask_decoded:
                    rgb = source_decoded.convert("RGB")
                    alpha = mask_decoded.getchannel("A") if mask_decoded.mode in {"LA", "RGBA"} else mask_decoded.convert("L")
                    if alpha.size != rgb.size:
                        raise ValueError(
                            f"Mask/image resolution mismatch for {camera_folder.as_posix()}: "
                            f"mask {source_mask} is {alpha.width}x{alpha.height}, image is "
                            f"{rgb.width}x{rgb.height}"
                        )
                    rgba = rgb.copy()
                    rgba.putalpha(alpha)
                    destination_image.parent.mkdir(parents=True, exist_ok=True)
                    # Keep the exact COLMAP filename, but use PNG bytes so alpha is lossless.
                    # BRUSH detects the image format from the bytes rather than the suffix.
                    rgba.save(destination_image, format="PNG")
            else:
                _copy_file(source_image, destination_image, args.copy_mode)

            manifest["views"].append(
                {
                    "colmap_name": image.name,
                    "camera_folder": camera_folder.as_posix(),
                    "camera_id": int(image.camera_id),
                    "source_image": str(source_image),
                    "output_image": str(destination_image.relative_to(temp_dir)),
                    "source_mask": str(source_mask) if source_mask is not None else None,
                    "output_mask": (
                        str(destination_mask.relative_to(temp_dir))
                        if destination_mask is not None
                        else None
                    ),
                }
            )

        manifest["masks"] = {
            "enabled": mask_mode != "none",
            "mode": mask_mode,
            "copied": sum(view["output_mask"] is not None for view in manifest["views"]),
            "embedded_alpha": (
                sum(view["source_mask"] is not None for view in manifest["views"])
                if mask_mode == "transparent"
                else 0
            ),
            "missing_camera_folders": missing_masks,
            "semantics": (
                "white/255=opaque, black/0=transparent"
                if mask_mode == "transparent"
                else "white/255=train, black/0=ignore"
            ),
        }

        selected_positions = None
        if args.npz_path is not None:
            npz_path = args.npz_path.resolve()
            with np.load(npz_path, allow_pickle=False) as npz:
                npz_frame = args.npz_frame if args.npz_frame is not None else args.frame
                keep, details = _npz_rows_for_static_frame(
                    npz, npz_frame, allow_nearest=args.npz_nearest_frame
                )
                selected_positions = np.asarray(npz["positions"])[keep].astype(np.float32)
                selected_colors = _normalize_colors(np.asarray(npz["colors"])[keep])
            _write_brush_init_ply(temp_dir / "init.ply", selected_positions, selected_colors)
            details["source"] = str(npz_path)
            details["bounds_min"] = selected_positions.min(axis=0).tolist()
            details["bounds_max"] = selected_positions.max(axis=0).tolist()
            details["camera_positive_depth"] = _visibility_summary(selected_positions, images)
            manifest["initialization"] = {"kind": "npz_init_ply", **details}
        else:
            manifest["initialization"] = {"kind": "colmap_points3D"}

        (temp_dir / "brush-dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temp_dir / "README.txt").write_text(
            _readme_text(output_dir, mask_mode, selected_positions is not None)
        )

        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output already exists: {output_dir}; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-colmap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True, help="Video frame to freeze")
    parser.add_argument("--images-root", type=Path, default=None, help="Default: DATA_DIR/images")
    parser.add_argument("--masks-root", type=Path, default=None, help="Default: DATA_DIR/masks if present")
    parser.add_argument("--skip-masks", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=("ignore", "transparent"),
        default="ignore",
        help=(
            "ignore: separate masks exclude pixels from loss (current BRUSH main); "
            "transparent: bake masks into image alpha and penalize outside opacity "
            "(recommended, including for BRUSH v0.3.0)"
        ),
    )
    parser.add_argument("--allow-missing-masks", action="store_true")
    parser.add_argument("--mask-frame-offset", type=int, default=0)
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--npz-path", type=Path, default=None)
    parser.add_argument("--npz-frame", type=int, default=None, help="Default: --frame")
    parser.add_argument(
        "--npz-nearest-frame",
        action="store_true",
        help="Allow the closest NPZ keyframe when the requested frame is absent",
    )
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_dataset(args)
    init = manifest["initialization"]
    print(f"Built BRUSH dataset: {args.output_dir.resolve()}")
    print(f"  views: {manifest['registered_views']}")
    print(f"  masks: {manifest['masks']['copied']}")
    if manifest["masks"]["embedded_alpha"]:
        print(f"  alpha-embedded images: {manifest['masks']['embedded_alpha']}")
    print(f"  initialization: {init['kind']}")
    if init["kind"] == "npz_init_ply":
        print(
            f"  NPZ rows: {init['selected_rows']:,} "
            f"({init['selected_dynamic_rows']:,} dynamic, {init['selected_static_rows']:,} static)"
        )
        depth = init["camera_positive_depth"]
        print(f"  median positive-depth fraction: {depth['positive_depth_fraction_median']:.3f}")
    print(f"  BRUSH commands: {args.output_dir.resolve() / 'README.txt'}")


if __name__ == "__main__":
    main()
