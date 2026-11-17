#!/usr/bin/env python3
"""Build a one-frame EasyVolcap/4K4D L3MHET camera-refinement dataset.

This prepares the data layout used by 4K4D's documented camera optimization
path: a static L3MHET model, ``OptimizableCamera``, and foreground masks.  The
input camera model remains the source of intrinsics/extrinsics; COLMAP points
are used only to derive a conservative scene bound.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file

from typing import Any

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def _load_colmap_io(repo_root: Path):
    module_path = trigs_file("io/colmap.py", repo_root)
    spec = importlib.util.spec_from_file_location("trigs_colmap_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_extension(path: Path) -> str:
    for extension in (".bin", ".txt"):
        if all(
            (path / f"{stem}{extension}").is_file()
            for stem in ("cameras", "images", "points3D")
        ):
            return extension
    raise FileNotFoundError(f"No complete COLMAP model found under {path}")


def _camera_folder(image_name: str) -> str:
    parts = Path(image_name).parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Expected a safe camera/image path, got {image_name!r}")
    return parts[0]


def _find_frame(root: Path, camera: str, frame: int, digits: int) -> Path:
    stem = f"{frame:0{digits}d}"
    folder = root / camera
    matches = [folder / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one frame {stem} under {folder}, found {matches}"
        )
    return matches[0]


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill only background components not connected to the image border."""
    import cv2

    foreground = np.asarray(binary, dtype=np.uint8)
    padded = np.pad(foreground, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = flood == 0
    filled = np.logical_or(padded != 0, holes)
    return filled[1:-1, 1:-1]


def _prepare_mask(
    source: Path,
    destination: Path,
    expected_size: tuple[int, int],
    *,
    threshold: int,
    fill_holes: bool,
    dilate_px: int,
) -> dict[str, Any]:
    import cv2

    with Image.open(source) as decoded:
        grayscale = np.asarray(decoded.convert("L"))
    if (grayscale.shape[1], grayscale.shape[0]) != expected_size:
        raise ValueError(
            f"Mask {source} is {grayscale.shape[1]}x{grayscale.shape[0]}, expected "
            f"{expected_size[0]}x{expected_size[1]}"
        )
    before = grayscale >= threshold
    after = _fill_holes(before) if fill_holes else before
    filled_pixels = int(np.count_nonzero(after & ~before))
    if dilate_px:
        diameter = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        after = cv2.dilate(after.astype(np.uint8), kernel, iterations=1) != 0
    if not after.any() or after.all():
        raise ValueError(
            f"Prepared mask must contain foreground and background pixels: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(after.astype(np.uint8) * 255, mode="L").save(
        destination, format="PNG"
    )
    return {
        "source": str(source),
        "foreground_fraction_before": float(before.mean()),
        "foreground_fraction_after": float(after.mean()),
        "filled_hole_pixels": filled_pixels,
    }


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copyfile(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def _camera_intrinsics(camera: Any) -> tuple[np.ndarray, np.ndarray]:
    model = (
        camera.model.decode() if isinstance(camera.model, bytes) else str(camera.model)
    )
    p = np.asarray(camera.params, dtype=np.float64)
    distortion = np.zeros(5, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, distortion[0] = p
        fx = fy = f
    elif model == "RADIAL":
        f, cx, cy, distortion[0], distortion[1] = p
        fx = fy = f
    elif model == "OPENCV":
        fx, fy, cx, cy, *dist = p
        distortion[:4] = dist[:4]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {model!r}; refusing a lossy conversion"
        )
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, distortion.reshape(5, 1)


def _yaml_matrix(key: str, matrix: np.ndarray) -> list[str]:
    # EasyVolcap's visual-hull code constructs image-coordinate tensors as
    # float32.  Writing camera matrices as float64 produces a deterministic
    # Double-vs-Float matmul failure before training begins.
    value = np.asarray(matrix, dtype=np.float32)
    data = ", ".join(f"{item:.8g}" for item in value.reshape(-1))
    return [
        f"{key}: !!opencv-matrix",
        f"  rows: {value.shape[0]}",
        f"  cols: {value.shape[1]}",
        "  dt: f",
        f"  data: [{data}]",
    ]


def _write_easyvolcap_cameras(
    output: Path,
    names: list[str],
    cameras_by_name: dict[str, dict[str, Any]],
) -> None:
    intri = ["%YAML:1.0", "---", "names:"] + [f'  - "{name}"' for name in names]
    extri = ["%YAML:1.0", "---", "names:"] + [f'  - "{name}"' for name in names]
    for name in names:
        row = cameras_by_name[name]
        intri += _yaml_matrix(f"K_{name}", row["K"])
        intri += [
            f"H_{name}: {float(row['height']):.1f}",
            f"W_{name}: {float(row['width']):.1f}",
        ]
        intri += _yaml_matrix(f"D_{name}", row["D"])
        extri += _yaml_matrix(f"Rot_{name}", row["R"])
        extri += _yaml_matrix(f"T_{name}", row["T"])
    (output / "intri.yml").write_text("\n".join(intri) + "\n")
    (output / "extri.yml").write_text("\n".join(extri) + "\n")


def _expanded_bounds(points: np.ndarray, margin: float) -> np.ndarray:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    extent = upper - lower
    pad = np.maximum(extent * margin, max(float(extent.max()), 1.0) * 1e-4)
    return np.stack((lower - pad, upper + pad))


def _config_text(data_root: Path, bounds: np.ndarray, experiment_name: str) -> str:
    bound_rows = ", ".join(
        "[" + ", ".join(f"{value:.12g}" for value in row) + "]" for row in bounds
    )
    quoted_data_root = json.dumps(str(data_root))
    quoted_experiment_name = json.dumps(experiment_name)
    return f"""configs:
    - configs/base.yaml
    - configs/models/l3mhet.yaml
    - configs/specs/mask.yaml
    - configs/specs/static.yaml
    - configs/specs/optcam.yaml
    - configs/specs/transient.yaml

exp_name: {quoted_experiment_name}

dataloader_cfg:
    dataset_cfg: &dataset_cfg
        data_root: {quoted_data_root}
        images_dir: images
        masks_dir: masks
        bounds: [{bound_rows}]
        view_sample: [0, null, 1]
        frame_sample: [0, 1, 1]
        use_aligned_cameras: false

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    model = args.reference_colmap.resolve()
    images_root = args.images_root.resolve()
    masks_root = args.masks_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    colmap_io = _load_colmap_io(repo_root)
    extension = _model_extension(model)
    cameras, images, points = colmap_io.read_model(str(model), extension)
    if not images or not points:
        raise ValueError(
            "The reference model must contain cameras, images, and NPZ-derived points"
        )

    registered = sorted(images.values(), key=lambda image: image.name)
    folders = [_camera_folder(image.name) for image in registered]
    if len(set(folders)) != len(folders):
        raise ValueError("Expected exactly one registered image per camera folder")

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    rows: list[dict[str, Any]] = []
    easycams: dict[str, dict[str, Any]] = {}
    try:
        for camera_name, image in zip(folders, registered):
            camera = cameras[image.camera_id]
            source_image = _find_frame(
                images_root, camera_name, args.frame, args.frame_digits
            )
            source_mask = _find_frame(
                masks_root,
                camera_name,
                args.frame + args.mask_frame_offset,
                args.frame_digits,
            )
            with Image.open(source_image) as decoded:
                actual_size = decoded.size
            expected_size = (int(camera.width), int(camera.height))
            if actual_size != expected_size:
                raise ValueError(
                    f"Image {source_image} is {actual_size}, camera expects {expected_size}"
                )

            extension_out = source_image.suffix.lower()
            relative = Path(camera_name) / f"000000{extension_out}"
            _copy_or_link(source_image, temp / "images" / relative, args.images_mode)
            mask_stats = _prepare_mask(
                source_mask,
                temp / "masks" / relative,
                expected_size,
                threshold=args.mask_threshold,
                fill_holes=args.fill_mask_holes,
                dilate_px=args.mask_dilate_px,
            )

            K, D = _camera_intrinsics(camera)
            easycams[camera_name] = {
                "K": K,
                "D": D,
                "height": camera.height,
                "width": camera.width,
                "R": colmap_io.qvec2rotmat(image.qvec),
                "T": np.asarray(image.tvec, dtype=np.float64).reshape(3, 1),
            }
            rows.append(
                {
                    "camera": camera_name,
                    "colmap_image": image.name,
                    "source_image": str(source_image),
                    "output_image": str(Path("images") / relative),
                    "output_mask": str(Path("masks") / relative),
                    **mask_stats,
                }
            )

        names = sorted(easycams)
        _write_easyvolcap_cameras(temp, names, easycams)
        xyz = np.stack([point.xyz for point in points.values()]).astype(np.float64)
        bounds = _expanded_bounds(xyz, args.bounds_margin)
        experiment_name = args.experiment_name or f"l3mhet_{output.name}_static"
        config = temp / "l3mhet_camera_refine.yaml"
        config.write_text(_config_text(output, bounds, experiment_name))
        manifest = {
            "format": "easyvolcap-l3mhet-camera-refine-v1",
            "method": "4K4D documented L3MHET + OptimizableCamera static-frame refinement",
            "reference_colmap": str(model),
            "frame": args.frame,
            "views": len(rows),
            "bounds": bounds.tolist(),
            "mask_threshold": args.mask_threshold,
            "fill_mask_holes": args.fill_mask_holes,
            "mask_dilate_px": args.mask_dilate_px,
            "images_mode": args.images_mode,
            "config": str(output / config.name),
            "cameras": rows,
        }
        (temp / "camera-refinement-dataset.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temp.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-colmap", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--mask-frame-offset", type=int, default=0)
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--mask-threshold", type=int, default=1)
    parser.add_argument("--fill-mask-holes", action="store_true")
    parser.add_argument("--mask-dilate-px", type=int, default=0)
    parser.add_argument("--bounds-margin", type=float, default=0.10)
    parser.add_argument(
        "--images-mode", choices=("copy", "hardlink", "symlink"), default="hardlink"
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args(argv)
    if not 0 <= args.mask_threshold <= 255:
        parser.error("--mask-threshold must be in [0, 255]")
    if args.mask_dilate_px < 0:
        parser.error("--mask-dilate-px must be non-negative")
    if args.bounds_margin < 0:
        parser.error("--bounds-margin must be non-negative")
    return args


def main() -> None:
    manifest = build(parse_args())
    summary = {key: value for key, value in manifest.items() if key != "cameras"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
