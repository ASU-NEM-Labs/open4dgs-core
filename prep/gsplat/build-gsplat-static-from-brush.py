#!/usr/bin/env python3
"""Build a gsplat COLMAP dataset using a BRUSH init.ply as points3D."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trigs_source import trigs_file


import numpy as np


def _load_colmap_io(repo_root: Path):
    module_path = trigs_file("io/colmap.py", repo_root)
    spec = importlib.util.spec_from_file_location("trigs_colmap_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_brush_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                count = int(text.rsplit(" ", 1)[1])
            elif text.startswith("property float "):
                properties.append(text.rsplit(" ", 1)[1])
            elif text == "end_header":
                break
        expected = ["x", "y", "z", "red", "green", "blue"]
        if properties != expected or count is None:
            raise ValueError(
                f"Expected binary XYZ + float RGB PLY, got properties={properties}"
            )
        rows = np.fromfile(handle, dtype="<f4", count=count * 6).reshape(count, 6)
        if handle.read(1):
            raise ValueError(f"PLY has unexpected trailing bytes: {path}")

    xyz = rows[:, :3].astype(np.float64)
    colors = rows[:, 3:6].astype(np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(colors).all():
        raise ValueError("PLY contains non-finite XYZ/RGB values")
    if colors.size and float(colors.max()) <= 1.5:
        colors *= 255.0
    rgb = np.rint(np.clip(colors, 0.0, 255.0)).astype(np.uint8)
    return xyz, rgb


def _remove_appledouble(root: Path) -> None:
    """Remove metadata sidecars created when writing through a macOS network mount."""
    for path in sorted(root.rglob("._*"), reverse=True):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()


def build(
    source: Path,
    output: Path,
    repo_root: Path,
    images_link_target: Path | None = None,
    images_mode: str = "symlink",
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    colmap_io = _load_colmap_io(repo_root.resolve())
    model = source / "sparse/0"
    cameras = colmap_io.read_cameras_binary(str(model / "cameras.bin"))
    source_images = colmap_io.read_images_binary(str(model / "images.bin"))
    xyz, rgb = _read_brush_xyz_rgb(source / "init.ply")

    images = {
        image_id: colmap_io.Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
        for image_id, image in source_images.items()
    }
    empty_i32 = np.empty((0,), dtype=np.int32)
    points = {
        index + 1: colmap_io.Point3D(
            id=index + 1,
            xyz=xyz[index],
            rgb=rgb[index],
            error=0.0,
            image_ids=empty_i32,
            point2D_idxs=empty_i32,
        )
        for index in range(len(xyz))
    }

    sparse = output / "sparse/0"
    sparse.mkdir(parents=True)
    colmap_io.write_cameras_binary(cameras, str(sparse / "cameras.bin"))
    colmap_io.write_images_binary(images, str(sparse / "images.bin"))
    colmap_io.write_points3D_binary(points, str(sparse / "points3D.bin"))

    source_images_dir = source / "images"
    link_target = images_link_target or source_images_dir
    output_images = output / "images"
    if images_mode == "symlink":
        os.symlink(link_target, output_images, target_is_directory=True)
    elif images_mode in {"hardlink", "copy"}:
        for source_image in source_images_dir.rglob("*"):
            if not source_image.is_file():
                continue
            destination = output_images / source_image.relative_to(source_images_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if images_mode == "hardlink":
                os.link(source_image, destination)
            else:
                shutil.copyfile(source_image, destination)
    else:
        raise ValueError(f"Unsupported images mode: {images_mode}")
    manifest = {
        "source": str(source),
        "images": len(images),
        "cameras": len(cameras),
        "points": len(points),
        "initialization": str(source / "init.ply"),
        "images_link_target": str(link_target),
        "images_mode": images_mode,
        "point_tracks": "intentionally empty",
    }
    (output / "gsplat-static.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _remove_appledouble(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--images-link-target",
        type=Path,
        default=None,
        help="Image directory path as seen by the machine that will train",
    )
    parser.add_argument(
        "--images-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source,
                args.output,
                args.repo_root,
                images_link_target=args.images_link_target,
                images_mode=args.images_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
