#!/usr/bin/env python3
"""Export raw L3MHET ray-distance depth and opacity without visualization.

Arguments before ``--`` belong to this script. Arguments after ``--`` are
passed to EasyVolcap's normal test configuration loader. Unlike the stock
DEPTH visualizer, this writes the uncurved floating-point ``dpt_map`` and
``acc_map`` tensors, so the result remains suitable for geometric checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def parse_local_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing raw-depth export directory.",
    )
    return parser.parse_args(argv)


def _scalar(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return int(value)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_plane(value: Any, height: int, width: int) -> np.ndarray:
    array = _numpy(value)
    if array.size != height * width:
        raise ValueError(
            f"Expected {height}x{width}={height * width} samples, got "
            f"shape {array.shape} ({array.size} values)"
        )
    return np.asarray(array, dtype=np.float32).reshape(height, width)


def _restore_full_image(
    array: np.ndarray, meta: Any, *, fill_value: float
) -> np.ndarray:
    if "orig_h" not in meta:
        return array
    orig_h = _scalar(meta.orig_h)
    orig_w = _scalar(meta.orig_w)
    crop_x = _scalar(meta.crop_x)
    crop_y = _scalar(meta.crop_y)
    output = np.full((orig_h, orig_w), fill_value, dtype=np.float32)
    h, w = array.shape
    if crop_x < 0 or crop_y < 0 or crop_x + w > orig_w or crop_y + h > orig_h:
        raise ValueError(
            f"Invalid crop ({crop_x},{crop_y},{w},{h}) in {orig_w}x{orig_h} image"
        )
    output[crop_y : crop_y + h, crop_x : crop_x + w] = array
    return output


def _restore_full_intrinsics(K: np.ndarray, meta: Any) -> np.ndarray:
    """Undo EasyVolcap's crop-origin shift after restoring a full image."""
    restored = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    if "orig_h" in meta:
        restored[0, 2] += _scalar(meta.crop_x)
        restored[1, 2] += _scalar(meta.crop_y)
    return restored


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    local_args = parse_local_args(sys.argv[1:separator])
    evc_args = sys.argv[separator + 1 :] if separator < len(sys.argv) else []

    output_dir = local_args.output_dir.resolve()
    if output_dir.exists() and not local_args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure EasyVolcap before importing its registries, matching the
    # upstream extract_optimized_cameras.py entrypoint ordering.
    sys.path.append(".")
    sys.argv = [sys.argv[0], "-t", "test", *evc_args]

    import torch
    from easyvolcap.engine import cfg
    from easyvolcap.scripts.main import test
    from easyvolcap.utils.data_utils import add_iter, to_cuda

    runner = test(cfg, dry_run=True)
    epoch = int(runner.load_network())
    runner.model.eval()
    dataset = runner.val_dataloader.dataset

    entries: list[dict[str, Any]] = []
    for batch in runner.val_dataloader:
        iteration = epoch * runner.ep_iter - 1
        batch = add_iter(batch, iteration, runner.total_iter)
        batch = to_cuda(batch)
        with (
            torch.inference_mode(),
            torch.no_grad(),
            torch.cuda.amp.autocast(
                enabled=runner.test_use_amp,
                cache_enabled=runner.test_amp_cached,
            ),
        ):
            output = runner.model(batch)

        height = _scalar(batch.meta.H)
        width = _scalar(batch.meta.W)
        depth = _restore_full_image(
            _image_plane(output.dpt_map, height, width),
            batch.meta,
            fill_value=np.nan,
        )
        opacity = _restore_full_image(
            _image_plane(output.acc_map, height, width),
            batch.meta,
            fill_value=0.0,
        )
        camera_index = _scalar(batch.meta.camera_index)
        frame_index = _scalar(batch.meta.frame_index)
        camera_name = str(dataset.camera_names[camera_index])
        relative = Path(camera_name) / f"frame-{frame_index:06d}.npz"
        destination = output_dir / relative
        _atomic_savez(
            destination,
            depth=depth.astype(np.float32, copy=False),
            opacity=opacity.astype(np.float32, copy=False),
            K=_restore_full_intrinsics(
                _numpy(batch.K).reshape(-1, 3, 3)[0], batch.meta
            ),
            R=_numpy(batch.R).reshape(-1, 3, 3)[0].astype(np.float64),
            T=_numpy(batch.T).reshape(-1, 3)[0].astype(np.float64),
            camera_index=np.asarray(camera_index, dtype=np.int32),
            frame_index=np.asarray(frame_index, dtype=np.int32),
        )
        entries.append(
            {
                "camera": camera_name,
                "camera_index": camera_index,
                "frame_index": frame_index,
                "path": relative.as_posix(),
                "height": int(depth.shape[0]),
                "width": int(depth.shape[1]),
                "valid_depth": int(np.isfinite(depth).sum()),
                "opacity_ge_0_5": int(np.count_nonzero(opacity >= 0.5)),
            }
        )
        print(f"exported {camera_name} frame {frame_index}: {destination}")

    manifest = {
        "format": "l3mhet-raw-depth-v1",
        "depth_semantics": "distance_along_normalized_world_ray",
        "use_z_depth": bool(getattr(runner.model, "use_z_depth", False)),
        "correct_pix": bool(getattr(runner.model, "correct_pix", True)),
        "checkpoint_epoch": epoch,
        "entries": entries,
    }
    if manifest["use_z_depth"]:
        raise RuntimeError(
            "This exporter expected L3MHET ray-distance depth, but the loaded "
            "model reports use_z_depth=True. No compatible manifest was written."
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "views": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
