#!/usr/bin/env python3
"""Measure, per camera, whether the trainer's learned time offset is
identifiable from the images at all.

Why this exists
----------------
`camera_time_offsets` in the TriGS trainer is one Adam-optimized scalar per
camera, regularized toward 0. On a camera whose footage carries almost no
temporal gradient in the training window (a static background camera, a
camera with only a still subject in frame), the *true* gradient of the loss
w.r.t. that offset is persistently tiny -- but Adam's per-parameter step is
normalized by the running RMS of the gradient, so it does not shrink to zero
for a persistently-tiny-but-nonzero signal (the `eps` floor keeps the step
near `lr` regardless). Left alone, such a camera drifts to the
`camera_time_offset_clamp_s` rail on noise, discarding a sync.json/ladder
prior that is often far better than the clamp itself.

This script measures, from the actual extracted images, the Fisher
information each camera's temporal gradient carries about its own offset,
turns it into a Cramer-Rao bound `sigma_photo` (in seconds), and combines it
with the ladder's own uncertainty `sigma_prior` (from
frame_timestamps.json, written by extract_4dv_sequence.py) into a MAP
shrinkage weight:

    s_c = sigma_prior_c^2 / (sigma_prior_c^2 + sigma_photo_c^2)   in [0, 1]

s_c -> 1 means the photometry can pin the offset far better than the ladder
prior (fully data-driven). s_c -> 0 means the opposite: the camera should
stay near its ladder/sync.json value no matter what Adam does with the raw
parameter. `train_from_colmap.py` resolves this per training position and
the trainer uses it as a *multiplicative* reparameterization of the learned
offset (`effective_offset = s_c * raw_param`), not just a regularizer --
see TrigsTrainer._camera_time_offset_scale.

Method
------
For a sparse, evenly-spaced sample of frame indices in [frame_start,
frame_end], load each camera's image at (f - frame_step) and (f +
frame_step), downsample, subsample onto a `--pixel-stride`-spaced grid, and
accumulate:

    diff(f)    =  I[f+frame_step] - I[f-frame_step]     (pixel_stride grid)
    dt         =  2 * frame_step / fps

The pixel_stride subsample matters: treating every downsampled pixel as an
independent Fisher-information observation drastically overstates the total
information, because real image content is spatially correlated (smooth
gradients, JPEG blocking, and `downsample`'s own bilinear resize smears
neighbors together further). Left dense, fisher inflates by orders of
magnitude and every camera's shrinkage saturates to ~1.0 regardless of
whether it has real motion -- pixel_stride is a coarse, unmeasured stand-in
for the true spatial correlation length, not a calibrated value.

For two independently-noisy frames, `E[diff^2] = true_signal_diff^2 +
2*sigma_noise^2`: the noise term alone is a large, strictly-positive
contribution, so a raw `sum(diff^2)` never goes small even for a perfectly
static, noisy camera. sigma_noise_c (single-frame sensor noise, in DN) is
estimated from the same pixel_stride-spaced samples as median(|diff|),
divided by sqrt(2)*Phi^-1(0.75) (~=0.954, the half-normal median-to-sigma
factor for diff ~ N(0, 2*sigma^2)) -- an approximation that assumes most
sampled pixels are near-static most of the time, not a calibrated
photometric measurement -- and used to bias-correct the sum before it
becomes Fisher information. It must use the *same* pixel set as the fisher
sum above; different pixel sets fed into the same bias-correction formula
give incoherent scale factors between the signal and noise terms.

    fisher_c = clip(sum(diff^2) - n_obs_c * 2 * sigma_noise_c^2, min=eps) / dt^2
    sigma_photo_c = sigma_noise_c / sqrt(fisher_c)   (Cramer-Rao bound, seconds)

    sigma_photo_c = sigma_noise_c / sqrt(fisher_c)   (Cramer-Rao bound, seconds)

Usage
-----
    python3 prep/sync/measure-offset-observability.py \\
        --data-dir /path/to/capture \\
        --frame-start 1 --frame-end 300 --output observability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Reuse the reference_export.json Camera/loader code (hyphenated filename,
# so imported by path rather than as a package).
# ---------------------------------------------------------------------------

_PREP = None


def prep():
    global _PREP
    if _PREP is not None:
        return _PREP
    import importlib.util
    init_dir = Path(__file__).resolve().parent.parent / "init"
    target = init_dir / "prep-roma-trigs-memory.py"
    if not target.exists():
        raise FileNotFoundError(f"cannot locate prep module at {target}")
    if str(init_dir) not in sys.path:
        sys.path.insert(0, str(init_dir))
    spec = importlib.util.spec_from_file_location("prep_roma_trigs_memory", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PREP = mod
    return mod


def _load_downsampled(
    images_root: Path, cameras: Sequence, frame_idx: int, image_pattern: str,
    downsample: int,
) -> List[np.ndarray]:
    def _load(cam) -> np.ndarray:
        path = images_root / cam.folder / image_pattern.format(frame=frame_idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing image for frame {frame_idx}: {path}")
        img = Image.open(path).convert("L")  # luma only: temporal gradient dominated by intensity
        if downsample > 1:
            img = img.resize(
                (max(1, img.width // downsample), max(1, img.height // downsample)),
                Image.BILINEAR,
            )
        return np.asarray(img, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=min(len(cameras), 8)) as ex:
        return list(ex.map(_load, cameras))


def sample_frame_indices(frame_start: int, frame_end: int, frame_step: int, max_samples: int) -> List[int]:
    lo, hi = frame_start + frame_step, frame_end - frame_step
    if hi < lo:
        raise ValueError(
            f"Window [{frame_start}, {frame_end}] is too short for --frame-step {frame_step}"
        )
    span = hi - lo
    n = min(max_samples, span + 1)
    if n <= 1:
        return [lo]
    return sorted({lo + round(i * span / (n - 1)) for i in range(n)})


def measure_observability(
    *,
    images_root: Path,
    cameras: Sequence,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    max_samples: int,
    fps: float,
    image_pattern: str,
    downsample: int,
    pixel_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fisher, sigma_noise_dn), both shape [n_cameras].

    ``fisher`` is bias-corrected: for two independently-noisy frames,
    ``E[diff^2] = true_signal_diff^2 + 2*sigma_noise^2`` -- the noise term
    alone is a large, strictly-positive contribution to a raw squared-
    difference sum, so a naive ``sum(diff^2)`` never goes small even for a
    perfectly static, noisy camera and would rank every camera as
    "observable". Subtracting the expected noise contribution recovers an
    (approximately) unbiased estimate of the actual signal power.
    """
    sample_points = sample_frame_indices(frame_start, frame_end, frame_step, max_samples)
    n_cams = len(cameras)
    raw_sq_sum = np.zeros(n_cams, dtype=np.float64)
    n_obs = np.zeros(n_cams, dtype=np.float64)
    abs_diff_samples: List[List[float]] = [[] for _ in range(n_cams)]
    dt_sec = 2.0 * frame_step / max(fps, 1e-6)

    for i, f in enumerate(sample_points):
        prev_images = _load_downsampled(images_root, cameras, f - frame_step, image_pattern, downsample)
        next_images = _load_downsampled(images_root, cameras, f + frame_step, image_pattern, downsample)
        for c in range(n_cams):
            diff = next_images[c] - prev_images[c]
            # Treating every pixel as an independent Fisher-information
            # observation wildly overstates the total information: real
            # image content is spatially correlated (smooth gradients, JPEG
            # blocking, further smeared by `downsample`'s bilinear resize),
            # so a dense per-pixel sum easily inflates fisher by 10^4-10^5x
            # and collapses sigma_photo toward zero for nearly every camera
            # regardless of real motion. Subsample onto a coarse, sparsely
            # spaced grid instead -- both for this sum and for the noise
            # estimate below (they must use the same pixel set, since the
            # noise estimate scales the bias correction subtracted from this
            # sum). PIXEL_STRIDE is a coarse, not-measured, stand-in for the
            # true spatial correlation length.
            sparse = diff[::pixel_stride, ::pixel_stride]
            raw_sq_sum[c] += float(np.sum(sparse ** 2))
            n_obs[c] += sparse.size
            abs_diff_samples[c].append(np.abs(sparse).ravel())
        print(f"[observability] sampled frame {f} ({i + 1}/{len(sample_points)})", flush=True)

    # median(|diff|) of a half-normal with diff ~ N(0, 2*sigma^2) equals
    # sqrt(2)*sigma*Phi^-1(0.75) ~= 0.95387*sigma; divide out that constant
    # to get an actual sigma estimate rather than just a raw statistic of
    # |diff|. (A raw percentile value is NOT sigma_noise_dn -- using it
    # directly as such badly underestimates true sigma and starves the bias
    # correction below, leaving every camera looking spuriously observable.)
    # Assumes most pixels in the sampled window are near-static most of the
    # time (background/subject-minority scenes), so the *median* pixel is
    # noise-dominated even though some pixels genuinely move.
    _median_to_sigma = float(np.sqrt(2.0)) * 0.6744897501960817
    sigma_noise_dn = np.array([
        float(np.median(np.concatenate(rows))) / _median_to_sigma if rows else 1.0
        for rows in abs_diff_samples
    ])
    # 8-bit quantization alone has a noise floor (uniform-quantization
    # variance = 1/12 DN^2, sigma ~= 0.29 DN). Without this floor, a very
    # quiet camera can drive the estimate to exactly 0 (most |diff| values
    # truncate to 0 once true noise is sub-DN); dividing by a zero
    # sigma_photo below would then read as *perfect* observability
    # regardless of how small fisher actually is -- the opposite of what a
    # quiet camera should report.
    sigma_noise_dn = np.maximum(sigma_noise_dn, 0.3)
    # Bias correction in raw DN^2 units, then convert to (DN/s)^2.
    noise_floor = n_obs * 2.0 * sigma_noise_dn ** 2
    signal_sq_sum = np.clip(raw_sq_sum - noise_floor, a_min=1e-6, a_max=None)
    fisher = signal_sq_sum / (dt_sec ** 2)
    return fisher, sigma_noise_dn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure per-camera identifiability of the learned time offset"
    )
    p.add_argument("--data-dir", required=True, type=Path,
                    help="Extracted dataset dir (has images/, reference_export.json, frame_timestamps.json)")
    p.add_argument("--reference-json", type=Path, default=None,
                    help="Defaults to <data-dir>/reference_export.json")
    p.add_argument("--images-root", type=Path, default=None,
                    help="Defaults to <data-dir>/images")
    p.add_argument("--frame-timestamps-json", type=Path, default=None,
                    help="Defaults to <data-dir>/frame_timestamps.json; supplies sigma_prior "
                         "(ladder.sigma_at_last_frame_ms) per camera")
    p.add_argument("--image-pattern", type=str, default="{frame:06d}.jpg")
    p.add_argument("--frame-start", type=int, required=True)
    p.add_argument("--frame-end", type=int, required=True)
    p.add_argument("--frame-step", type=int, default=1,
                    help="Frame offset used for the centered temporal difference")
    p.add_argument("--max-samples", type=int, default=40,
                    help="Cap on evaluated sample points across the window")
    p.add_argument("--downsample", type=int, default=4,
                    help="Integer downsample factor applied before computing gradients")
    p.add_argument("--pixel-stride", type=int, default=4,
                    help="Extra spatial stride (post-downsample) so summed pixels are far "
                         "enough apart to approximate independent Fisher-information samples; "
                         "a dense per-pixel sum drastically overstates information for "
                         "spatially-correlated real image content")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--default-sigma-prior-ms", type=float, default=4.0,
                    help="Used only if a camera has no ladder block in frame_timestamps.json")
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    reference_json = (args.reference_json or (data_dir / "reference_export.json")).expanduser().resolve()
    images_root = (args.images_root or (data_dir / "images")).expanduser().resolve()
    frame_timestamps_json = (
        args.frame_timestamps_json or (data_dir / "frame_timestamps.json")
    ).expanduser().resolve()

    module = prep()
    cameras, _ = module.load_reference_export(reference_json)
    print(f"[observability] {len(cameras)} cameras from {reference_json}")

    fisher, sigma_noise_dn = measure_observability(
        images_root=images_root,
        cameras=cameras,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
        max_samples=args.max_samples,
        fps=args.fps,
        image_pattern=args.image_pattern,
        downsample=args.downsample,
        pixel_stride=args.pixel_stride,
    )

    ladder_sigma_ms: Dict[str, float] = {}
    if frame_timestamps_json.exists():
        raw = json.loads(frame_timestamps_json.read_text(encoding="utf-8"))
        for folder, entry in (raw.get("cameras") or {}).items():
            ladder = entry.get("ladder")
            if ladder and "sigma_at_last_frame_ms" in ladder:
                ladder_sigma_ms[str(folder)] = float(ladder["sigma_at_last_frame_ms"])
    else:
        print(f"[observability] [warning] {frame_timestamps_json} not found; "
              f"using --default-sigma-prior-ms={args.default_sigma_prior_ms} for every camera")

    cameras_out: Dict[str, dict] = {}
    rows = []
    for i, cam in enumerate(cameras):
        sigma_prior_ms = ladder_sigma_ms.get(str(cam.folder), float(args.default_sigma_prior_ms))
        sigma_prior_s = sigma_prior_ms / 1000.0
        sigma_photo_s = (
            float(sigma_noise_dn[i]) / np.sqrt(fisher[i]) if fisher[i] > 0 else float("inf")
        )
        shrinkage = (sigma_prior_s ** 2) / (sigma_prior_s ** 2 + sigma_photo_s ** 2) if np.isfinite(sigma_photo_s) else 0.0
        cameras_out[str(cam.folder)] = {
            "fisher": float(fisher[i]),
            "sigma_noise_dn": float(sigma_noise_dn[i]),
            "sigma_photo_ms": float(sigma_photo_s * 1000.0) if np.isfinite(sigma_photo_s) else None,
            "sigma_prior_ms": sigma_prior_ms,
            "shrinkage": float(shrinkage),
        }
        rows.append((cam.folder, shrinkage, sigma_photo_s * 1000.0 if np.isfinite(sigma_photo_s) else float("inf"), sigma_prior_ms))

    payload = {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "frame_step": args.frame_step,
        "fps": args.fps,
        "downsample": args.downsample,
        "cameras": cameras_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows.sort(key=lambda r: r[1])
    print(f"\n[observability] wrote {args.output}")
    print(f"{'folder':>8} {'shrinkage':>10} {'sigma_photo_ms':>15} {'sigma_prior_ms':>15}")
    sentinel = ("...", None, None, None)
    preview_rows = rows[:8] + ([sentinel] if len(rows) > 16 else []) + rows[-8:]
    for folder, shrinkage, sigma_photo_ms, sigma_prior_ms in preview_rows:
        if folder == "...":
            print("     ...")
            continue
        print(f"{folder:>8} {shrinkage:10.3f} {sigma_photo_ms:15.3f} {sigma_prior_ms:15.3f}")
    n_low = sum(1 for _, s, _, _ in rows if s < 0.2)
    print(f"\n{n_low}/{len(rows)} cameras have shrinkage < 0.2 "
          f"(will stay near their ladder/sync.json prior during training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
