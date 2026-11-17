"""
Step 2c: Generate foreground masks for multi-camera 4D capture.

Two modes:

  1. PRODUCTION (Background Matting V2) — requires a clean background plate
     Compares each frame against the background to extract foreground alpha.
     This is what FreeTimeGS/4dv.ai uses.

  2. FALLBACK (SAM2 automatic) — no background plate needed
     Uses SAM2's automatic mask generator to find the subject.
     Less precise edges but works without a background plate.

Background plate capture:
  Before your 4D capture session, take one shot from each camera with
  an empty scene (no subject). Save as:
    data_dir/background/cam01.jpg
    data_dir/background/cam02.jpg
    ...

Output:
  data_dir/masks/cam01/000001.png   (0-255 alpha matte)
  data_dir/masks/cam01/000002.png
  data_dir/foregrounds/cam01/000001.png  (edge-decontaminated RGBA)
  ...

Usage:
    # Production: Background Matting V2 (requires background plate)
    python 02c_generate_masks.py \
        --data-dir /path/to/data \
        --start-frame 1 --end-frame 60

    # The same command also writes edge-decontaminated foreground RGBA to
    # data_dir/foregrounds for arbitrary-background composite training.

    # Fallback: SAM2 automatic (no background needed)
    python 02c_generate_masks.py \
        --data-dir /path/to/data \
        --start-frame 1 --end-frame 60 \
        --method sam2

    # Process specific cameras only
    python 02c_generate_masks.py \
        --data-dir /path/to/data \
        --start-frame 1 --end-frame 60 \
        --cameras 01_C0004,03_C0004

Prerequisites:
    # Shared environment: torch, torchvision, numpy, opencv-python, pillow,
    # and tqdm (see prep/SETUP.md). For Background Matting V2's supported
    # TorchScript path, only the checkpoint is needed; the source checkout is
    # required only for the legacy raw-state-dict fallback. Set BGMV2_REPO to
    # that checkout when using raw weights.
    mkdir -p ~/BackgroundMattingV2/checkpoints
    uv pip install --python "$CONDA_PREFIX/bin/python" gdown
    gdown https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf \
        -O ~/BackgroundMattingV2/checkpoints/torchscript_resnet50_fp32.pth

    # For SAM2 fallback:
    git clone https://github.com/facebookresearch/sam2.git ~/sam2_repo
    export CUDA_HOME="$CONDA_PREFIX"
    export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3090; use 12.0 for RTX 50-series
    uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation -e ~/sam2_repo
    curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
        -o ~/sam2_repo/checkpoints/sam2.1_hiera_large.pt

    # See prep/SETUP.md for all dependencies, checkpoint alternatives, and
    # smoke tests. The editable SAM2 checkout supplies its Hydra config files.
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# =============================================================================
# Background Matting V2
# =============================================================================

REFINE_ATTRS = (
    "backbone_scale", "refine_mode", "refine_sample_pixels", "refine_threshold"
)


def _refine_targets(model):
    """Where each refinement setting actually lives, for this model object.

    A TorchScript module exposes the four settings flat, the way upstream's
    inference_images.py sets them. A MattingRefine instance keeps only
    backbone_scale at the top level and holds the rest on its Refiner
    submodule under different names, so a flat setattr there is silently inert.
    """
    refiner = getattr(model, "refiner", None)
    targets = {"backbone_scale": (model, "backbone_scale")}
    for flat, nested in (("refine_mode", "mode"),
                         ("refine_sample_pixels", "sample_pixels"),
                         ("refine_threshold", "threshold")):
        if hasattr(model, flat):
            targets[flat] = (model, flat)
        elif refiner is not None and hasattr(refiner, nested):
            targets[flat] = (refiner, nested)
        else:
            targets[flat] = (None, None)
    return targets


def report_refine_config(model, label):
    """Print the refinement settings actually in force.

    backbone_scale is the one to watch: at 0.25 the base network sees 960x540
    of a 4K frame, so a bow that is 8px wide arrives as 2px and is gone before
    the refiner ever runs.
    """
    print(f"[BGMv2] {label} refinement config:")
    for attr, (obj, name) in _refine_targets(model).items():
        if obj is None:
            print(f"  {attr:<22} <not exposed by this model>")
        else:
            where = "" if obj is model else "  (on .refiner)"
            print(f"  {attr:<22} {getattr(obj, name)}{where}")


def apply_refine_overrides(model, backbone_scale=None, refine_mode=None,
                           refine_sample_pixels=None, refine_threshold=None):
    """Set the refinement settings, wherever this model keeps them."""
    requested = {
        "backbone_scale": backbone_scale,
        "refine_mode": refine_mode,
        "refine_sample_pixels": refine_sample_pixels,
        "refine_threshold": refine_threshold,
    }
    targets = _refine_targets(model)
    applied = False
    for attr, value in requested.items():
        if value is None:
            continue
        obj, name = targets[attr]
        if obj is None:
            print(f"[BGMv2] WARNING: --{attr.replace('_', '-')} ignored, "
                  f"this model does not expose it")
            continue
        setattr(obj, name, value)
        applied = True
    return applied


def load_bgmv2_model(checkpoint_path, device="cuda"):
    """Load Background Matting V2 (handles both TorchScript and standard weights)."""
    try:
        # Attempt to load as TorchScript first
        model = torch.jit.load(checkpoint_path).to(device).eval()
        print(f"[BGMv2] Loaded TorchScript model from {checkpoint_path}")
        report_refine_config(model, "TorchScript")
        return model
    except Exception:
        print(f"[BGMv2] File is not TorchScript. Loading as standard PyTorch weights...")
        
        # Add the cloned repo to Python's path. BGMV2_REPO keeps the legacy
        # raw-weight source tree separate from the default TorchScript cache.
        import sys
        repo_path = os.environ.get(
            "BGMV2_REPO", os.path.expanduser("~/BackgroundMattingV2")
        )
        if not os.path.isdir(repo_path):
            raise RuntimeError(
                "BGMv2 raw weights need a source checkout. Clone "
                "BackgroundMattingV2 and set BGMV2_REPO to its path, or "
                "use the supported TorchScript checkpoint."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            
        # Import the correct architecture class
        from model import MattingRefine
        
        # Initialize the standard ResNet50 BGMv2 architecture
        model = MattingRefine(
            backbone='resnet50',
            backbone_scale=0.5,
            refine_mode='full',
            refine_sample_pixels=80000
        ).to(device).eval()
        
        # Load the weights into the architecture
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[BGMv2] Successfully loaded PyTorch weights from {checkpoint_path}")
        report_refine_config(model, "MattingRefine")
        return model


def run_bgmv2(model, src_img, bgr_img, device="cuda"):
    """
    Run Background Matting V2 on a single image.

    Args:
        model: TorchScript BGMv2 model
        src_img: source image as numpy [H, W, 3] uint8 BGR
        bgr_img: background image as numpy [H, W, 3] uint8 BGR

    Returns:
        alpha: [H, W] float32 in [0, 1]
        foreground: [H, W, 3] float32 BGR in [0, 1]
    """
    src = torch.from_numpy(src_img[:, :, ::-1].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    bgr = torch.from_numpy(bgr_img[:, :, ::-1].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    src = src.to(device)
    bgr = bgr.to(device)

    with torch.no_grad():
        pha, fgr = model(src, bgr)[:2]

    alpha = pha[0, 0].clamp(0, 1).cpu().numpy()
    foreground = (
        fgr[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1].copy()
    )
    return alpha, foreground


BG_EXTS = ('.jpg', '.png', '.JPG', '.jpeg', '.JPEG', '.PNG')


def find_background_image(background_dir, cam_name):
    """Resolve one camera's clean plate, across the two layouts in use.

      background/0000.jpg            flat, one file per camera
      reference_images/0000/*.jpg    a directory per camera, mirroring images/

    Returns None if neither exists.
    """
    for ext in BG_EXTS:
        candidate = os.path.join(background_dir, f"{cam_name}{ext}")
        if os.path.exists(candidate):
            return candidate

    cam_dir = os.path.join(background_dir, cam_name)
    if os.path.isdir(cam_dir):
        frames = sorted(
            f for f in os.listdir(cam_dir) if f.endswith(BG_EXTS)
        )
        if frames:
            return os.path.join(cam_dir, frames[0])

    return None


def load_gate(gate_dir, cam_name, shape):
    """Load a 3D subject-volume gate for one camera, or None if absent.

    Built by build_volume_gate.py. Zeroes anything whose ray misses the capture
    volume, which a clean plate cannot do: crew and set dressing change between
    plate and take, so their residual is large and legitimate.
    """
    if not gate_dir:
        return None

    for ext in ('.png', '.jpg'):
        path = os.path.join(gate_dir, f"{cam_name}{ext}")
        if os.path.exists(path):
            gate = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gate is None:
                return None
            if gate.shape != shape:
                gate = cv2.resize(gate, (shape[1], shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            return gate
    return None


def write_mask_atomic(mask_path, mask):
    """Write a mask without leaving a partial final PNG after interruption."""
    path = Path(mask_path)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), mask):
            raise OSError(f"OpenCV failed to write mask: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_edge_corrected_foreground(
    source_bgr, foreground_bgr, alpha, *, solid_alpha=0.98,
    blend_alpha=0.90, zero_alpha=0.02,
):
    """Keep source detail inside and use BGMv2 foreground at soft boundaries.

    BGMv2's foreground estimate is valuable where the captured RGB is mixed
    with the clean plate, but replacing the entire subject can soften texture.
    Use the BGMv2 foreground throughout the uncertain boundary, then blend to
    the original source between ``blend_alpha`` and ``solid_alpha``. Pixels
    with tiny alpha are set to black: they do not materially affect the
    composite and the uniform exterior keeps the lossless PNGs compact.
    Retaining a smooth blend near the solid interior avoids a colour seam.
    """
    blend_alpha = float(blend_alpha)
    solid_alpha = float(solid_alpha)
    zero_alpha = float(zero_alpha)
    if not (0.0 <= zero_alpha <= blend_alpha < solid_alpha <= 1.0):
        raise ValueError(
            "foreground edge thresholds must satisfy "
            "0 <= zero_alpha <= blend_alpha < solid_alpha <= 1; got "
            f"{zero_alpha}, {blend_alpha}, {solid_alpha}"
        )

    source = np.asarray(source_bgr, dtype=np.float32) / 255.0
    foreground = np.asarray(foreground_bgr, dtype=np.float32)
    matte = np.asarray(alpha, dtype=np.float32)
    if source.shape != foreground.shape or source.shape[:2] != matte.shape:
        raise ValueError(
            "source, foreground, and alpha shapes must match; got "
            f"{source.shape}, {foreground.shape}, {matte.shape}"
        )

    t = np.clip((matte - blend_alpha) / (solid_alpha - blend_alpha), 0.0, 1.0)
    # Smoothstep prevents a visible derivative change at either edge of the band.
    source_weight = t * t * (3.0 - 2.0 * t)
    corrected = (
        source * source_weight[..., None]
        + foreground * (1.0 - source_weight[..., None])
    )
    corrected[matte <= zero_alpha] = 0.0
    return np.rint(np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)


def tighten_soft_alpha(alpha, gamma=1.0):
    """Suppress low-confidence fringe while preserving a soft antialiased edge.

    A gamma above one contracts only the fractional part of the matte: zero and
    one remain unchanged. This is preferable to a hard threshold, which creates
    stair-stepped silhouettes and loses hair and motion-blurred detail.
    """
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"mask alpha gamma must be greater than zero, got {gamma}")
    matte = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    return np.power(matte, gamma).astype(np.float32, copy=False)


def pack_foreground_rgba(foreground_bgr, alpha_u8):
    """Pack straight BGR colour and its matte as an OpenCV BGRA PNG.

    The RGB channels remain straight (not premultiplied) for training. Adding
    the alpha channel makes ordinary image viewers display the *actual*
    composite instead of showing low-alpha edge colour as fully opaque.
    """
    foreground = np.asarray(foreground_bgr, dtype=np.uint8)
    alpha = np.asarray(alpha_u8, dtype=np.uint8)
    if foreground.shape[:2] != alpha.shape or foreground.shape[-1] != 3:
        raise ValueError(
            "foreground RGB and alpha shapes must match; got "
            f"{foreground.shape} and {alpha.shape}"
        )
    return np.dstack((foreground, alpha))


def close_mask_gaps(mask, radius):
    """Fill small dark gaps without replacing BGMv2's soft exterior edge."""
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = (mask >= 128).astype(np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return np.maximum(mask, closed * 255).astype(np.uint8)


def process_camera_bgmv2(
    cam_name, images_dir, background_dir, output_dir,
    start_frame, end_frame, model, device="cuda", close_radius=0,
    gate_dir=None, foregrounds_output_dir=None,
    foreground_blend_alpha=0.90, foreground_solid_alpha=0.98,
    foreground_zero_alpha=0.02, mask_alpha_gamma=1.0, overwrite=False,
):
    """Process all frames for one camera using Background Matting V2."""
    cam_mask_dir = os.path.join(output_dir, cam_name)
    os.makedirs(cam_mask_dir, exist_ok=True)
    cam_foreground_dir = None
    if foregrounds_output_dir is not None:
        cam_foreground_dir = os.path.join(foregrounds_output_dir, cam_name)
        os.makedirs(cam_foreground_dir, exist_ok=True)
    gate = None

    # Load background image for this camera
    bgr_path = find_background_image(background_dir, cam_name)

    if bgr_path is None:
        print(f"  {cam_name}: No background image found, skipping")
        return 0

    bgr_img = cv2.imread(bgr_path)
    if bgr_img is None:
        print(f"  {cam_name}: Failed to load background image")
        return 0

    saved = 0
    for frame_idx in range(start_frame, end_frame + 1):
        mask_path = os.path.join(cam_mask_dir, f"{frame_idx:06d}.png")
        foreground_path = (
            os.path.join(cam_foreground_dir, f"{frame_idx:06d}.png")
            if cam_foreground_dir is not None else None
        )
        if not overwrite and os.path.exists(mask_path) and (
            foreground_path is None or os.path.exists(foreground_path)
        ):
            saved += 1
            continue

        src_path = os.path.join(images_dir, cam_name, f"{frame_idx:06d}.jpg")
        if not os.path.exists(src_path):
            continue

        src_img = cv2.imread(src_path)
        if src_img is None:
            continue

        # Resize background to match source if needed
        if bgr_img.shape[:2] != src_img.shape[:2]:
            bgr_resized = cv2.resize(bgr_img, (src_img.shape[1], src_img.shape[0]))
        else:
            bgr_resized = bgr_img

        alpha, foreground = run_bgmv2(model, src_img, bgr_resized, device)

        if os.path.exists(mask_path) and not overwrite:
            # Resume foreground generation against the exact matte that will be
            # used for training, even if it came from an earlier invocation.
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"Failed to read existing mask: {mask_path}")
        else:
            alpha = tighten_soft_alpha(alpha, mask_alpha_gamma)
            mask = np.rint(alpha * 255).clip(0, 255).astype(np.uint8)
            mask = close_mask_gaps(mask, close_radius)

            # Gate last, so closing cannot push coverage back outside the volume.
            if gate_dir:
                if gate is None:
                    gate = load_gate(gate_dir, cam_name, mask.shape)
                if gate is not None:
                    mask = np.minimum(mask, gate)

            write_mask_atomic(mask_path, mask)

        if foreground_path is not None and (
            overwrite or not os.path.exists(foreground_path)
        ):
            corrected = build_edge_corrected_foreground(
                src_img,
                foreground,
                mask.astype(np.float32) / 255.0,
                blend_alpha=foreground_blend_alpha,
                solid_alpha=foreground_solid_alpha,
                zero_alpha=foreground_zero_alpha,
            )
            write_mask_atomic(
                foreground_path, pack_foreground_rgba(corrected, mask)
            )
        saved += 1

    return saved


def count_camera_masks(output_dir, cam_name, start_frame, end_frame):
    """Count masks in the requested range (including masks from earlier runs)."""
    cam_mask_dir = os.path.join(output_dir, cam_name)
    return sum(
        os.path.exists(os.path.join(cam_mask_dir, f"{frame_idx:06d}.png"))
        for frame_idx in range(start_frame, end_frame + 1)
    )


def camera_has_pending_frames(
    images_dir, output_dir, cam_name, start_frame, end_frame,
    foregrounds_output_dir=None, overwrite=False,
):
    """Return whether a source frame is missing any requested output."""
    if overwrite:
        return True
    cam_image_dir = os.path.join(images_dir, cam_name)
    cam_mask_dir = os.path.join(output_dir, cam_name)
    for frame_idx in range(start_frame, end_frame + 1):
        name = f"{frame_idx:06d}"
        if (
            os.path.exists(os.path.join(cam_image_dir, f"{name}.jpg"))
            and (
                not os.path.exists(os.path.join(cam_mask_dir, f"{name}.png"))
                or (
                    foregrounds_output_dir is not None
                    and not os.path.exists(
                        os.path.join(foregrounds_output_dir, cam_name, f"{name}.png")
                    )
                )
            )
        ):
            return True
    return False


def process_bgmv2_in_camera_workers(
    cam_names, images_dir, output_dir, start_frame, end_frame, retries,
    foregrounds_output_dir=None, overwrite=False,
):
    """Run each camera in a fresh process to contain native-library failures.

    Long BGMv2 jobs exercise CUDA, cuDNN, OpenCV, and libjpeg hundreds of
    thousands of times. A fresh process per camera prevents native state or
    allocator fragmentation from accumulating across the entire capture. The
    normal per-file resume behavior also makes retrying a worker inexpensive.
    """
    total_masks = 0
    script_path = os.path.abspath(__file__)

    for camera_number, cam_name in enumerate(cam_names, start=1):
        existing = count_camera_masks(
            output_dir, cam_name, start_frame, end_frame
        )
        if not camera_has_pending_frames(
            images_dir, output_dir, cam_name, start_frame, end_frame,
            foregrounds_output_dir, overwrite,
        ):
            print(
                f"[BGMv2] Camera {camera_number}/{len(cam_names)} {cam_name}: "
                f"already complete ({existing} masks), skipping",
                flush=True,
            )
            total_masks += existing
            continue

        command = [
            sys.executable, "-u", script_path, *sys.argv[1:],
            "--cameras", cam_name, "--camera-worker",
        ]
        for attempt in range(retries + 1):
            print(
                f"[BGMv2] Camera {camera_number}/{len(cam_names)} {cam_name}"
                + (f" (retry {attempt}/{retries})" if attempt else ""),
                flush=True,
            )
            result = subprocess.run(command)
            if result.returncode == 0:
                break
            if attempt == retries:
                if result.returncode < 0:
                    reason = f"signal {-result.returncode}"
                else:
                    reason = f"exit code {result.returncode}"
                print(
                    f"\nERROR: BGMv2 worker for camera {cam_name} failed with "
                    f"{reason} after {retries + 1} attempt(s).",
                    file=sys.stderr,
                )
                print(
                    "Completed masks are intact. Rerun the same command to "
                    "resume from the first missing frame.",
                    file=sys.stderr,
                )
                return total_masks, False

        total_masks += count_camera_masks(
            output_dir, cam_name, start_frame, end_frame
        )

    return total_masks, True


# =============================================================================
# SAM2 Automatic Segmentation (fallback)
# =============================================================================

def process_camera_sam2(
    cam_name, images_dir, output_dir,
    start_frame, end_frame,
    sam2_checkpoint, sam2_config,
    device="cuda", gate_dir=None
):
    """
    Process all frames for one camera using SAM2 automatic mask generation.

    Strategy: For each frame, generate all masks, pick the largest one
    that covers less than 80% of the image (the subject, not the background).
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    cam_mask_dir = os.path.join(output_dir, cam_name)
    os.makedirs(cam_mask_dir, exist_ok=True)

    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        min_mask_region_area=10000,
    )

    saved = 0
    for frame_idx in range(start_frame, end_frame + 1):
        mask_path = os.path.join(cam_mask_dir, f"{frame_idx:06d}.png")
        if os.path.exists(mask_path):
            saved += 1
            continue

        src_path = os.path.join(images_dir, cam_name, f"{frame_idx:06d}.jpg")
        if not os.path.exists(src_path):
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        image_area = h * w

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks = mask_generator.generate(img_rgb)

        if not masks:
            write_mask_atomic(mask_path, np.zeros((h, w), dtype=np.uint8))
            saved += 1
            continue

        # Sort by area (largest first)
        masks.sort(key=lambda x: x['area'], reverse=True)

        # Pick largest mask that's not the whole image
        selected_mask = None
        for m in masks:
            if m['area'] < image_area * 0.80:
                selected_mask = m
                break

        if selected_mask is None:
            selected_mask = masks[1] if len(masks) > 1 else masks[0]

        mask_binary = selected_mask['segmentation'].astype(np.uint8) * 255
        gate = load_gate(gate_dir, cam_name, mask_binary.shape)
        if gate is not None:
            mask_binary = np.minimum(mask_binary, gate)
        write_mask_atomic(mask_path, mask_binary)
        saved += 1

    del mask_generator, sam2_model
    torch.cuda.empty_cache()

    return saved


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate foreground masks for multi-camera 4D capture"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=60)
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: data_dir/masks)")
    parser.add_argument(
        "--foregrounds-output-dir", default=None,
        help=(
            "BGMv2 edge-decontaminated straight-RGBA output directory "
            "(default: data_dir/foregrounds)"
        ),
    )
    parser.add_argument(
        "--no-save-foregrounds", action="store_true",
        help="Generate alpha masks only; do not save BGMv2 foreground RGBA",
    )
    parser.add_argument(
        "--foreground-blend-alpha", type=float, default=0.90,
        help=(
            "At or below this alpha, use BGMv2 foreground RGB; blend to the "
            "source above it (default: 0.90)"
        ),
    )
    parser.add_argument(
        "--foreground-solid-alpha", type=float, default=0.98,
        help="Above this alpha, retain the original source RGB (default: 0.98)",
    )
    parser.add_argument(
        "--foreground-zero-alpha", type=float, default=0.02,
        help=(
            "At or below this alpha, store black to keep foreground PNGs "
            "compact (default: 0.02)"
        ),
    )
    parser.add_argument(
        "--mask-alpha-gamma", type=float, default=1.0,
        help=(
            "Power applied to the BGMv2 alpha matte. Values above 1 tighten "
            "low-confidence soft fringe without hard-thresholding; try 1.5 "
            "for dark-background contamination (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Regenerate and replace existing masks and foregrounds. Required "
            "when changing alpha/foreground settings for an existing dataset."
        ),
    )
    parser.add_argument("--method", default="auto",
                        choices=["auto", "bgmv2", "sam2"],
                        help="auto: bgmv2 if background exists, else sam2")
    parser.add_argument("--background-dir", default=None,
                        help="Dir with background images (default: data_dir/background)")
    parser.add_argument("--bgmv2-checkpoint", default=None)
    parser.add_argument(
        "--close-radius", type=int, default=0,
        help=(
            "BGMv2-only morphological closing radius in pixels; 0 disables it. "
            "At 4K, 16 gives a 33x33 kernel and is a useful starting point."
        ),
    )
    parser.add_argument(
        "--gate-dir", default=None,
        help=(
            "Dir of per-camera 3D subject-volume gates from build_volume_gate.py. "
            "Anything outside the capture volume is zeroed, which removes crew "
            "and set dressing that a clean plate cannot."
        ),
    )
    parser.add_argument(
        "--backbone-scale", type=float, default=None,
        help=(
            "BGMv2 base-network scale. Upstream recommends 0.25 for 4K, but "
            "that gives the base net 960x540 and thin structures such as a bow "
            "vanish before refinement. Raise it to keep them."
        ),
    )
    parser.add_argument("--refine-mode", default=None,
                        choices=["full", "sampling", "thresholding"],
                        help="BGMv2 refinement mode")
    parser.add_argument("--refine-sample-pixels", type=int, default=None,
                        help="Pixels refined in 'sampling' mode (upstream default 80000)")
    parser.add_argument("--refine-threshold", type=float, default=None,
                        help="Error threshold in 'thresholding' mode; lower refines more")
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--device", default="cuda")
    isolation = parser.add_mutually_exclusive_group()
    isolation.add_argument(
        "--isolate-cameras", dest="isolate_cameras", action="store_true",
        help=(
            "Run each BGMv2 camera in a fresh process (default). This contains "
            "native CUDA/OpenCV failures during very long jobs."
        ),
    )
    isolation.add_argument(
        "--no-isolate-cameras", dest="isolate_cameras", action="store_false",
        help="Run all BGMv2 cameras in one process",
    )
    parser.set_defaults(isolate_cameras=True)
    parser.add_argument(
        "--camera-retries", type=int, default=1,
        help="Retries after a BGMv2 camera worker fails (default: 1)",
    )
    parser.add_argument("--camera-worker", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.close_radius < 0:
        parser.error("--close-radius must be zero or greater")
    if args.camera_retries < 0:
        parser.error("--camera-retries must be zero or greater")
    if args.mask_alpha_gamma <= 0:
        parser.error("--mask-alpha-gamma must be greater than zero")
    if not (
        0.0 <= args.foreground_zero_alpha
        <= args.foreground_blend_alpha
        < args.foreground_solid_alpha <= 1.0
    ):
        parser.error(
            "foreground thresholds must satisfy "
            "0 <= zero <= blend < solid <= 1"
        )

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    output_dir = args.output_dir or os.path.join(data_dir, "masks")
    foregrounds_output_dir = None
    if not args.no_save_foregrounds:
        foregrounds_output_dir = (
            args.foregrounds_output_dir
            or os.path.join(data_dir, "foregrounds")
        )
    background_dir = args.background_dir or os.path.join(data_dir, "background")
    os.makedirs(output_dir, exist_ok=True)
    if foregrounds_output_dir is not None:
        os.makedirs(foregrounds_output_dir, exist_ok=True)

    if args.cameras:
        cam_names = args.cameras.split(",")
    else:
        cam_names = sorted([
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ])

    print(f"Cameras: {len(cam_names)}")
    print(f"Frames: {args.start_frame} to {args.end_frame}")

    # Determine method
    method = args.method
    if method == "auto":
        if os.path.exists(background_dir) and len(os.listdir(background_dir)) > 0:
            method = "bgmv2"
            print(f"[Auto] Background directory found → Background Matting V2")
        else:
            method = "sam2"
            print(f"[Auto] No background directory → SAM2 automatic segmentation")
            print(f"  TIP: For better masks, capture a background plate and save to:")
            print(f"       {background_dir}/cam_name.jpg")

    # Pre-flight. A plate or gate that fails to resolve is silent per frame, so
    # check the naming lines up before spending a model load on it.
    if method == "bgmv2":
        found = [c for c in cam_names if find_background_image(background_dir, c)]
        print(f"[BGMv2] Clean plates resolved: {len(found)}/{len(cam_names)}")
        if not found:
            print(f"\nERROR: no clean plate matched any camera in {background_dir}")
            print("  Looked for, per camera <cam>:")
            print(f"    {background_dir}/<cam>.jpg")
            print(f"    {background_dir}/<cam>/<first image>")
            if os.path.isdir(background_dir):
                entries = sorted(os.listdir(background_dir))[:5]
                print(f"  That dir actually contains: {', '.join(entries) or '(empty)'}")
                print(f"  Camera folders are named:   {', '.join(cam_names[:5])}")
            else:
                print(f"  That dir does not exist.")
            sys.exit(1)
        elif len(found) < len(cam_names):
            missing = [c for c in cam_names if c not in found]
            print(f"  WARNING: no plate for {len(missing)} camera(s), "
                  f"they will be skipped: {', '.join(missing[:8])}"
                  f"{' ...' if len(missing) > 8 else ''}")

    if args.gate_dir:
        gated = [
            c for c in cam_names
            if any(os.path.exists(os.path.join(args.gate_dir, f"{c}{e}"))
                   for e in ('.png', '.jpg'))
        ]
        print(f"[Gate] Volume gates resolved: {len(gated)}/{len(cam_names)}")
        if not gated:
            print(f"\nERROR: no gate matched any camera in {args.gate_dir}")
            if os.path.isdir(args.gate_dir):
                entries = sorted(
                    f for f in os.listdir(args.gate_dir) if f.endswith(('.png', '.jpg'))
                )[:5]
                print(f"  That dir contains:        {', '.join(entries) or '(none)'}")
                print(f"  Camera folders are named: {', '.join(cam_names[:5])}")
                print("  Gates are named from the COLMAP image names. Rebuild them "
                      "against\n  a model whose names match your image folders.")
            else:
                print(f"  That dir does not exist.")
            sys.exit(1)

    total_masks = 0

    if method == "bgmv2" and args.isolate_cameras and not args.camera_worker:
        print(
            "[BGMv2] Camera isolation enabled; each camera runs in a fresh "
            "process and existing masks are resumed.",
            flush=True,
        )
        total_masks, succeeded = process_bgmv2_in_camera_workers(
            cam_names, images_dir, output_dir,
            args.start_frame, args.end_frame, args.camera_retries,
            foregrounds_output_dir, args.overwrite,
        )
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Method: {method}")
        print(f"Cameras: {len(cam_names)}")
        print(f"Total masks: {total_masks}")
        print(f"Output: {output_dir}")
        if foregrounds_output_dir is not None:
            print(f"Foreground RGBA: {foregrounds_output_dir}")
        if not succeeded:
            sys.exit(1)
        return

    if method == "bgmv2":
        checkpoint = args.bgmv2_checkpoint
        if checkpoint is None:
            for path in [
                os.path.expanduser("~/BackgroundMattingV2/checkpoints/torchscript_resnet50_fp32.pth"),
                os.path.expanduser("~/BackgroundMattingV2/checkpoints/pytorch_resnet50.pth"),
            ]:
                if os.path.exists(path):
                    checkpoint = path
                    break

        if checkpoint is None:
            print("ERROR: BGMv2 checkpoint not found.")
            print("  mkdir -p ~/BackgroundMattingV2/checkpoints")
            print("  uv pip install --python \"$CONDA_PREFIX/bin/python\" gdown")
            print("  gdown https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf \\")
            print("    -O ~/BackgroundMattingV2/checkpoints/torchscript_resnet50_fp32.pth")
            sys.exit(1)

        print(f"[BGMv2] Checkpoint: {checkpoint}")
        if args.close_radius > 0:
            kernel_size = args.close_radius * 2 + 1
            print(
                f"[BGMv2] Mask closing: radius={args.close_radius}px "
                f"({kernel_size}x{kernel_size} kernel)"
            )
        model = load_bgmv2_model(checkpoint, args.device)

        if apply_refine_overrides(
            model, args.backbone_scale, args.refine_mode,
            args.refine_sample_pixels, args.refine_threshold
        ):
            report_refine_config(model, "after overrides")

        for cam_name in tqdm(cam_names, desc="BGMv2 masks"):
            n = process_camera_bgmv2(
                cam_name, images_dir, background_dir, output_dir,
                args.start_frame, args.end_frame, model, args.device,
                args.close_radius, args.gate_dir,
                foregrounds_output_dir,
                args.foreground_blend_alpha,
                args.foreground_solid_alpha,
                args.foreground_zero_alpha,
                args.mask_alpha_gamma,
                args.overwrite,
            )
            total_masks += n

        del model
        torch.cuda.empty_cache()

    elif method == "sam2":
        checkpoint = args.sam2_checkpoint
        if checkpoint is None:
            for path in [
                os.path.expanduser("~/checkpoints/sam2.1_hiera_large.pt"),
                os.path.expanduser("~/sam2_repo/checkpoints/sam2.1_hiera_large.pt"),
            ]:
                if os.path.exists(path):
                    checkpoint = path
                    break

        if checkpoint is None:
            print("ERROR: SAM2 checkpoint not found.")
            print("  # Install SAM2 first; see prep/SETUP.md")
            print("  curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \\")
            print("    -o ~/sam2_repo/checkpoints/sam2.1_hiera_large.pt")
            sys.exit(1)

        print(f"[SAM2] Checkpoint: {checkpoint}")

        for cam_name in tqdm(cam_names, desc="SAM2 masks"):
            n = process_camera_sam2(
                cam_name, images_dir, output_dir,
                args.start_frame, args.end_frame,
                checkpoint, args.sam2_config, args.device, args.gate_dir
            )
            total_masks += n
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Cameras: {len(cam_names)}")
    print(f"Total masks: {total_masks}")
    print(f"Output: {output_dir}")
    if method == "bgmv2" and foregrounds_output_dir is not None:
        print(f"Foreground RGBA: {foregrounds_output_dir}")


if __name__ == "__main__":
    main()
