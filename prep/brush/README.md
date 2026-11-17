# BRUSH static diagnostic export

`build-brush-static-dataset.py` freezes one synchronized multi-camera frame
into a standard COLMAP dataset that BRUSH can train. It preserves the camera
poses and intrinsics from the reference COLMAP model, copies the selected
video frame under the image names recorded in that model, mirrors foreground
masks, and can convert a matching NPZ instant to `init.ply`.

Build two datasets for the useful A/B test:

```bash
# Control: static BRUSH trainer + COLMAP sparse initialization.
python prep/brush/build-brush-static-dataset.py \
  --data-dir . \
  --reference-colmap sparse/dynamic_4dv_v8 \
  --frame 1 \
  --mask-frame-offset 0 \
  --output-dir brush-static-f000001-colmap

# Same cameras/images/masks, but initialize BRUSH from the frame-1 NPZ points.
python prep/brush/build-brush-static-dataset.py \
  --data-dir . \
  --reference-colmap sparse/dynamic_4dv_v8 \
  --npz-path keyframes_trigs-f26-k3-v13.npz \
  --frame 1 \
  --mask-frame-offset 0 \
  --mask-mode transparent \
  --output-dir brush-static-f000001-npz
```

Frame 1 is a keyframe for a sequence starting at 1 with stride 3. The exporter
refuses to mix an image frame with a different NPZ keyframe unless
`--npz-nearest-frame` is explicitly supplied.

Each output contains a `README.txt` with commands for both the v0.3.0 release
and the current BRUSH main branch. With the v0.3.0 release binary, use:

```bash
brush /absolute/path/to/brush-static-f000001-npz \
  --with-viewer \
  --total-steps 30000 \
  --max-resolution 1920 \
  --export-path /absolute/path/to/brush-static-f000001-npz/brush_exports
```

`--mask-mode transparent` writes lossless RGBA image data under the exact
filenames stored by COLMAP. Mask value 255 is opaque and 0 is transparent, so
BRUSH penalizes opacity outside the foreground rather than merely ignoring the
background. This mode works with v0.3.0 and is recommended for an isolated
actor/object.

The alternative `--mask-mode ignore` writes a nested `masks/` tree where white
pixels contribute to RGB loss and black pixels are ignored. Use that mode only
with current BRUSH main: v0.3.0 cannot pair nested per-camera mask paths. Omit
masks entirely with `--skip-masks` for a control.

Interpret the A/B result as follows:

| COLMAP init | NPZ init | Likely conclusion |
|---|---|---|
| clean | clean | The TriGS temporal/training path is the main suspect. |
| clean | messy | The NPZ selection, coordinates, or point distribution is suspect. |
| messy | clean | The reference COLMAP point cloud is weak, but the NPZ initialization is viable. |
| messy | messy | First inspect cameras, synchronization, distortion, and images. |

The generated `brush-dataset.json` records every source-to-output mapping, NPZ
selection counts, point bounds, and the fraction of selected NPZ points in
front of each COLMAP camera. A very low positive-depth fraction is a strong
coordinate-frame warning.
