# Open4DGS-Core

Open4DGS-Core prepares synchronized multi-camera video, initializes dynamic Gaussian scenes, and exports TRiGS reconstructions. Training is provided by [Open4DGS-Trigs](https://github.com/ASU-NEM-Labs/open4dgs-trigs), the companion AGPL-3.0 repository.

Start with the [two-repository installation guide](SETUP.md). In the development repository, the trainer also lives under `train/trigs/`.

## Release status and support

This is an **initial research release**. Interfaces, defaults, and file formats may
change as the project develops. The roadmap describes planned work without
committed delivery dates.

The main Linux setup was exercised on **Ubuntu 24.04, Python 3.11, CUDA 12.8,
and an NVIDIA RTX 3090**. The trainer setup also documents a reported Windows
11 / RTX PRO 6000 Blackwell configuration. These are specific tested or reported
configurations, not a guarantee for every GPU or operating system. **WSL 2 has
not yet been validated for this release.** See the setup guides for details.

The split repositories have passed local dependency coverage checks and the
trainer package builds successfully. A fresh end-to-end GPU run using only the
split repositories has not yet been completed. Reconstruction quality, memory
requirements, and runtime depend on the capture, masks, calibration, and GPU;
large scenes may need substantial system RAM and GPU memory.

Please use GitHub issues for reproducible bugs and installation problems. Include
both repository versions, your environment, the command, and a redacted error
log. Support is best effort; response times and feature delivery are not
guaranteed. For suspected security problems, see [SECURITY.md](SECURITY.md).

## Example datasets

Two example datasets are planned for the companion
[Open4DGS-Datasets repository](https://github.com/ASU-NEM-Labs/open4dgs-datasets).
Dataset downloads, usage instructions, and dataset-specific licenses will be
provided there as they become available. The datasets are distributed separately
from these source repositories.

## Roadmap

- [x] Initial Code Release
- [ ] Prep / Training time improvements
- [ ] Unity Plugin with sample project
- [ ] Web SDKs (sparkjs-based and threejs-based)
- [ ] 8th Wall sample project
- [ ] Compression and Streaming

## Setup

- [Preparation setup](prep/SETUP.md) covers video tools, COLMAP, RoMa, MEMFOF, and optional masking dependencies. RoMa v2 and MEMFOF are the default initialization backends.
- [Trainer setup](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md) covers the training environment and CUDA extensions. See also the [trainer README](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/README.md).

The documented setup was exercised on Ubuntu 24.04 with Python 3.11, CUDA 12.8 and an RTX 3090. See the setup guides for validation limits, other GPU architectures, and optional environments. Model weights are downloaded separately. RoMa v2 uses DINOv3, which has its own [license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), distinct from the RoMa v2 code license.

### Preparation environment on Ubuntu

Install the system tools, create a Python 3.11 Conda environment, and install the common Python packages and the default matching and optical-flow backends:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build curl ca-certificates \
  libgl1 libglib2.0-0 ffmpeg tmux
conda create -n trigs python=3.11 pip -y
conda activate trigs
conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3090; use the target for your GPU

uv pip install --python "$CONDA_PREFIX/bin/python" \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$CONDA_PREFIX/bin/python" \
  numpy scipy pillow opencv-python tqdm imageio matplotlib pycolmap
uv pip install --python "$CONDA_PREFIX/bin/python" \
  "git+https://github.com/Parskatt/RoMaV2.git@v2.0.1" \
  "git+https://github.com/msu-video-group/memfof.git@a51de9fc59c6fe20ba08e079372c7b583d58a712"
```

This assumes [Conda and uv are installed](prep/SETUP.md#2-create-the-shared-python-environment). Alignment also needs the standalone COLMAP executable; `pycolmap` does not provide it. Follow the [COLMAP build instructions](prep/SETUP.md#3-install-the-colmap-command-line-program) and their validation notes. Install [mask dependencies](prep/SETUP.md#5-install-mask-generation-dependencies-as-needed) for the chosen mask method. For trainer installation, follow [Open4DGS-Trigs setup](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md), including its Windows instructions.

If preparation packages are installed into the trainer's environment, `uv run` can remove packages outside the trainer lockfile, including RoMa and MEMFOF. Activate the environment and run its console scripts directly, or use `uv run --no-sync` after installation. See [environment guidance](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md#sharing-an-environment-with-preparation-tools).

## Pipeline

The examples below use `REPO` for the absolute path to this checkout and assume commands run from the capture data directory. Replace the bracketed values with your own paths and capture settings. Frame ranges and frame rates must match the footage; use each script's `--help` for further options. The detailed, current training defaults and timing workflow are in [docs/run-pipeline.md](docs/run-pipeline.md).

```bash
export REPO=/absolute/path/to/open4dgs-core
cd /absolute/path/to/capture
```

### 1. Extract frames

For timecode-synchronized footage organized in camera subfolders, choose a common start timecode present in each video. Take names such as `C0007.MP4` correspond to `--take 7`.

```bash
python "$REPO/prep/extract/extract_synced_sequence.py" \
  --input-dir /path/to/camera-videos --output-dir images \
  --start-timecode HH:MM:SS:FF --num-frames <FRAME_COUNT> \
  --force-rotation n --take 7 --skip-existing
```

See [extraction tools](prep/extract/) for other capture formats.

### 2. Align cameras

Extract representative alignment frames in the same camera layout, then build a COLMAP reference model:

```bash
python "$REPO/prep/align/build-colmap-ref-trigs.py" \
  --data-dir /path/to/align --ref-frame 1
cp -r /path/to/align/sparse ./sparse
cp /path/to/align/reference_export.json ./reference_export.json
```

Inspect the model analysis and registered cameras; the stage guide suggests a mean reprojection error below one pixel as a target. Verify calibration for the capture rather than relying on that number alone.

### 3. Generate masks

For Background Matting V2, provide a clean background plate for each camera under `background/`. The clean-plate utility pairs sorted source files with sorted camera folders; inspect its pairing before use.

```bash
python "$REPO/prep/mask/extract_clean_plate.py" \
  --source-root /path/to/clean-plates --images-dir images \
  --output-dir background --frame-number 0

python "$REPO/prep/mask/generate_masks.py" --data-dir . \
  --start-frame <FIRST_FRAME> --end-frame <LAST_FRAME> \
  --method bgmv2 --backbone-scale 0.5 --refine-mode thresholding \
  --refine-threshold 0.015 --foregrounds-output-dir foregrounds \
  --foreground-blend-alpha 0.90 --foreground-solid-alpha 0.98 \
  --foreground-zero-alpha 0.02 --mask-alpha-gamma 1.0
```

The script also supports SAM 2 when clean plates are unavailable; see [mask setup](prep/SETUP.md#5-install-mask-generation-dependencies-as-needed).

### 4. Solve camera timing and rolling shutter

For cameras without genlock, the optional visual solver estimates sub-frame offsets from motion. The example also models rolling shutter. The 15 ms readout was measured for a Sony RX0 II rig at 60 fps; measure the readout for your own camera and sensor mode.

```bash
python "$REPO/prep/sync/calculate-sync.py" \
  --reference-json reference_export.json --images-root images \
  --image-pattern '{frame:06d}.jpg' --fps 60 \
  --frame-start <FIRST_FRAME> --frame-end <LAST_FRAME> \
  --camera-neighbors 12 --phase-estimator direct --rs-mode fixed \
  --rs-readout-ms 15.0 --out-dir sync_run --output-sync sync_run/sync.json

python "$REPO/prep/sync/make_timing_candidate.py" \
  --resolve-dir sync_run --readout-ms 15 --weak-cameras \
  --unresolved-rs --out sync_run/timing_candidate.json
```

The solver validates the visual estimate and can publish a timecode-only fallback; inspect its report. The timing export can stop when quality gates fail. See [sync documentation](prep/sync/README.md) and [the pipeline guide](docs/run-pipeline.md#1-capture-timing) for fallbacks and optional inputs. For genlocked, global-shutter footage, skip this stage and omit the timing argument below.

### 5. Initialize points

RoMa v2 matches views and MEMFOF tracks motion across frames. The default is 60 fps and a five-frame keyframe stride, with additional sub-keyframes for fast motion. Pass `--fps 30` for 30 fps footage.

```bash
python "$REPO/prep/init/prep-roma-trigs-memory.py" \
  --reference-json reference_export.json --images-root images \
  --frame-start <FIRST_FRAME> --frame-end <LAST_FRAME> \
  --timing-json sync_run/timing_candidate.json --output-path keyframes.npz
```

Omit `--timing-json` for genlocked, global-shutter footage; `--sync-json` is also supported for a sync-only input. The NPZ carries initialization and capture metadata into training.

### 6. Train and evaluate

With the trainer environment installed and active:

```bash
trigs-train-colmap --data-dir . --output-dir runs/example \
  --npz-path keyframes.npz --timing-json sync_run/timing_candidate.json \
  --config.n-steps 30000

trigs-eval-colmap --data-dir . --runs runs/example --step -1 \
  --out-dir runs/example/eval --shutter captured
```

Omit `--timing-json` when no timing file is needed. The trainer takes the frame range, stride, fps, and target population from the NPZ. See the [trainer guide](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/README.md) and [pipeline guide](docs/run-pipeline.md#3-training) for run behavior and options.

### 7. Export

Export PLYs using the trainer, or create a timeline file using the delivery tool:

```bash
trigs-export-plys --ckpt-path runs/example/trigs_ckpt.pt --output-dir output_plys \
  --n-frames <FRAME_COUNT> --fps 60

python "$REPO/deliver/make-trigs-timeline.py" from-checkpoint \
  --output output.trigsbin --checkpoint runs/example/trigs_ckpt.pt \
  --frame-rate 60 --total-frames <FRAME_COUNT> \
  --chunk-duration-seconds 2.5 --chunk-overlap-seconds 0 \
  --tensor-dtype motion32 --overwrite --fade-frames 0
```

Use the actual frame rate and frame count for the capture. Some final COLMAP checkpoints do not contain the training configuration; when the run changes deformation settings, use a snapshot saved with `--config.checkpoint-dir` to preserve those settings for PLY export. See each command's `--help` for export options.

## Inspecting preparation and training

Two viewers cover different stages of the pipeline:

| Viewer | Repository | Input | Rendering |
| --- | --- | --- | --- |
| [NPZ initialization viewer](#npz-initialization-viewer) | Open4DGS-Core | Prepared `keyframes.npz` | Point cloud in the browser; no CUDA required |
| [Interactive trainer viewer](https://github.com/ASU-NEM-Labs/open4dgs-trigs#interactive-trainer-viewer) | Open4DGS-Trigs | Trained `.pt` checkpoints or live training snapshots | Gaussian rendering on an NVIDIA CUDA GPU |

### NPZ initialization viewer

Use [prep/init/npz-viewer.py](prep/init/npz-viewer.py) to inspect the output of
`prep-roma-trigs-memory.py` before training. From the `open4dgs-core` checkout
root, with the preparation environment active:

```bash
python prep/init/npz-viewer.py \
  --npz /path/to/capture/keyframes.npz \
  --port 8765 --max-points 100000
```

Open **http://localhost:8765**. The script also attempts to open your browser
automatically; add `--no-browser` when running remotely. Keep the terminal open
and press `Ctrl+C` to stop the server.

The Python server needs **NumPy**, and the browser needs **WebGL**. CUDA,
PyTorch, and the trainer package are not required for this viewer. Its page loads
Three.js and fonts from external CDNs, so the browser needs internet access for
those assets. The input must contain `positions` and `times`; the preparation
output also supplies colors, motion, source frames, and other useful metadata.

- **Navigate:** left-drag to orbit, right- or middle-drag to pan, and scroll to
  zoom. Camera presets, scene rotation, and **Reset Up** help orient the scene.
- **Inspect time:** use the frame slider or left/right arrow keys. Press Space
  to play/pause; adjust playback FPS in the sidebar. Frames represent the
  available initialization source-frame groups, not rendered trained motion.
- **Inspect motion and classification:** switch between RGB, time and velocity
  heatmaps, static/dynamic and saved-mask views. Toggle velocity vectors and
  inspect per-frame statistics.
- **Plan preparation regions:** adjust the stage-capsule and detail-sphere
  previews, then use **Copy ROMA args** to copy their settings into a preparation
  command. Previewing regions does not edit the input NPZ.

`--max-points` caps the points displayed per source-frame group; lower it if the
browser becomes slow. The current CLI default is 1,400,000, so the example uses
an explicit smaller limit. This subsamples the display, not the saved NPZ. The
server still loads the full NPZ into RAM, and the browser prefetches frames, so
large captures can need substantial memory even with a lower point limit.

For a remote host, run the server there with `--no-browser`, then forward the
port from your local computer:

```bash
ssh -N -L 8765:localhost:8765 user@remote-host
```

Open **http://localhost:8765** locally. Use trusted machines and protect the
server ports from public access; see [SECURITY.md](SECURITY.md). Run
`python prep/init/npz-viewer.py --help` for the complete CLI options.

### Interactive trainer viewer

The trainer's browser viewer supports playback, visibility and motion diagnostics,
and following a running training job through periodically saved snapshots. With
the trainer environment installed and active:

```bash
python -m trigs.core.viewer --ckpt /path/to/run/trigs_ckpt.pt --port 8080

# Alternatively, follow training snapshots:
python -m trigs.core.viewer --watch /path/to/run/viewer_snapshots --port 8080
```

Open **http://localhost:8080**. This viewer needs CUDA on the machine running
Python. See the [trainer viewer guide](https://github.com/ASU-NEM-Labs/open4dgs-trigs#interactive-trainer-viewer)
for controls, snapshot settings, memory considerations, and matching the viewer's
visibility threshold to the training configuration.

## The `.trigsbin` intermediate format

The export workflow above produces `.trigsbin` files for transferring reconstructions between tools and for inspection and editing. The format stores full-size scene tensors, so files can be large. **It is an intermediate format and is not streamable.**

A separate delivery format with compression and streaming support is planned. These capabilities are roadmap items and are not features of the current `.trigsbin` format.

## Edit reconstructions in SuperSplat

Our [customized SuperSplat editor](https://cmlapps.com/trigs-editor/) supports loading and editing `.trigsbin` files.

Its customized sphere and box tools let you delete content **inside or outside** a selected volume over a specified **frame range**. This is particularly useful for removing the surrounding environment from reconstructions trained without masks.

## Citations and credits

Open4DGS-Trigs builds on [TriGSVanilla](https://github.com/OpsiClear-4DGS/TriGSVanilla), an independent implementation of [TRiGS](https://arxiv.org/abs/2604.00538). The [trainer citations](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/README.md#citations-and-credits) also cover 3D Gaussian Splatting, FreeTimeGS, LongVolCap's temporal hierarchy, and gsplat.

Preparation uses [RoMa v2](https://arxiv.org/abs/2511.15706), [DINOv3](https://arxiv.org/abs/2508.10104), [MEMFOF](https://arxiv.org/abs/2506.23151), [Farnebäck optical flow](https://www.ida.liu.se/ext/WITAS-ev/Computer_Vision_Technologies/PaperInfo/farneback03.html), and [Lucas–Kanade tracking](https://publications.ri.cmu.edu/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai). When used, cite [RoMa v1](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html), [Background Matting V2](https://arxiv.org/abs/2012.07810), and [SAM 2](https://arxiv.org/abs/2408.00714). See [COLMAP's citation instructions](https://colmap.github.io/#citation) for its research references.

## License

Open4DGS-Core is licensed under the [MIT License](LICENSE), except for separately licensed third-party material. The separate Open4DGS-Trigs project is licensed under [AGPL-3.0](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/LICENSE), with incorporated code and notices described in its [third-party notices](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/THIRD_PARTY_NOTICES.md). Third-party software and model weights retain their own licenses.
