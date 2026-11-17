# Running the pipeline

Four steps: capture timing (sync and rolling shutter), point preparation, training, evaluation. The defaults
include speed-aware sub-keyframes,
official-rate motion learning rates, 20 cameras per step. Pass a flag only to change something.

Set `REPO=/absolute/path/to/open4dgs-core` and run from the capture directory
(the one holding `images/`, `sparse/` and `reference_export.json`).

## 1. Capture timing

Readout-aware sync solve, then a timing file for prep and training. The timing file takes every camera the
solve placed from the solve; cameras it could not place start at 0 (timecode only) and the trainer can learn
from there.

```bash
python "$REPO/prep/sync/calculate-sync.py" --reference-json reference_export.json --images-root images \
  --image-pattern '{frame:06d}.jpg' --fps 60 --frame-start 1 --frame-end <LAST_FRAME> \
  --camera-neighbors 12 --phase-estimator direct --rs-mode fixed --rs-readout-ms 15.0 \
  --out-dir sync_run --output-sync sync_run/sync.json
```

```bash
python "$REPO/prep/sync/make_timing_candidate.py" --resolve-dir sync_run --readout-ms 15 \
  --weak-cameras --unresolved-rs --out sync_run/timing_candidate.json
```

- `--rs-readout-ms 15` / `--readout-ms 15` is the row readout measured on the Sony RX0 II rig at 60 fps. Readout
  depends on the sensor mode (resolution, frame rate, crop), not on shutter speed; measure it for other cameras.
- `--sync-json` is optional: it supplies starting values for cameras the solve could not place (e.g. from an
  earlier solve). Solved cameras always come from the solve, so a copy of `sync_candidate.json` adds nothing.
- `--unresolved-rs` applies the readout to cameras the solve could not place. Leave it off only when their
  `sync.json` values came from an older solve that did not model the readout.
- When a gate fails, no timing file is written. `--waive-gate G2_censoring` writes it anyway, marked
  experimental.

## 2. Point preparation

```bash
python "$REPO/prep/init/prep-roma-trigs-memory.py" --reference-json reference_export.json --images-root images \
  --frame-start <FIRST> --frame-end <LAST> --timing-json sync_run/timing_candidate.json \
  --output-path keyframes.npz
```

Filled in automatically:

| Setting | Default |
|---|---|
| `--image-pattern` | `{frame:06d}.jpg` or `.png`, whichever exists |
| `--input-frame-start/-end` | the frames every camera has (tracking reads past the keyframe range) |
| `--temporal-frame-delta` | `--frame-step` (5) |
| `--subkeyframe-min-speed` | 4.0 spacings/frame; off when temporal tracking is off |
| `--memfof-compose-hops` | on with the MEMFOF backend |

Keyframes are every 5th frame. Wherever fast content cannot be followed across 5 frames, prep adds rows for
the frames in between (tracked one frame each way) and the trainer gives them one-frame lifetimes. Each
extra frame costs about as much as a keyframe. `--subkeyframe-min-speed 0` turns this off.

Backends: RoMa v2 (spatial) and MEMFOF (temporal) by default; see [prep/SETUP.md](../prep/SETUP.md).
`--temporal-backend roma` tracks with RoMa instead. Review both models' licences before commercial use.

## 3. Training

```bash
trigs-train-colmap --data-dir . --output-dir runs/<name> --npz-path keyframes.npz \
  --timing-json sync_run/timing_candidate.json --config.n-steps <STEPS>
```

Filled in automatically:

| Setting | Default |
|---|---|
| `--start-frame/--end-frame` | the NPZ's source frame range |
| `--keyframe-step`, `--config.keyframe-stride`, `--config.fps` | the NPZ's; a different explicit value stops the run |
| `--config.target-count` | `target_count_per_keyframe` (25,862) x keyframes in the window; sub-keyframe rows are added on top |
| learning rates | the `lr_*` values are the effective rates at any batch size (`batch_lr_scaling=False`) |

`--config.n-steps` (default 30,000) is the normal-training budget: about 100 visits per frame for 300 frames,
enough for slow and still content.

Fast content needs more: the trainer tops up every frame that has short-lived
(sub-keyframe) splats to 387 visits. The extra steps are added to
`n-steps` and mixed into the run; other frames keep their normal visits. The log prints the added steps,
e.g. `Fast frames: 22/61 topped up toward 387 visits (normal 98); +6,082 steps`. On by default
(`--config.frame-sampling-fast-visits 387`); `0` turns it off. Chunked training applies it the same way.

With it on, TensorBoard `capacity/*` tracks where relocation moves splats: the share of live splats centred
on fast frames and, for the other frames, the fraction of their starting splats they keep. The run ends with
a one-line summary. The ratio includes splats that normal training prunes anyway, so it is low even
without relocation shifting anything; judge it against a run with the top-up off.

Outputs in `runs/<name>/`: `trigs_ckpt.pt`, viewer snapshots every 1,000 steps (last 3 kept;
`--config.viewer-snapshot-keep-every 3000` keeps more), TensorBoard logs in `tb/`, and `run_args.json` (the
resolved inputs, read by the evaluator).

### Long sequences: chunked training

When the whole model does not fit on the GPU, train it in time windows. The model stays in host RAM, and
each visit puts one window's splats on the GPU (splats live in that window plus every static splat):

```bash
trigs-train-colmap --data-dir . --output-dir runs/<name> --npz-path keyframes.npz \
  --timing-json sync_run/timing_candidate.json --config.n-steps <STEPS> \
  --config.temporal-chunk-len-s 3.0 --config.temporal-chunk-visit-steps 1000
```

- `--config.temporal-chunk-visit-steps 1000` cycles through the windows in shuffled 1,000-step visits.
  Learning-rate decay, relocation and Adam state then follow the whole run, as in unchunked training. Without
  it, training makes a few long passes over the windows; inspect temporal coverage carefully when using that mode.
- `--config.temporal-chunk-len-s` sets window length in seconds. GPU memory scales with it; time per step
  does not. Windows overlap by `--config.temporal-chunk-overlap-s` (0.5 s), and the overlap frames are trained by
  both windows. At 3.0 s that is about 20% more frame visits: for a given per-frame level, set
  `n-steps` to about 1.2 x frames x visits.
- The fast-frame top-up, the `capacity/*` watch, PSNR evals and viewer snapshots (the whole model, written
  between visits at the same global-step cadence) all work as in unchunked runs.
- The console shows one progress bar for the whole run (global step and ETA). Its label names the window in
  training, and each visit's start line stays in the log.
- `--config.train-seed` is unchunked only.

Diagnostics are off by default:

| Flag | Adds |
|---|---|
| `--config.diagnostics` | per-relocation counts, init/budget statistics, loader checks (DEBUG log level) |
| `--config.opacity-diag-every 100` | opacity-drain summaries (TensorBoard and console) |
| `--config.record-relocation-lineage` | per-row relocation generation/donor counts in checkpoints |
| `--config.tb-stats-every 25` | denser TensorBoard statistics (default 500) |
| `--config.checkpoint-dir <dir>` | resumable checkpoints every 25% of the run |
| `--config.train-seed <n>` | a fixed seed for frame/camera sampling and relocation draws |

Upgrading old commands: drop every `--config.lr-*` flag that was divided by sqrt(20). Those values now apply
as given, so passing them makes every learning rate 4.5x too small. `--config.batch-lr-scaling` restores the
old rule for old commands.

## 4. Evaluation

```bash
trigs-eval-colmap --data-dir . --runs runs/<name> --step -1 --out-dir runs/<name>/eval \
  --shutter captured
```

`--step -1` evaluates the final checkpoint; any kept snapshot step also works. The frame range comes from
the run's `run_args.json`. Add `--crop-cams`, `--crop-boxes` and `--crop-frames` for close-up crops.

## Known issue

Trained subjects are slightly transparent: against a coloured background you can see through the person a
little (less visible on black). Opacity tuning is open.
