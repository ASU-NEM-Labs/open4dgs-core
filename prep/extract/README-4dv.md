# 4DV.ai JCam import

`extract_4dv_sequence.py` adapts a 4DV capture without changing the raw MP4s.
It reads every per-frame timestamp from `record_logs`, identifies the common
multi-camera interval, extracts a contiguous sequence into the usual
`images/<camera>/<frame>.jpg` layout, and writes `sync.json` using the PLUS
convention used by RoMa and the COLMAP TriGS trainer:

```text
actual_capture_time = logical_frame_time + camera_offset_seconds
```

The importer also:

- writes every modeled source timestamp and source-frame mapping to
  `frame_timestamps.json`;
- normalizes the empty-stage snapshots into `reference_images`;
- converts the manifest intrinsics/extrinsics into `reference_export.json` and
  a COLMAP text model under `sparse/0`;
- adjusts both pixels and calibration consistently for `l`, `r`, `u`, or `n`
  rotations;
- verifies MP4/log frame counts and media dimensions with `ffprobe`;
- reports whether one constant offset per camera is adequate.

## Per-camera clock ladder

`record_logs` are short (often ~100 frames) relative to a full take. Each
camera free-runs its own clock -- on a 96-camera rig, measured period spread
is ~16.60-16.65ms (60.06-60.25fps) with a start phase spread of a full frame
period, so a single logical instant is observed by at most one or two
cameras. `fit_camera_ladder()` turns each camera's logged frames into a
robust `time_ns(frame_number) = t1 + (frame_number - 1) * period` model:

- **Min-latency (lower-envelope) regression.** Encoder save times are
  one-sided -- a frame can be delayed by encode/write contention but never
  saved before capture -- so the fit is iteratively refit to the
  lowest-residual frames instead of an ordinary least-squares mean fit.
- **IDR frames are dropped before fitting.** On this rig they save a
  measured +1.87ms later than P frames -- a NAL-type-dependent encode-latency
  artifact with nothing to do with capture time.
- **Uncertainty by block bootstrap** (not i.i.d.): residuals are
  autocorrelated (lag-1 ~= +0.72, a queueing signature), so a plain bootstrap
  understates the true uncertainty by roughly 2x.

This lets extraction go **past** the logged prefix: `--num-frames` is capped
by the confirmed MP4 length (via `ffprobe`, unless `--skip-media-probe`), not
by `record_logs` length. A separate confidence gate applies once the window
extends past the log: the model's 1-sigma uncertainty at the last selected
frame (`--max-ladder-sigma-ms`, default 4.0) must stay under budget, or the
run raises unless `--allow-ladder-extrapolation` is passed. Check
`import_report.json`'s `ladder_sigma_at_last_frame_ms` per camera before
trusting a long extrapolated window -- on the cello take, extrapolating to a
300-frame window pushed the worst camera to ~5.85ms, so
`--max-ladder-sigma-ms 6.0` was needed there.

`frame_timestamps.json`'s per-camera `timestamp_ns` list carries the full
per-frame modeled ladder (not one constant offset); `sync.json` still carries
one legacy scalar per camera for RoMa/import_report, now evaluated at the
window **midpoint** rather than frame 0 to roughly halve its worst-case error
for a long window. `train_from_colmap.py` reads the per-frame
`frame_timestamps.json` directly when present, falling back to `sync.json`
only for datasets that don't have it.

An MP4 may contain frames after the timestamped `record_logs` prefix. Those
trailing frames are reported but only extractable via ladder extrapolation
(above), not raw log lookup. An MP4 shorter than its log remains a hard
error.

Start with an audit. `--num-frames` is intentionally omitted so the audit uses
the complete common interval:

```bash
python prep/extract/extract_4dv_sequence.py \
  --capture-dir /path/to/4dv-capture \
  --output-dir /path/to/processed-capture \
  --fps 60 \
  --default-rot n \
  --dry-run
```

Review the reported output-frame count, ladder fit RMS, and ladder sigma at
the last frame. Then run the same command without `--dry-run` to decode. Do
not add `--allow-unstable-timestamps` or `--allow-ladder-extrapolation` until
the audit has been reviewed; they suppress the safety gates but do not make a
noisy clock model correct.

The importer preserves the provided 4DV calibration as a pose-only starting
model. Validate it before RoMa; each new capture may require a fresh
empty-stage COLMAP solve because the manifest poses are not epipolar-consistent
with the captured images:

```bash
python prep/align/build-colmap-ref-trigs.py \
  --data-dir /path/to/processed-capture \
  --reference-images-dir /path/to/processed-capture/reference_images \
  --ref-frame 1 \
  --require-all-cameras
```

The alignment wrapper selects the largest retained numbered COLMAP model; model
`0` is not necessarily the largest. With `--require-all-cameras`, incomplete
alignment fails before replacing `sparse/0` or `reference_export.json`. Preserve
the manifest-derived files when comparing calibrations.

