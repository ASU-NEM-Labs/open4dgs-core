# Sub-frame multicamera synchronization

`calculate-sync.py` is the primary entry point. It scans the capture, chooses
time-distributed frames with per-camera motion coverage, estimates visual phase,
validates the circular sync graph, and conditionally
publishes `sync.json`.

The conservative default runs one visual pass (one frame per time bin, with
per-camera coverage repair). Expensive adaptive expansion is opt-in with
`--adaptive-rounds 1`. If enabled, later rounds reuse every completed
frame/pair measurement instead of recomputing the earlier round.
Each round also refuses more than 3,000 uncached frame/pair jobs by default;
raise `--max-new-measurements` only after reviewing the printed work-unit count.

```bash
python3 prep/sync/calculate-sync.py \
  --reference-json /data/capture/reference_export.json \
  --images-root /data/capture/images \
  --fps 60 \
  --frame-start 1 \
  --frame-end 417 \
  --stage-capsule-a=-0.094,-2.495,-0.111 \
  --stage-capsule-b=-0.206,0.595,0.711 \
  --stage-capsule-radius 1.0 \
  --out-dir /data/capture/sync_run \
  --output-sync /data/capture/sync.json
```

The `=` form for negative capsule endpoints avoids command-line parsers
mistaking the value for another option. The shell wrapper accepts the same
values through `STAGE_CAPSULE_A`, `STAGE_CAPSULE_B`, and
`STAGE_CAPSULE_RADIUS`:

```bash
FRAME_START=1 FRAME_END=417 \
REFERENCE_JSON=reference_export.json IMAGES_ROOT=images \
prep/sync/prep-calculate-sync.sh
```

Set `DISABLE_STAGE_CAPSULE=1` in the shell wrapper, or pass
`--disable-stage-capsule` to Python, for a diagnostic run without the capsule.

## Output and publication policy

All diagnostics remain in `--out-dir`, including:

- `frame_motion_scan_by_camera.csv`
- `selected_frames.json`
- `visual_sync_pair_observations.csv`
- `visual_sync_measurement_cache.csv` (persistent superset used for resume)
- `visual_sync_pair_summary.csv`
- `sync_candidate.json`
- `sync_robust.json` (current copy of the candidate, retained for compatibility)
- `robust.log` (current robust-solver console output)
- `sync_robust_report.json`
- `sync_pipeline_report.json`

When every camera survives the degree/uncertainty checks and the verdict is
`GOOD` or `EXCELLENT`, the validated visual candidate is published. Otherwise
the candidate remains in `sync_run`, while `--output-sync` receives a complete
all-zero timecode-only baseline. The pipeline report identifies this as
`output_mode: timecode_only_zero` and `fallback_used: true`. Use
`--disable-zero-fallback` (or `DISABLE_ZERO_FALLBACK=1`) to preserve the old
fail-without-output behavior. Use `--publish-on-failure` only when an explicitly
untrusted visual candidate is needed for investigation.

The final JSON uses PLUS convention:

```text
actual_time_sec = nominal_time_sec + offset_sec[camera]
```

Visual phase is unobservable in genuinely static footage. In that case use an
audio waveform, sync flash/LED, exposure signal, or genlock rather than lowering
the visual information thresholds.

The production default is `--phase-estimator linear`: it computes robust phase
per spatial image cell and bootstraps those cells for uncertainty. The optional
`hybrid` and `direct` modes retain the nonlinear objective for diagnostics, but
any optimum touching the allowed phase boundary is censored instead of treated
as a precise half-frame measurement.

To resume an interrupted run with exactly the same inputs and estimator
settings, pass `--resume-observations` (or set `RESUME_OBSERVATIONS=1` in the
shell wrapper). This also reuses the existing motion scan. Do not resume after
changing calibration, images, scan range/settings, the stage capsule, or
estimator thresholds.

The visual estimator checkpoints each completed frame/pair (including rejected
measurements) to the measurement cache using an atomic replacement. A crash
during measurement therefore leaves earlier completed pairs available for
`--resume-observations` (`--resume` when running `visual-epipolar-sync.py` directly).
Runs made with older versions only saved the cache after all measurements, so
their interrupted work may not be recoverable.

For native crashes such as `SIGSEGV`, the visual estimator enables Python's
fault handler before importing native libraries. Save the fatal traceback,
the printed runtime versions, and `visual_sync_measurement_cache.progress.json`
(next to the cache), which identifies the current frame and camera pair.
The traceback identifies the active Python call; it does not by itself prove
which native library caused memory corruption. An intermittent crash alone
does not establish a race condition.
