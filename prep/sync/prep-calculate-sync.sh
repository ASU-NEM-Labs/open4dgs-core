#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

REFERENCE_JSON="${REFERENCE_JSON:-reference_export.json}"
IMAGES_ROOT="${IMAGES_ROOT:-images}"
IMAGE_PATTERN="${IMAGE_PATTERN:-}"
if [[ -z "$IMAGE_PATTERN" ]]; then
  IMAGE_PATTERN='{frame:06d}.jpg'
fi
FPS="${FPS:-60}"
FRAME_START="${FRAME_START:?Set FRAME_START to the first image frame to scan}"
FRAME_END="${FRAME_END:?Set FRAME_END to the last image frame to scan}"
FRAME_STEP="${FRAME_STEP:-8}"
OUT_DIR="${OUT_DIR:-sync_run}"
OUTPUT_SYNC="${OUTPUT_SYNC:-sync.json}"
RS_MODE="${RS_MODE:-report}"

STAGE_CAPSULE_A="${STAGE_CAPSULE_A:--0.094,-2.495,-0.111}"
STAGE_CAPSULE_B="${STAGE_CAPSULE_B:--0.206,0.595,0.711}"
STAGE_CAPSULE_RADIUS="${STAGE_CAPSULE_RADIUS:-1.000}"

args=(
  --reference-json "$REFERENCE_JSON"
  --images-root "$IMAGES_ROOT"
  --image-pattern "$IMAGE_PATTERN"
  --fps "$FPS"
  --frame-start "$FRAME_START"
  --frame-end "$FRAME_END"
  --frame-step "$FRAME_STEP"
  --out-dir "$OUT_DIR"
  --output-sync "$OUTPUT_SYNC"
  --rs-mode "$RS_MODE"
  "--stage-capsule-a=$STAGE_CAPSULE_A"
  "--stage-capsule-b=$STAGE_CAPSULE_B"
  --stage-capsule-radius "$STAGE_CAPSULE_RADIUS"
)

if [[ "${DISABLE_STAGE_CAPSULE:-0}" == "1" ]]; then
  args+=(--disable-stage-capsule)
fi

if [[ "${RESUME_OBSERVATIONS:-0}" == "1" ]]; then
  args+=(--resume-observations)
fi

if [[ -n "${ADAPTIVE_ROUNDS:-}" ]]; then
  args+=(--adaptive-rounds "$ADAPTIVE_ROUNDS")
fi

if [[ "${DISABLE_ZERO_FALLBACK:-0}" == "1" ]]; then
  args+=(--disable-zero-fallback)
fi

exec python3 "$SCRIPT_DIR/calculate-sync.py" "${args[@]}" "$@"
