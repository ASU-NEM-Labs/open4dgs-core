#!/usr/bin/env python3
"""Merge static/background rows from an already-computed anchor NPZ onto a
dynamic-only coverage NPZ, without re-running the (expensive) RoMa prep.

Use case: a coverage prep whose capsule config filtered out the background has
no `is_static` rows. Rather than re-run the 8h prep, lift the static rows from
a previously-computed anchor NPZ (e.g. the `--hybrid-static-from anchor` file)
and append them to the coverage NPZ so the single-npz training path
(`build_canonical_from_npz`, `--npz-path`) renders them always-on.

The appended static rows are forced to: is_static=True, velocity=0,
has_velocity=False, dynamic_allowed=False, dynamic_protect=False. Their
times/durations are irrelevant (static rows render with gamma=1, undeformed).

Only rows tagged `is_static` in the anchor are taken, so the anchor's dense
*dynamic* priors are NOT carried over.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Row-aligned fields we know how to synthesize for static background rows.
_ROW_FIELDS = {
    "positions", "colors", "velocities", "times", "durations",
    "has_velocity", "is_static", "dynamic_protect", "dynamic_allowed",
    "stage_keep", "source_frames",
    # --centred-velocity / --rigid-cluster-motion outputs.
    "accelerations", "has_acceleration", "track_flags",
}
# Coverage-length arrays that are NOT per row (only ambiguous if a count
# happens to equal the row count).
_METADATA_FIELDS = {"keyframe_frames"}


def _static_block(key, cov_arr, m, anchor_pos, anchor_col, static_time, static_dur, src_start):
    """Build the (m, ...) block of appended static rows for one field,
    matching the coverage array's trailing shape and dtype."""
    trailing = cov_arr.shape[1:]
    dt = cov_arr.dtype
    if key == "positions":
        return anchor_pos.astype(dt, copy=False)
    if key == "colors":
        return anchor_col.astype(dt, copy=False)
    if key == "times":
        return np.full((m,) + trailing, static_time, dtype=dt)
    if key == "durations":
        return np.full((m,) + trailing, static_dur, dtype=dt)
    if key in ("is_static", "stage_keep"):
        return np.ones((m,) + trailing, dtype=dt)  # True
    if key in (
        "velocities", "has_velocity", "dynamic_protect", "dynamic_allowed",
        "accelerations", "has_acceleration", "track_flags",
    ):
        return np.zeros((m,) + trailing, dtype=dt)  # 0 / False / no tracked endpoints
    if key == "source_frames":
        return np.full((m,) + trailing, src_start, dtype=dt)
    # Unknown row-aligned field: keep counts consistent, warn.
    print(f"  WARNING: unknown row-aligned field {key!r}; filling static rows with zeros")
    return np.zeros((m,) + trailing, dtype=dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", required=True, type=Path, help="Dynamic coverage NPZ (kept whole).")
    ap.add_argument("--anchor", required=True, type=Path, help="NPZ holding static/background rows.")
    ap.add_argument("--output", required=True, type=Path, help="Merged NPZ to write.")
    ap.add_argument("--anchor-all-static", action="store_true",
                    help="Treat ALL anchor rows as static (use only if the anchor has no is_static field).")
    ap.add_argument("--static-time", type=float, default=0.5, help="Placeholder normalized time for static rows (ignored at render).")
    ap.add_argument("--static-duration", type=float, default=0.05, help="Placeholder duration for static rows (ignored at render).")
    args = ap.parse_args()

    cov = dict(np.load(args.coverage, allow_pickle=True))
    anc = dict(np.load(args.anchor, allow_pickle=True))

    n_cov = int(np.asarray(cov["positions"]).shape[0])
    n_anc = int(np.asarray(anc["positions"]).shape[0])
    cov_static = int(
        np.asarray(cov.get("is_static", np.zeros((n_cov,), dtype=bool)))
        .reshape(-1)
        .astype(bool)
        .sum()
    )
    print(f"coverage: {n_cov:,} rows ({args.coverage.name})")
    print(f"anchor:   {n_anc:,} rows ({args.anchor.name})")

    # --- pick static rows from the anchor -------------------------------
    if args.anchor_all_static:
        static_mask = np.ones((n_anc,), dtype=bool)
        print("  --anchor-all-static: taking ALL anchor rows as static")
    elif "is_static" in anc:
        static_mask = np.asarray(anc["is_static"]).reshape(-1).astype(bool)
        print(f"  anchor is_static rows: {int(static_mask.sum()):,} / {n_anc:,}")
    else:
        raise SystemExit(
            "Anchor NPZ has no 'is_static' field. Re-run with --anchor-all-static "
            "if every anchor row is background, or pick a different anchor."
        )

    m = int(static_mask.sum())
    if m == 0:
        raise SystemExit("No static rows found in anchor; nothing to merge.")

    anchor_pos = np.asarray(anc["positions"])[static_mask].astype(np.float32)
    anchor_col = np.asarray(anc["colors"])[static_mask].astype(np.float32)

    src_start = int(np.asarray(cov["source_frames"]).min()) if "source_frames" in cov else int(args.static_time)

    # --- build merged dict ----------------------------------------------
    merged: dict = {}
    for key, cov_arr in cov.items():
        cov_arr = np.asarray(cov_arr)
        # Every coverage-length array is per row (an unknown one is zero-filled
        # with a warning), so no field is left with the coverage row count.
        is_row = cov_arr.ndim >= 1 and cov_arr.shape[0] == n_cov and key not in _METADATA_FIELDS
        if is_row:
            block = _static_block(
                key, cov_arr, m, anchor_pos, anchor_col,
                args.static_time, args.static_duration, src_start,
            )
            merged[key] = np.concatenate([cov_arr, block], axis=0)
        else:
            # scalar / metadata (fps, frame ranges, keyframe_frames, ...) -> keep coverage's
            merged[key] = cov_arr

    n_out = int(np.asarray(merged["positions"]).shape[0])
    for key, arr in merged.items():
        arr = np.asarray(arr)
        if key in _METADATA_FIELDS and arr.ndim >= 1 and arr.shape[0] == n_cov:
            print(f"  NOTE: metadata field {key!r} has the coverage row count {n_cov}; kept as metadata.")

    np.savez_compressed(args.output, **merged)
    print(
        f"\nwrote {args.output}  "
        f"(coverage {n_cov:,}: {cov_static:,} static + "
        f"{n_cov - cov_static:,} dynamic; anchor {m:,} static; "
        f"total {n_out:,} rows)"
    )


if __name__ == "__main__":
    main()
