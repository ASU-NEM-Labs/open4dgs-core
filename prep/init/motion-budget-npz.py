#!/usr/bin/env python3
"""Apply the motion-aware cap (motion_budget.py) to an existing prep NPZ.

Same pass as prep's --motion-budget: each keyframe's still rows are thinned to
--target-points-per-frame and its moving rows are kept (or capped by
--moving-per-frame; 0 = no cap). Per-row arrays are subset; everything else is
copied unchanged. An NPZ assembled with a per-frame cap has already thinned its
moving rows, so this can only lower still density there; to keep more moving
rows, re-run prep with --motion-budget.

Example:
  python motion-budget-npz.py in.npz out.npz --target-points-per-frame 30000
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("prep_roma_trigs_memory", HERE / "prep-roma-trigs-memory.py")
PREP = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = PREP
_spec.loader.exec_module(PREP)

import motion_budget as mb  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--target-points-per-frame", type=int, default=70000,
                   help="Cap on still rows per keyframe (0 = none).")
    p.add_argument("--moving-per-frame", type=int, default=0, help="Cap on moving rows (0 = none).")
    p.add_argument("--min-speed", type=float, default=0.2,
                   help="Spacings per video frame (see prep --motion-budget-min-speed).")
    p.add_argument("--knn", type=int, default=16)
    p.add_argument("--spacing-points", type=int, default=70000,
                   help="Density at which the speed threshold's spacing is measured "
                        "(see prep --motion-budget-spacing-points); keep it equal to prep's.")
    p.add_argument("--dedup-radius", type=float, default=0.0)
    p.add_argument("--dedup-radius-max", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats-path", default=None, help="Optional JSON with per-keyframe counts.")
    a = p.parse_args()

    with np.load(a.input, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    n = len(data["positions"])
    rows = {k: v for k, v in data.items() if v.ndim >= 1 and len(v) == n}
    missing = [k for k in ("velocities", "has_velocity", "track_flags", "source_frames", "is_static", "colors")
               if k not in rows]
    if missing:
        raise SystemExit(f"{a.input} lacks per-row arrays {missing}")
    units = str(data.get("velocity_units", "world_per_video_frame"))
    per_frame = "world_per_video_frame" in units
    # Per-keyframe-interval velocities: the trainer converts them with keyframe_step, so do the same.
    frame_delta = int(data.get("keyframe_step", 1))
    out, stats = PREP.apply_motion_budget(
        rows, frame_delta=frame_delta, velocity_per_frame=per_frame,
        dedup_radius=a.dedup_radius, dedup_radius_max=a.dedup_radius_max,
        params=mb.MotionBudgetParams(a.moving_per_frame, a.target_points_per_frame, a.min_speed, a.knn, a.seed,
                                     a.spacing_points),
    )
    data.update(out)
    np.savez_compressed(a.output, **data)
    print(f"wrote {a.output}: {len(out['positions']):,} rows")
    if a.stats_path:
        Path(a.stats_path).write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
