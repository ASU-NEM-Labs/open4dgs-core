import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_motion_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import motion_budget as mb  # noqa: E402
import rigid_motion as rm  # noqa: E402
from test_motion_budget import _scene  # noqa: E402


def _arrays(s, unit):
    n = len(s["pos"])
    flags = np.where(s["has_vel"], rm.TRACK_FWD | rm.TRACK_BWD, 0).astype(np.uint8)
    return {
        "positions": s["pos"].astype(np.float32),
        "colors": np.zeros((n, 3), np.float32),
        "velocities": (s["vel"] * unit).astype(np.float32),
        "accelerations": np.zeros((n, 3), np.float32),
        "has_acceleration": np.zeros(n, bool),
        "has_velocity": s["has_vel"].copy(),
        "track_flags": flags,
        "source_frames": s["sf"].reshape(-1, 1).astype(np.int32),
        "is_static": s["is_static"].copy(),
        "durations": np.arange(n, dtype=np.float32).reshape(-1, 1),
    }


@pytest.mark.parametrize("per_frame", [True, False])
def test_apply_motion_budget_reads_stored_units_and_keeps_rows_aligned(per_frame):
    s = _scene()
    d = 2
    unit = 1.0 if per_frame else float(d)
    arrays = _arrays(s, unit)
    params = mb.MotionBudgetParams(moving_per_frame=500, still_per_frame=300, min_speed_spacings=0.5)
    out, st = MODULE.apply_motion_budget(
        arrays, frame_delta=d, velocity_per_frame=per_frame,
        dedup_radius=0.0, dedup_radius_max=0.0, params=params,
    )
    kf = {p["keyframe"]: p for p in st["per_keyframe"]}
    assert kf[3]["moving_in"] == s["n_p"] and kf[1]["moving_in"] == 0
    n = len(out["positions"])
    assert all(len(v) == n for v in out.values())
    # Every kept row carries its own data: durations were set to the row index.
    idx = out["durations"].reshape(-1).astype(int)
    np.testing.assert_array_equal(out["positions"], arrays["positions"][idx])
    np.testing.assert_array_equal(out["velocities"], arrays["velocities"][idx])
    assert out["is_static"].sum() == s["n_s"]
    # The test planes are flat, so voxel thinning stalls and the quota fallback applies.
    assert (kf[3]["moving_kept"], kf[3]["still_kept"], kf[1]["still_kept"]) == (500, 300, 300)


def test_motion_budget_args_do_not_change_the_cache_fingerprint(tmp_path, monkeypatch):
    ref = tmp_path / "reference.json"
    ref.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "prep", "--reference-json", str(ref), "--images-root", str(tmp_path),
        "--output-path", str(tmp_path / "out.npz"), "--frame-start", "1", "--frame-end", "3",
    ])
    args = MODULE.parse_args()
    assert args.motion_budget is True                      # production default
    args.motion_budget = False
    fp = lambda: MODULE.build_run_fingerprint(args)[0]
    off = fp()
    args.target_points_per_frame = 999
    assert fp() != off                                     # without the budget, assembly caps
    args.motion_budget = True
    on = fp()
    assert on != off                                       # the budget changes assembly (no cap)
    args.target_points_per_frame = 12345                   # ... and moves the cap to finalize
    args.motion_budget_moving_per_frame = 123
    args.motion_budget_min_speed = 2.0
    assert fp() == on


def test_motion_budget_npz_tool_subsets_rows_and_copies_metadata(tmp_path, monkeypatch):
    s = _scene()
    arrays = _arrays(s, 1.0)
    src = tmp_path / "in.npz"
    np.savez_compressed(src, **arrays, keyframe_frames=np.array([1, 3], np.int32),
                        velocity_units=np.asarray("world_per_video_frame"), fps=np.asarray(30.0, np.float32))
    tool_spec = importlib.util.spec_from_file_location("motion_budget_npz", SCRIPT.parent / "motion-budget-npz.py")
    tool = importlib.util.module_from_spec(tool_spec)
    tool_spec.loader.exec_module(tool)
    dst = tmp_path / "out.npz"
    monkeypatch.setattr(sys, "argv", ["tool", str(src), str(dst), "--moving-per-frame", "500",
                                      "--target-points-per-frame", "300", "--min-speed", "0.5"])
    tool.main()
    z = np.load(dst)
    assert len(z["positions"]) == 300 + 800 + s["n_s"]
    assert all(len(z[k]) == len(z["positions"]) for k in arrays)
    np.testing.assert_array_equal(z["keyframe_frames"], [1, 3])
    assert str(z["velocity_units"]) == "world_per_video_frame"
