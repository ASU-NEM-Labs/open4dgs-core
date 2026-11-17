import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "merge_static_into_coverage.py"
SPEC = importlib.util.spec_from_file_location("merge_static_into_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _coverage(n):
    rng = np.random.default_rng(0)
    return {
        "positions": rng.normal(size=(n, 3)).astype(np.float32),
        "colors": rng.random((n, 3)).astype(np.float32),
        "velocities": rng.normal(size=(n, 3)).astype(np.float32),
        "accelerations": rng.normal(size=(n, 3)).astype(np.float32),
        "has_acceleration": np.ones(n, bool),
        "track_flags": np.full(n, 3, np.uint8),
        "has_velocity": np.ones(n, bool),
        "times": np.zeros((n, 1), np.float32),
        "durations": np.full((n, 1), 0.1, np.float32),
        "source_frames": np.full((n, 1), 5, np.int32),
        "is_static": np.zeros(n, bool),
        "dynamic_protect": np.zeros(n, bool),
        "dynamic_allowed": np.ones(n, bool),
        "stage_keep": np.ones(n, bool),
        "keyframe_frames": np.array([5, 10], np.int32),
        "fps": np.asarray(30.0, np.float32),
        "velocity_units": np.asarray("world_per_video_frame"),
    }


def test_merge_appends_acceleration_fields_per_row(tmp_path, monkeypatch):
    n, m = 40, 7
    cov = _coverage(n)
    anchor = {
        "positions": np.arange(3 * (m + 3), dtype=np.float32).reshape(-1, 3),
        "colors": np.full((m + 3, 3), 0.5, np.float32),
        "is_static": np.r_[np.ones(m, bool), np.zeros(3, bool)],
    }
    np.savez(tmp_path / "cov.npz", **cov)
    np.savez(tmp_path / "anc.npz", **anchor)
    out = tmp_path / "out.npz"
    monkeypatch.setattr(sys, "argv", ["merge", "--coverage", str(tmp_path / "cov.npz"),
                                      "--anchor", str(tmp_path / "anc.npz"), "--output", str(out)])
    MODULE.main()
    z = np.load(out)
    for k in cov:
        if cov[k].ndim >= 1 and len(cov[k]) == n:
            assert len(z[k]) == n + m, k                      # every per-row field got the static rows
    np.testing.assert_array_equal(z["accelerations"][:n], cov["accelerations"])
    assert not z["accelerations"][n:].any() and not z["has_acceleration"][n:].any()
    assert not z["track_flags"][n:].any() and z["is_static"][n:].all()
    np.testing.assert_array_equal(z["positions"][n:], anchor["positions"][:m])
    np.testing.assert_array_equal(z["keyframe_frames"], [5, 10])   # metadata untouched
    assert str(z["velocity_units"]) == "world_per_video_frame"
