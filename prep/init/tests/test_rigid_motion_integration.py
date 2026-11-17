import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_rigid_integration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import rigid_motion as rm  # noqa: E402
from test_rigid_motion import _endpoints, _scene  # noqa: E402


def _arrays(s, unit):
    n = len(s["pos"])
    return {
        "positions": s["pos"].astype(np.float32),
        "colors": s["col"].astype(np.float32),
        "velocities": (s["vel"] * unit).astype(np.float32),
        "accelerations": (s["acc"] * unit * unit).astype(np.float32),
        "has_acceleration": s["has_acc"].copy(),
        "has_velocity": np.ones(n, bool),
        "track_flags": s["flags"].copy(),
        "source_frames": s["sf"].reshape(-1, 1).astype(np.int32),
        "is_static": np.zeros(n, bool),
        "durations": np.full((n, 1), 0.01, np.float32),
    }


@pytest.mark.parametrize("per_frame", [True, False])
def test_apply_rigid_cluster_motion_round_trips_stored_units(per_frame):
    s = _scene()
    d = s["d"]
    unit = 1.0 if per_frame else float(d)
    arrays = _arrays(s, unit)
    out = MODULE.apply_rigid_cluster_motion(
        arrays, frame_delta=d, velocity_per_frame=per_frame, speed_adaptive_duration=True,
        one_frame_duration=0.01, duration_min_scale=0.2, duration_max_scale=1.0,
        params=rm.RigidMotionParams(),
    )
    assert out["rows_changed"] > 0
    rows = s["limb_rows"]
    vel = arrays["velocities"].astype(np.float64) / unit
    acc = arrays["accelerations"].astype(np.float64) / (unit * unit)
    e_f, e_b = _endpoints(s, vel, acc, rows)
    spacing = rm.median_spacing(s["pos"][s["sf"] == d])
    assert np.median(np.linalg.norm(e_b - s["true_bwd"], axis=1)) / spacing < 1.0
    assert np.median(np.linalg.norm(e_f - s["true_fwd"], axis=1)) / spacing < 1.0
    # Durations of the changed keyframe follow the assembly's speed-adaptive rule on
    # the new velocities; keyframes with no changed rows keep theirs.
    mid = np.nonzero(s["sf"] == d)[0]
    expect = MODULE.scale_duration_by_speed(
        arrays["velocities"][mid], arrays["has_velocity"][mid], 0.01, 0.2, 1.0,
    )
    np.testing.assert_allclose(arrays["durations"][mid], expect)
    assert np.allclose(arrays["durations"][s["sf"] == 0], 0.01)


def test_production_defaults():
    argv = sys.argv
    try:
        sys.argv = ["prep", "--reference-json", "x.json", "--images-root", "imgs", "--output-path", "o.npz", "--frame-start", "1", "--frame-end", "9"]
        args = MODULE.parse_args()
    finally:
        sys.argv = argv
    assert args.temporal_backend == "memfof"
    assert args.rigid_cluster_motion is True
    assert args.temporal_frame_delta == args.frame_step == 5
    assert args.subkeyframe_min_speed == 4.0
    assert args.memfof_checkpoint == "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"


def test_rigid_args_do_not_change_the_cache_fingerprint():
    argv = sys.argv
    base = ["prep", "--reference-json", str(SCRIPT), "--images-root", "imgs", "--output-path", "o.npz", "--frame-start", "1", "--frame-end", "9"]
    try:
        sys.argv = base
        a = MODULE.parse_args()
        sys.argv = base + ["--no-rigid-cluster-motion", "--rigid-min-tracks", "20"]
        b = MODULE.parse_args()
        sys.argv = base + ["--temporal-backend", "roma"]
        c = MODULE.parse_args()
    finally:
        sys.argv = argv
    fa, fb, fc = (MODULE.build_run_fingerprint(x)[0] for x in (a, b, c))
    assert fa == fb          # finalize-only options reuse the per-frame cache
    assert fa != fc          # the temporal backend changes every checkpoint
