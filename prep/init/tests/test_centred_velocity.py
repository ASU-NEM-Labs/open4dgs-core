import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_centred_velocity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _parabola(n=50, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 3)).astype(np.float32)
    v = rng.normal(size=(n, 3)).astype(np.float32) * 0.05
    a = rng.normal(size=(n, 3)).astype(np.float32) * 0.01
    at = lambda t: x0 + v * t + 0.5 * a * t * t
    return x0, v, a, at


def test_centred_velocity_and_acceleration_are_exact_for_a_parabola():
    x0, v, a, at = _parabola()
    src = np.arange(len(x0))
    pos_tn, accel, has = MODULE.combine_bidirectional_tracks(
        at(0.0), at(1.0), src, at(0.0), at(-1.0), src, 1,
    )
    assert has.all()
    np.testing.assert_allclose(pos_tn - at(0.0), v, atol=1e-5)          # centred, not v + a/2
    np.testing.assert_allclose(accel, a, atol=1e-5)


def test_forward_only_points_keep_the_forward_difference():
    x0, v, a, at = _parabola()
    src_f = np.arange(len(x0))
    src_b = src_f[::2]                                                 # odd points lost going back
    pos_tn, accel, has = MODULE.combine_bidirectional_tracks(
        at(0.0), at(1.0), src_f, at(0.0)[src_b], at(-1.0)[src_b], src_b, 1,
    )
    assert has[::2].all() and not has[1::2].any()
    np.testing.assert_allclose(pos_tn[1::2], at(1.0)[1::2], atol=1e-6)
    assert np.all(accel[1::2] == 0)


def test_each_direction_differences_against_its_own_x_t():
    """The backward track triangulates X(t) from different cameras; a shared
    offset in that X(t) must not leak into the acceleration."""
    x0, v, a, at = _parabola()
    src = np.arange(len(x0))
    shift = np.float32(0.003)
    _, accel, _ = MODULE.combine_bidirectional_tracks(
        at(0.0), at(1.0), src, at(0.0) + shift, at(-1.0) + shift, src, 1,
    )
    np.testing.assert_allclose(accel, a, atol=1e-5)


def test_rows_recover_acceleration_by_exact_position():
    x0, v, a, at = _parabola()
    has = np.ones(len(x0), bool)
    rows = np.concatenate([x0[[5, 2, 9]], np.array([[9.0, 9.0, 9.0]], np.float32)])
    accel, found = MODULE.accelerations_for_rows(rows, x0, a, has)
    np.testing.assert_allclose(accel[:3], a[[5, 2, 9]])
    assert found.tolist() == [True, True, True, False]
    assert np.all(accel[3] == 0)
