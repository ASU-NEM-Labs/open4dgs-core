import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_memory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _dense_line():
    x = np.arange(20, dtype=np.float64) * 0.01
    trusted = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    two_view = np.stack([x + 0.005, np.zeros_like(x), np.zeros_like(x)], axis=1)
    floaters = np.asarray(
        [[0.0, 0.50, 0.0], [0.0, 0.80, 0.0], [0.0, 1.10, 0.0]],
        dtype=np.float64,
    )
    points = np.concatenate([trusted, two_view, floaters], axis=0)
    support = np.asarray([3] * len(trusted) + [2] * (len(two_view) + len(floaters)))
    return points, support


def test_keeps_dense_two_view_surface_and_rejects_sparse_floaters():
    points, support = _dense_line()
    keep, info = MODULE.filter_two_view_by_three_view_core(
        [points],
        support,
        enabled=True,
        radius=0.0,
        radius_factor=2.5,
        support_knn=4,
    )

    assert keep[:40].all()
    assert not keep[40:].any()
    assert info["two_view_kept"] == 20
    assert info["two_view_rejected"] == 3


def test_requires_support_at_both_temporal_endpoints():
    points_t, support = _dense_line()
    points_tnext = points_t.copy()
    # Make five otherwise-valid 2-view rows form a sparse temporal spray.
    points_tnext[20:25, 1] += np.arange(1, 6, dtype=np.float64) * 0.25

    keep, _ = MODULE.filter_two_view_by_three_view_core(
        [points_t, points_tnext],
        support,
        enabled=True,
        radius=0.0,
        radius_factor=2.5,
        support_knn=4,
    )

    assert keep[:20].all()
    assert not keep[20:25].any()
    assert keep[25:40].all()

