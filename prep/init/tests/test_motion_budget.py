import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motion_budget as mb  # noqa: E402

STEP = 0.01


def _plane(nx, ny, x0=0.0):
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), -1).reshape(-1, 2) * STEP
    return np.c_[g[:, 0] + x0, g[:, 1], np.zeros(len(g))]


def _scene():
    """Keyframe 1: a still wall only (one row with a wild velocity).
    Keyframe 3: the same wall plus a fast patch, part of it untracked.
    Plus static rows that must be left alone."""
    wall = _plane(40, 40)                                   # 1600 rows
    patch = _plane(30, 30, x0=5.0)                          # 900 rows, far from the wall
    pos = np.concatenate([wall, wall, patch, _plane(5, 10, x0=-5.0)])
    n_w, n_p, n_s = len(wall), len(patch), 50
    sf = np.r_[np.full(n_w, 1), np.full(n_w, 3), np.full(n_p, 3), np.full(n_s, 0)]
    vel = np.zeros((len(pos), 3))
    vel[2 * n_w:2 * n_w + n_p, 0] = 0.05                    # fast: several spacings / frame
    vel[123, 0] = 0.5                                       # one wild track on the calm wall
    has_vel = np.ones(len(pos), bool)
    untracked = np.arange(2 * n_w, 2 * n_w + n_p)[::7]      # ~1/7 of the patch untracked
    has_vel[untracked] = False
    vel[untracked] = 0.0
    is_static = np.r_[np.zeros(2 * n_w + n_p, bool), np.ones(n_s, bool)]
    return dict(pos=pos, vel=vel, has_vel=has_vel, sf=sf, is_static=is_static,
                n_w=n_w, n_p=n_p, n_s=n_s, untracked=untracked)


def _first(pos, col, pri, target):
    return np.arange(target)


def _run(s, **kw):
    params = mb.MotionBudgetParams(**{"moving_per_frame": 500, "still_per_frame": 300,
                                      "min_speed_spacings": 0.5, **kw})
    return mb.motion_budget_select(
        positions=s["pos"], colors=np.zeros_like(s["pos"]), priorities=np.ones(len(s["pos"])),
        speed_per_frame=np.linalg.norm(s["vel"], axis=1), has_velocity=s["has_vel"],
        source_frames=s["sf"], is_static=s["is_static"], params=params, thin_fn=_first,
    )


def test_fast_keyframe_keeps_more_rows_than_calm_keyframe():
    s = _scene()
    keep, st = _run(s)
    kf = {p["keyframe"]: p for p in st["per_keyframe"]}
    assert kf[1]["moving_in"] == 0                          # the wild track does not flip a row
    assert (kf[1]["moving_kept"], kf[1]["still_kept"]) == (0, 300)
    assert kf[3]["moving_in"] == s["n_p"]                   # untracked patch rows count as moving
    assert (kf[3]["moving_kept"], kf[3]["still_kept"]) == (500, 300)
    sf_kept = s["sf"][keep]
    assert (sf_kept == 3).sum() == 800 and (sf_kept == 1).sum() == 300


def test_moving_rows_uncapped_by_default_while_still_rows_are_capped():
    s = _scene()
    keep, st = _run(s, moving_per_frame=0)
    kf = {p["keyframe"]: p for p in st["per_keyframe"]}
    assert (kf[3]["moving_kept"], kf[3]["still_kept"]) == (s["n_p"], 300)    # all 900 movers kept
    assert (kf[1]["moving_kept"], kf[1]["still_kept"]) == (0, 300)
    assert mb.MotionBudgetParams().moving_per_frame == 0


def test_static_rows_untouched_and_order_preserved():
    s = _scene()
    keep, st = _run(s)
    static = np.nonzero(s["is_static"])[0]
    assert np.isin(static, keep).all() and st["static_rows"] == s["n_s"]
    assert (np.diff(keep) > 0).all()


def test_classes_below_quota_or_uncapped_are_kept_whole():
    s = _scene()
    assert len(_run(s, moving_per_frame=5000, still_per_frame=5000)[0]) == len(s["pos"])
    assert len(_run(s, moving_per_frame=0, still_per_frame=0)[0]) == len(s["pos"])


def test_neighbourhood_speed_ignores_untracked_and_outliers():
    pos = _plane(10, 10)
    speed = np.full(len(pos), 0.03)
    has = np.ones(len(pos), bool)
    speed[42] = 9.0                                         # outlier
    has[17] = False; speed[17] = 0.0                        # untracked
    med = mb.neighbourhood_speed(pos, speed, has, knn=8)
    np.testing.assert_allclose(med, 0.03)


def test_spacing_measured_at_budgeted_density():
    pos = _plane(100, 100)                                  # 10k rows at STEP
    np.testing.assert_allclose(mb.keyframe_spacing(pos, 10_000), STEP, rtol=1e-6)
    np.testing.assert_allclose(mb.keyframe_spacing(pos, 2_500), 2 * STEP, rtol=1e-6)   # 1/4 rows -> 2x
    np.testing.assert_allclose(mb.keyframe_spacing(pos, 50_000), STEP, rtol=1e-6)     # never below actual


def test_quota_enforced_when_thinning_overshoots():
    s = _scene()
    keep, st = mb.motion_budget_select(
        positions=s["pos"], colors=np.zeros_like(s["pos"]), priorities=np.ones(len(s["pos"])),
        speed_per_frame=np.linalg.norm(s["vel"], axis=1), has_velocity=s["has_vel"],
        source_frames=s["sf"], is_static=s["is_static"],
        params=mb.MotionBudgetParams(moving_per_frame=500, still_per_frame=300),
        thin_fn=lambda pos, col, pri, target: np.arange(len(pos)),   # thins nothing
    )
    kf = {p["keyframe"]: p for p in st["per_keyframe"]}
    assert (kf[3]["moving_kept"], kf[3]["still_kept"], kf[1]["still_kept"]) == (500, 300, 300)


def test_still_cap_does_not_move_the_speed_threshold():
    """Lowering the still cap must not reclassify movers (spacing is measured at
    spacing_points, not at the cap)."""
    s = _scene()
    a = _run(s, still_per_frame=300, spacing_points=800)[1]
    b = _run(s, still_per_frame=30, spacing_points=800)[1]
    assert [p["moving_in"] for p in a["per_keyframe"]] == [p["moving_in"] for p in b["per_keyframe"]]
    assert [p["spacing"] for p in a["per_keyframe"]] == [p["spacing"] for p in b["per_keyframe"]]
