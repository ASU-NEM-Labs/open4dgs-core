import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rigid_motion as rm  # noqa: E402


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def _limb_samples(n, rng):
    """Points on a cylinder (radius 0.04, length 0.3) along +x from a pivot at the origin."""
    s = rng.uniform(0, 0.3, n)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.c_[s, 0.04 * np.cos(th), 0.04 * np.sin(th)]


def _scene(seed=0, sticky_frac=0.3, untracked_frac=0.3, d=2):
    """Keyframes 0, d, 2d. A limb rotates 25 deg per step about its pivot (a stop-like
    swing seen from the middle keyframe); a static background wall sits behind it.
    Each keyframe is an independent sampling (like per-keyframe triangulation)."""
    rng = np.random.default_rng(seed)
    pivot = np.array([0.0, 0.0, 1.0])
    R_step = _rot([0, 0, 1], 25.0)
    poses = {0: R_step.T, d: np.eye(3), 2 * d: R_step}              # limb rotation at each keyframe
    wall_xy = np.stack(np.meshgrid(np.arange(-0.2, 0.6, 0.011), np.arange(-0.3, 0.3, 0.011)), -1).reshape(-1, 2)
    wall = np.c_[wall_xy, np.full(len(wall_xy), 1.3)]
    pos, col, sf, truth = [], [], [], {}
    for k, R in poses.items():
        local = _limb_samples(700, rng)
        limb = local @ R.T + pivot
        pos += [limb, wall]
        col += [np.tile([0.6, 0.4, 0.3], (len(limb), 1)), np.tile([0.2, 0.2, 0.25], (len(wall), 1))]
        sf += [np.full(len(limb), k), np.full(len(wall), k)]
        if k == d:
            truth["limb_local"] = local
            truth["n_limb"] = len(limb)
    pos, col, sf = np.concatenate(pos), np.concatenate(col), np.concatenate(sf)
    vel = np.zeros_like(pos); acc = np.zeros_like(pos)
    has_acc = np.zeros(len(pos), bool); flags = np.zeros(len(pos), np.uint8)

    mid = np.nonzero(sf == d)[0]
    limb_rows = mid[: truth["n_limb"]]
    wall_rows = mid[truth["n_limb"]:]
    local = truth["limb_local"]
    true_fwd = local @ poses[2 * d].T + pivot
    true_bwd = local @ poses[0].T + pivot
    x = pos[limb_rows]
    u = rng.uniform(size=len(limb_rows))
    good = u < 1 - sticky_frac - untracked_frac
    sticky = (u >= 1 - sticky_frac - untracked_frac) & (u < 1 - untracked_frac)
    # Correct two-way tracks: centred velocity + acceleration from the true endpoints.
    e_f, e_b = true_fwd.copy(), true_bwd.copy()
    # Sticky tracks: forward correct, backward stuck where the limb is at keyframe d.
    e_b[sticky] = x[sticky]
    tr = good | sticky
    vel[limb_rows[tr]] = (e_f[tr] - e_b[tr]) / (2 * d)
    acc[limb_rows[tr]] = (e_f[tr] - 2 * x[tr] + e_b[tr]) / (d * d)
    has_acc[limb_rows[tr]] = True
    flags[limb_rows[tr]] = rm.TRACK_FWD | rm.TRACK_BWD
    # Untracked limb rows keep a slow inherited velocity and no acceleration.
    vel[limb_rows[~tr]] = 0.002
    # Background: tracked, not moving.
    flags[wall_rows] = rm.TRACK_FWD | rm.TRACK_BWD
    has_acc[wall_rows] = True
    return dict(pos=pos, col=col, sf=sf, vel=vel, acc=acc, has_acc=has_acc, flags=flags,
                limb_rows=limb_rows, wall_rows=wall_rows, true_fwd=true_fwd, true_bwd=true_bwd,
                good=good, sticky=sticky, d=d)


def _run(s, **kw):
    return rm.rigid_cluster_motion(
        positions=s["pos"], colors=s["col"], velocities=s["vel"], accelerations=s["acc"],
        has_acceleration=s["has_acc"], track_flags=s["flags"], source_frames=s["sf"],
        is_static=np.zeros(len(s["pos"]), bool), frame_delta=s["d"],
        params=rm.RigidMotionParams(**kw),
    )


def _endpoints(s, vel, acc, rows):
    d = s["d"]; x = s["pos"][rows]
    return x + d * vel[rows] + 0.5 * acc[rows] * d * d, x - d * vel[rows] + 0.5 * acc[rows] * d * d


def test_kabsch_batch_matches_single():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 6, 3))
    R = _rot([1, 2, 3], 40.0); t = np.array([0.1, -0.2, 0.3])
    b = a @ R.T + t
    rb, tb = rm.kabsch_batch(a, b)
    for i in range(5):
        r1, t1 = rm.kabsch(a[i], b[i])
        np.testing.assert_allclose(rb[i], r1, atol=1e-8)
        np.testing.assert_allclose(tb[i], t1, atol=1e-8)
        np.testing.assert_allclose(r1, R, atol=1e-8)


def test_sticky_and_untracked_limb_rows_recover_the_rigid_motion():
    s = _scene()
    vel, acc, has_acc, flags, changed, stats = _run(s)
    spacing = rm.median_spacing(s["pos"][s["sf"] == s["d"]])
    rows = s["limb_rows"]
    e_f, e_b = _endpoints(s, vel, acc, rows)
    err_b = np.linalg.norm(e_b - s["true_bwd"], axis=1) / spacing
    err_f = np.linalg.norm(e_f - s["true_fwd"], axis=1) / spacing
    # Before: sticky rows were ~10+ spacings off going back, untracked rows had no motion.
    e_f0, e_b0 = _endpoints(s, s["vel"], s["acc"], rows)
    before_b = np.linalg.norm(e_b0 - s["true_bwd"], axis=1) / spacing
    assert np.median(before_b[s["sticky"]]) > 3                     # near the pivot points barely move
    assert np.median(err_b) < 1.0 and np.percentile(err_b, 90) < 2.0
    assert np.median(err_f) < 1.0 and np.percentile(err_f, 90) < 2.0
    assert np.mean(has_acc[rows]) > 0.97      # a few tip rows may sit in clusters too small to fit
    assert stats.fits["bwd"] > 0 and stats.fits["fwd"] > 0
    wrong = s["sticky"] & (before_b > 3.0)          # stuck tracks clearly beyond the 2-spacing tolerance
    assert np.mean((flags[rows[wrong]] & rm.RIGID_BWD) > 0) > 0.95


def test_correct_tracks_and_static_background_are_left_alone():
    s = _scene()
    vel, acc, has_acc, flags, changed, stats = _run(s)
    good_rows = s["limb_rows"][s["good"]]
    np.testing.assert_allclose(vel[good_rows], s["vel"][good_rows], atol=1e-6)
    np.testing.assert_allclose(acc[good_rows], s["acc"][good_rows], atol=1e-6)
    assert not changed[s["wall_rows"]].any()
    np.testing.assert_array_equal(vel[s["wall_rows"]], s["vel"][s["wall_rows"]])


def test_still_surface_sharing_clusters_with_a_moving_limb_is_left_alone():
    # A still panel 2 cm behind the limb (inside the same voxel clusters), tracked as
    # still at every keyframe. The limb's fit must not carry the panel along: a panel
    # point moved by the limb's motion lands in empty space at the neighbour keyframe.
    s = _scene()
    d = s["d"]
    xs, ys = np.meshgrid(np.arange(0.0, 0.3, 0.011), np.arange(-0.06, 0.06, 0.011))
    panel = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 1.0 + 0.06)]
    pos, col, sf = [s["pos"]], [s["col"]], [s["sf"]]
    for k in (0, d, 2 * d):
        pos.append(panel); col.append(np.tile([0.2, 0.5, 0.2], (len(panel), 1))); sf.append(np.full(len(panel), k))
    n0 = len(s["pos"])
    s["pos"], s["col"], s["sf"] = np.concatenate(pos), np.concatenate(col), np.concatenate(sf)
    extra = len(s["pos"]) - n0
    s["vel"] = np.r_[s["vel"], np.zeros((extra, 3))]
    s["acc"] = np.r_[s["acc"], np.zeros((extra, 3))]
    s["has_acc"] = np.r_[s["has_acc"], np.ones(extra, bool)]
    s["flags"] = np.r_[s["flags"], np.full(extra, rm.TRACK_FWD | rm.TRACK_BWD, np.uint8)]
    vel, acc, has_acc, flags, changed, stats = _run(s)
    panel_rows = np.arange(n0, len(s["pos"]))
    spacing_mid = rm.median_spacing(s["pos"][s["sf"] == d])
    moved = np.linalg.norm(vel[panel_rows], axis=1) * d / spacing_mid
    # Near the pivot "still" and "moving with the limb" are within tolerance of each
    # other; anything the pass touches there may move at most ~the inlier tolerance.
    assert changed[panel_rows].mean() < 0.02
    assert moved.max() < 1.5
    rows = s["limb_rows"]
    e_f, e_b = _endpoints(s, vel, acc, rows)
    spacing = rm.median_spacing(s["pos"][s["sf"] == d])
    assert np.median(np.linalg.norm(e_b - s["true_bwd"], axis=1)) / spacing < 1.0


def test_static_consensus_never_overwrites_motion():
    # Most tracks stuck: the best consensus is "not moving". That fit must be skipped,
    # not used to replace the minority of correct tracks.
    s = _scene(sticky_frac=0.85, untracked_frac=0.0)
    vel, acc, has_acc, flags, changed, stats = _run(s, min_inlier_frac=0.3)
    good_rows = s["limb_rows"][s["good"]]
    np.testing.assert_allclose(vel[good_rows], s["vel"][good_rows], atol=1e-6)


def test_disabled_when_no_neighbour_keyframes():
    s = _scene()
    only_mid = s["sf"] == s["d"]
    out = rm.rigid_cluster_motion(
        positions=s["pos"][only_mid], colors=s["col"][only_mid], velocities=s["vel"][only_mid],
        accelerations=s["acc"][only_mid], has_acceleration=s["has_acc"][only_mid],
        track_flags=s["flags"][only_mid], source_frames=s["sf"][only_mid],
        is_static=np.zeros(int(only_mid.sum()), bool), frame_delta=s["d"],
    )
    assert not out[4].any()


def test_track_flags_for_rows():
    tracked = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    rows = np.array([[1, 0, 0], [5, 5, 5], [2, 0, 0], [0, 0, 0]], np.float32)
    flags = rm.track_flags_for_rows(rows, tracked, np.array([True, False, True]))
    assert flags.tolist() == [rm.TRACK_FWD, 0, rm.TRACK_FWD | rm.TRACK_BWD, rm.TRACK_FWD | rm.TRACK_BWD]


def test_memfof_flow_sampling_rescales_to_full_resolution():
    rb = pytest.importorskip("roma_backends")
    h, w, scale = 54, 96, 0.5                     # reduced grid for a 108x192 image
    flow = np.zeros((2, h, w), np.float32)
    flow[0] = 3.0                                 # +3 px in reduced pixels = +6 full-res px
    flow[1] = -1.5                                # -3 full-res px
    uvs = np.array([[10.0, 20.0], [150.5, 90.25]], np.float32)
    disp = rb.sample_flow_field(flow, scale, uvs)
    np.testing.assert_allclose(disp, [[6.0, -3.0], [6.0, -3.0]], atol=1e-5)
    backend = object.__new__(rb.MemfofTemporalBackend)
    dense = rb.MemfofTemporalBackend.make_dense(flow.astype(np.float16), scale, 108, 192)
    out, cert = backend.sample_at_uvs(dense, uvs)
    np.testing.assert_allclose(out, uvs + [6.0, -3.0], atol=1e-3)
    assert (cert == 1).all()
    with pytest.raises(RuntimeError):
        backend.sample_at_uvs(dense, uvs, reverse=True)
