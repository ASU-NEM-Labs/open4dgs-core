import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT = Path(__file__).resolve().parents[1] / "downsize-npz.py"
_spec = importlib.util.spec_from_file_location("downsize_npz", SCRIPT)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def _npz(path, n_static=700, seed=1):
    rng = np.random.default_rng(seed)
    kf = np.repeat(np.arange(1, 40, 3), rng.integers(3000, 9000, 13))       # 13 keyframes, uneven sizes
    n_dyn = len(kf); n = n_dyn + n_static
    sf = np.concatenate([kf, np.zeros(n_static, int)]).reshape(-1, 1).astype(np.int32)
    pos = np.concatenate([rng.random((n_dyn, 3)), rng.random((n_static, 3)) + 5]).astype(np.float32)
    idx = np.arange(n, dtype=np.float32)
    d = dict(positions=pos, colors=np.stack([idx] * 3, 1), velocities=np.stack([idx] * 3, 1),
             times=((sf - 1) / 60.0).astype(np.float32), is_static=np.arange(n) >= n_dyn,
             track_flags=(np.arange(n) % 7).astype(np.uint8), source_frames=sf,
             keyframe_frames=np.arange(1, 40, 3).astype(np.int32), fps=np.float32(60),
             velocity_units=np.array("world_per_video_frame"))
    d["times"][n_dyn:] = 0
    np.savez_compressed(path, **d)
    return d


def _run(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_rows_stay_aligned_and_metadata_is_untouched(tmp_path):
    d = _npz(tmp_path / "in.npz")
    n = len(d["positions"])
    _run(tmp_path / "in.npz", tmp_path / "out.npz", "--target-count", 9000, "--static-budget", 1000)
    o = np.load(tmp_path / "out.npz")
    keep = o["colors"][:, 0].astype(int)
    assert len(keep) == 9000 and np.all(np.diff(keep) > 0)
    for k, v in d.items():
        if v.ndim >= 1 and v.shape[0] == n:
            assert np.array_equal(o[k], v[keep]), k
        else:
            assert np.array_equal(o[k], v), k
    assert int(o["is_static"].sum()) == 700                      # under the static budget: all kept
    assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zipfile.ZipFile(tmp_path / "out.npz").infolist())


def test_keyframe_quotas_are_proportional_and_exact():
    counts = np.array([3000, 8000, 4500, 7], np.int64)
    q = ds.keyframe_quotas(counts, 5000)
    assert q.sum() == 5000 and np.all(q >= 1) and np.all(q <= counts)
    # The trainer's rule: floors, then the whole remainder (here 2) to the largest fraction.
    assert list(q) == [967, 2579, 1452, 2]


def test_static_rows_over_budget_are_capped(tmp_path):
    _npz(tmp_path / "in.npz", n_static=3000)
    _run(tmp_path / "in.npz", tmp_path / "out.npz", "--target-count", 9000, "--static-budget", 1000)
    o = np.load(tmp_path / "out.npz")
    assert int(o["is_static"].sum()) == 1000 and len(o["positions"]) == 9000


def test_keyframe_stride_keeps_only_grid_keyframes(tmp_path):
    _npz(tmp_path / "in.npz")
    _run(tmp_path / "in.npz", tmp_path / "out.npz", "--target-count", 9000, "--static-budget", 1000,
         "--keyframe-stride", 9)
    o = np.load(tmp_path / "out.npz")
    sf = o["source_frames"].reshape(-1)
    assert sorted(set(sf[~o["is_static"]].tolist())) == [1, 10, 19, 28, 37]
    assert int(o["is_static"].sum()) == 700 and len(sf) == 9000


def test_even_selection_spaces_points_more_evenly(tmp_path):
    _npz(tmp_path / "in.npz")
    cv = {}
    for even in (False, True):
        out = tmp_path / f"out{even}.npz"
        _run(tmp_path / "in.npz", out, "--target-count", 9000, "--static-budget", 1000, *(["--even"] if even else []))
        o = np.load(out)
        p = o["positions"][(o["source_frames"].reshape(-1) == 1) & ~o["is_static"]]
        nn = cKDTree(p).query(p, k=2)[0][:, 1]
        cv[even] = nn.std() / nn.mean()
    assert cv[True] < cv[False]


def _motion_npz(path, seed=2):
    """Keyframes with a still sheet (most rows) and a moving patch; velocities per video frame."""
    rng = np.random.default_rng(seed)
    pos, vel, sf, hv = [], [], [], []
    for f in range(1, 40, 3):
        still = np.c_[rng.random((6000, 2)), np.zeros(6000)]
        mover = np.c_[rng.random((1500, 2)) * 0.3 + 2.0, np.zeros(1500)]
        pos += [still, mover]
        vel += [np.zeros((6000, 3)), np.tile([0.05, 0.0, 0.0], (1500, 1))]
        sf += [np.full(7500, f)]
        hv += [np.ones(7500, bool)]
    pos = np.concatenate(pos).astype(np.float32); n = len(pos)
    d = dict(positions=pos, velocities=np.concatenate(vel).astype(np.float32),
             colors=np.stack([np.arange(n, dtype=np.float32)] * 3, 1), has_velocity=np.concatenate(hv),
             source_frames=np.concatenate(sf).reshape(-1, 1).astype(np.int32), is_static=np.zeros(n, bool),
             times=((np.concatenate(sf) - 1) / 60.0).reshape(-1, 1).astype(np.float32),
             keyframe_step=np.asarray(3, np.int32), velocity_units=np.array("world_per_video_frame"))
    np.savez_compressed(path, **d)
    return d


def test_slow_keep_shifts_the_budget_to_moving_rows(tmp_path):
    d = _motion_npz(tmp_path / "in.npz")
    moving_in = d["positions"][:, 0] > 1.5
    _run(tmp_path / "in.npz", tmp_path / "off.npz", "--target-count", 20000, "--static-budget", 0)
    _run(tmp_path / "in.npz", tmp_path / "on.npz", "--target-count", 20000, "--static-budget", 0, "--slow-keep", 0.25)
    off, on = np.load(tmp_path / "off.npz"), np.load(tmp_path / "on.npz")
    assert len(off["positions"]) == len(on["positions"]) == 20000
    frac = lambda z: float(np.mean(z["positions"][:, 0] > 1.5))  # noqa: E731
    assert abs(frac(off) - moving_in.mean()) < 0.02                 # proportional: 20% movers
    # weights 1 : 0.25 over 1500 : 6000 rows -> movers get 1500 / (1500 + 1500) = half
    assert abs(frac(on) - 0.5) < 0.02
    # still rows remain in every keyframe
    sf = on["source_frames"].reshape(-1)
    for f in np.unique(sf):
        assert np.sum((sf == f) & (on["positions"][:, 0] < 1.5)) > 100


def test_slow_keep_one_is_unchanged(tmp_path):
    _motion_npz(tmp_path / "in.npz")
    _run(tmp_path / "in.npz", tmp_path / "a.npz", "--target-count", 20000, "--static-budget", 0)
    _run(tmp_path / "in.npz", tmp_path / "b.npz", "--target-count", 20000, "--static-budget", 0, "--slow-keep", 1.0)
    assert np.array_equal(np.load(tmp_path / "a.npz")["colors"], np.load(tmp_path / "b.npz")["colors"])
