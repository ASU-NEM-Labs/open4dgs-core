import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chain_targets as ct  # noqa: E402

CLI = Path(__file__).resolve().parents[1] / "chain-targets-npz.py"


def _scene(step=3, n_kf=15, a=(0.0, -0.002, 0.0), v0=(0.03, 0.01, 0.0), seed=0):
    """A dense 2-D sheet of points translating with constant acceleration (per frame),
    re-sampled at random positions on every keyframe, plus a still floor."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); v0 = np.asarray(v0)
    pos, vel, acc, frames, static = [], [], [], [], []
    for k in range(n_kf):
        f = 1 + k * step
        t = f - 1
        shift = v0 * t + 0.5 * a * t * t
        p = np.c_[rng.random((4000, 2)), np.zeros(4000)] + shift
        pos.append(p); vel.append(np.tile(v0 + a * t, (4000, 1))); acc.append(np.tile(a, (4000, 1)))
        frames.append(np.full(4000, f)); static.append(np.zeros(4000, bool))
        q = np.c_[rng.random((500, 2)), np.full(500, -1.0)]
        pos.append(q); vel.append(np.zeros((500, 3))); acc.append(np.zeros((500, 3)))
        frames.append(np.full(500, f)); static.append(np.ones(500, bool))
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    return cat(pos), cat(vel), cat(acc), cat(frames), cat(static)


def test_chains_follow_exact_motion_and_stop_at_a_broken_keyframe():
    pos, vel, acc, frames, static = _scene()
    mid = 1 + 7 * 3
    query = np.nonzero((frames == mid) & ~static)[0][:200]
    params = ct.ChainParams(window_frames=12, tau_spacings=3.0)
    offsets, chains = ct.build_chains(pos, vel, acc, frames, ~static, query, 3, params)
    assert list(offsets) == [-12, -9, -6, -3, 3, 6, 9, 12]
    # Random re-sampling leaves the odd gap wider than tau, so a few chains stop early.
    assert np.mean(~np.isnan(chains[..., 0])) > 0.95
    for i, o in enumerate(offsets):
        expect = pos[query] + vel[query] * o + 0.5 * acc[query] * o * o
        # Linking snaps to the nearest re-sampled point, so error is about one spacing.
        assert np.nanmedian(np.linalg.norm(chains[:, i] - expect, axis=1)) < 0.03

    # Teleport the sheet at frame mid+9: chains forward stop there, backward are untouched.
    broken = (frames == mid + 9) & ~static
    pos2 = pos.copy(); pos2[broken] += 5.0
    _, chains2 = ct.build_chains(pos2, vel, acc, frames, ~static, query, 3, params)
    valid = ~np.isnan(chains2[..., 0])
    fwd, bwd = ct.contiguous_span(valid, offsets)
    assert fwd.max() == 6 and np.mean(fwd == 6) > 0.95
    assert np.mean(bwd == 12) > 0.95


def test_cycle_check_rejects_links_whose_motion_disagrees():
    pos, vel, acc, frames, static = _scene()
    mid = 1 + 7 * 3
    query = np.nonzero((frames == mid) & ~static)[0][:200]
    # The next keyframe's rows claim the opposite velocity: forward prediction still lands,
    # but their backward prediction does not return to the source.
    nxt = (frames == mid + 3) & ~static
    vel2 = vel.copy(); vel2[nxt] *= -1.0
    on = ct.ChainParams(window_frames=6, tau_spacings=3.0, cycle_check=True)
    off = ct.ChainParams(window_frames=6, tau_spacings=3.0, cycle_check=False)
    offsets, with_check = ct.build_chains(pos, vel2, acc, frames, ~static, query, 3, on)
    _, without = ct.build_chains(pos, vel2, acc, frames, ~static, query, 3, off)
    col = list(offsets).index(3)
    assert np.isnan(with_check[:, col, 0]).all()
    assert not np.isnan(without[:, col, 0]).any()


def test_cli_annotates_a_row_subset_through_the_dense_file(tmp_path):
    pos, vel, acc, frames, static = _scene()
    n = len(pos)
    dense = dict(positions=pos.astype(np.float32), velocities=vel.astype(np.float32),
                 accelerations=acc.astype(np.float32), has_acceleration=np.ones(n, bool),
                 source_frames=frames.reshape(-1, 1).astype(np.int32), is_static=static,
                 colors=np.zeros((n, 3), np.float32), keyframe_step=np.asarray(3, np.int32),
                 velocity_units=np.asarray("world_per_video_frame"))
    np.savez_compressed(tmp_path / "dense.npz", **dense)
    keep = np.sort(np.random.default_rng(1).choice(n, n // 10, replace=False))
    np.savez_compressed(tmp_path / "small.npz", **{k: (v[keep] if v.ndim and len(v) == n else v) for k, v in dense.items()})
    r = subprocess.run([sys.executable, str(CLI), str(tmp_path / "dense.npz"), str(tmp_path / "small.npz"),
                        str(tmp_path / "out.npz"), "--window-frames", "9"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "out.npz")
    assert z["chain_pos"].shape == (len(keep), 6, 3)
    assert list(z["chain_offsets_frames"]) == [-9, -6, -3, 3, 6, 9]
    st = z["is_static"]
    assert np.isnan(z["chain_pos"][st]).all()
    # Interior keyframes of the moving sheet chain both ways; the result is aligned to the small file's rows.
    sf = z["source_frames"].reshape(-1)
    inner = ~st & (sf >= 10) & (sf <= 34)
    assert np.mean(~np.isnan(z["chain_pos"][inner][..., 0])) > 0.95
    o = 3
    expect = z["positions"][inner] + z["velocities"][inner] * o + 0.5 * z["accelerations"][inner] * o * o
    got = z["chain_pos"][inner][:, list(z["chain_offsets_frames"]).index(o)]
    assert np.nanmedian(np.linalg.norm(got - expect, axis=1)) < 0.03
