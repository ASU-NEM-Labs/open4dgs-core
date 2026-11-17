import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _arrays(n=4):
    values = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    return {
        "positions": values.copy(),
        "colors": (values / max(n * 3, 1)).copy(),
        "velocities": (values * 0.01).copy(),
        "times": np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(n, 1),
        "durations": np.full((n, 1), 0.1, dtype=np.float32),
        "has_velocity": np.ones(n, dtype=bool),
        "source_frames": np.full((n, 1), 7, dtype=np.int32),
        "is_static": np.zeros(n, dtype=bool),
        "dynamic_protect": np.zeros(n, dtype=bool),
        "dynamic_allowed": np.ones(n, dtype=bool),
        "stage_keep": np.ones(n, dtype=bool),
        "accelerations": (values * 0.001).copy(),
        "has_acceleration": np.ones(n, dtype=bool),
        "track_flags": np.full(n, 3, dtype=np.uint8),
    }


def test_checkpoint_round_trip_preserves_values_dtypes_and_stats(tmp_path):
    path = tmp_path / "frame.npz"
    arrays = _arrays()
    stats = {"per_frame": [{"frame_idx": 7}], "static": {"static": 2}}
    MODULE.write_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="run",
        input_fingerprint="input",
        velocity_units="world_per_video_frame",
        arrays=arrays,
        stats_delta=stats,
    )

    loaded = MODULE.load_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="run",
        input_fingerprint="input",
    )

    assert loaded is not None
    assert loaded["stats_delta"] == stats
    assert loaded["velocity_units"] == "world_per_video_frame"
    for name, expected in arrays.items():
        actual = loaded["arrays"][name]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_checkpoint_rejects_stale_and_corrupt_files(tmp_path):
    path = tmp_path / "frame.npz"
    MODULE.write_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="run",
        input_fingerprint="input",
        velocity_units="world_per_video_frame",
        arrays=_arrays(),
        stats_delta={},
    )
    assert MODULE.load_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="other",
        input_fingerprint="input",
    ) is None
    assert MODULE.load_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="run",
        input_fingerprint="other",
    ) is None

    path.write_bytes(b"not an npz")
    assert MODULE.load_frame_checkpoint(
        path,
        frame_idx=7,
        run_fingerprint="run",
        input_fingerprint="input",
    ) is None


def test_atomic_checkpoint_failure_does_not_replace_existing_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "frame.npz"
    path.write_bytes(b"old")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(np, "savez", fail)
    try:
        MODULE._atomic_savez(path, value=np.asarray([1]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected simulated failure")

    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cache_cleanup_failure_is_nonfatal(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    active = cache_root / "run"
    active.mkdir(parents=True)

    def fail(_path):
        raise OSError("simulated NFS cleanup race")

    monkeypatch.setattr(MODULE.shutil, "rmtree", fail)
    assert MODULE._cleanup_completed_cache(active, cache_root) is False
    assert active.exists()


def test_finish_completed_cache_retains_cache_when_requested(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    active = cache_root / "run"
    active.mkdir(parents=True)
    checkpoint = active / "frame-0000000007.npz"
    checkpoint.write_bytes(b"checkpoint")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cleanup must not run with keep_cache=True")

    monkeypatch.setattr(MODULE, "_cleanup_completed_cache", fail_if_called)

    assert MODULE._finish_completed_cache(active, cache_root, keep_cache=True)
    assert checkpoint.read_bytes() == b"checkpoint"


def test_disk_static_merge_matches_voxels_and_preserves_dynamic_rows(tmp_path):
    arrays = _arrays(6)
    arrays["positions"][:] = np.asarray(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.01, 0.01],
            [0.21, 0.01, 0.01],
            [0.22, 0.01, 0.01],
            [1.00, 1.00, 1.00],
            [2.00, 2.00, 2.00],
        ],
        dtype=np.float32,
    )
    arrays["is_static"][:4] = True
    arrays["has_velocity"][:4] = False
    raw = MODULE._create_memmap_arrays(tmp_path, "raw-test", 6)
    for name in arrays:
        raw[name][:] = arrays[name]
    MODULE._flush_memmaps(raw)

    merged = MODULE._disk_backed_static_merge(
        raw,
        work_dir=tmp_path,
        radius=0.1,
        max_points=0,
        seed=42,
        source_frame_start=0,
        block_rows=2,
    )

    expected_pos, expected_col = MODULE._voxel_merge_static_rows(
        arrays["positions"][:4], arrays["colors"][:4], radius=0.1
    )
    np.testing.assert_allclose(merged["positions"][:2], expected_pos)
    np.testing.assert_allclose(merged["colors"][:2], expected_col)
    np.testing.assert_array_equal(merged["positions"][2:], arrays["positions"][4:])
    np.testing.assert_array_equal(merged["velocities"][2:], arrays["velocities"][4:])
    assert merged["is_static"][:2].all()
    assert not merged["is_static"][2:].any()
