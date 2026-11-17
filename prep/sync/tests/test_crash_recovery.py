"""Regression coverage for invalid native inputs and interrupted measurements."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch

import numpy as np
import pytest

from test_sync_math import load_script


@pytest.fixture
def visual():
    return load_script("visual_crash_test", "visual-epipolar-sync.py")


@pytest.mark.parametrize("failed_point,status", [
    ([np.nan, 40], 1), ([np.inf, 40], 1), ([1e30, 40], 1), ([41, 40], 0),
])
def test_lk_drops_failed_points_before_backward_and_next_step(visual, failed_point, status):
    camera = visual.Camera("a", "a", np.eye(3), np.zeros(3),
                           "PINHOLE", 100, 100, np.array([50, 50, 50, 50]))
    sizes = []

    def load_gray(path):
        return np.full((100, 100), int(path.stem), dtype=np.uint8)

    def lk(prev, nxt, points, unused, **kwargs):
        assert np.all(np.isfinite(points))
        assert np.all((points >= 0) & (points < 100))
        sizes.append(len(points))
        result = points.copy()
        result[:, 0, 0] += int(nxt[0, 0]) - int(prev[0, 0])
        valid = np.ones((len(points), 1), np.uint8)
        if len(points) == 3:
            result[1, 0] = failed_point
            valid[1] = status
        elif len(points) == 2:
            # This point passes forward tracking but fails the backward check.
            valid[1] = 0
        return result, valid, None

    with patch.object(visual, "load_gray", side_effect=load_gray), \
            patch.object(visual.cv2, "calcOpticalFlowPyrLK", side_effect=lk):
        velocity, acceleration, valid, _, baseline = visual.sparse_multiframe_motion(
            Path("."), camera, 10, "{frame}.jpg",
            np.array([[20, 20], [40, 40], [60, 60]], np.float32), [2], 1, 40)
    assert sizes == [3, 2, 1, 1] * 2
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(velocity[0], [1, 0])
    np.testing.assert_allclose(acceleration[0], [0, 0])
    np.testing.assert_array_equal(baseline, [2, 0, 0])


def test_lk_stops_when_all_tracks_fail(visual):
    camera = visual.Camera("a", "a", np.eye(3), np.zeros(3),
                           "PINHOLE", 100, 100, np.array([50, 50, 50, 50]))
    with patch.object(visual, "load_gray", return_value=np.zeros((100, 100), np.uint8)), \
            patch.object(visual.cv2, "calcOpticalFlowPyrLK", return_value=(
                np.full((1, 1, 2), np.nan, np.float32), np.zeros((1, 1), np.uint8), None)) as lk:
        _, _, valid, _, _ = visual.sparse_multiframe_motion(
            Path("."), camera, 10, "{frame}.jpg", np.array([[20, 20]], np.float32), [2], 1, 40)
    assert lk.call_count == 2  # One forward attempt in each direction; no backward calls.
    assert not valid.any()


def test_failed_checkpoint_replacement_preserves_previous_csv(visual, tmp_path):
    path = tmp_path / "cache.csv"
    old = [{"frame": 10, "cam_a": 0, "cam_b": 1, "error": "few_matches"}]
    visual.write_observations_atomic(path, old)
    with patch.object(visual.os, "replace", side_effect=OSError("interrupted replacement")):
        with pytest.raises(OSError):
            visual.write_observations_atomic(path, old + [dict(old[0], frame=11)])
    assert visual.read_cached_observations(path) == old
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("separate_cache", [False, True])
def test_abrupt_exit_preserves_completed_pairs_and_resume_skips_them(visual, tmp_path, separate_cache):
    # Exit without unwinding Python frames, as a native crash would. No GPU/model required.
    script = textwrap.dedent("""\
        import importlib.util, os, sys
        from pathlib import Path
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location('visual', sys.argv[1])
        visual = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = visual
        spec.loader.exec_module(visual)
        visual.load_reference_export = lambda _: ([SimpleNamespace(folder=str(i)) for i in range(3)], None)
        visual.parse_pairs = lambda *args: [(0, 1), (0, 2)]
        visual.load_roma_backend_factory = lambda: lambda **kwargs: SimpleNamespace(name='fake')
        def estimate(*args, **kwargs):
            frame, pair = args[4], args[5]
            if os.environ['TEST_RESUME'] == 'fresh_crash':
                os._exit(92)
            if os.environ['TEST_RESUME'] == '0' and pair == (0, 2):
                os._exit(91)
            if os.environ['TEST_RESUME'] == '1' and pair == (0, 1):
                raise AssertionError('completed pair was recomputed')
            return dict(frame=frame, cam_a=pair[0], cam_b=pair[1], error='few_matches')
        visual.estimate_pair_frame = estimate
        visual.aggregate_pair_rows = lambda *args, **kwargs: sys.exit(0)
        sys.argv = ['visual', *sys.argv[2:]]
        visual.main()
    """)
    active = tmp_path / "pairs.csv"
    cache = tmp_path / "cache.csv" if separate_cache else active
    argv = [sys.executable, "-c", script, visual.__file__,
            "--reference-json", "unused.json", "--images-root", ".",
            "--fps", "60", "--frames", "10", "--device", "cpu", "--pair-csv", str(active)]
    if separate_cache:
        argv += ["--cache-csv", str(cache)]
        # An existing compatible row outside this request must survive resumption.
        visual.write_observations_atomic(cache, [dict(frame=9, cam_a=0, cam_b=1, error="old")])
        argv += ["--resume"]
    failed = subprocess.run(argv, env={**os.environ, "TEST_RESUME": "0"}, capture_output=True, text=True)
    assert failed.returncode == 91, failed.stderr
    rows = visual.read_cached_observations(cache)
    assert {(r["frame"], r["cam_b"]) for r in rows} == ({(9, 1), (10, 1)} if separate_cache else {(10, 1)})
    progress = json.loads(cache.with_suffix(".progress.json").read_text())
    assert (progress["status"], progress["frame"], progress["cam_b"]) == ("measuring", 10, 2)
    resumed = subprocess.run(argv + ["--resume"], env={**os.environ, "TEST_RESUME": "1"},
                             capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert len(visual.read_cached_observations(cache)) == (3 if separate_cache else 2)
    assert len(visual.read_cached_observations(active)) == 2

    # Starting again without --resume must discard that old cache even if the
    # process dies in its very first measurement.
    fresh = subprocess.run([arg for arg in argv if arg != "--resume"],
                           env={**os.environ, "TEST_RESUME": "fresh_crash"},
                           capture_output=True, text=True)
    assert fresh.returncode == 92, fresh.stderr
    assert visual.read_cached_observations(cache) == []
