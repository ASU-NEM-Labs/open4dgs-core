import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


SYNC_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SYNC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SyncMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visual = load_script("visual_sync_test", "visual-epipolar-sync.py")
        cls.robust = load_script("robust_sync_test", "robust-sync-graph.py")
        cls.calculate = load_script("calculate_sync_test", "calculate-sync.py")

    def test_circular_graph_crosses_frame_boundary(self):
        period = 1.0 / 60.0
        truth = {"a": 0.49 * period, "b": -0.48 * period,
                 "c": 0.10 * period, "d": -0.05 * period}
        edges = []
        for a, b in [("a", "b"), ("a", "c"), ("a", "d"),
                     ("b", "c"), ("b", "d"), ("c", "d")]:
            delta = float(self.robust.wrap_period(truth[a] - truth[b], period))
            edges.append((a, b, delta, 1e8))
        solved = self.robust.solve_component(edges, list(truth), period)
        error = np.asarray([
            float(self.robust.wrap_period(solved[k] - truth[k], period)) for k in truth
        ])
        error = self.robust.wrap_period(error - np.median(error), period)
        self.assertLess(float(np.max(np.abs(error))), 1e-9)

    def test_contradictory_sparse_pair_is_rejected(self):
        rows = [
            {"cam_a": 5, "cam_b": 12, "folder_a": "06", "folder_b": "13",
             "frame": frame, "delta_b_frames": delta, "n_used": 100,
             "mad_frames": 0.10, "phase_se_frames": 0.05}
            for frame, delta in enumerate([-0.468, 0.238, 0.304], 1)
        ]
        result = self.visual.aggregate_pair_rows(
            rows, 60.0, 40, min_frames=3, max_mad_frames=0.15,
            phase_period_frames=1.0)
        self.assertEqual(result, [])

    def test_cached_observations_restore_numeric_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "observations.csv"
            path.write_text(
                "cam_a,cam_b,frame,delta_b_frames,n_used,direct_phase_boundary,error\n"
                "1,2,17,0.125,80,False,\n"
                "2,3,18,,0,True,few_spatial_cells:1\n",
                encoding="utf-8")
            rows = self.visual.read_cached_observations(path)
        self.assertEqual(rows[0]["cam_a"], 1)
        self.assertEqual(rows[0]["frame"], 17)
        self.assertAlmostEqual(rows[0]["delta_b_frames"], 0.125)
        self.assertFalse(rows[0]["direct_phase_boundary"])
        self.assertNotIn("delta_b_frames", rows[1])
        self.assertTrue(rows[1]["direct_phase_boundary"])

    def test_motion_scan_round_trip(self):
        samples = [self.calculate.MotionSample(9, "0001", 1.25, 0.2, 4, 1.1)]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "scan.csv"
            self.calculate.write_scan_csv(path, samples)
            loaded = self.calculate.read_scan_csv(path)
        self.assertEqual(loaded[0].frame, 9)
        self.assertEqual(loaded[0].camera, "0001")
        self.assertAlmostEqual(loaded[0].speed_px_per_frame, 1.25)

    def test_zero_sync_contains_every_camera(self):
        cameras = [{"folder": "01"}, {"folder": "2"}, {"folder": "0044"}]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sync.json"
            result = self.calculate.write_zero_sync(path, cameras)
            written = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result, {"0001": 0.0, "0002": 0.0, "0044": 0.0})
        self.assertEqual(written, result)

    def test_symmetric_search_recovers_known_phase(self):
        visual = self.visual
        rng = np.random.default_rng(4)
        n, focal, cx, cy = 180, 900.0, 500.0, 400.0
        depth = rng.uniform(4, 7, n)
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-0.8, 0.8, n)
        vx = rng.uniform(-0.01, 0.01, n)
        vy = rng.uniform(0.012, 0.026, n)
        params = np.array([focal, focal, cx, cy])
        identity = np.eye(3)
        cam_a = visual.Camera("a", "01", identity, np.zeros(3),
                              "PINHOLE", 1000, 800, params)
        cam_b = visual.Camera("b", "02", identity, np.array([-1.0, 0, 0]),
                              "PINHOLE", 1000, 800, params)
        true_delta = 0.38

        def project(camera, time):
            px = x + vx * time + camera.t[0]
            py = y + vy * time
            return np.stack([focal * px / depth + cx,
                             focal * py / depth + cy], axis=1).astype(np.float32)

        uv_a = project(cam_a, +0.5 * true_delta)
        uv_b = project(cam_b, -0.5 * true_delta)
        certainty = np.ones(n, np.float32)
        old_load, old_matches = visual.load_rgb, visual.run_roma_matches
        old_motion = visual.sparse_multiframe_motion
        try:
            visual.load_rgb = lambda path: np.zeros((800, 1000, 3), np.uint8)
            visual.run_roma_matches = lambda *args, **kwargs: (
                uv_a.copy(), uv_b.copy(), certainty.copy())

            def fake_motion(*args, **kwargs):
                velocity = np.stack([focal * vx / depth, focal * vy / depth], axis=1).astype(np.float32)
                return (velocity, np.zeros_like(velocity), np.ones(n, bool),
                        np.zeros(n, np.float32), np.full(n, 2, np.int16))

            visual.sparse_multiframe_motion = fake_motion
            result = visual.estimate_pair_frame(
                None, [cam_a, cam_b], pathlib.Path("."), "{frame:06d}.jpg",
                10, (0, 1), "cpu", 1000, 0.1, 0.1, 0.5, 0.0, 0.1, 1.0,
                [1, 2, 4, 8], 3.0, 40.0, 101, 1.0, 6, 32,
                "linear", 0.15, "off",
                None, None, None)
            direct_result = visual.estimate_pair_frame(
                None, [cam_a, cam_b], pathlib.Path("."), "{frame:06d}.jpg",
                10, (0, 1), "cpu", 1000, 0.1, 0.1, 0.5, 0.0, 0.1, 1.0,
                [1, 2, 4, 8], 3.0, 40.0, 101, 1.0, 6, 32,
                "direct", 0.15, "off",
                None, None, None)
            uv_a = project(cam_a, +0.40)
            uv_b = project(cam_b, -0.40)
            censored_result = visual.estimate_pair_frame(
                None, [cam_a, cam_b], pathlib.Path("."), "{frame:06d}.jpg",
                10, (0, 1), "cpu", 1000, 0.1, 0.1, 0.5, 0.0, 0.1, 1.0,
                [1, 2, 4, 8], 3.0, 40.0, 101, 1.0, 6, 32,
                "direct", 0.15, "off",
                None, None, None)
        finally:
            visual.load_rgb = old_load
            visual.run_roma_matches = old_matches
            visual.sparse_multiframe_motion = old_motion
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["delta_b_frames"], true_delta, places=2)
        self.assertAlmostEqual(result["delta_b_frames_raw_median"], true_delta, places=2)
        self.assertEqual(result["phase_estimator"], "linear")
        self.assertNotIn("error", direct_result)
        self.assertAlmostEqual(direct_result["delta_b_frames"], true_delta, places=2)
        self.assertEqual(direct_result["phase_estimator"], "direct")
        self.assertEqual(censored_result.get("error"), "phase_search_boundary")


if __name__ == "__main__":
    unittest.main()
