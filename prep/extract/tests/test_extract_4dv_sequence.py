from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "extract_4dv_sequence.py"
SPEC = importlib.util.spec_from_file_location("extract_4dv_sequence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FourDvImportTests(unittest.TestCase):
    def test_parse_record_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "JCam000_test.log"
            path.write_text(
                "Frame 1 saved at 20250812_011321_793645224 type 1\n"
                "Frame 2 saved at 20250812_011321_809716216 type 5\n",
                encoding="utf-8",
            )
            rows = MODULE.parse_record_log(path)
        self.assertEqual([row.frame_number for row in rows], [1, 2])
        self.assertEqual([row.frame_type for row in rows], [1, 5])
        self.assertEqual(rows[1].timestamp_ns - rows[0].timestamp_ns, 16_070_992)

    def _source(self, camera_index: int, phase_ns: int, count: int = 20):
        root = Path("/capture")
        camera_id = f"JCam{camera_index:03d}"
        start = MODULE.parse_timestamp_ns("20250812_011321_000000000")
        frames = tuple(
            MODULE.LogFrame(i + 1, start + phase_ns + round(i * 1e9 / 60.0), 1)
            for i in range(count)
        )
        return MODULE.CameraSource(
            camera_id=camera_id,
            camera_index=camera_index,
            folder=f"{camera_index:04d}",
            video_path=root / "videos" / f"{camera_id}_x.mp4",
            log_path=root / "record_logs" / f"{camera_id}_x.log",
            snapshot_path=root / "snapshot" / "reference" / "images" / f"{camera_id}_x.png",
            frames=frames,
        )

    def test_timeline_integer_aligns_and_centers_offsets(self):
        sources = (
            self._source(0, 0),
            self._source(1, 5_000_000),
            self._source(2, 21_000_000),
        )
        timeline = MODULE.build_timeline(sources, fps=60.0, num_frames=10)
        starts = [row.source_start_frame for row in timeline.selections]
        offsets = [row.sync_offset_sec for row in timeline.selections]
        self.assertEqual(starts, [2, 2, 1])
        self.assertAlmostEqual(sorted(offsets)[1], 0.0, places=7)
        self.assertLess(max(abs(value) for value in offsets), 1.0 / 120.0)

    def test_timeline_keeps_observed_frames_when_clock_is_slightly_fast(self):
        source = self._source(0, 0, count=100)
        # Make the last timestamp 5 ms earlier than a perfect 60 Hz clock.
        frames = list(source.frames)
        frames[-1] = MODULE.LogFrame(
            frames[-1].frame_number, frames[-1].timestamp_ns - 5_000_000, 1
        )
        source = MODULE.CameraSource(
            source.camera_id, source.camera_index, source.folder,
            source.video_path, source.log_path, source.snapshot_path, tuple(frames),
        )
        timeline = MODULE.build_timeline((source,), fps=60.0, num_frames=100)
        self.assertEqual(timeline.num_frames, 100)

    def test_rotate_calibration_left(self):
        k = [[100.0, 0.0, 199.5], [0.0, 120.0, 99.5], [0.0, 0.0, 1.0]]
        e = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
        new_k, new_e, width, height = MODULE.rotate_calibration(k, e, "l")
        self.assertEqual((width, height), (200, 400))
        self.assertEqual(new_k, [[120.0, 0.0, 99.5], [0.0, 100.0, 199.5], [0.0, 0.0, 1.0]])
        self.assertEqual(new_e, [[0.0, 1.0, 0.0, 2.0], [-1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 3.0]])

    def test_manifest_calibration_writes_colmap_text_model(self):
        source = self._source(0, 0)
        timeline = MODULE.build_timeline((source,), fps=60.0, num_frames=3)
        manifest = {
            "intri": {"session_id": "i", "scale": 1.0, "data": [{
                "camera_id": "JCam000",
                "intrinsic": [[100.0, 0.0, 199.5], [0.0, 100.0, 99.5], [0.0, 0.0, 1.0]],
            }]},
            "extri": {"session_id": "e", "data": [{
                "camera_id": "JCam000",
                "extrinsic": [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            }]},
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            MODULE.write_manifest_calibration(
                manifest, timeline, {"JCam000": "n"}, output, "jpg"
            )
            reference = json.loads((output / "reference_export.json").read_text())
            cameras_text = (output / "sparse" / "0" / "cameras.txt").read_text()
            images_text = (output / "sparse" / "0" / "images.txt").read_text()
        self.assertEqual(reference["cameras"][0]["folder"], "0000")
        self.assertIn("1 PINHOLE 400 200", cameras_text)
        self.assertIn("0000/000001.jpg", images_text)

    def test_sequence_completion_rejects_partial_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "000001.jpg").touch()
            self.assertFalse(MODULE._sequence_is_complete(path, count=2, extension="jpg"))
            (path / "000002.jpg").touch()
            self.assertTrue(MODULE._sequence_is_complete(path, count=2, extension="jpg"))

    def test_media_may_have_unlogged_trailing_frames(self):
        source = self._source(0, 0, count=100)
        probe = MODULE.MediaProbe(3840, 2160, 60.0, 690)
        self.assertEqual(MODULE.unlogged_trailing_frame_count(source, probe), 590)

    def test_media_cannot_be_shorter_than_log(self):
        source = self._source(0, 0, count=100)
        probe = MODULE.MediaProbe(3840, 2160, 60.0, 99)
        with self.assertRaisesRegex(ValueError, "references frame 100"):
            MODULE.unlogged_trailing_frame_count(source, probe)

    # ---- camera ladder ---------------------------------------------------

    def _frames(self, *, count, period_ns, phase_ns=0, idr_every=None, idr_bias_ns=0):
        rows = []
        for i in range(count):
            frame_type = 5 if idr_every and (i % idr_every == 0) else 1
            t = phase_ns + round(i * period_ns)
            if frame_type == 5:
                t += idr_bias_ns
            rows.append(MODULE.LogFrame(i + 1, t, frame_type))
        return tuple(rows)

    def test_fit_camera_ladder_ignores_idr_encode_latency_bias(self):
        period_ns = 1e9 / 60.0
        frames = self._frames(
            count=100, period_ns=period_ns, idr_every=33, idr_bias_ns=2_000_000,
        )
        ladder = MODULE.fit_camera_ladder(frames)
        self.assertAlmostEqual(ladder.period_ns, period_ns, delta=50.0)
        # Frame 1 is itself one of the biased IDR rows; the fit is built only
        # from P frames, so it should recover the *true*, unbiased line (0),
        # not the biased raw timestamp recorded for frame 1.
        self.assertAlmostEqual(ladder.time_ns(1), 0.0, delta=1000.0)
        self.assertIsNotNone(ladder.idr_bias_ms)
        self.assertAlmostEqual(ladder.idr_bias_ms, 2.0, delta=0.5)
        # The fit itself should not be dragged toward the excluded IDR rows.
        self.assertLess(ladder.fit_rms_ms, 0.1)

    def test_fit_camera_ladder_min_latency_resists_one_sided_delay(self):
        # Every 5th frame is saved 10ms late (one-sided: never early). A plain
        # OLS fit would be pulled up toward the mean delay; the min-latency
        # (lower-envelope) estimator should stay near the true, undelayed line.
        period_ns = 1e9 / 60.0
        rows = []
        for i in range(120):
            t = round(i * period_ns)
            if i % 5 == 0 and i > 0:
                t += 10_000_000
            rows.append(MODULE.LogFrame(i + 1, t, 1))
        ladder = MODULE.fit_camera_ladder(tuple(rows))
        self.assertAlmostEqual(ladder.period_ns, period_ns, delta=200.0)
        self.assertAlmostEqual(ladder.time_ns(1), 0.0, delta=1_000_000.0)

    def test_fit_camera_ladder_requires_minimum_frames(self):
        frames = self._frames(count=5, period_ns=1e9 / 60.0)
        with self.assertRaisesRegex(ValueError, "at least 8"):
            MODULE.fit_camera_ladder(frames)

    # ---- ladder-based build_timeline extensions ---------------------------

    def test_build_timeline_rejects_window_past_log_without_video_counts(self):
        source = self._source(0, 0, count=100)
        with self.assertRaisesRegex(ValueError, "at most 100"):
            MODULE.build_timeline((source,), fps=60.0, num_frames=150)

    def test_build_timeline_extrapolates_clean_ladder_with_video_counts(self):
        # A camera clock that free-runs exactly on-model should extrapolate
        # past its logged 100 frames with negligible sigma, no override flag
        # needed, as long as the video is confirmed long enough.
        source = self._source(0, 0, count=100)
        timeline = MODULE.build_timeline(
            (source,), fps=60.0, num_frames=150,
            video_frame_counts={source.camera_id: 200},
        )
        self.assertEqual(timeline.num_frames, 150)
        self.assertEqual(timeline.selections[0].extrapolated_frames, 50)
        self.assertLess(timeline.selections[0].sigma_last_ms, 4.0)

    def test_build_timeline_extrapolation_capped_by_video_frame_count(self):
        source = self._source(0, 0, count=100)
        with self.assertRaisesRegex(ValueError, "at most 120"):
            MODULE.build_timeline(
                (source,), fps=60.0, num_frames=150,
                video_frame_counts={source.camera_id: 120},
            )

    def test_build_timeline_sigma_gate_blocks_noisy_extrapolation(self):
        # A camera whose per-frame period genuinely varies block-to-block
        # (not one outlier, and not a clean alternation that averages out
        # within every block) has a nonzero *bootstrap* rate uncertainty, so
        # a distant enough extrapolation must eventually cross the sigma
        # budget. An irrational-frequency sine keeps the per-block offset
        # deterministic but not aliased with the block size.
        period_ns = 1e9 / 60.0
        rows = []
        t = 0.0
        for i in range(100):
            rows.append(MODULE.LogFrame(i + 1, round(t), 1))
            block_offset = 6_000.0 * math.sin((i // 10) * 2.399963)
            t += period_ns + block_offset
        source = MODULE.CameraSource(
            camera_id="JCam000", camera_index=0, folder="0000",
            video_path=Path("/capture/videos/JCam000_x.mp4"),
            log_path=Path("/capture/record_logs/JCam000_x.log"),
            snapshot_path=Path("/capture/snapshot/reference/images/JCam000_x.png"),
            frames=tuple(rows),
        )
        with self.assertRaisesRegex(ValueError, "max_ladder_sigma_ms"):
            MODULE.build_timeline(
                (source,), fps=60.0, num_frames=50_000,
                video_frame_counts={source.camera_id: 100_000},
                max_ladder_sigma_ms=4.0,
            )
        # The same request succeeds once extrapolation is explicitly allowed.
        timeline = MODULE.build_timeline(
            (source,), fps=60.0, num_frames=50_000,
            video_frame_counts={source.camera_id: 100_000},
            max_ladder_sigma_ms=4.0,
            allow_ladder_extrapolation=True,
        )
        self.assertEqual(timeline.num_frames, 50_000)


if __name__ == "__main__":
    unittest.main()
