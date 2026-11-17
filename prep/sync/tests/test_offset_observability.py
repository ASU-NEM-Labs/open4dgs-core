import importlib.util
import pathlib
import sys
import tempfile
import unittest
from dataclasses import dataclass

import numpy as np
from PIL import Image


SYNC_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SYNC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _FakeCamera:
    folder: str


class OffsetObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script("offset_observability_test", "measure-offset-observability.py")

    def test_sample_frame_indices_spans_window_and_respects_step(self):
        idx = self.mod.sample_frame_indices(1, 100, frame_step=2, max_samples=10)
        self.assertEqual(idx[0], 1 + 2)
        self.assertEqual(idx[-1], 100 - 2)
        self.assertLessEqual(len(idx), 10)
        self.assertEqual(idx, sorted(set(idx)))

    def test_sample_frame_indices_rejects_window_shorter_than_step(self):
        with self.assertRaises(ValueError):
            self.mod.sample_frame_indices(1, 3, frame_step=5, max_samples=10)

    def _write_camera_frames(self, images_root, folder, frames_by_index):
        # PNG (lossless): JPEG's DCT quantization crushes small per-pixel
        # noise on a near-flat image even at quality=100, which would corrupt
        # the injected noise model this test depends on.
        cam_dir = images_root / folder
        cam_dir.mkdir(parents=True, exist_ok=True)
        for idx, arr in frames_by_index.items():
            Image.fromarray(arr.astype(np.uint8), mode="L").save(
                cam_dir / f"{idx:06d}.png"
            )

    def test_moving_camera_gets_higher_shrinkage_than_static_camera(self):
        # Camera "moving": a bright bar sweeps across the frame over time
        # (large, clean dI/dt). Camera "static": constant image plus tiny
        # noise (near-zero true dI/dt). The Fisher/shrinkage pipeline should
        # rank "moving" as far more identifiable than "static", regardless
        # of the exact calibration constants.
        rng = np.random.default_rng(0)
        h, w = 64, 64
        frame_indices = list(range(1, 21))

        moving_frames = {}
        static_frames = {}
        for i in frame_indices:
            bar = np.zeros((h, w), dtype=np.float64)
            col = int((i / max(frame_indices)) * (w - 8))
            bar[:, col:col + 8] = 200.0
            moving_frames[i] = np.clip(bar + rng.normal(0, 1.0, size=(h, w)), 0, 255)
            static_frames[i] = np.clip(
                np.full((h, w), 100.0) + rng.normal(0, 1.0, size=(h, w)), 0, 255
            )

        with tempfile.TemporaryDirectory() as tmp:
            images_root = pathlib.Path(tmp) / "images"
            self._write_camera_frames(images_root, "0000", moving_frames)
            self._write_camera_frames(images_root, "0001", static_frames)
            cameras = [_FakeCamera(folder="0000"), _FakeCamera(folder="0001")]

            # pixel_stride=1: this test is about the bias-correction/
            # discrimination logic, not the pixel-independence heuristic
            # (see test_pixel_stride_reduces_dense_pixel_overcounting below).
            # At only 64x64, a stride=4 grid leaves too few points (256) for
            # the bias-corrected sum to reliably average down to ~0 for a
            # genuinely static camera -- a real frame has ~2000x more points
            # after downsampling, where that averaging is no longer an issue.
            fisher, sigma_noise = self.mod.measure_observability(
                images_root=images_root,
                cameras=cameras,
                frame_start=1,
                frame_end=20,
                frame_step=1,
                max_samples=10,
                fps=60.0,
                image_pattern="{frame:06d}.png",
                downsample=1,
                pixel_stride=1,
            )

        self.assertGreater(fisher[0], fisher[1] * 5)

        sigma_prior_s = 0.004  # 4ms, a typical ladder prior
        sigma_photo = sigma_noise / np.sqrt(np.maximum(fisher, 1e-12))
        shrinkage = (sigma_prior_s ** 2) / (sigma_prior_s ** 2 + sigma_photo ** 2)
        self.assertGreater(shrinkage[0], shrinkage[1])
        self.assertLess(shrinkage[1], 0.3)

    def test_pixel_stride_reduces_dense_pixel_overcounting(self):
        # Real image content is spatially correlated (smooth gradients, JPEG
        # blocking); treating every pixel as an independent Fisher
        # observation drastically overstates fisher. A coarser pixel_stride
        # must reduce fisher roughly in proportion to the retained point
        # count, on the SAME underlying data -- this is what stops every
        # real camera's shrinkage from saturating to ~1.0 regardless of
        # actual motion (see the bug this test was written to catch).
        rng = np.random.default_rng(1)
        h, w = 64, 64
        frame_indices = list(range(1, 21))
        frames = {}
        for i in frame_indices:
            bar = np.zeros((h, w), dtype=np.float64)
            col = int((i / max(frame_indices)) * (w - 8))
            bar[:, col:col + 8] = 200.0
            frames[i] = np.clip(bar + rng.normal(0, 1.0, size=(h, w)), 0, 255)

        with tempfile.TemporaryDirectory() as tmp:
            images_root = pathlib.Path(tmp) / "images"
            self._write_camera_frames(images_root, "0000", frames)
            cameras = [_FakeCamera(folder="0000")]

            fisher_dense, _ = self.mod.measure_observability(
                images_root=images_root, cameras=cameras,
                frame_start=1, frame_end=20, frame_step=1, max_samples=10,
                fps=60.0, image_pattern="{frame:06d}.png", downsample=1,
                pixel_stride=1,
            )
            fisher_strided, _ = self.mod.measure_observability(
                images_root=images_root, cameras=cameras,
                frame_start=1, frame_end=20, frame_step=1, max_samples=10,
                fps=60.0, image_pattern="{frame:06d}.png", downsample=1,
                pixel_stride=4,
            )

        # stride=4 in 2D keeps ~1/16 of the points; fisher should drop by
        # roughly that factor (not be unaffected, and not collapse to ~0).
        ratio = fisher_dense[0] / fisher_strided[0]
        self.assertGreater(ratio, 8.0)
        self.assertLess(ratio, 32.0)


if __name__ == "__main__":
    unittest.main()
