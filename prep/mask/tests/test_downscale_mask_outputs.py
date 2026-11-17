#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "downscale_mask_outputs.py"
SPEC = importlib.util.spec_from_file_location("downscale_mask_outputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DownscaleMaskOutputsTests(unittest.TestCase):
    def test_sizes_match_trainer_parser_floor_division(self):
        mask = np.zeros((2161, 3841), dtype=np.uint8)
        self.assertEqual(MODULE.downscale_mask(mask, 2).shape, (1080, 1920))

    def test_mask_is_box_averaged(self):
        mask = np.array([[0, 255, 255, 255], [0, 255, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(MODULE.downscale_mask(mask, 2), [[128, 255]])

    def test_transparent_neighbours_do_not_darken_edge_colour(self):
        # One opaque white pixel with three fully transparent (zeroed) pixels.
        fg = np.zeros((2, 2, 4), dtype=np.uint8)
        fg[0, 0] = (255, 255, 255, 255)
        out = MODULE.downscale_foreground(fg, 2)
        np.testing.assert_array_equal(out[0, 0, :3], (255, 255, 255))
        self.assertEqual(int(out[0, 0, 3]), 64)

    def test_composite_of_small_copy_equals_shrunk_full_composite(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (64, 96, 3)).astype(np.uint8)
        alpha = rng.integers(0, 256, (64, 96)).astype(np.uint8)
        rgb[alpha < 5] = 0  # generate_masks zeroes colour where the matte is ~0
        fg = np.dstack((rgb, alpha))
        bg = np.array([30.0, 200.0, 90.0])

        def composite(f):
            a = f[..., 3:].astype(np.float64) / 255.0
            return f[..., :3] * a + bg * (1.0 - a)

        want = cv2.resize(composite(fg), (48, 32), interpolation=cv2.INTER_AREA)
        got = composite(MODULE.downscale_foreground(fg, 2))
        self.assertLess(np.abs(got - want).max(), 1.5)  # uint8 rounding only

    def test_rgb_foreground_uses_mask_as_weight(self):
        fg = np.zeros((2, 2, 3), dtype=np.uint8)
        fg[0, 0] = 200
        weight = np.zeros((2, 2), dtype=np.uint8)
        weight[0, 0] = 255
        np.testing.assert_array_equal(MODULE.downscale_foreground(fg, 2, weight)[0, 0], 200)
        np.testing.assert_array_equal(MODULE.downscale_foreground(fg, 2)[0, 0], 50)

    def test_main_writes_suffixed_dirs_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for name in ("masks", "foregrounds"):
                (data / name / "01").mkdir(parents=True)
            for frame in (1, 2, 3):
                cv2.imwrite(str(data / "masks" / "01" / f"{frame:06d}.png"),
                            np.full((8, 12), 255, np.uint8))
                cv2.imwrite(str(data / "foregrounds" / "01" / f"{frame:06d}.png"),
                            np.full((8, 12, 4), 255, np.uint8))
            args = ["--data-dir", str(data), "--factor", "2", "--workers", "1",
                    "--start-frame", "1", "--end-frame", "2"]
            self.assertEqual(MODULE.main(args), 0)

            for name, shape in (("masks_f2", (4, 6)), ("foregrounds_f2", (4, 6, 4))):
                written = sorted(p.name for p in (data / name / "01").iterdir())
                self.assertEqual(written, ["000001.png", "000002.png"])
                img = cv2.imread(str(data / name / "01" / "000001.png"), cv2.IMREAD_UNCHANGED)
                self.assertEqual(img.shape, shape)

            tasks, skipped, missing = MODULE.build_tasks(
                data, "masks", "foregrounds", 2, 1, 2, "{frame:06d}.png", False)
            self.assertEqual((len(tasks), skipped, missing), (0, 4, []))

    def test_missing_requested_frame_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "masks" / "01").mkdir(parents=True)
            cv2.imwrite(str(data / "masks" / "01" / "000001.png"), np.zeros((4, 4), np.uint8))
            code = MODULE.main(["--data-dir", str(data), "--foregrounds-dir", "",
                                "--start-frame", "1", "--end-frame", "2", "--workers", "1"])
            self.assertEqual(code, 1)
            self.assertFalse((data / "masks_f2" / "01" / "000001.png").exists())


if __name__ == "__main__":
    unittest.main()
