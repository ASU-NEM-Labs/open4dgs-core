#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "generate_masks.py"
SPEC = importlib.util.spec_from_file_location("generate_masks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GenerateMasksTests(unittest.TestCase):
    def test_atomic_mask_write_produces_readable_png_without_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000001.png"
            expected = np.arange(20, dtype=np.uint8).reshape(4, 5)

            MODULE.write_mask_atomic(path, expected)

            np.testing.assert_array_equal(
                cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), expected
            )
            self.assertEqual(list(path.parent.glob(".*.tmp-*.png")), [])

    def test_edge_corrected_foreground_preserves_interior_and_cleans_edge(self):
        source = np.array(
            [[
                [200, 150, 100], [200, 150, 100],
                [200, 150, 100], [200, 150, 100],
            ]],
            dtype=np.uint8,
        )
        foreground = np.array(
            [[
                [0.2, 0.4, 0.8], [0.2, 0.4, 0.8],
                [0.2, 0.4, 0.8], [0.2, 0.4, 0.8],
            ]],
            dtype=np.float32,
        )
        alpha = np.array([[0.0, 0.5, 0.94, 1.0]], dtype=np.float32)

        corrected = MODULE.build_edge_corrected_foreground(
            source, foreground, alpha, blend_alpha=0.90,
            solid_alpha=0.98, zero_alpha=0.02,
        )

        np.testing.assert_array_equal(corrected[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(corrected[0, 1], [51, 102, 204])
        np.testing.assert_array_equal(corrected[0, 3], source[0, 3])
        self.assertTrue(np.all(corrected[0, 2] > np.array([51, 102, 100])))
        self.assertTrue(np.all(corrected[0, 2] < np.array([201, 151, 205])))

    def test_soft_alpha_gamma_tightens_fractional_values_only(self):
        alpha = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)

        tightened = MODULE.tighten_soft_alpha(alpha, gamma=2.0)

        np.testing.assert_allclose(tightened, [[0.0, 0.0625, 0.25, 1.0]])

    def test_foreground_rgba_keeps_straight_rgb_and_embeds_alpha(self):
        foreground = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        alpha = np.array([[0, 128]], dtype=np.uint8)

        rgba = MODULE.pack_foreground_rgba(foreground, alpha)

        np.testing.assert_array_equal(rgba[..., :3], foreground)
        np.testing.assert_array_equal(rgba[..., 3], alpha)

    def test_camera_worker_retries_after_native_process_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "masks"
            (images / "cam00").mkdir(parents=True)
            output.mkdir()
            (images / "cam00" / "000001.jpg").touch()

            results = [SimpleNamespace(returncode=-11), SimpleNamespace(returncode=0)]
            with mock.patch.object(MODULE.subprocess, "run", side_effect=results) as run:
                with mock.patch.object(sys, "argv", ["generate_masks.py", "--data-dir", "."]):
                    total, succeeded = MODULE.process_bgmv2_in_camera_workers(
                        ["cam00"], str(images), str(output), 1, 1, retries=1
                    )

            self.assertTrue(succeeded)
            self.assertEqual(total, 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args.args[0][-3:],
                ["--cameras", "cam00", "--camera-worker"],
            )

    def test_complete_camera_does_not_start_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "masks"
            (images / "cam00").mkdir(parents=True)
            (output / "cam00").mkdir(parents=True)
            (images / "cam00" / "000001.jpg").touch()
            (output / "cam00" / "000001.png").touch()

            with mock.patch.object(MODULE.subprocess, "run") as run:
                total, succeeded = MODULE.process_bgmv2_in_camera_workers(
                    ["cam00"], str(images), str(output), 1, 1, retries=1
                )

            self.assertTrue(succeeded)
            self.assertEqual(total, 1)
            run.assert_not_called()

    def test_missing_foreground_starts_worker_even_when_mask_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "masks"
            foregrounds = root / "foregrounds"
            (images / "cam00").mkdir(parents=True)
            (output / "cam00").mkdir(parents=True)
            foregrounds.mkdir()
            (images / "cam00" / "000001.jpg").touch()
            (output / "cam00" / "000001.png").touch()

            with mock.patch.object(
                MODULE.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as run:
                with mock.patch.object(
                    sys, "argv", ["generate_masks.py", "--data-dir", "."]
                ):
                    total, succeeded = MODULE.process_bgmv2_in_camera_workers(
                        ["cam00"], str(images), str(output), 1, 1, retries=0,
                        foregrounds_output_dir=str(foregrounds),
                    )

            self.assertTrue(succeeded)
            self.assertEqual(total, 1)
            run.assert_called_once()

    def test_overwrite_starts_worker_even_when_outputs_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "masks"
            foregrounds = root / "foregrounds"
            for parent in (images, output, foregrounds):
                (parent / "cam00").mkdir(parents=True)
            (images / "cam00" / "000001.jpg").touch()
            (output / "cam00" / "000001.png").touch()
            (foregrounds / "cam00" / "000001.png").touch()

            with mock.patch.object(
                MODULE.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as run:
                with mock.patch.object(
                    sys, "argv", ["generate_masks.py", "--data-dir", "."]
                ):
                    _, succeeded = MODULE.process_bgmv2_in_camera_workers(
                        ["cam00"], str(images), str(output), 1, 1, retries=0,
                        foregrounds_output_dir=str(foregrounds), overwrite=True,
                    )

            self.assertTrue(succeeded)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
