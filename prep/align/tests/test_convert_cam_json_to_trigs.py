from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "convert-cam-json-to-trigs.py"
SPEC = importlib.util.spec_from_file_location("convert_cam_json_to_trigs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [
                1.0 - 2.0 * qy**2 - 2.0 * qz**2,
                2.0 * qx * qy - 2.0 * qw * qz,
                2.0 * qx * qz + 2.0 * qw * qy,
            ],
            [
                2.0 * qx * qy + 2.0 * qw * qz,
                1.0 - 2.0 * qx**2 - 2.0 * qz**2,
                2.0 * qy * qz - 2.0 * qw * qx,
            ],
            [
                2.0 * qx * qz - 2.0 * qw * qy,
                2.0 * qy * qz + 2.0 * qw * qx,
                1.0 - 2.0 * qx**2 - 2.0 * qy**2,
            ],
        ],
        dtype=np.float64,
    )


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


class ConvertCamJsonToTrigsTests(unittest.TestCase):
    def test_rotation_to_qvec_round_trips_colmap_convention(self) -> None:
        rotations = [
            np.eye(3),
            axis_angle_rotation(np.asarray([1.0, 0.0, 0.0]), np.deg2rad(165.0)),
            axis_angle_rotation(np.asarray([0.2, -0.7, 0.4]), np.deg2rad(73.0)),
            axis_angle_rotation(np.asarray([1.0, 2.0, 3.0]), np.deg2rad(179.0)),
        ]

        for rotation in rotations:
            with self.subTest(rotation=rotation):
                qvec = MODULE._rotation_to_qvec(rotation)
                reconstructed = qvec_to_rotation(qvec)
                np.testing.assert_allclose(reconstructed, rotation, atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.norm(qvec)), 1.0, places=12)
                self.assertGreaterEqual(float(qvec[0]), 0.0)

    def test_rejects_non_rotation_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotation must be 3x3"):
            MODULE._rotation_to_qvec(np.eye(4))

    def test_rejects_video_camera_identity_mismatch(self) -> None:
        records = [
            {
                "camera_id": 3,
                "video": "videos/dense/04.mp4",
            }
        ]
        with self.assertRaisesRegex(ValueError, "does not match camera_id"):
            MODULE._camera_records(records)

    def test_detects_real_first_frame_and_extension(self) -> None:
        records = [
            {"camera_id": 0, "video": "videos/dense/00.mp4"},
            {"camera_id": 1, "video": "videos/dense/01.mp4"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            images_root = Path(temp)
            for folder in ("00", "01"):
                camera_dir = images_root / folder
                camera_dir.mkdir()
                (camera_dir / "000001.jpg").touch()
                (camera_dir / "000002.jpg").touch()

            frame_index, extension = MODULE._detect_reference_image(
                records, images_root
            )

        self.assertEqual(frame_index, 1)
        self.assertEqual(extension, "jpg")

    def test_rejects_inconsistent_camera_image_sequences(self) -> None:
        records = [
            {"camera_id": 0, "video": "videos/dense/00.mp4"},
            {"camera_id": 1, "video": "videos/dense/01.mp4"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            images_root = Path(temp)
            (images_root / "00").mkdir()
            (images_root / "01").mkdir()
            (images_root / "00" / "000001.jpg").touch()
            (images_root / "01" / "000000.png").touch()

            with self.assertRaisesRegex(ValueError, "do not share one first frame"):
                MODULE._detect_reference_image(records, images_root)


if __name__ == "__main__":
    unittest.main()
