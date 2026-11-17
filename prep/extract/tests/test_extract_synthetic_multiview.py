from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "extract_synthetic_multiview.py"
SPEC = importlib.util.spec_from_file_location("extract_synthetic_multiview", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SyntheticMultiviewExtractionTests(unittest.TestCase):
    def test_discovers_videos_in_natural_camera_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("10.mp4", "2.mp4", "01.mov", ".hidden.mp4", "notes.txt"):
                (root / name).touch()
            videos = MODULE.discover_videos(root)
        self.assertEqual([path.stem for path in videos], ["01", "2", "10"])

    def test_png_command_is_one_based_and_passthrough(self):
        info = MODULE.VideoInfo(Path("/video/00.mp4"), "00", 704, 1280, "30/1")
        command, extension = MODULE.build_ffmpeg_command(
            info=info,
            camera_dir=Path("/output/00"),
            image_format="png",
            start_number=1,
            digits=6,
            jpeg_quality=2,
            png_compression=3,
            max_frames=None,
            ffmpeg="ffmpeg",
        )
        self.assertEqual(extension, "png")
        self.assertIn("passthrough", command)
        self.assertEqual(command[-2:], ["1", "/output/00/%06d.png"])

    def test_jpeg_command_uses_requested_frame_limit(self):
        info = MODULE.VideoInfo(Path("/video/20.mp4"), "20", 704, 1280, "30/1")
        command, extension = MODULE.build_ffmpeg_command(
            info=info,
            camera_dir=Path("/output/20"),
            image_format="jpeg",
            start_number=1,
            digits=6,
            jpeg_quality=2,
            png_compression=3,
            max_frames=121,
            ffmpeg="ffmpeg",
        )
        self.assertEqual(extension, "jpg")
        self.assertIn("mjpeg", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "121")
        self.assertEqual(command[-1], "/output/20/%06d.jpg")

    def test_default_first_output_is_frame_one(self):
        args = MODULE.parse_args(
            ["--videos-dir", "/videos", "--output-dir", "/images"]
        )
        self.assertEqual(args.start_number, 1)
        self.assertEqual(args.format, "png")


if __name__ == "__main__":
    unittest.main()
