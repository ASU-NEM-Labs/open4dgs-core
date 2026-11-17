import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "extract_synced_sequence.py"
SPEC = importlib.util.spec_from_file_location("extract_synced_sequence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_filename_take_number_uses_exact_take_token():
    assert MODULE.filename_take_number("camera_C0005_timestamp.MP4") == 5
    assert MODULE.filename_take_number("camera_C00050_timestamp.MP4") is None
    assert MODULE.filename_take_number("camera.mp4") is None


def test_flat_capture_filters_requested_take(tmp_path):
    wanted_b = tmp_path / "camera-b_C0005_timestamp.MP4"
    wanted_a = tmp_path / "camera-a_C0005_timestamp.MP4"
    other = tmp_path / "camera-a_C0004_timestamp.MP4"
    for path in (wanted_b, wanted_a, other):
        path.touch()

    tasks = MODULE.discover_video_tasks(str(tmp_path), 5)

    assert tasks == [
        (1, wanted_a.name, str(wanted_a)),
        (2, wanted_b.name, str(wanted_b)),
    ]


def test_legacy_untagged_flat_capture_still_uses_all_mp4s(tmp_path):
    first = tmp_path / "camera-a.mp4"
    second = tmp_path / "camera-b.MP4"
    first.touch()
    second.touch()

    tasks = MODULE.discover_video_tasks(str(tmp_path), 5)

    assert [task[2] for task in tasks] == [str(first), str(second)]


def test_camera_folders_filter_requested_take(tmp_path):
    cam_b = tmp_path / "cam-b"
    cam_a = tmp_path / "cam-a"
    cam_b.mkdir()
    cam_a.mkdir()
    wanted_a = cam_a / "C0005.MP4"
    wanted_b = cam_b / "clip_C0005_proxy.MP4"
    (cam_a / "C0004.MP4").touch()
    wanted_a.touch()
    wanted_b.touch()

    tasks = MODULE.discover_video_tasks(str(tmp_path), 5)

    assert tasks == [
        (1, "cam-a", str(wanted_a)),
        (2, "cam-b", str(wanted_b)),
    ]
