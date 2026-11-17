import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_defaults", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _parse(monkeypatch, tmp_path, *extra):
    ref = tmp_path / "reference.json"
    ref.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "prep", "--reference-json", str(ref), "--images-root", str(tmp_path / "images"),
        "--output-path", str(tmp_path / "out.npz"), "--frame-start", "10", "--frame-end", "20", *extra,
    ])
    return MODULE.parse_args()


def _images(tmp_path, spans, ext=".jpg"):
    for cam, (lo, hi) in spans.items():
        d = tmp_path / "images" / cam
        d.mkdir(parents=True)
        for f in range(lo, hi + 1):
            (d / f"{f:06d}{ext}").write_bytes(b"")
        (d / "._000001.jpg").write_bytes(b"")  # macOS sidecar: must not count as a frame


def test_pattern_and_common_frame_range_are_detected(tmp_path, monkeypatch):
    _images(tmp_path, {"01": (1, 30), "02": (3, 28)})
    args = _parse(monkeypatch, tmp_path)
    assert args.image_pattern == "{frame:06d}.jpg"
    assert (args.input_frame_start, args.input_frame_end) == (3, 28)  # frames every camera has
    assert args.temporal_frame_delta == args.frame_step == 5
    assert args.subkeyframe_min_speed == MODULE.SUBKEYFRAME_MIN_SPEED_DEFAULT


def test_explicit_values_win_and_png_is_detected(tmp_path, monkeypatch):
    _images(tmp_path, {"01": (1, 30)}, ext=".png")
    args = _parse(monkeypatch, tmp_path, "--input-frame-start", "5", "--temporal-frame-delta", "2",
                  "--subkeyframe-min-speed", "0")
    assert args.image_pattern == "{frame:06d}.png"
    assert (args.input_frame_start, args.input_frame_end) == (5, 30)
    assert args.temporal_frame_delta == 2 and args.subkeyframe_min_speed == 0.0


def test_no_images_keeps_the_old_fallbacks(tmp_path, monkeypatch):
    args = _parse(monkeypatch, tmp_path)
    assert args.image_pattern == "{frame:06d}.png"
    assert args.input_frame_start is None and args.input_frame_end is None  # main() falls back to --frame-*


def test_subkeyframes_switch_off_when_they_cannot_run(tmp_path, monkeypatch):
    args = _parse(monkeypatch, tmp_path, "--disable-temporal-tracking")
    assert args.subkeyframe_min_speed == 0.0


def test_compose_hops_follow_the_temporal_backend(tmp_path, monkeypatch):
    assert _parse(monkeypatch, tmp_path).memfof_compose_hops is True
    assert _parse(monkeypatch, tmp_path, "--temporal-backend", "roma").memfof_compose_hops is False


def test_last_keyframe_tracks_into_images_past_the_clip_end():
    frames = list(range(66, 127, 5))                                   # 66 .. 126
    assert MODULE.trackable_keyframes(frames, 5, 583)[-1] == 126         # images exist up to 583
    assert MODULE.trackable_keyframes(frames, 5, 126)[-1] == 121         # no image past the clip: drop 126
    assert MODULE.trackable_keyframes(frames, 1, 126)[-1] == 121 or 126 in MODULE.trackable_keyframes(frames, 1, 127)
