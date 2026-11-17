from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "make-trigs-timeline.py"
SPEC = importlib.util.spec_from_file_location("make_trigs_timeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packager
SPEC.loader.exec_module(packager)


def test_trainer_windows_become_overlapping_train_and_disjoint_core_ranges():
    meta = {
        "trainerWindows": [
            [0.0, 2.5],
            [2.0, 4.5],
            [4.0, 6.5],
            [6.0, 8.5],
            [8.0, 10.5],
            [10.0, 12.5],
            [12.0, 14.483333333333333],
        ],
        "trainerOwnershipBounds": [2.25, 4.25, 6.25, 8.25, 10.25, 12.25],
    }

    _, _, schedules, total_frames, source = packager.build_single_checkpoint_split_plan(
        meta=meta,
        frame_rate=60.0,
        total_frames=None,
        chunk_duration_s=None,
        chunk_overlap_s=0.5,
    )

    assert source == "trainer_metrics"
    assert total_frames == 870
    assert [
        (s.train_start, s.train_end, s.core_start, s.core_end) for s in schedules
    ] == [
        (0, 151, 0, 135),
        (120, 271, 135, 255),
        (240, 391, 255, 375),
        (360, 511, 375, 495),
        (480, 631, 495, 615),
        (600, 751, 615, 735),
        (720, 870, 735, 870),
    ]


def test_temporal_support_mask_uses_support_intersection_not_center_only():
    arrays = {
        "times": np.asarray([0.1, 0.9, 1.4], dtype=np.float32),
        "log_durations": np.log(
            np.asarray([0.05, 0.2, 0.05], dtype=np.float32)
        ),
    }
    flags = {"is_static": np.asarray([False, False, True])}

    selected = packager.temporal_support_mask(
        arrays,
        flags,
        t_start=0.0,
        t_end=0.5,
        opacity_threshold=0.05,
        include_static=False,
    )
    assert selected.tolist() == [True, True, False]

    selected_with_static = packager.temporal_support_mask(
        arrays,
        flags,
        t_start=0.0,
        t_end=0.5,
        opacity_threshold=0.05,
        include_static=True,
    )
    assert selected_with_static.tolist() == [True, True, True]


def test_fixed_duration_plan_is_available_without_trainer_windows():
    windows, bounds, schedules, total_frames, source = (
        packager.build_single_checkpoint_split_plan(
            meta={"n_frames": 10},
            frame_rate=2.0,
            total_frames=None,
            chunk_duration_s=2.0,
            chunk_overlap_s=0.5,
        )
    )

    assert source == "fixed_duration"
    assert total_frames == 10
    assert windows == [[0.0, 2.0], [1.5, 3.5], [3.0, 4.5]]
    assert bounds == [1.75, 3.25]
    assert schedules[0].core_start == 0
    assert schedules[-1].core_end == 10


def _small_checkpoint(path: Path) -> None:
    n = 4
    params = {
        "means": torch.zeros(n, 3),
        "scales": torch.zeros(n, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacities": torch.zeros(n),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 0, 3),
        "times": torch.tensor([[0.75], [0.10], [0.75], [1.40]]),
        "log_durations": torch.log(torch.tensor([[1.5], [0.05], [0.20], [0.05]])),
        "anchors": torch.zeros(n, 3),
        "zeta_base_r": torch.zeros(n, 3),
        "zeta_base_t": torch.zeros(n, 3),
        "zeta_c0_r": torch.zeros(n, 3),
        "zeta_c0_t": torch.zeros(n, 3),
        "zeta_c1_r": torch.zeros(n, 3),
        "zeta_c1_t": torch.zeros(n, 3),
        "zeta_c2_r": torch.zeros(n, 3),
        "zeta_c2_t": torch.zeros(n, 3),
        "is_static": torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        "dynamic_protect": torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
    }
    torch.save(
        {
            "params": params,
            "fps": 10.0,
            "n_frames": 16,
            "metrics": {
                "windows": [[0.0, 1.0], [0.5, 1.5]],
                "ownership_bounds": [0.75],
                "entry_diagnostics": [{"cull_threshold": 0.05}],
            },
        },
        path,
    )


def _read_tensor(path: Path, manifest: dict, spec: dict) -> np.ndarray:
    dtype = np.dtype("<f4" if spec["dtype"] == "float32" else "<f2")
    payload_offset = int(manifest["container"]["payloadOffset"])
    with path.open("rb") as f:
        f.seek(payload_offset + int(spec["byteOffset"]))
        data = f.read(int(spec["byteLength"]))
    return np.frombuffer(data, dtype=dtype).reshape(spec["shape"])


def test_from_checkpoint_global_mode_preserves_time_and_writes_global_static(tmp_path):
    checkpoint = tmp_path / "trigs_ckpt.pt"
    output = tmp_path / "timeline.trigsbin"
    _small_checkpoint(checkpoint)

    assert packager.main(
        [
            "from-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--frame-rate",
            "10",
            "--time-domain",
            "global",
            "--tensor-dtype",
            "motion32",
        ]
    ) == 0

    _, manifest = packager.read_trigsbin_manifest(output)
    assert manifest["static"]["count"] == 1
    assert manifest["timeline"]["totalFrames"] == 16
    assert manifest["semantics"]["temporalCullThreshold"] == 0.05
    assert manifest["semantics"]["dynamicTimesDomain"] == "global"
    assert [clip["dynamic"]["count"] for clip in manifest["clips"]] == [2, 2]
    assert manifest["clips"][0]["global"]["coreFrameEndExclusive"] == 8
    assert manifest["clips"][1]["global"]["coreFrameStart"] == 8
    assert manifest["clips"][1]["training"]["globalFrameStart"] == 0
    assert manifest["clips"][1]["local"]["coreFrameStart"] == 8
    assert manifest["clips"][1]["training"]["scheduleFrameCount"] == 16
    assert manifest["clips"][1]["training"]["frameCount"] == 7

    first_times = _read_tensor(
        output,
        manifest,
        manifest["clips"][0]["dynamic"]["tensors"]["times"],
    )
    second_times = _read_tensor(
        output,
        manifest,
        manifest["clips"][1]["dynamic"]["tensors"]["times"],
    )
    np.testing.assert_allclose(first_times, np.asarray([0.10, 0.75], dtype=np.float32))
    np.testing.assert_allclose(second_times, np.asarray([0.75, 1.40], dtype=np.float32))


def test_from_checkpoint_defaults_to_clip_local_time_and_guards_player_clock(tmp_path):
    checkpoint = tmp_path / "trigs_ckpt.pt"
    output = tmp_path / "timeline-local.trigsbin"
    _small_checkpoint(checkpoint)

    assert packager.main(
        [
            "from-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--frame-rate",
            "10",
        ]
    ) == 0

    _, manifest = packager.read_trigsbin_manifest(output)
    assert manifest["semantics"]["dynamicTimesDomain"] == "clip-local"
    assert manifest["clips"][1]["training"]["globalFrameStart"] == 5
    assert manifest["clips"][1]["training"]["scheduleFrameCount"] == 11
    assert manifest["clips"][1]["training"]["frameCount"] == 7
    second_times = _read_tensor(
        output,
        manifest,
        manifest["clips"][1]["dynamic"]["tensors"]["times"],
    )
    np.testing.assert_allclose(second_times, np.asarray([0.25, 0.90], dtype=np.float32))

    tensor_span = float(second_times.max() - second_times.min())
    expected_span = (
        manifest["clips"][1]["training"]["frameCount"] - 1
    ) / manifest["timeline"]["frameRate"]
    assert 0.85 <= tensor_span / expected_span <= 1.15


def test_from_checkpoint_rejects_silent_timeline_truncation(tmp_path):
    checkpoint = tmp_path / "trigs_ckpt.pt"
    output = tmp_path / "truncated.trigsbin"
    _small_checkpoint(checkpoint)

    with pytest.raises(SystemExit, match="would truncate the trainer timeline"):
        packager.main(
            [
                "from-checkpoint",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--frame-rate",
                "10",
                "--total-frames",
                "10",
            ]
        )
