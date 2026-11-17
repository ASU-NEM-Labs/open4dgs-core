import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "build_4dv_snapshot_pose_candidate.py"
SPEC = importlib.util.spec_from_file_location("build_4dv_snapshot_pose_candidate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_proper_similarity_row_recovers_synthetic_transform():
    source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    theta = np.deg2rad(25.0)
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    target = 1.7 * source @ rotation + np.asarray([3.0, -2.0, 0.5])

    scale_fit, rotation_fit, translation_fit = MODULE.proper_similarity_row(
        source, target
    )

    recovered = scale_fit * source @ rotation_fit + translation_fit
    assert np.allclose(recovered, target, atol=1e-10)
    assert np.linalg.det(rotation_fit) > 0.999


def test_candidate_replaces_only_requested_pose_and_preserves_intrinsics():
    base_cameras = []
    snapshot_rows = []
    for index, center in enumerate(
        [np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]),
         np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])]
    ):
        rotation = np.eye(3)
        translation = -rotation @ center
        base_cameras.append({
            "folder": f"{index:04d}", "R": rotation.tolist(),
            "t": translation.tolist(), "params": [1000.0, 1000.0, 500.0, 500.0],
        })
        snapshot_rows.append({
            "camera_id": f"JCam{index:03d}",
            "extrinsic": np.column_stack([rotation, translation]).tolist(),
        })
    rotated = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    snapshot_rows[3]["extrinsic"] = np.column_stack(
        [rotated, -rotated @ np.asarray([0.0, 0.0, 1.0])]
    ).tolist()
    base = {"cameras": base_cameras}
    record = {"extri": {"data": snapshot_rows}}

    candidate, diagnostics = MODULE.build_candidate(base, record, ["0003"])

    assert candidate["cameras"][0]["R"] == base["cameras"][0]["R"]
    assert candidate["cameras"][3]["params"] == base["cameras"][3]["params"]
    assert diagnostics["replacements"]["0003"]["rotation_delta_deg"] > 80.0


def test_candidate_can_replace_complete_projection_from_snapshot():
    base_cameras = []
    snapshot_extrinsics = []
    snapshot_intrinsics = []
    for index, center in enumerate(
        [np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]),
         np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])]
    ):
        rotation = np.eye(3)
        translation = -rotation @ center
        base_cameras.append({
            "folder": f"{index:04d}", "R": rotation.tolist(),
            "t": translation.tolist(), "camera_model_name": "OPENCV",
            "params": [1200.0, 1190.0, 500.0, 500.0, 0.4, -0.6, 0.0, 0.0],
        })
        snapshot_extrinsics.append({
            "camera_id": f"JCam{index:03d}",
            "extrinsic": np.column_stack([rotation, translation]).tolist(),
        })
        snapshot_intrinsics.append({
            "camera_id": f"JCam{index:03d}",
            "intrinsic": [[1000.0, 0.0, 499.5], [0.0, 1001.0, 499.5], [0.0, 0.0, 1.0]],
        })
    base = {"cameras": base_cameras}
    record = {
        "extri": {"data": snapshot_extrinsics},
        "intri": {"data": snapshot_intrinsics},
    }

    candidate, diagnostics = MODULE.build_candidate(
        base, record, ["0003"], replace_projection=True
    )

    assert candidate["cameras"][0]["camera_model_name"] == "OPENCV"
    assert candidate["cameras"][3]["camera_model_name"] == "PINHOLE"
    assert candidate["cameras"][3]["params"] == [1000.0, 1001.0, 499.5, 499.5]
    assert diagnostics["replacements"]["0003"]["projection_replaced"] is True
