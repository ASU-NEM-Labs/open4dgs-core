import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_match_debug", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_match_debug_grids_accumulate_and_write_stage_panels(tmp_path: Path):
    camera = MODULE.Camera(
        name="camera 0",
        folder="0000",
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        camera_model_name="PINHOLE",
        width=32,
        height=24,
        params=np.array([20.0, 20.0, 16.0, 12.0], dtype=np.float64),
    )
    grids = MODULE._make_match_debug_grids([camera], cell_px=8)
    MODULE._accumulate_match_debug(
        grids["raw"][0],
        np.array([[1.0, 1.0], [2.0, 2.0], [31.0, 23.0], [-1.0, 2.0]]),
        cell_px=8,
    )
    MODULE._accumulate_match_debug(
        grids["final_accepted"][0], np.array([[2.0, 2.0]]), cell_px=8
    )

    report = MODULE._write_match_debug_grids(
        tmp_path,
        [camera],
        [np.full((24, 32, 3), 80, dtype=np.uint8)],
        grids,
    )

    assert report["per_camera"]["0000"]["raw"] == 3
    assert report["per_camera"]["0000"]["final_accepted"] == 1
    assert (tmp_path / "0000.jpg").is_file()
    saved = json.loads((tmp_path / "coverage.json").read_text())
    assert saved == report


def test_spatial_audit_writes_component_outcomes_without_filtering(tmp_path: Path):
    camera = MODULE.Camera(
        name="camera 23",
        folder="0023",
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        camera_model_name="PINHOLE",
        width=3840,
        height=2160,
        params=np.array([2000.0, 2000.0, 1920.0, 1080.0], dtype=np.float64),
    )
    records = [
        {
            "component_id": 12,
            "uv": [2200.0, 300.0],
            "support": 3,
            "status": "reprojection",
            "position": [1.0, 2.0, 3.0],
            "mean_reprojection_px": 4.0,
            "max_reprojection_px": 7.0,
            "reprojection_allowed_px": 2.5,
        },
        {
            "component_id": 13,
            "uv": [2210.0, 310.0],
            "support": 4,
            "status": "final_accepted",
            "position": [4.0, 5.0, 6.0],
            "mean_reprojection_px": 0.5,
            "max_reprojection_px": 1.0,
            "reprojection_allowed_px": 2.5,
            "triangulation_angle_deg": 12.0,
        },
    ]

    report = MODULE._write_spatial_audit(
        tmp_path,
        frame_idx=2,
        camera_idx=23,
        camera=camera,
        roi=(2020.0, 130.0, 2420.0, 620.0),
        pair_observation_counts={
            "raw": 100,
            "ransac_kept": 60,
            "calibrated_rejected": 20,
            "calibrated_kept": 40,
        },
        records=records,
    )

    assert report["components_in_roi"] == 2
    assert report["component_status_counts"]["reprojection"] == 1
    assert report["component_status_counts"]["final_accepted"] == 1
    saved_report = json.loads(
        (tmp_path / "spatial-audit-camera0023.json").read_text()
    )
    assert saved_report == report
    with np.load(tmp_path / "spatial-audit-camera0023.npz") as data:
        assert data["position"].shape == (2, 3)
        assert data["status_code"].tolist() == [4, 8]


def test_points_in_roi_is_inclusive_and_rejects_nonfinite_points():
    points = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
            [9.99, 20.0],
            [30.01, 40.0],
            [np.nan, 25.0],
        ]
    )
    assert MODULE._points_in_roi(points, (10.0, 20.0, 30.0, 40.0)).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
