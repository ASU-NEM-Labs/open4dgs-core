import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "audit_detail_region.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_detail_region", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _report(**updates):
    summary = {
        "pair_success_ratio": 0.95,
        "match_yield_ratio": 0.70,
        "calibrated_inlier_ratio_2_5px": 0.90,
        "fitted_inlier_ratio": 0.92,
        "detail_three_plus_track_ratio": 0.80,
        "temporal_flow_p90_px_per_frame": 0.1,
        "predicted_sync_shift_p90_px": 0.05,
    }
    summary.update(updates)
    return {"summary": summary}


def test_classification_separates_matcher_calibration_tracks_and_timing():
    assert MODULE.classify_report(
        _report(match_yield_ratio=0.01)
    )["primary_cause"] == "roma_correspondence_failure"
    assert MODULE.classify_report(
        _report(calibrated_inlier_ratio_2_5px=0.20, fitted_inlier_ratio=0.85)
    )["primary_cause"] == "camera_calibration_mismatch"
    assert MODULE.classify_report(
        _report(detail_three_plus_track_ratio=0.10)
    )["primary_cause"] == "multiview_track_assembly_failure"
    assert MODULE.classify_report(
        _report(predicted_sync_shift_p90_px=3.0)
    )["primary_cause"] == "timing_or_motion_failure"


def test_classification_detects_well_observed_camera_outlier_before_tracks():
    report = _report(detail_three_plus_track_ratio=0.10)
    report["per_camera"] = [
        {
            "folder": "0026",
            "match_observations": 7308,
            "calibrated_inlier_ratio_2_5px": 0.0,
        }
    ]

    result = MODULE.classify_report(report)

    assert result["primary_cause"] == "camera_calibration_outlier_subset"
    assert "0026" in result["explanation"]


def test_explicit_camera_pairs_use_folders_and_require_visibility():
    cameras = [
        SimpleNamespace(folder="0026"),
        SimpleNamespace(folder="0079"),
        SimpleNamespace(folder="0008"),
    ]

    assert MODULE.select_explicit_camera_pairs(
        cameras, [0, 1], [("0079", "0026")]
    ) == [(0, 1)]
    with pytest.raises(ValueError, match="do not see"):
        MODULE.select_explicit_camera_pairs(cameras, [0, 1], [("0026", "0008")])


def test_calibrated_fundamental_explains_synthetic_correspondences():
    intrinsic = np.asarray(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    )
    camera_a = SimpleNamespace(R=np.eye(3), t=np.zeros(3), _K=intrinsic)
    # Camera B is translated one world unit to the right, so t = -R*C.
    camera_b = SimpleNamespace(
        R=np.eye(3), t=np.asarray([-1.0, 0.0, 0.0]), _K=intrinsic
    )
    world = np.asarray(
        [[0.0, 0.0, 5.0], [0.4, 0.2, 4.0], [-0.5, -0.3, 6.0], [0.8, -0.1, 7.0]]
    )

    def project(camera):
        local = (camera.R @ world.T).T + camera.t
        homogeneous = (intrinsic @ local.T).T
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    points_a = project(camera_a)
    points_b = project(camera_b)
    fundamental = MODULE.calibrated_fundamental(camera_a, camera_b)
    a = np.column_stack([points_a, np.ones(len(points_a))])
    b = np.column_stack([points_b, np.ones(len(points_b))])
    residual = np.abs(np.sum(b * (a @ fundamental.T), axis=1))

    assert np.max(residual) < 1e-10


def test_signed_camera_axis_search_is_complete_and_contains_opengl_to_opencv():
    transforms = MODULE.signed_camera_axis_transforms()

    assert len(transforms) == 48
    matrices = [matrix for _label, matrix in transforms]
    assert any(np.array_equal(matrix, np.diag([1.0, -1.0, -1.0])) for matrix in matrices)
    assert all(np.allclose(matrix @ matrix.T, np.eye(3)) for matrix in matrices)


def test_comparison_cameras_are_aligned_by_folder_not_array_order():
    primary = [
        SimpleNamespace(folder="0000", width=100, height=50),
        SimpleNamespace(folder="0001", width=100, height=50),
    ]
    comparison = [
        SimpleNamespace(folder="0001", width=100, height=50, marker="b"),
        SimpleNamespace(folder="0000", width=100, height=50, marker="a"),
    ]

    aligned = MODULE.align_cameras_by_folder(primary, comparison)

    assert [camera.marker for camera in aligned] == ["a", "b"]


def test_comparison_cameras_reject_incompatible_image_geometry():
    primary = [SimpleNamespace(folder="0000", width=100, height=50)]
    comparison = [SimpleNamespace(folder="0000", width=50, height=100)]

    with pytest.raises(ValueError, match="incompatible image sizes"):
        MODULE.align_cameras_by_folder(primary, comparison)
