import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_detail", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_detail_sphere_reserves_target_share_from_large_background():
    rng = np.random.default_rng(7)
    detail = rng.normal(0.0, 0.03, size=(30, 3))
    background = rng.normal(2.0, 0.4, size=(970, 3))
    positions = np.concatenate([detail, background]).astype(np.float32)
    colors = np.zeros_like(positions)
    priorities = np.arange(len(positions), dtype=np.float32)

    _, _, _, kept_idx, info = MODULE._priority_voxel_to_target_with_detail_sphere(
        positions,
        colors,
        priorities,
        target_points=100,
        min_radius=0.0,
        max_radius=0.0,
        center=np.zeros(3),
        radius=0.15,
        target_fraction=0.25,
    )

    # The voxel search is approximate, but the detail region must receive its
    # reserved quarter instead of competing with 970 background candidates.
    assert info["input_detail"] == 30
    assert info["detail_quota"] == 25
    assert info["kept_detail"] >= 25
    assert 90 <= len(kept_idx) <= 110


class _ProjectingCamera(SimpleNamespace):
    def project_world_point(self, _center):
        return self.uv


def test_detail_sphere_camera_pairs_complete_visible_camera_graph():
    cameras = [
        _ProjectingCamera(width=100, height=80, uv=np.asarray([50.0, 40.0]), center_world=np.asarray([0.0, 0.0, 1.0])),
        _ProjectingCamera(width=100, height=80, uv=np.asarray([15.0, 20.0]), center_world=np.asarray([0.1, 0.0, 1.0])),
        _ProjectingCamera(width=100, height=80, uv=np.asarray([95.0, 20.0]), center_world=np.asarray([-1.0, 0.0, 0.0])),
        _ProjectingCamera(width=100, height=80, uv=None, center_world=np.asarray([0.0, 0.0, -1.0])),
    ]

    visible, pairs = MODULE.choose_detail_sphere_camera_pairs(
        cameras,
        np.zeros(3),
        min_image_margin_px=10.0,
    )

    assert visible == [0, 1]
    assert pairs == [(0, 1)]


def test_detail_sphere_camera_pairs_reject_opposite_surface_views():
    cameras = [
        _ProjectingCamera(
            width=100,
            height=80,
            uv=np.asarray([50.0, 40.0]),
            center_world=np.asarray([0.0, 0.0, 2.0]),
        ),
        _ProjectingCamera(
            width=100,
            height=80,
            uv=np.asarray([50.0, 40.0]),
            center_world=np.asarray([1.0, 0.0, 2.0]),
        ),
        _ProjectingCamera(
            width=100,
            height=80,
            uv=np.asarray([50.0, 40.0]),
            center_world=np.asarray([0.0, 0.0, -2.0]),
        ),
    ]

    visible, pairs = MODULE.choose_detail_sphere_camera_pairs(
        cameras,
        np.zeros(3),
        max_view_angle_deg=45.0,
    )

    assert visible == [0, 1, 2]
    assert pairs == [(0, 1)]


def test_camera_undistorted_pixel_plane_round_trip():
    camera = MODULE.Camera(
        name="test",
        folder="0000",
        R=np.eye(3),
        t=np.zeros(3),
        camera_model_name="OPENCV",
        width=1920,
        height=1080,
        params=np.asarray(
            [1200.0, 1180.0, 960.0, 540.0, -0.22, 0.06, 0.001, -0.002]
        ),
    )
    ideal_px = np.asarray(
        [[960.0, 540.0], [300.0, 140.0], [1650.0, 900.0]], dtype=np.float64
    )
    xy = np.column_stack(
        [
            (ideal_px[:, 0] - camera._K[0, 2]) / camera._K[0, 0],
            (ideal_px[:, 1] - camera._K[1, 2]) / camera._K[1, 1],
            np.ones(len(ideal_px)),
        ]
    )
    distorted, _ = MODULE.cv2.projectPoints(
        xy.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera._K,
        camera._D,
    )
    recovered = camera.undistort_image_points_pixel(distorted.reshape(-1, 2))

    # OpenCV's iterative undistortion is sub-hundredth-pixel here, including
    # the deliberately strong distortion near the image corners.
    assert np.allclose(recovered, ideal_px, atol=1e-2)


def test_calibrated_pair_epipolar_distance_accepts_true_geometry():
    params = np.asarray([1000.0, 1000.0, 960.0, 540.0])
    camera_a = MODULE.Camera(
        name="a", folder="0000", R=np.eye(3), t=np.zeros(3),
        camera_model_name="PINHOLE", width=1920, height=1080, params=params,
    )
    camera_b = MODULE.Camera(
        name="b", folder="0001", R=np.eye(3), t=np.asarray([-1.0, 0.0, 0.0]),
        camera_model_name="PINHOLE", width=1920, height=1080, params=params,
    )
    world = np.asarray(
        [[0.0, 0.0, 5.0], [0.4, 0.2, 4.0], [-0.5, -0.3, 6.0]],
        dtype=np.float64,
    )
    points_a = np.stack([camera_a.project_world_point(point) for point in world])
    points_b = np.stack([camera_b.project_world_point(point) for point in world])

    residual = MODULE.calibrated_pair_epipolar_distance_px(
        camera_a, camera_b, points_a, points_b
    )

    assert np.max(residual) < 1e-8


def test_projected_detail_crop_stays_square_and_inside_image():
    camera = MODULE.Camera(
        name="crop-test",
        folder="0000",
        R=np.eye(3),
        t=np.zeros(3),
        camera_model_name="PINHOLE",
        width=1920,
        height=1080,
        params=np.asarray([1000.0, 1000.0, 960.0, 540.0]),
    )
    bounds = MODULE.project_detail_sphere_crop_bounds(
        camera,
        np.asarray([-4.0, -2.0, 5.0]),
        0.5,
        scale=1.25,
        min_size_px=256,
    )

    assert bounds is not None
    x0, y0, x1, y1 = bounds
    assert 0 <= x0 < x1 <= camera.width
    assert 0 <= y0 < y1 <= camera.height
    assert x1 - x0 == y1 - y0
