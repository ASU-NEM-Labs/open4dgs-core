import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "solve_4dv_dynamic_calibration.py"
SPEC = importlib.util.spec_from_file_location("solve_4dv_dynamic_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _camera_rows():
    centers = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.5]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    base = []
    extrinsics = []
    intrinsics = []
    for index, center in enumerate(centers):
        rotation = np.eye(3)
        translation = -rotation @ center
        base.append({
            "name": f"JCam{index:03d}",
            "folder": f"{index:04d}",
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "camera_model_name": "OPENCV",
            "width": 1920,
            "height": 1080,
            "params": [1200.0, 1190.0, 959.5, 539.5, 0.1, -0.1, 0.0, 0.0],
        })
        extrinsics.append({
            "camera_id": f"JCam{index:03d}",
            "extrinsic": np.column_stack([rotation, translation]).tolist(),
        })
        focal_x = 1500.0 + 100.0 * index
        focal_y = 1490.0 + 100.0 * index
        intrinsics.append({
            "camera_id": f"JCam{index:03d}",
            "intrinsic": [
                [focal_x, 0.0, 959.5],
                [0.0, focal_y, 539.5],
                [0.0, 0.0, 1.0],
            ],
        })
    return base, extrinsics, intrinsics


def test_effective_prior_preserves_base_distortion_and_centres():
    base_cameras, snapshot_extrinsics, capture_intrinsics = _camera_rows()
    base = {"cameras": base_cameras}
    snapshot = {"extri": {"data": snapshot_extrinsics}}
    capture = {"intri": {"data": capture_intrinsics}}

    prior, diagnostics = MODULE.build_effective_prior(base, capture, snapshot)

    assert diagnostics["common_cameras"] == 4
    assert all(camera["camera_model_name"] == "OPENCV" for camera in prior["cameras"])
    assert prior["cameras"][0]["params"] == base_cameras[0]["params"]
    assert prior["cameras"][3]["params"] == base_cameras[3]["params"]
    assert prior["dynamic_calibration_prior"]["projection_source"] == (
        "base_reference_full_distortion_model"
    )
    assert prior["dynamic_calibration_prior"]["optical_centres_fixed"] is True
    assert prior["dynamic_calibration_prior"]["pose_source"] == (
        "base_reference_image_calibration"
    )


def test_snapshot_pose_prior_is_explicit_not_default():
    base_cameras, snapshot_extrinsics, capture_intrinsics = _camera_rows()
    theta = np.deg2rad(12.0)
    snapshot_rotation = np.asarray([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    for row in snapshot_extrinsics:
        matrix = np.asarray(row["extrinsic"], dtype=np.float64)
        center = MODULE.camera_center(matrix[:, :3], matrix[:, 3])
        row["extrinsic"] = np.column_stack([
            snapshot_rotation, -snapshot_rotation @ center,
        ]).tolist()
    base = {"cameras": base_cameras}
    snapshot = {"extri": {"data": snapshot_extrinsics}}
    capture = {"intri": {"data": capture_intrinsics}}

    default_prior, _ = MODULE.build_effective_prior(base, capture, snapshot)
    snapshot_prior, _ = MODULE.build_effective_prior(
        base, capture, snapshot, pose_prior_source="snapshot"
    )

    assert np.allclose(default_prior["cameras"][0]["R"], np.eye(3))
    assert not np.allclose(snapshot_prior["cameras"][0]["R"], np.eye(3))


def test_connected_components_reports_isolated_cameras():
    components = MODULE.connected_components(6, [(0, 1), (1, 2), (3, 4)])
    assert components == [[0, 1, 2], [3, 4], [5]]


def test_calibrated_angular_residual_is_zero_for_consistent_rays():
    points = np.asarray(
        [[0.2, 0.3, 4.0], [0.7, -0.1, 3.0], [-0.2, 0.5, 5.0]],
        dtype=np.float64,
    )
    center_a = np.asarray([0.0, 0.0, 0.0])
    center_b = np.asarray([1.0, 0.0, 0.0])
    rays_a = points - center_a
    rays_b = points - center_b
    rays_a /= np.linalg.norm(rays_a, axis=1, keepdims=True)
    rays_b /= np.linalg.norm(rays_b, axis=1, keepdims=True)
    packed = {
        "camera_a": np.zeros(len(points), dtype=np.int64),
        "camera_b": np.ones(len(points), dtype=np.int64),
        "rays_a": rays_a,
        "rays_b": rays_b,
        "baseline": np.repeat([[1.0, 0.0, 0.0]], len(points), axis=0),
        "focal": np.full(len(points), 1000.0),
    }

    correct = MODULE.residuals_numpy(np.stack([np.eye(3), np.eye(3)]), packed)
    theta = np.deg2rad(5.0)
    wrong_b = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    wrong = MODULE.residuals_numpy(np.stack([np.eye(3), wrong_b]), packed)

    assert np.max(correct) < 1e-10
    assert np.median(wrong) > 10.0


def test_packed_with_centers_recomputes_each_observation_baseline():
    packed = {
        "camera_a": np.asarray([0, 2, 0], dtype=np.int64),
        "camera_b": np.asarray([1, 1, 2], dtype=np.int64),
        "baseline": np.zeros((3, 3), dtype=np.float64),
        "rays_a": np.zeros((3, 3), dtype=np.float64),
    }
    centers = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-2.0, 1.0, 4.0],
    ])

    updated = MODULE.packed_with_centers(packed, centers)

    assert np.allclose(updated["baseline"], np.asarray([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, -1.0],
        [-2.0, 1.0, 4.0],
    ]))
    assert updated["rays_a"] is packed["rays_a"]
    assert np.allclose(packed["baseline"], 0.0)


def test_packed_with_calibration_applies_only_selected_focal_scale():
    rays = np.asarray([
        [0.2, 0.1, 1.0],
        [-0.3, 0.2, 1.0],
    ])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    packed = {
        "camera_a": np.asarray([0, 1], dtype=np.int64),
        "camera_b": np.asarray([1, 0], dtype=np.int64),
        "baseline": np.zeros((2, 3), dtype=np.float64),
        "rays_a": rays.copy(),
        "rays_b": rays[::-1].copy(),
        "focal": np.asarray([1000.0, 1000.0]),
    }
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scales = np.asarray([[2.0, 4.0], [1.0, 1.0]])

    updated = MODULE.packed_with_calibration(packed, centers, scales)

    adjusted_xy = updated["rays_a"][:, :2] / updated["rays_a"][:, 2:3]
    assert np.allclose(adjusted_xy[0], [0.1, 0.025])
    assert np.allclose(adjusted_xy[1], [-0.3, 0.2])
    assert np.allclose(updated["focal"], 1000.0 * (8.0 ** 0.25))


def test_failed_camera_neighbor_selection_uses_training_consistency():
    centers = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    points = np.asarray([
        [0.2, 0.3, 4.0], [0.4, -0.2, 3.5],
        [-0.3, 0.1, 5.0], [0.1, 0.5, 4.5],
    ])
    rays_0 = points / np.linalg.norm(points, axis=1, keepdims=True)
    rays_1 = points - centers[1]
    rays_1 /= np.linalg.norm(rays_1, axis=1, keepdims=True)
    rays_2 = np.repeat([[1.0, 0.0, 0.01]], len(points), axis=0)
    rays_2 /= np.linalg.norm(rays_2, axis=1, keepdims=True)
    blocks = []
    for neighbor, neighbor_rays in ((1, rays_1), (2, rays_2)):
        blocks.append(MODULE.MatchBlock(
            frame=1, camera_a=0, camera_b=neighbor,
            rays_a=rays_0, rays_b=neighbor_rays,
            certainty=np.ones(len(points)),
            train_mask=np.ones(len(points), dtype=bool),
            raw_matches=len(points), fitted_ratio=1.0,
            relative_rotation=None, relative_translation=None, pose_inliers=0,
        ))
    matrix = np.diag([1000.0, 1000.0, 1.0])
    train = MODULE.pack_observations(
        blocks, centers, [matrix] * 3, train=True
    )
    args = argparse.Namespace(
        failed_pose_min_neighbor_inlier_ratio=0.5,
        failed_pose_max_neighbor_p90_px=8.0,
        failed_pose_min_neighbors=1,
    )

    keep, report = MODULE.select_consistent_failed_camera_neighbors(
        np.stack([np.eye(3)] * 3), centers, train, blocks,
        ["0000", "0001", "0002"], [0], args,
    )

    assert report["per_camera"][0]["selected_neighbors"] == ["0001"]
    assert np.all(train["pair_id"][keep] == 0)


def test_failed_camera_neighbor_selection_never_forces_bad_edge():
    centers = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    ray = np.asarray([[0.0, 0.0, 1.0]])
    bad_rays = (
        np.asarray([[0.0, 1.0, 0.01]]),
        np.asarray([[1.0, 0.0, 0.01]]),
    )
    bad_rays = tuple(
        value / np.linalg.norm(value, axis=1, keepdims=True)
        for value in bad_rays
    )
    blocks = [
        MODULE.MatchBlock(
            frame=1, camera_a=0, camera_b=neighbor,
            rays_a=ray, rays_b=bad_ray,
            certainty=np.ones(1), train_mask=np.asarray([True]),
            raw_matches=1, fitted_ratio=1.0,
            relative_rotation=None, relative_translation=None, pose_inliers=0,
        )
        for neighbor, bad_ray in zip((1, 2), bad_rays)
    ]
    matrix = np.diag([1000.0, 1000.0, 1.0])
    train = MODULE.pack_observations(blocks, centers, [matrix] * 3, train=True)
    args = argparse.Namespace(
        failed_pose_min_neighbor_inlier_ratio=0.9,
        failed_pose_max_neighbor_p90_px=8.0,
        failed_pose_min_neighbors=2,
    )

    keep, report = MODULE.select_consistent_failed_camera_neighbors(
        np.stack([np.eye(3)] * 3), centers, train, blocks,
        ["0000", "0001", "0002"], [0], args,
    )

    row = report["per_camera"][0]
    assert row["selected_neighbors"] == []
    assert row["minimum_neighbors_satisfied"] is False
    assert not np.any(keep)


def test_validation_rejects_a_solution_that_does_not_improve():
    points = np.asarray([[0.2, 0.3, 4.0], [0.7, -0.1, 3.0]], dtype=np.float64)
    rays_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    shifted = points - np.asarray([1.0, 0.0, 0.0])
    rays_b = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    validation = {
        "camera_a": np.zeros(2, dtype=np.int64),
        "camera_b": np.ones(2, dtype=np.int64),
        "rays_a": rays_a,
        "rays_b": rays_b,
        "baseline": np.repeat([[1.0, 0.0, 0.0]], 2, axis=0),
        "focal": np.full(2, 1000.0),
        "weight": np.ones(2),
        "pair_id": np.zeros(2, dtype=np.int64),
    }
    block = MODULE.MatchBlock(
        frame=1,
        camera_a=0,
        camera_b=1,
        rays_a=rays_a,
        rays_b=rays_b,
        certainty=np.ones(2),
        train_mask=np.asarray([True, False]),
        raw_matches=2,
        fitted_ratio=1.0,
        relative_rotation=None,
        relative_translation=None,
        pose_inliers=0,
    )
    args = argparse.Namespace(
        camera_pairs=(),
        min_camera_validation_observations=1,
        min_camera_validation_inlier_ratio=0.25,
        min_explicit_pair_validation_inlier_ratio=0.35,
        min_validation_inlier_ratio=0.5,
        max_validation_p90_px=8.0,
        min_validation_improvement=0.15,
    )
    rotations = np.stack([np.eye(3), np.eye(3)])

    report = MODULE.validation_report(
        rotations, rotations, validation, [block], ["0000", "0001"], args
    )

    assert report["gates"]["overall_inlier_ratio"] is True
    assert report["gates"]["material_improvement"] is False
    assert report["passed"] is False


def test_validation_rejects_camera_with_bad_p90_despite_enough_inliers():
    # Nine perfect held-out observations and one severe outlier give a 90%
    # 2.5 px inlier ratio, but the interpolated p90 remains well above 5 px.
    # The old ratio-only camera gate would publish this camera.
    points = np.repeat([[0.2, 0.3, 4.0]], 10, axis=0)
    rays_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    shifted = points - np.asarray([1.0, 0.0, 0.0])
    rays_b = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    rays_b[-1] = np.asarray([0.0, 1.0, 0.01])
    rays_b[-1] /= np.linalg.norm(rays_b[-1])
    validation = {
        "camera_a": np.zeros(10, dtype=np.int64),
        "camera_b": np.ones(10, dtype=np.int64),
        "rays_a": rays_a,
        "rays_b": rays_b,
        "baseline": np.repeat([[1.0, 0.0, 0.0]], 10, axis=0),
        "focal": np.full(10, 1000.0),
        "weight": np.ones(10),
        "pair_id": np.zeros(10, dtype=np.int64),
    }
    block = MODULE.MatchBlock(
        frame=1, camera_a=0, camera_b=1,
        rays_a=rays_a, rays_b=rays_b, certainty=np.ones(10),
        train_mask=np.zeros(10, dtype=bool), raw_matches=10,
        fitted_ratio=1.0, relative_rotation=None,
        relative_translation=None, pose_inliers=0,
    )
    args = argparse.Namespace(
        camera_pairs=(), min_camera_validation_observations=1,
        min_camera_validation_inlier_ratio=0.65,
        max_camera_validation_p90_px=5.0,
        min_explicit_pair_validation_inlier_ratio=0.0,
        min_validation_inlier_ratio=0.0,
        max_validation_p90_px=10000.0,
        min_validation_improvement=0.0,
    )
    rotations = np.stack([np.eye(3), np.eye(3)])

    report = MODULE.validation_report(
        rotations, rotations, validation, [block], ["0000", "0001"], args
    )

    assert report["failed_cameras"]
    assert "p90" in report["failed_cameras"][0]["failure_reasons"]
    assert report["gates"]["per_camera"] is False


def test_validation_does_not_pass_a_pruned_required_explicit_pair():
    ray = np.asarray([[0.0, 0.0, 1.0]])
    validation = {
        "camera_a": np.asarray([0], dtype=np.int64),
        "camera_b": np.asarray([2], dtype=np.int64),
        "rays_a": ray,
        "rays_b": ray,
        "baseline": np.asarray([[1.0, 0.0, 0.0]]),
        "focal": np.asarray([1000.0]),
        "weight": np.ones(1),
        "pair_id": np.asarray([1], dtype=np.int64),
    }
    blocks = [
        MODULE.MatchBlock(
            frame=1, camera_a=0, camera_b=1, rays_a=ray, rays_b=ray,
            certainty=np.ones(1), train_mask=np.asarray([True]),
            raw_matches=1, fitted_ratio=1.0, relative_rotation=None,
            relative_translation=None, pose_inliers=0,
        ),
        MODULE.MatchBlock(
            frame=1, camera_a=0, camera_b=2, rays_a=ray, rays_b=ray,
            certainty=np.ones(1), train_mask=np.asarray([False]),
            raw_matches=1, fitted_ratio=1.0, relative_rotation=None,
            relative_translation=None, pose_inliers=0,
        ),
    ]
    args = argparse.Namespace(
        camera_pairs=(("0000", "0001"),),
        min_camera_validation_observations=1,
        min_camera_validation_inlier_ratio=0.0,
        min_explicit_pair_validation_inlier_ratio=0.0,
        min_validation_inlier_ratio=0.0,
        max_validation_p90_px=10000.0,
        min_validation_improvement=0.0,
    )
    rotations = np.stack([np.eye(3)] * 3)

    report = MODULE.validation_report(
        rotations, rotations, validation, blocks,
        ["0000", "0001", "0002"], args,
    )

    assert report["gates"]["explicit_pairs"] is False
    assert report["missing_explicit_validation_pairs"] == [["0000", "0001"]]


def test_pair_pose_wahba_initialization_recovers_absolute_rotations():
    centers = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [0.0, 1.0, 0.2],
        [-0.5, -0.4, 0.8],
    ])
    rotations = []
    for degrees in (5.0, -7.0, 11.0, -4.0):
        theta = np.deg2rad(degrees)
        rotations.append(np.asarray([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]))
    rotations = np.stack(rotations)
    blocks = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            relative_rotation = rotations[right] @ rotations[left].T
            relative_translation = rotations[right] @ (
                centers[left] - centers[right]
            )
            relative_translation /= np.linalg.norm(relative_translation)
            blocks.append(MODULE.MatchBlock(
                frame=1,
                camera_a=left,
                camera_b=right,
                rays_a=np.zeros((1, 3)),
                rays_b=np.zeros((1, 3)),
                certainty=np.ones(1),
                train_mask=np.asarray([True]),
                raw_matches=100,
                fitted_ratio=0.8,
                relative_rotation=relative_rotation,
                relative_translation=relative_translation,
                pose_inliers=80,
            ))

    # One high-confidence pair decomposition is deliberately wrong.  The
    # synchronized graph should reject it through cycle consistency instead of
    # letting it define a camera through a spanning-tree path.
    bad_theta = np.deg2rad(70.0)
    bad_rotation = np.asarray([
        [np.cos(bad_theta), -np.sin(bad_theta), 0.0],
        [np.sin(bad_theta), np.cos(bad_theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    for block_index in (1,):
        blocks[block_index].relative_rotation = (
            bad_rotation @ blocks[block_index].relative_rotation
        )
        blocks[block_index].relative_translation *= -1.0

    # Deliberately supply an unusable prior.  Pair-pose initialization must be
    # image/geometry-derived, rather than using the old rotations to choose an
    # Essential-matrix translation sign.
    bad_prior = np.repeat(np.diag([-1.0, -1.0, 1.0])[None, :, :], 4, axis=0)
    initialized, diagnostics = MODULE.initialize_rotations_from_pair_poses(
        blocks, centers, bad_prior, 40
    )

    assert diagnostics["initialized_cameras"] == 4
    for actual, expected in zip(initialized, rotations):
        assert MODULE.rotation_delta_degrees(actual, expected) < 0.1


def test_solved_colmap_uses_the_same_poses_and_projection_as_reference():
    base_cameras, _, capture_intrinsics = _camera_rows()
    reference = {"cameras": []}
    colmap = MODULE._load_colmap_io_module()
    input_cameras = {}
    input_images = {}
    for index, (base_camera, intrinsic_row) in enumerate(
        zip(base_cameras, capture_intrinsics), start=1
    ):
        folder = base_camera["folder"]
        theta = np.deg2rad(float(index))
        rotation = np.asarray([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.asarray([0.1 * index, -0.2 * index, 0.3 * index])
        reference["cameras"].append({
            **base_camera,
            "R": rotation.tolist(),
            "t": translation.tolist(),
        })
        input_cameras[index] = colmap.Camera(
            id=index,
            model="PINHOLE",
            width=1920,
            height=1080,
            params=np.asarray([1000.0, 1000.0, 959.5, 539.5]),
        )
        input_images[index] = colmap.Image(
            id=index,
            qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            camera_id=index,
            name=f"{folder}/000001.jpg",
            xys=np.zeros((0, 2)),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
    with tempfile.TemporaryDirectory() as temp_name:
        input_path = Path(temp_name) / "input"
        output_path = Path(temp_name) / "output"
        input_path.mkdir()
        colmap.write_model(input_cameras, input_images, {}, str(input_path), ext=".bin")

        summary = MODULE.write_solved_colmap(input_path, output_path, reference)
        written_cameras, written_images, _ = colmap.read_model(
            str(output_path), ext=".bin"
        )

    assert summary["cameras"] == 4
    by_folder = {
        Path(image.name).parent.name: image for image in written_images.values()
    }
    reference_by_folder = {camera["folder"]: camera for camera in reference["cameras"]}
    for folder, image in by_folder.items():
        camera = written_cameras[image.camera_id]
        expected = reference_by_folder[folder]
        assert np.allclose(image.qvec2rotmat(), expected["R"], atol=1e-10)
        assert np.allclose(image.tvec, expected["t"], atol=1e-12)
        assert camera.model == "OPENCV"
        assert np.allclose(camera.params, expected["params"], atol=1e-12)
