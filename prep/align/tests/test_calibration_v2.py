import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "calibration_v2.py"
SPEC = importlib.util.spec_from_file_location("calibration_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DRIVER_SCRIPT = Path(__file__).resolve().parents[1] / "run_calibration_v2.py"
DRIVER_SPEC = importlib.util.spec_from_file_location("run_calibration_v2", DRIVER_SCRIPT)
DRIVER = importlib.util.module_from_spec(DRIVER_SPEC)
sys.modules[DRIVER_SPEC.name] = DRIVER
DRIVER_SPEC.loader.exec_module(DRIVER)


def _camera(params=(800.0, 810.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        model=SimpleNamespace(name="OPENCV"),
        params=np.asarray(params, dtype=np.float64),
    )


def _make_database(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
          height INTEGER, params BLOB, prior_focal_length INTEGER);
        CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
        CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
        CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
        CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
        CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER,
          data BLOB, config INTEGER, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
        """
    )
    for image_id in (1, 2, 3):
        points = np.asarray(
            [[20.0 + image_id, 30.0], [100.0, 100.0], [200.0, 150.0], [210.0, 160.0]],
            dtype=np.float32,
        )
        con.execute("INSERT INTO cameras VALUES(?,?,?,?,?,?)", (
            image_id, 4, 640, 480,
            np.asarray([800, 800, 320, 240, 0, 0, 0, 0], dtype=np.float64).tobytes(), 1,
        ))
        con.execute("INSERT INTO images VALUES(?,?,?)", (image_id, f"{image_id:04d}/000001.jpg", image_id))
        con.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (image_id, 4, 2, points.tobytes()))
        descriptors = np.zeros((4, 128), dtype=np.uint8)
        con.execute("INSERT INTO descriptors VALUES(?,?,?,?)", (image_id, 4, 128, descriptors.tobytes()))

    def add_pair(a, b, rows):
        values = np.asarray(rows, dtype=np.uint32)
        pair_id = MODULE.pair_id_from_image_ids(a, b)
        payload = (pair_id, len(values), 2, values.tobytes())
        con.execute("INSERT INTO matches VALUES(?,?,?,?)", payload)
        con.execute(
            "INSERT INTO two_view_geometries VALUES(?,?,?,?,?,?,?,?,?,?)",
            (*payload, 2, None, None, None, None, None),
        )

    # keypoint 0 is a complete three-view cycle. Keypoint 1 is only a chain.
    # keypoint 2 creates a duplicate-camera component and must be rejected.
    add_pair(1, 2, [[0, 0], [1, 1], [2, 2], [3, 2]])
    add_pair(2, 3, [[0, 0], [1, 1], [2, 2]])
    add_pair(1, 3, [[0, 0]])
    con.commit()
    con.close()


def test_pose_convention_projection_and_triangulation():
    camera = _camera()
    point = np.asarray([0.15, -0.1, 4.0])
    rotations = [np.eye(3), np.eye(3), np.eye(3)]
    translations = [np.zeros(3), np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, -0.8, 0.0])]
    observations = []
    for rotation, translation in zip(rotations, translations):
        pixel, valid = MODULE.project_opencv(point[None], rotation, translation, camera)
        assert valid[0]
        normalized = MODULE.undistort_pixels(pixel, camera)[0]
        observations.append((rotation, translation, normalized))
    recovered = MODULE.triangulate_normalized(observations)
    assert np.allclose(recovered, point, atol=1e-9)
    wrong_pixel, _ = MODULE.project_opencv(
        point[None], rotations[1], translations[1] + np.asarray([0.05, 0.0, 0.0]), camera
    )
    correct_pixel, _ = MODULE.project_opencv(point[None], rotations[1], translations[1], camera)
    assert np.linalg.norm(wrong_pixel - correct_pixel) > 5.0


def test_nominal_seed_registers_every_database_camera(tmp_path):
    pytest.importorskip("pycolmap")
    database = tmp_path / "database.db"
    _make_database(database)
    specs = {
        f"{image_id:04d}": MODULE.CameraSpec(
            folder=f"{image_id:04d}",
            width=640,
            height=480,
            model="OPENCV",
            params=(800, 800, 320, 240, 0, 0, 0, 0),
            rotation=np.eye(3),
            translation=np.asarray([float(image_id), 0.0, 0.0]),
        )
        for image_id in (1, 2, 3)
    }

    reconstruction = MODULE.build_nominal_reconstruction(database, specs)

    assert reconstruction.num_images() == 3
    assert len(reconstruction.cameras) == 3
    for image_id in (1, 2, 3):
        pose = MODULE.image_cam_from_world(reconstruction.images[image_id])
        assert np.asarray(pose.translation) == pytest.approx([image_id, 0.0, 0.0])


def test_clean_plate_core_pose_transplant_preserves_full_camera_set(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    database = tmp_path / "database.db"
    _make_database(database)
    specs = {
        f"{image_id:04d}": MODULE.CameraSpec(
            folder=f"{image_id:04d}", width=640, height=480, model="OPENCV",
            params=(800, 800, 320, 240, 0, 0, 0, 0),
            rotation=np.eye(3), translation=np.zeros(3),
        )
        for image_id in (1, 2, 3)
    }
    destination = MODULE.build_nominal_reconstruction(database, specs)
    source = pycolmap.Reconstruction()
    source.add_camera_with_trivial_rig(MODULE.create_opencv_camera(
        2, 640, 480, specs["0002"].params
    ))
    source.add_image_with_trivial_frame(pycolmap.Image(
        name="0002/000001.jpg", camera_id=2, image_id=2
    ))
    source.images[2].frame.rig_from_world = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.eye(3)), np.asarray([1.5, -0.2, 0.1])
    )
    source.register_frame(source.images[2].frame_id)

    report = MODULE.transplant_registered_poses(source, destination)

    assert destination.num_images() == 3
    assert report["transplanted_folders"] == ["0002"]
    assert np.asarray(MODULE.image_cam_from_world(
        destination.images[2]
    ).translation) == pytest.approx([1.5, -0.2, 0.1])


def test_distortion_round_trip_and_invertibility_gate():
    camera = _camera((800, 810, 320, 240, 0.08, -0.02, 0.001, -0.001))
    world = np.asarray([[0.2, -0.1, 3.0], [-0.8, 0.5, 4.0], [1.0, 0.7, 5.0]])
    pixels, _ = MODULE.project_opencv(world, np.eye(3), np.zeros(3), camera)
    normalized = MODULE.undistort_pixels(pixels, camera)
    assert np.allclose(normalized, world[:, :2] / world[:, 2:3], atol=2e-7)
    assert MODULE.distortion_grid_invertible(camera.params, 640, 480)
    assert not MODULE.distortion_grid_invertible(
        [800, 810, 320, 240, -4.0, 0.0, 0.0, 0.0], 640, 480
    )


def test_crop_coordinate_mapping_round_trip():
    transform = MODULE.crop_transform((100, 50, 900, 650), 1000)
    points = np.asarray([[100, 50], [500, 300], [899.5, 649.5]], dtype=np.float64)
    cropped = MODULE.transform_points(points, transform)
    recovered = MODULE.transform_points(cropped, np.linalg.inv(transform))
    assert np.allclose(recovered, points, atol=1e-10)


def test_exact_track_construction_rejects_chain_and_duplicate_camera(tmp_path):
    database = tmp_path / "database.db"
    _make_database(database)
    tracks, report = MODULE.build_exact_tracks(database, min_views=3, require_cycle=True)
    assert len(tracks) == 1
    assert tracks[0].observations == ((1, 0), (2, 0), (3, 0))
    assert report["non_cycle_components"] == 1
    assert report["duplicate_camera_components"] == 1


def test_split_is_deterministic_ratio_faithful_and_never_repaired():
    keypoints = {image_id: np.asarray([[320.0, 240.0]] * 100) for image_id in (1, 2, 3)}
    sizes = {image_id: (640, 480) for image_id in (1, 2, 3)}

    def tracks():
        return [
            MODULE.ExactTrack(f"{index:024x}", tuple((image_id, index) for image_id in (1, 2, 3)), ())
            for index in range(100)
        ]

    first = tracks()
    first_report = MODULE.assign_track_splits(
        first, keypoints, sizes, seed=7,
        min_validation_per_camera=20, min_test_per_camera=20,
    )
    second = tracks()
    MODULE.assign_track_splits(
        second, keypoints, sizes, seed=7,
        min_validation_per_camera=20, min_test_per_camera=20,
    )
    assert first_report["tracks"] == {"fit": 70, "validation": 15, "test": 15}
    assert [row.split for row in first] == [row.split for row in second]
    assert not first_report["ready_for_model_selection"]
    assert not first_report["ready_for_final_test"]


def test_track_serialization_and_fit_filter_are_split_isolated(tmp_path):
    database = tmp_path / "database.db"
    _make_database(database)
    tracks, _ = MODULE.build_exact_tracks(database, min_views=2, require_cycle=False)
    for index, track in enumerate(tracks):
        track.split = "fit" if index == 0 else "test"
    con = sqlite3.connect(database)
    keypoints = MODULE.read_keypoints(con)
    con.close()
    names = {image_id: f"{image_id:04d}/000001.jpg" for image_id in (1, 2, 3)}
    sizes = {image_id: (640, 480) for image_id in (1, 2, 3)}
    path = tmp_path / "tracks.npz"
    MODULE.save_tracks(path, tracks, keypoints, names, sizes, {"source": "synthetic"})
    store = MODULE.load_tracks(path)
    assert store.metadata["source"] == "synthetic"
    assert len(list(store.iter_split("fit"))) == 1
    report = MODULE.filter_database_to_fit_tracks(database, tmp_path / "fit.db", tracks)
    assert report["verified_fit_matches"] > 0
    con = sqlite3.connect(tmp_path / "fit.db")
    kept = sum(row[0] for row in con.execute("SELECT rows FROM two_view_geometries"))
    con.close()
    assert kept == report["verified_fit_matches"]


def test_topology_fit_keeps_rejected_evidence_but_removes_holdout_nodes(tmp_path):
    database = tmp_path / "database.db"
    _make_database(database)
    tracks, _ = MODULE.build_exact_tracks(database, min_views=2, require_cycle=False)
    assert len(tracks) == 2
    for track in tracks:
        track.split = "validation" if (1, 0) in track.observations else "fit"
    destination = tmp_path / "topology-fit.db"

    report = MODULE.filter_database_to_topology_fit(database, destination, tracks)

    con = sqlite3.connect(destination)
    try:
        pair_id = MODULE.pair_id_from_image_ids(1, 2)
        rows, blob = con.execute(
            "SELECT rows,data FROM two_view_geometries WHERE pair_id=?", (pair_id,)
        ).fetchone()
    finally:
        con.close()
    kept = np.frombuffer(blob, dtype=np.uint32).reshape(rows, 2)
    assert (0, 0) not in {tuple(row) for row in kept}
    assert {(1, 1), (2, 2), (3, 2)}.issubset({tuple(row) for row in kept})
    assert report["removed_holdout_verified_matches"] == 3


def test_apriltag_corners_are_appended_without_geometry(tmp_path):
    database = tmp_path / "database.db"
    _make_database(database)
    root = tmp_path / "reference_images"
    for image_id in (1, 2, 3):
        folder = root / f"{image_id:04d}"
        folder.mkdir(parents=True)
        canvas = np.full((480, 640), 255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 180
        )
        canvas[140:320, 230:410] = marker
        assert cv2.imwrite(str(folder / "000001.jpg"), canvas)
    report = MODULE.augment_apriltag_features(database, root)
    assert report["geometry_assumed"] is False
    assert report["detections"] == 3
    assert report["corners_appended"] == 12
    con = sqlite3.connect(database)
    rows = con.execute("SELECT rows FROM keypoints WHERE image_id=1").fetchone()[0]
    descriptor_rows = con.execute("SELECT rows FROM descriptors WHERE image_id=1").fetchone()[0]
    con.close()
    assert rows == descriptor_rows == 8


def test_paired_bootstrap_requires_safe_p90_improvement():
    ids = np.asarray([f"{index:024x}" for index in range(200)])
    baseline = np.linspace(0.2, 2.0, len(ids))
    improved = baseline * 0.75
    result = MODULE.paired_bootstrap_p90_improvement(
        baseline, improved, ids, ids, seed=11, samples=500
    )
    assert result["accepted"]
    harmed = MODULE.paired_bootstrap_p90_improvement(
        baseline, baseline + 0.2, ids, ids, seed=11, samples=500
    )
    assert not harmed["accepted"]


def test_synthetic_focal_and_radial_perturbation_recovery(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    rng = np.random.default_rng(17)
    xyz = np.column_stack([
        rng.uniform(-1.5, 1.5, 180),
        rng.uniform(-1.0, 1.0, 180),
        rng.uniform(3.0, 6.0, 180),
    ])
    true_params = np.asarray([820.0, 795.0, 320.0, 240.0, 0.065, -0.018, 0.0, 0.0])
    true_camera = _camera(true_params)
    pixels, valid = MODULE.project_opencv(xyz, np.eye(3), np.zeros(3), true_camera)
    assert np.all(valid)
    model = tmp_path / "model"
    model.mkdir()
    seeded_params = [760.0, 750.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]
    (model / "cameras.txt").write_text(
        "# Camera list\n1 OPENCV 640 480 " + " ".join(map(str, seeded_params)) + "\n",
        encoding="utf-8",
    )
    image_entries = " ".join(
        f"{pixels[index, 0]} {pixels[index, 1]} {index + 1}" for index in range(len(xyz))
    )
    (model / "images.txt").write_text(
        "# Image list\n1 1 0 0 0 0 0 0 1 0001/000001.jpg\n" + image_entries + "\n",
        encoding="utf-8",
    )
    point_rows = ["# 3D point list"]
    for index, point in enumerate(xyz):
        point_rows.append(
            f"{index + 1} {point[0]} {point[1]} {point[2]} 128 128 128 0 1 {index}"
        )
    (model / "points3D.txt").write_text("\n".join(point_rows) + "\n", encoding="utf-8")
    reconstruction = pycolmap.Reconstruction(model)
    spec = MODULE.CameraSpec(
        folder="0001", width=640, height=480, model="OPENCV",
        params=tuple(true_params), rotation=np.eye(3), translation=np.zeros(3),
    )
    focal_report = MODULE.refine_intrinsics_blockwise(
        reconstruction, {"0001": spec}, "focal",
        focal_log_sigma=1.0,
    )
    report = MODULE.refine_intrinsics_blockwise(
        reconstruction, {"0001": spec}, "radial",
        focal_log_sigma=1.0, radial_sigma=10.0,
    )
    recovered = np.asarray(reconstruction.cameras[1].params)
    assert focal_report["0001"]["status"] == "updated"
    assert report["0001"]["status"] == "updated"
    assert recovered[:2] == pytest.approx(true_params[:2], abs=0.15)
    assert recovered[4:6] == pytest.approx(true_params[4:6], abs=2e-4)
    assert report["0001"]["fit_p90_after_px"] < 0.01


def test_synthetic_timestamp_offset_recovery():
    pycolmap = pytest.importorskip("pycolmap")
    camera = pycolmap.Camera.create_from_model_name(1, "OPENCV", 800, 640, 480)
    camera.params = np.asarray([800, 810, 320, 240, 0.01, -0.002, 0, 0])
    times = np.linspace(0.0, 2.0, 81)
    landmarks = np.empty((len(times), 4, 3), dtype=np.float64)
    for landmark in range(4):
        landmarks[:, landmark, 0] = -0.5 + 0.3 * landmark + 0.25 * np.sin(2.1 * times + landmark)
        landmarks[:, landmark, 1] = -0.3 + 0.2 * landmark + 0.18 * np.cos(1.7 * times + 0.2 * landmark)
        landmarks[:, landmark, 2] = 4.0 + 0.1 * np.sin(times + landmark)
    observation_times = np.repeat(np.linspace(0.3, 1.7, 24), 4)
    landmark_indices = np.tile(np.arange(4), 24)
    true_offset = 0.037
    from scipy.interpolate import CubicSpline
    sampled = CubicSpline(times, landmarks, axis=0)(observation_times + true_offset)
    points = sampled[np.arange(len(observation_times)), landmark_indices]
    pixels, _ = MODULE.project_opencv(points, np.eye(3), np.zeros(3), camera)
    result = MODULE.estimate_time_offset_from_trajectories(
        times, landmarks, observation_times, landmark_indices, pixels,
        np.eye(3), np.zeros(3), camera,
        initial_offset=0.0, offset_bounds=(-0.1, 0.1),
    )
    assert result["success"]
    assert result["offset_seconds"] == pytest.approx(true_offset, abs=2e-5)
    assert result["median_after_px"] < 1e-5

    theta = 0.025
    perturbed_rotation = cv2.Rodrigues(np.asarray([0.0, theta, -0.01]))[0]
    joint = MODULE.refine_pose_and_time_from_trajectories(
        times, landmarks, observation_times, landmark_indices, pixels,
        perturbed_rotation, np.asarray([0.04, -0.03, 0.02]), camera,
        initial_offset=0.0, offset_bounds=(-0.1, 0.1),
        rotation_prior_sigma_rad=1.0, translation_prior_sigma=5.0,
        offset_prior_sigma_sec=1.0,
    )
    assert joint["success"]
    assert joint["offset_seconds"] == pytest.approx(true_offset, abs=2e-4)
    assert np.asarray(joint["t"]) == pytest.approx(np.zeros(3), abs=2e-3)
    assert joint["median_after_px"] < 0.05
    assert joint["certifies_static_calibration"] is False


def test_audit_recognizes_relocated_mount_and_refreshes_pristine_metadata(tmp_path):
    data = tmp_path / "linux-data"
    reference_root = data / "reference_images"
    for index in range(96):
        path = reference_root / f"{index:04d}" / "000001.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"plate-{index}".encode())
    nominal = data / "reference_export-4dv-manifest.json"
    timestamps = data / "frame_timestamps.json"
    capture = data / "capture.json"
    snapshot = data / "record.json"
    nominal.write_text('{"kind":"nominal"}', encoding="utf-8")
    timestamps.write_text('{"kind":"timestamps"}', encoding="utf-8")
    capture.write_text('{"kind":"capture"}', encoding="utf-8")
    snapshot.write_text('{"kind":"snapshot"}', encoding="utf-8")
    out = data / "runs" / "calibration-v2"
    out.mkdir(parents=True)
    old_manifest = MODULE.input_manifest(
        reference_root, {"nominal_reference": nominal, "timestamps": timestamps}
    )
    MODULE.atomic_write_json(out / "input-manifest.json", old_manifest)
    MODULE.atomic_write_json(out / DRIVER.RUN_MARKER, {
        "schema_version": 1,
        "pipeline": "clean-plate-calibration-v2",
        "data_dir": "/Users/example/mounted/data",
        "reference_root": "/Users/example/mounted/data/reference_images",
        "nominal_reference": "/Users/example/mounted/data/reference_export-4dv-manifest.json",
        "policy": {},
    })
    args = SimpleNamespace(
        data_dir=str(data), out_dir=str(out), nominal_reference=str(nominal),
        capture_manifest=str(capture), snapshot_record=str(snapshot),
        timestamps=None, force_rehash=False,
    )

    DRIVER.command_audit(args)

    marker = json.loads((out / DRIVER.RUN_MARKER).read_text(encoding="utf-8"))
    refreshed = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert marker["schema_version"] == 2
    assert marker["data_dir"] == str(data.resolve())
    assert marker["path_aliases"][0]["data_dir"] == "/Users/example/mounted/data"
    assert set(refreshed["declared_inputs"]) == {
        "nominal_reference", "timestamps", "capture_manifest", "snapshot_record"
    }
    aggregate = refreshed["aggregate_sha256"]
    args.capture_manifest = None
    args.snapshot_record = None
    DRIVER.command_audit(args)
    sticky = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert sticky["aggregate_sha256"] == aggregate
    assert set(sticky["declared_inputs"]) == {
        "nominal_reference", "timestamps", "capture_manifest", "snapshot_record"
    }


def test_colmap_option_selection_supports_legacy_and_modern_namespaces():
    modern = []
    selected = DRIVER.append_compatible_option(
        modern,
        {"--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"},
        ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"), 1,
    )
    assert selected == "--FeatureExtraction.use_gpu"
    assert modern == ["--FeatureExtraction.use_gpu", "1"]
    legacy = []
    selected = DRIVER.append_compatible_option(
        legacy, {"--SiftExtraction.use_gpu"},
        ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"), 1,
    )
    assert selected == "--SiftExtraction.use_gpu"
    assert legacy == ["--SiftExtraction.use_gpu", "1"]
    with pytest.raises(RuntimeError, match="required options"):
        DRIVER.append_compatible_option([], set(), ("--Missing.option",), 1)


def test_incremental_options_accept_legacy_api_without_structureless_fallback():
    class LegacyOptions:
        def __init__(self):
            self.multiple_models = True
            self.min_model_size = 10
            self.ba_refine_focal_length = True
            self.ba_refine_principal_point = True
            self.ba_refine_extra_params = True
            self.mapper = SimpleNamespace(
                init_min_num_inliers=0,
                abs_pose_min_num_inliers=0,
                abs_pose_max_error=0.0,
                abs_pose_refine_focal_length=True,
                abs_pose_refine_extra_params=True,
                filter_max_reproj_error=0.0,
                tri_min_angle=0.0,
            )
            self.triangulation = SimpleNamespace(min_angle=0.0)

        def check(self):
            return True

    fake_pycolmap = SimpleNamespace(IncrementalPipelineOptions=LegacyOptions)
    args = SimpleNamespace(
        mapper_seed=7,
        init_min_inliers=50,
        abs_pose_min_inliers=20,
        abs_pose_max_error=2.0,
        mapper_filter_error=2.0,
        min_tri_angle=1.0,
    )
    options, report = DRIVER.configure_incremental_options(fake_pycolmap, args)

    assert report["structure_less_registration_fallback"] is None
    assert options.multiple_models is False
    assert options.mapper.abs_pose_refine_focal_length is False
    assert options.mapper.abs_pose_refine_extra_params is False
    assert options.ba_refine_focal_length is False
    assert options.triangulation.min_angle == 1.0


def test_targeted_pair_reimport_removes_only_affected_pairs(tmp_path):
    database = tmp_path / "targeted.db"
    _make_database(database)
    affected = MODULE.pair_id_from_image_ids(1, 2)
    untouched = MODULE.pair_id_from_image_ids(2, 3)

    count = DRIVER.targeted.clear_pairs_for_reimport(database, [affected, affected])

    assert count == 1
    con = sqlite3.connect(database)
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM matches WHERE pair_id=?", (affected,)
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE pair_id=?", (affected,)
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM matches WHERE pair_id=?", (untouched,)
        ).fetchone()[0] == 1
        assert con.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE pair_id=?", (untouched,)
        ).fetchone()[0] == 1
    finally:
        con.close()


def test_targeted_verification_merge_never_discards_prior_inliers(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    _make_database(source)
    shutil.copyfile(source, destination)
    pair_id = MODULE.pair_id_from_image_ids(1, 2)
    newly_verified = np.asarray([[0, 0], [3, 3]], dtype=np.uint32)
    con = sqlite3.connect(destination)
    con.execute(
        "UPDATE two_view_geometries SET rows=?,cols=2,data=? WHERE pair_id=?",
        (len(newly_verified), newly_verified.tobytes(), pair_id),
    )
    con.commit()
    con.close()

    report = DRIVER.targeted.merge_verified_pair_rows(source, destination, [pair_id])

    con = sqlite3.connect(destination)
    row = con.execute(
        "SELECT rows,data FROM two_view_geometries WHERE pair_id=?", (pair_id,)
    ).fetchone()
    con.close()
    merged = np.frombuffer(row[1], dtype=np.uint32).reshape(row[0], 2)
    assert {tuple(value) for value in merged} == {
        (0, 0), (1, 1), (2, 2), (3, 2), (3, 3)
    }
    assert report["prior_verified_rows_preserved"] == 4
    assert report["new_verified_rows_added"] == 1
