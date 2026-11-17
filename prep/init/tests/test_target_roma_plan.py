from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "prep/init/build-target-roma-plan.py"
ROMA_PATH = ROOT / "prep/init/prep-roma-trigs-memory.py"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLAN = _load("build_target_roma_plan", PLAN_PATH)
ROMA = _load("prep_roma_target_plan", ROMA_PATH)

WIDTH, HEIGHT = 640, 480
FOCAL = 600.0
TARGET = np.array([0.0, 0.0, 0.0])
RADIUS = 0.15


def _camera(index: int, position: np.ndarray, height_offset: float = 0.0):
    """A camera at ``position`` looking at the origin, optionally mis-aimed."""
    forward = -position / np.linalg.norm(position)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.stack([right, true_up, forward])
    t = -R @ position
    # Shifting the principal point aims the camera up or down, which is how we
    # reproduce a target that hangs off the top edge.
    return {
        "name": f"{index:04d}",
        "folder": f"{index:04d}",
        "R": R.tolist(),
        "t": t.tolist(),
        "camera_model_name": "PINHOLE",
        "width": WIDTH,
        "height": HEIGHT,
        "params": [FOCAL, FOCAL, WIDTH / 2.0, HEIGHT / 2.0 + height_offset],
    }


def _reference(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"cameras": rows}))
    return path


def _ring(count: int, radius: float = 3.0, height_offset: float = 0.0):
    rows = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        position = np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)])
        rows.append(_camera(i, position, height_offset))
    return rows


def _args(reference: Path, output: Path, **overrides):
    argv = [
        "--reference-json", str(reference),
        "--output-plan", str(output),
        "--target-center", "0", "0", "0",
        "--target-radius", str(RADIUS),
        "--target-frame", "2",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return PLAN.build_parser().parse_args(argv)


def test_fully_visible_target_reports_full_fraction(tmp_path: Path):
    camera = PLAN.PlanCamera(0, _camera(0, np.array([0.0, 0.0, 3.0])))
    view = PLAN.target_view(camera, TARGET, RADIUS, 0.1)
    assert view is not None
    assert view["visible_fraction"] == pytest.approx(1.0)


def test_target_behind_camera_is_rejected(tmp_path: Path):
    camera = PLAN.PlanCamera(0, _camera(0, np.array([0.0, 0.0, 3.0])))
    view = PLAN.target_view(camera, np.array([0.0, 0.0, 100.0]), RADIUS, 0.1)
    assert view is None


def test_partially_clipped_target_is_measured_not_discarded(tmp_path: Path):
    """The head hangs off the top edge for most of this rig; that must count."""
    # Aiming the camera down pushes the target above the top of the sensor.
    camera = PLAN.PlanCamera(0, _camera(0, np.array([0.0, 0.0, 3.0]), height_offset=-225.0))
    view = PLAN.target_view(camera, TARGET, RADIUS, 0.05)
    assert view is not None
    assert 0.0 < view["visible_fraction"] < 1.0
    assert PLAN.target_view(camera, TARGET, RADIUS, 0.99) is None


def test_crop_stays_inside_the_image_when_target_is_clipped(tmp_path: Path):
    camera = PLAN.PlanCamera(0, _camera(0, np.array([0.0, 0.0, 3.0]), height_offset=-225.0))
    view = PLAN.target_view(camera, TARGET, RADIUS, 0.05)
    assert view is not None
    crop = PLAN.square_crop(camera, view, padding=0.35, minimum=64, maximum=512)
    assert crop is not None
    x0, y0, x1, y1 = crop
    assert 0 <= x0 < x1 <= WIDTH
    assert 0 <= y0 < y1 <= HEIGHT
    assert x1 - x0 == y1 - y0


def test_plan_is_accepted_by_the_roma_loader(tmp_path: Path):
    """The contract that matters: prep-roma must ingest what we emit."""
    reference = _reference(tmp_path, _ring(8))
    output = tmp_path / "plan.json"
    plan = PLAN.build_plan(_args(reference, output, max_view_angle_deg=90))
    output.write_text(json.dumps(plan))
    assert plan["entries"]
    cameras, _ = ROMA.load_reference_export(reference)
    loaded = ROMA.load_depth_guide_plan(output, cameras, reference)
    assert len(loaded["entries"]) == len(plan["entries"])
    assert int(loaded["target_frame"]) == 2


def test_pairs_are_built_from_all_visible_cameras_not_a_neighbour_list(tmp_path: Path):
    """Every viable pair must be considered, which is the point of this planner."""
    reference = _reference(tmp_path, _ring(8))
    plan = PLAN.build_plan(
        _args(reference, tmp_path / "plan.json", max_view_angle_deg=180)
    )
    visible = plan["summary"]["cameras_containing_target"]
    assert visible == 8
    assert plan["summary"]["entries"] == visible * (visible - 1) // 2


def test_view_angle_window_is_enforced(tmp_path: Path):
    reference = _reference(tmp_path, _ring(12))
    plan = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "plan.json",
            min_view_angle_deg=20,
            max_view_angle_deg=40,
            min_pairs_per_camera=0,
        )
    )
    assert plan["entries"]
    for entry in plan["entries"]:
        assert 20 <= entry["view_angle_deg"] <= 40


def test_starved_cameras_are_topped_up_past_the_angle_cap(tmp_path: Path):
    """A camera facing the far side of the target is far in angle from every
    other camera, so the cap starves exactly the views nothing else covers."""
    reference = _reference(tmp_path, _ring(12))
    capped = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "a.json",
            max_view_angle_deg=40,
            min_pairs_per_camera=0,
        )
    )
    topped = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "b.json",
            max_view_angle_deg=40,
            min_pairs_per_camera=5,
        )
    )

    def counts(plan):
        tally: dict[int, int] = {}
        for entry in plan["entries"]:
            for cam in (entry["cam_a"], entry["cam_b"]):
                tally[cam] = tally.get(cam, 0) + 1
        return tally

    assert min(counts(capped).values()) < 5
    assert min(counts(topped).values()) >= 5
    # Only starved cameras get new pairs; the cap still governs the rest.
    assert len(topped["entries"]) > len(capped["entries"])
    assert {(e["cam_a"], e["cam_b"]) for e in capped["entries"]} <= {
        (e["cam_a"], e["cam_b"]) for e in topped["entries"]
    }


def test_min_view_angle_is_never_violated_by_top_up(tmp_path: Path):
    """Topping up may exceed the max, but a degenerate baseline is useless."""
    reference = _reference(tmp_path, _ring(16))
    plan = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "plan.json",
            min_view_angle_deg=30,
            max_view_angle_deg=35,
            min_pairs_per_camera=8,
        )
    )
    assert plan["entries"]
    for entry in plan["entries"]:
        assert entry["view_angle_deg"] >= 30


def test_max_pairs_keeps_the_highest_scoring(tmp_path: Path):
    reference = _reference(tmp_path, _ring(10))
    full = PLAN.build_plan(
        _args(reference, tmp_path / "a.json", max_view_angle_deg=180)
    )
    capped = PLAN.build_plan(
        _args(reference, tmp_path / "b.json", max_view_angle_deg=180, max_pairs=5)
    )
    assert len(capped["entries"]) == 5
    assert [e["id"] for e in capped["entries"]] == [
        e["id"] for e in full["entries"][:5]
    ]


def test_too_few_visible_cameras_is_an_error(tmp_path: Path):
    """Failing loudly beats emitting a plan that cannot form a single pair."""
    reference = _reference(tmp_path, _ring(4, height_offset=-400.0))
    with pytest.raises(ValueError, match="no pair can"):
        PLAN.build_plan(_args(reference, tmp_path / "plan.json"))


def test_excluded_cameras_never_appear_in_the_plan(tmp_path: Path):
    """A camera with bad calibration consumes pairs and returns nothing."""
    reference = _reference(tmp_path, _ring(8))
    plan = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "plan.json",
            max_view_angle_deg=180,
            exclude_cameras=3,
        )
    )
    assert plan["summary"]["cameras_containing_target"] == 7
    for entry in plan["entries"]:
        assert 3 not in (entry["cam_a"], entry["cam_b"])
    assert any(
        row.get("reason") == "excluded" and row["camera_index"] == 3
        for row in plan["cameras_rejected"]
    )


def test_exclusion_redistributes_the_minimum_pair_guarantee(tmp_path: Path):
    """Dropping a camera must not leave its former partners starved."""
    reference = _reference(tmp_path, _ring(10))
    plan = PLAN.build_plan(
        _args(
            reference,
            tmp_path / "plan.json",
            max_view_angle_deg=40,
            min_pairs_per_camera=5,
            exclude_cameras=4,
        )
    )
    tally: dict[int, int] = {}
    for entry in plan["entries"]:
        for cam in (entry["cam_a"], entry["cam_b"]):
            tally[cam] = tally.get(cam, 0) + 1
    assert 4 not in tally
    assert min(tally.values()) >= 5
