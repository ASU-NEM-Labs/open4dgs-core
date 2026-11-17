from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "prep/init/audit-l3mhet-density-grid.py"
EXPORT_PATH = ROOT / "prep/easyvolcap/export-l3mhet-density-grid.py"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load("audit_l3mhet_density_grid", AUDIT_PATH)
EXPORT = _load("export_l3mhet_density_grid", EXPORT_PATH)

CENTER = np.array([1.30, -0.20, 1.10])
RADIUS = 0.09
VOXEL = 0.004


def _sphere_grid(tmp_path: Path) -> Path:
    """An occupancy field that is 1 inside a sphere and 0 outside."""
    lo = CENTER - 0.16
    hi = CENTER + 0.16
    xs, ys, zs = EXPORT.grid_axes(lo, hi, VOXEL)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    distance = np.sqrt(
        (gx - CENTER[0]) ** 2 + (gy - CENTER[1]) ** 2 + (gz - CENTER[2]) ** 2
    )
    # A soft edge so marching cubes has a gradient to interpolate along.
    occupancy = np.clip((RADIUS - distance) / VOXEL + 0.5, 0.0, 1.0)
    path = tmp_path / "grid.npz"
    np.savez_compressed(
        path,
        density=occupancy.astype(np.float16) * 100,
        occupancy=occupancy.astype(np.float16),
        xs=xs,
        ys=ys,
        zs=zs,
        bbox_min=lo,
        bbox_max=hi,
        voxel_size=np.asarray(VOXEL),
        metadata=np.asarray(json.dumps({"format": "l3mhet-density-grid-v1"})),
    )
    return path


def _sphere_points(count: int, seed: int, hemisphere: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    if hemisphere:
        direction[:, 1] = -np.abs(direction[:, 1])
    return CENTER[None] + RADIUS * direction


def _npz(tmp_path: Path, positions: np.ndarray) -> Path:
    path = tmp_path / "roma.npz"
    np.savez_compressed(path, positions=positions.astype(np.float32))
    return path


def test_grid_axes_is_inclusive_and_regular():
    xs, ys, zs = EXPORT.grid_axes(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.25]), 0.05
    )
    assert (xs.size, ys.size, zs.size) == (21, 11, 6)
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(ys), 0.05)


def test_grid_axes_rejects_degenerate_box():
    with pytest.raises(ValueError):
        EXPORT.grid_axes(np.array([1.0, 0, 0]), np.array([0.0, 1, 1]), 0.01)
    with pytest.raises(ValueError):
        EXPORT.grid_axes(np.zeros(3), np.ones(3), 0.0)


def test_isosurface_recovers_world_coordinates(tmp_path: Path):
    """A coordinate-order bug here would silently misplace every seed."""
    grid = AUDIT.load_grid(_sphere_grid(tmp_path))
    verts = AUDIT.isosurface(grid, grid["occupancy"].astype(np.float32), 0.5)
    assert len(verts) > 500
    radius = np.linalg.norm(verts - CENTER[None], axis=1)
    assert np.median(radius) == pytest.approx(RADIUS, abs=VOXEL)
    assert radius.max() - radius.min() < 3 * VOXEL


def test_covered_surface_reports_agreement_not_novelty(tmp_path: Path):
    grid_path = _sphere_grid(tmp_path)
    npz_path = _npz(tmp_path, _sphere_points(20000, seed=0, hemisphere=False))
    args = AUDIT.build_parser().parse_args(
        [
            "--grid-path", str(grid_path),
            "--npz-path", str(npz_path),
            "--thresholds", "0.5",
            "--near-radius", "0.01",
        ]
    )
    entry = AUDIT.audit(args)["thresholds"][0]
    assert entry["agreement_fraction"] > 0.95
    assert entry["agreement_dist_p50"] < 2 * VOXEL
    assert entry["roma_covered_fraction"] > 0.95


def test_uncovered_surface_is_reported_as_novel_and_located(tmp_path: Path):
    """The half RoMa never solved must surface as novel geometry, in place."""
    grid_path = _sphere_grid(tmp_path)
    covered = _sphere_points(20000, seed=1, hemisphere=True)
    npz_path = _npz(tmp_path, covered)
    args = AUDIT.build_parser().parse_args(
        [
            "--grid-path", str(grid_path),
            "--npz-path", str(npz_path),
            "--thresholds", "0.5",
            "--near-radius", "0.01",
        ]
    )
    entry = AUDIT.audit(args)["thresholds"][0]
    assert 0.3 < entry["agreement_fraction"] < 0.7
    assert entry["novel_vertices"] > 100
    # Novel geometry must sit on the side RoMa was missing (+y), not anywhere.
    assert entry["novel_centroid"][1] > CENTER[1]


def test_write_points_emits_only_novel_geometry(tmp_path: Path):
    grid_path = _sphere_grid(tmp_path)
    npz_path = _npz(tmp_path, _sphere_points(20000, seed=2, hemisphere=True))
    output = tmp_path / "seeds.npz"
    args = AUDIT.build_parser().parse_args(
        [
            "--grid-path", str(grid_path),
            "--npz-path", str(npz_path),
            "--thresholds", "0.5",
            "--threshold", "0.5",
            "--near-radius", "0.01",
            "--write-points", str(output),
        ]
    )
    report = AUDIT.audit(args)
    assert report["written"]["points"] > 100
    with np.load(output, allow_pickle=False) as archive:
        seeds = np.asarray(archive["positions"], dtype=np.float64)
    assert len(seeds) == report["written"]["points"]
    # Every emitted seed lies on the sphere and away from RoMa's hemisphere.
    radius = np.linalg.norm(seeds - CENTER[None], axis=1)
    assert np.abs(radius - RADIUS).max() < 3 * VOXEL
    assert seeds[:, 1].mean() > CENTER[1]


def test_grid_without_roma_overlap_is_rejected(tmp_path: Path):
    """Without overlap there is nothing to validate against, so refuse."""
    grid_path = _sphere_grid(tmp_path)
    npz_path = _npz(tmp_path, _sphere_points(100, seed=3, hemisphere=False) + 10.0)
    args = AUDIT.build_parser().parse_args(
        ["--grid-path", str(grid_path), "--npz-path", str(npz_path)]
    )
    with pytest.raises(ValueError, match="nothing to validate"):
        AUDIT.audit(args)
