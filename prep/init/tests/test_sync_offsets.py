import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_sync", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _camera(folder: str, name: str):
    return MODULE.Camera(
        name=name,
        folder=folder,
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        camera_model_name="PINHOLE",
        width=1920,
        height=1080,
        params=np.asarray([1000.0, 1000.0, 960.0, 540.0], dtype=np.float64),
    )


def test_missing_folder_key_does_not_borrow_zero_based_camera_index():
    cameras = [
        _camera("01", "Camera 1"),
        _camera("02", "Camera 2"),
        _camera("03", "Camera 3"),
    ]
    sync_data = {"0001": 0.001, "0002": 0.002}

    offset, key = MODULE._lookup_sync_offset_for_camera(sync_data, cameras, 2)

    assert offset == 0.0
    assert key is None


def test_padded_folder_key_is_resolved():
    cameras = [_camera("03", "Camera 3")]

    offset, key = MODULE._lookup_sync_offset_for_camera(
        {"0003": 0.003}, cameras, 0
    )

    assert offset == pytest.approx(0.003)
    assert key == "0003"


def test_duplicate_sync_key_mapping_is_rejected(tmp_path):
    cameras = [
        _camera("01", "shared"),
        _camera("02", "shared"),
    ]
    sync_path = tmp_path / "sync.json"
    sync_path.write_text(json.dumps({"shared": 0.001}), encoding="utf-8")

    with pytest.raises(ValueError, match="matched multiple cameras"):
        MODULE.load_sync_offsets_for_cameras(str(sync_path), cameras)
