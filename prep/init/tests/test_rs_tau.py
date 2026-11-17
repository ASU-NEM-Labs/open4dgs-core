"""Prep's rolling-shutter capture-time term (rs_tau_frames): legacy content reference unchanged, midpoint
reference = the shared trigs.capture_timing convention, cache fingerprint unchanged when no timing file is used."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "prep-roma-trigs-memory.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prep_roma_trigs_rs_tau", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_content_reference_matches_legacy_formula():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1080, 500); residual = 0.123; sign = -1.0; readout = 0.42; W = 1080.0; ref = 611.3
    legacy = residual + (sign * readout * (u - ref) / max(W, 1.0))   # the pre-change expression
    new = MODULE.rs_tau_frames(residual, sign, readout, u, W, "content", content_ref=ref)
    assert np.max(np.abs(new - legacy)) < 1e-12


def test_midpoint_reference_is_the_shared_convention():
    v = np.array([0.0, 270.0, 540.0, 810.0, 1080.0, -10.0, 1100.0])
    c_frames, readout_frames = 0.12, 0.9          # 2 ms and 15 ms at 60 fps
    tau = MODULE.rs_tau_frames(c_frames, 1.0, readout_frames, v, 1080.0, "midpoint")
    frac = np.clip(v / 1080.0 - 0.5, -0.5, 0.5)
    assert np.allclose(tau, c_frames + readout_frames * frac)
    assert tau[2] == pytest.approx(c_frames)       # the middle row carries exactly the camera constant


def test_zero_sign_or_readout_leaves_only_the_residual():
    v = np.linspace(0, 1080, 7)
    assert np.all(MODULE.rs_tau_frames(0.3, 0.0, 0.9, v, 1080.0, "midpoint") == 0.3)
    assert np.all(MODULE.rs_tau_frames(0.3, 1.0, 0.0, v, 1080.0, "content", content_ref=500.0) == 0.3)


def test_axis_index_and_reference_validation():
    assert MODULE._rs_axis_index("column") == 0 and MODULE._rs_axis_index("row") == 1
    with pytest.raises(ValueError):
        MODULE._rs_axis_index("diagonal")
    with pytest.raises(ValueError):
        MODULE.rs_tau_frames(0.0, 1.0, 0.9, np.zeros(3), 1080.0, "image")


def test_fingerprint_has_timing_hash_only_when_used(tmp_path):
    ref = tmp_path / "reference_export.json"; ref.write_text("{}")
    timing = tmp_path / "timing.json"; timing.write_text("{}")
    base = dict(reference_json=str(ref), sync_json=None, rotation_json=None, depth_guide_plan=None, motion_budget=False)
    _, desc0 = MODULE.build_run_fingerprint(argparse.Namespace(**base, timing_json=None))
    _, desc1 = MODULE.build_run_fingerprint(argparse.Namespace(**base, timing_json=str(timing)))
    assert "timing_sha256" not in desc0 and "capture_timing_sha256" not in desc0
    assert "timing_sha256" in desc1 and "capture_timing_sha256" in desc1
