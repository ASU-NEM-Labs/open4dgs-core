"""make_timing_candidate.py on a synthetic readout-aware sync run: solved cameras come from the re-solve,
cameras it could not place come from an optional sync.json or start at 0 (timecode)."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "make_timing_candidate.py"


def _resolve_dir(tmp_path: Path, censored: float = 0.0) -> Path:
    d = tmp_path / "sync_run"
    d.mkdir()
    solved = {"0001": 0.0, "0002": 0.0005, "0003": -0.0010}           # seconds, PLUS convention
    (d / "sync_candidate.json").write_text(json.dumps(solved))
    (d / "sync_robust_report.json").write_text(json.dumps({
        "verdict": "EXCELLENT", "cameras_demoted": [], "unconstrained_cameras": ["0004"],
        "per_camera": {k: {"degree": 4, "sigma_ms": 0.1, "bias_ms": 0.0} for k in solved},
        "split_half": {"uncertainty_ms": {k: 0.1 for k in solved}},
    }))
    (d / "sync_pipeline_report.json").write_text(json.dumps({"phase_measurement": {
        "censored_phase_fraction": censored,
        "visual_report": {"rolling_shutter": {"mode": "fixed"},
                          "offsets_plus_sec": {**solved, "0004": 0.0}},
    }}))
    return d


def _run(tmp_path: Path, *extra: str):
    out = tmp_path / "timing.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--resolve-dir", str(tmp_path / "sync_run"),
                        "--readout-ms", "15", "--weak-cameras", "--out", str(out), *extra],
                       capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_without_sync_json_unplaced_cameras_start_at_zero(tmp_path):
    _resolve_dir(tmp_path)
    r, t = _run(tmp_path, "--unresolved-rs")
    assert r.returncode == 0, r.stderr
    cams = t["cameras"]
    assert sorted(cams) == ["0001", "0002", "0003", "0004"]
    gauge = t["gauge_camera"]
    assert cams[gauge]["c_ms"] == 0.0
    assert cams["0002"]["c_ms"] - cams["0003"]["c_ms"] == 1.5           # solved differences kept
    assert cams["0004"]["status"] == "unresolved" and cams["0004"]["c_ms"] == 0.0 and cams["0004"]["rs"] is True
    assert "timecode" in r.stdout


def test_copied_candidate_as_sync_json_gives_the_same_file(tmp_path):
    d = _resolve_dir(tmp_path)
    _, plain = _run(tmp_path, "--unresolved-rs", "--gauge", "01")
    (tmp_path / "timing.json").unlink()
    (tmp_path / "timing.report.json").unlink()
    r, copied = _run(tmp_path, "--unresolved-rs", "--gauge", "01", "--sync-json", str(d / "sync_candidate.json"))
    assert r.returncode == 0, r.stderr
    assert {k: v["c_ms"] for k, v in copied["cameras"].items()} == {k: v["c_ms"] for k, v in plain["cameras"].items()}


def test_sync_json_value_is_used_for_an_unplaced_camera(tmp_path):
    _resolve_dir(tmp_path)
    sync = tmp_path / "sync.json"
    sync.write_text(json.dumps({"0001": 0.0, "0004": 0.002}))
    r, t = _run(tmp_path, "--sync-json", str(sync), "--gauge", "01")
    assert r.returncode == 0, r.stderr
    assert t["cameras"]["0004"]["c_ms"] == 2.0 and t["cameras"]["0004"]["rs"] is False


def test_failed_gate_writes_nothing_unless_waived(tmp_path):
    _resolve_dir(tmp_path, censored=0.38)
    r, t = _run(tmp_path)
    assert t is None and "GATES FAILED" in r.stdout
    (tmp_path / "timing.report.json").unlink()
    r, t = _run(tmp_path, "--waive-gate", "G2_censoring")
    assert r.returncode == 0 and t["experimental"] is True and t["waiver"]["waived_gates"] == ["G2_censoring"]
