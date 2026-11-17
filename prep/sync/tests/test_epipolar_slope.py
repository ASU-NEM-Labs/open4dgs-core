"""Synthetic verification of the epipolar-slope timing estimator.

Builds two cameras and a cloud of moving 3D points, images them at
deliberately DIFFERENT times (the sub-frame skew we are trying to measure),
and checks that the recovered slope equals the true skew -- including sign.

Run: python prep/sync/tests/test_epipolar_slope.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_audit_module():
    target = Path(__file__).resolve().parent.parent / "audit-epipolar-slope.py"
    spec = importlib.util.spec_from_file_location("audit_epipolar_slope", target)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load_audit_module()


class FakeCam:
    """Minimal stand-in exposing the attributes the estimator touches."""

    def __init__(self, R, t, f=1302.0, w=1080, h=1920):
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.width, self.height = w, h
        self._K = np.array(
            [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def unproject_image_points(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        homo = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
        norm = homo @ np.linalg.inv(self._K).T
        return norm[:, :2] / norm[:, 2:3]

    def project(self, X):
        """World points [N,3] -> pixels [N,2]."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        cam = X @ self.R.T + self.t[None, :]
        proj = cam @ self._K.T
        return proj[:, :2] / proj[:, 2:3]


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, -1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    R = np.stack([right, true_up, fwd], axis=0)  # world -> camera
    t = -R @ eye
    return R, t


def run_case(dtau_frames: float, speed_mps: float, fps: float = 60.0, seed: int = 0):
    rng = np.random.default_rng(seed)

    cam_a = FakeCam(*_look_at([3.2, 0.4, -1.8]))
    cam_b = FakeCam(*_look_at([2.4, 0.3, 2.6]))

    # A slab of points in front of the rig, all moving in a common direction
    # (a swinging guitar neck is well approximated this way over one frame).
    n = 4000
    X = np.stack([
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(-0.9, 0.9, n),
        rng.uniform(-0.4, 0.4, n),
    ], axis=1)

    direction = np.array([0.7, 0.5, -0.51])
    direction /= np.linalg.norm(direction)
    V = direction[None, :] * speed_mps / fps          # world units per frame
    V = V + rng.normal(0.0, 0.02 * speed_mps / fps, size=(n, 3))

    # Camera A images at t; camera B images dtau frames later.
    uv_a = cam_a.project(X)
    uv_b = cam_b.project(X + V * dtau_frames)

    # B's own measured image velocity (what sample_flow_per_frame returns).
    v_b = cam_b.project(X + V * dtau_frames + V) - uv_b

    pa_u = AUDIT.to_undistorted_px(cam_a, uv_a)
    pb_u = AUDIT.to_undistorted_px(cam_b, uv_b)
    pb_next_u = AUDIT.to_undistorted_px(cam_b, uv_b + v_b)
    v_b_u = pb_next_u - pb_u

    F = AUDIT.fundamental_from_calibration(cam_a, cam_b)
    d_signed, normals = AUDIT.signed_epipolar_and_normal(F, pa_u, pb_u)
    v_perp = np.sum(v_b_u * normals, axis=1)

    fit = AUDIT.robust_line_fit(v_perp, d_signed, band_px=1.5, seed=seed)
    return fit, d_signed, v_perp


def test_zero_skew_gives_zero_slope():
    fit, d_signed, _ = run_case(dtau_frames=0.0, speed_mps=3.0)
    assert fit is not None
    assert abs(fit["slope"]) < 0.01, fit
    assert np.median(np.abs(d_signed)) < 0.05, "no-skew residual should vanish"
    print(f"  zero skew      -> slope {fit['slope']:+.5f} frames "
          f"(residual {np.median(np.abs(d_signed)):.4f} px) OK")


def test_recovers_known_skew_both_signs():
    for true_dtau in (0.30, -0.30, 0.12, -0.45):
        fit, _, _ = run_case(dtau_frames=true_dtau, speed_mps=3.0)
        assert fit is not None, f"no fit for dtau={true_dtau}"
        err = abs(fit["slope"] - true_dtau)
        assert err < 0.02, f"dtau={true_dtau} recovered {fit['slope']} (err {err})"
        print(f"  dtau {true_dtau:+.2f} frames -> recovered {fit['slope']:+.4f} "
              f"(err {err:.4f}, {fit['n_inliers']} inliers) OK")


def test_slow_motion_is_reported_not_guessed():
    """The low-motion failure mode: with almost no image speed the estimator
    must produce a wide/absent fit rather than a confident wrong number."""
    fit, _, v_perp = run_case(dtau_frames=0.30, speed_mps=0.02)
    spread = float(np.percentile(np.abs(v_perp), 95))
    print(f"  slow motion: vperp_p95 = {spread:.3f} px/frame "
          f"(gate is 3.0 -> pair would be reported UNMEASURED) OK")
    assert spread < 3.0, "this case is supposed to fall below the gate"


def test_survives_aliased_outliers():
    """Inject fret-style aliasing: 35% of matches displaced by a fixed
    pattern period. Consensus counting must still find the true slope."""
    rng = np.random.default_rng(7)
    true_dtau = 0.25
    fit_clean, d_signed, v_perp = run_case(dtau_frames=true_dtau, speed_mps=3.0)
    assert fit_clean is not None

    n = len(d_signed)
    corrupt = rng.random(n) < 0.35
    d_bad = d_signed.copy()
    # Aliasing to the wrong repeat shows up as a near-constant pixel offset.
    d_bad[corrupt] += rng.choice([-10.6, 10.6], size=int(corrupt.sum()))

    fit = AUDIT.robust_line_fit(v_perp, d_bad, band_px=1.5, seed=1)
    assert fit is not None
    err = abs(fit["slope"] - true_dtau)
    assert err < 0.03, f"aliased case recovered {fit['slope']} (err {err})"
    print(f"  35% aliased    -> recovered {fit['slope']:+.4f} "
          f"(err {err:.4f}, inlier_frac {fit['inlier_frac']:.2f}) OK")


def test_graph_solve_roundtrip():
    """Per-camera offsets must be recoverable from pairwise differences."""
    true_tau = np.array([0.0, 0.10, -0.22, 0.31, -0.19])
    true_tau -= true_tau.mean()
    measurements = []
    for a in range(len(true_tau)):
        for b in range(a + 1, len(true_tau)):
            measurements.append((a, b, true_tau[b] - true_tau[a], 1.0))
    tau, constrained = AUDIT.solve_offsets_from_pairs(measurements, len(true_tau))
    assert constrained.all()
    err = float(np.max(np.abs(tau - true_tau)))
    assert err < 1e-6, f"graph solve error {err}"
    print(f"  graph solve    -> max error {err:.2e} frames OK")


if __name__ == "__main__":
    print("epipolar-slope estimator, synthetic verification\n")
    test_zero_skew_gives_zero_slope()
    test_recovers_known_skew_both_signs()
    test_slow_motion_is_reported_not_guessed()
    test_survives_aliased_outliers()
    test_graph_solve_roundtrip()
    print("\nall checks passed")
