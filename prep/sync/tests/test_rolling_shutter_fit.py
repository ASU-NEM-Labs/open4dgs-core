"""Synthetic verification of the rolling-shutter readout estimator.

Simulates a rig where each camera has BOTH a per-camera sub-frame offset and a
per-column readout with a per-camera sign, then checks that the joint fit
recovers rho and does not invent one when RS is absent.

Run: python prep/sync/tests/test_rolling_shutter_fit.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name: str, filename: str):
    target = Path(__file__).resolve().parent.parent / filename
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RS = _load("audit_rolling_shutter", "audit-rolling-shutter.py")
AUDIT = sys.modules["audit_epipolar_slope"]

FPS = 60.0
W = 1080.0


def simulate(rho_ms: float, n_pairs: int = 12, n_frames: int = 4,
             n_match: int = 900, noise_px: float = 0.4, seed: int = 0):
    """Build observations exactly as stage 1 would record them."""
    rng = np.random.default_rng(seed)
    rho_frames = rho_ms / 1000.0 * FPS

    n_cams = 2 * n_pairs
    signs = np.where(rng.random(n_cams) < 0.68, 1.0, -1.0)   # ~30 L / 14 R split
    tau = rng.normal(0.0, 0.0015, n_cams) * FPS              # ~1.5 ms, in frames

    P, FR, UA, UB, VP, Y, SA, SB = [], [], [], [], [], [], [], []
    for pid in range(n_pairs):
        ia, ib = 2 * pid, 2 * pid + 1
        for fi in range(n_frames):
            # Subject occupies a different column band each frame, and the two
            # cameras frame it differently -- this is what gives the column
            # term leverage inside a group.
            ca = rng.uniform(250, 830)
            cb = rng.uniform(250, 830)
            u_a = np.clip(rng.normal(ca, 110, n_match), 0, W - 1)
            u_b = np.clip(rng.normal(cb, 110, n_match), 0, W - 1)
            v_perp = rng.normal(0.0, 6.0, n_match)
            v_perp[np.abs(v_perp) < 1.0] = 1.5      # keep usable motion

            g = (signs[ib] * (u_b - W / 2) - signs[ia] * (u_a - W / 2)) / W
            dt = (tau[ib] - tau[ia]) + rho_frames * g
            d = dt * v_perp + rng.normal(0.0, noise_px, n_match)

            P.append(np.full(n_match, pid)); FR.append(np.full(n_match, fi))
            UA.append(u_a); UB.append(u_b); VP.append(v_perp); Y.append(d)
            SA.append(np.full(n_match, signs[ia])); SB.append(np.full(n_match, signs[ib]))

    cat = lambda xs: np.concatenate(xs)
    return (cat(P).astype(np.int64), cat(FR).astype(np.int64), cat(UA), cat(UB),
            cat(VP), cat(Y), cat(SA), cat(SB))


def run(rho_ms: float, **kw):
    pid, fr, u_a, u_b, v_perp, y, s_a, s_b = simulate(rho_ms, **kw)
    key = pid * 1_000_000 + fr
    _, group = np.unique(key, return_inverse=True)
    n_groups = int(group.max()) + 1
    g = (s_b * (u_b - W / 2) - s_a * (u_a - W / 2)) / W
    x_rho = g * v_perp
    m0 = RS.joint_fit(group, v_perp, x_rho, y, n_groups, False, 2.0)
    m1 = RS.joint_fit(group, v_perp, x_rho, y, n_groups, True, 2.0)
    est = m1["rho_frames"] / FPS * 1000.0
    se = m1["se_rho_frames"] / FPS * 1000.0
    impr = (1 - m1["resid_robust_rms_px"] / m0["resid_robust_rms_px"]) * 100
    return est, se, impr, m0, m1, (g, group, y, v_perp, s_a, s_b)


def test_recovers_known_readout():
    for true_ms in (7.0, 4.0, 12.0):
        est, se, impr, *_ = run(true_ms)
        err = abs(est - true_ms)
        assert err < 0.5, f"rho={true_ms}: recovered {est:.3f} +/- {se:.3f} (err {err:.3f})"
        assert impr > 1.0, f"rho={true_ms}: residual barely improved ({impr:.2f}%)"
        print(f"  rho {true_ms:5.1f} ms -> recovered {est:+7.3f} +/- {se:.3f} ms "
              f"(err {err:.3f}, resid {impr:+5.1f}%) OK")


def test_no_rs_reports_zero():
    est, se, impr, *_ = run(0.0)
    assert abs(est) < 3.0 * max(se, 1e-9) or abs(est) < 0.4, \
        f"invented rho={est:.3f} +/- {se:.3f} with no RS present"
    print(f"  no RS         -> recovered {est:+7.3f} +/- {se:.3f} ms "
          f"({abs(est)/max(se,1e-9):.1f} sigma, resid {impr:+5.1f}%) OK")


def test_permutation_null_separates_signal():
    """The real null test: scrambling the per-camera l/r signs must destroy the
    signal. This is what the old same-vs-opposite residual check failed to do."""
    rng = np.random.default_rng(5)
    pid, fr, u_a, u_b, v_perp, y, s_a, s_b = simulate(7.0, n_pairs=20, seed=5)
    key = pid * 1_000_000 + fr
    _, group = np.unique(key, return_inverse=True)
    n_groups = int(group.max()) + 1

    n_cams = 2 * 20
    signs = np.zeros(n_cams)
    cam_a = 2 * pid
    cam_b = 2 * pid + 1
    signs[cam_a] = s_a
    signs[cam_b] = s_b

    def refit(sg):
        gg = (sg[cam_b] * (u_b - W / 2) - sg[cam_a] * (u_a - W / 2)) / W
        f = RS.joint_fit(group, v_perp, gg * v_perp, y, n_groups, True, 2.0, iters=8)
        return abs(f["rho_frames"] / FPS * 1000.0)

    true_rho = refit(signs)
    null = np.array([refit(rng.permutation(signs)) for _ in range(60)])
    pval = float(np.mean(null >= true_rho))
    assert pval < 0.05, f"permutation failed to separate: p={pval:.3f}"
    assert true_rho > 3 * np.median(null), \
        f"true rho {true_rho:.3f} not clear of null median {np.median(null):.3f}"
    print(f"  permutation   -> true |rho| {true_rho:.3f} ms vs null median "
          f"{np.median(null):.3f} ms, p={pval:.4f} OK")


def test_old_residual_crosscheck_has_no_power():
    """Documents WHY the same-vs-opposite residual comparison was removed: the
    sign cancellation lives in the mean of the column term, which per-group
    dtau absorbs, so the leftover scatter is identical either way."""
    _, _, _, m0, _, (g, group, y, v_perp, s_a, s_b) = run(7.0, n_pairs=40, seed=3)
    r0 = y - m0["dtau_frames"][group] * v_perp
    opp = (s_a * s_b) < 0
    ratio = np.sqrt(np.mean(r0[opp] ** 2)) / np.sqrt(np.mean(r0[~opp] ** 2))
    assert 0.9 < ratio < 1.1, f"expected no power, got ratio {ratio:.3f}"
    print(f"  removed check -> opposite/same RMS ratio {ratio:.3f} (~1.0, no power) OK")


def test_dtau_absorbs_per_camera_offsets():
    """rho must be unaffected by large per-camera constants -- that is what
    makes the estimate robust to the sync problem it sits on top of."""
    est_a, se_a, *_ = run(7.0, seed=11)
    # Same scene, 5x larger per-camera offsets.
    rng = np.random.default_rng(11)
    pid, fr, u_a, u_b, v_perp, y, s_a, s_b = simulate(7.0, seed=11)
    key = pid * 1_000_000 + fr
    _, group = np.unique(key, return_inverse=True)
    big = rng.normal(0.0, 0.0075, int(group.max()) + 1) * FPS   # ~7.5 ms per group
    y2 = y + big[group] * v_perp
    g = (s_b * (u_b - W / 2) - s_a * (u_a - W / 2)) / W
    m1 = RS.joint_fit(group, v_perp, g * v_perp, y2, int(group.max()) + 1, True, 2.0)
    est_b = m1["rho_frames"] / FPS * 1000.0
    assert abs(est_a - est_b) < 0.3, \
        f"rho moved {abs(est_a-est_b):.3f} ms when offsets grew (should be ~0)"
    print(f"  offset immunity-> rho {est_a:+.3f} vs {est_b:+.3f} ms with 5x larger "
          f"per-group offsets OK")


if __name__ == "__main__":
    print("rolling-shutter readout estimator, synthetic verification\n")
    test_recovers_known_readout()
    test_no_rs_reports_zero()
    test_permutation_null_separates_signal()
    test_old_residual_crosscheck_has_no_power()
    test_dtau_absorbs_per_camera_offsets()
    print("\nall checks passed")
