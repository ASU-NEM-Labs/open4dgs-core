import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "roma_backends.py"
SPEC = importlib.util.spec_from_file_location("roma_backends_spatial_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_directional_spatial_sampler_reserves_every_grid_cell():
    h = w = 8
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    warp = torch.stack(
        [
            (xx + 0.5) * (2.0 / w) - 1.0,
            (yy + 0.5) * (2.0 / h) - 1.0,
        ],
        dim=-1,
    ).unsqueeze(0)
    # Make the bottom-right globally dominant. Spatial anchor locations must
    # still cover all quadrants rather than following confidence.
    overlap = (yy * 10.0 + xx).view(1, h, w, 1)

    src, dst, cert = MODULE.RoMaV2Backend._directional_spatial_samples(
        warp,
        overlap,
        num=8,
        grid=2,
        h_src=80,
        w_src=80,
        h_dst=80,
        w_dst=80,
    )

    assert len(src) == len(dst) == len(cert) == 8
    quadrants = (src[:, 1] >= 40).astype(np.int32) * 2 + (src[:, 0] >= 40)
    assert np.bincount(quadrants, minlength=4).tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(src, dst, atol=1e-5)


def test_directional_spatial_anchors_do_not_change_with_pair_confidence():
    h = w = 8
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    warp = torch.stack(
        [
            (xx + 0.5) * (2.0 / w) - 1.0,
            (yy + 0.5) * (2.0 / h) - 1.0,
        ],
        dim=-1,
    ).unsqueeze(0)
    overlap_a = (yy * 10.0 + xx).view(1, h, w, 1)
    overlap_b = torch.flip(overlap_a, dims=(1, 2))

    src_a, _, _ = MODULE.RoMaV2Backend._directional_spatial_samples(
        warp, overlap_a, num=12, grid=2,
        h_src=80, w_src=80, h_dst=80, w_dst=80,
    )
    src_b, _, _ = MODULE.RoMaV2Backend._directional_spatial_samples(
        warp, overlap_b, num=12, grid=2,
        h_src=80, w_src=80, h_dst=80, w_dst=80,
    )

    np.testing.assert_array_equal(src_a, src_b)


def test_directional_spatial_sampler_handles_nondivisible_quota():
    warp = torch.zeros((1, 6, 6, 2), dtype=torch.float32)
    overlap = torch.arange(36, dtype=torch.float32).view(1, 6, 6, 1)

    src, dst, cert = MODULE.RoMaV2Backend._directional_spatial_samples(
        warp,
        overlap,
        num=7,
        grid=2,
        h_src=60,
        w_src=60,
        h_dst=60,
        w_dst=60,
    )

    assert src.shape == (7, 2)
    assert dst.shape == (7, 2)
    assert cert.shape == (7,)
