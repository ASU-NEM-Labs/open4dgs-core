#!/usr/bin/env python3
"""Suppress off-surface (floater) primitives in a COPY of a TriGS checkpoint.

Selection is identical to audit_checkpoint_floaters.py --criterion npz-distance:
a dynamic primitive whose distance to the NEAREST clean-NPZ point exceeds the
threshold is not on any real surface. Suppression sets its opacity logit to -20
(sigmoid ~ 2e-9). Row count and tensor shapes are unchanged, so the output loads
anywhere the original does, and the original file is never touched.

Purpose: measure what the flagged cohort actually contributes. Load original and
filtered checkpoints side by side; if the subject is unchanged and only the
floaters disappear, the cohort was parasitic and suppression is the answer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    sys.exit("SciPy is required:  uv pip install scipy")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--npz", required=True, type=Path,
                    help="Clean init NPZ (same one the audit used).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dist", type=float, default=None,
                    help="Suppress dynamic rows farther than this (m) from the "
                         "nearest NPZ point.")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Alternative: suppress the N most off-surface dynamic "
                         "rows (same semantics as the audit).")
    args = ap.parse_args()

    if (args.dist is None) == (args.top_n is None):
        sys.exit("Give exactly one of --dist or --top-n")
    if args.checkpoint.resolve() == args.output.resolve():
        sys.exit("Refusing to overwrite --checkpoint; choose a different --output")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    means = params["means"].detach().cpu().numpy().astype(np.float64)
    n = means.shape[0]

    if "is_static" in params:
        dyn = params["is_static"].detach().cpu().reshape(-1).numpy() <= 0.5
    else:
        dyn = np.ones((n,), dtype=bool)

    npz = np.load(args.npz, allow_pickle=True)
    pos = np.asarray(npz["positions"], dtype=np.float64)
    d, _ = cKDTree(pos).query(means[dyn], k=1, workers=-1)
    d = np.asarray(d).reshape(-1)

    dyn_idx = np.flatnonzero(dyn)
    if args.top_n is not None:
        m = min(int(args.top_n), d.shape[0])
        pick = np.argsort(d)[::-1][:m]
        thresh = float(d[pick].min()) if m else float("nan")
        print(f"suppressing top-{m} off-surface dynamic rows "
              f"(distance >= {thresh:.4f} m)")
    else:
        pick = np.flatnonzero(d > float(args.dist))
        print(f"suppressing {len(pick):,} dynamic rows with distance > "
              f"{args.dist:.4f} m")
    flag_rows = dyn_idx[pick]

    if len(flag_rows) == 0:
        sys.exit("Nothing selected; not writing an identical checkpoint.")

    op = params["opacities"].detach().clone()
    flat = op.reshape(-1) if op.numel() == n else op.reshape(n, -1)[:, 0]
    before = torch.sigmoid(flat[flag_rows])
    print(f"  their current opacity: median {before.median():.4f}  "
          f"max {before.max():.4f}")
    if op.numel() == n:
        op.reshape(-1)[flag_rows] = -20.0
    else:
        op.reshape(n, -1)[flag_rows, :] = -20.0
    params["opacities"] = op

    ckpt.setdefault("postprocess", {})["offsurface_suppress"] = {
        "source": str(args.checkpoint),
        "npz": str(args.npz),
        "suppressed": int(len(flag_rows)),
        "dist": args.dist,
        "top_n": args.top_n,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(args.output)
    print(f"wrote {args.output}  ({len(flag_rows):,} rows suppressed, "
          f"shapes unchanged, original untouched)")


if __name__ == "__main__":
    main()
