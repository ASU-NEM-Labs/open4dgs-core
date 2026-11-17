#!/usr/bin/env python3
"""Suppress isolated floater primitives in a trained TriGS checkpoint.

Why post-hoc
------------
The trainer's removal mechanisms -- `lambda_reg` (opacity pressure) and
`relocate_tau_alpha` (recycle threshold) -- are population-wide and keyed on
opacity. Opacity does not separate a floater from a faint-but-real detail splat,
so raising either one destroys detail before it clears the floaters (measured:
lambda_reg 0.1 and tau_alpha 0.02 both put 30-65% of the model into a 100-step
kill/resurrect cycle and blurred the subject).

The property that *does* separate them is spatial isolation: real surfaces are
dense point clouds, floaters sit alone in empty space. That needs a neighbour
query, which is cheap here and impossible to express as a per-primitive loss.

What it does
------------
Flags primitives whose distance to their k-th nearest neighbour is far above the
population norm, and suppresses them by driving their opacity logit to a large
negative value. Opacity is zeroed rather than rows deleted so every tensor keeps
its shape -- no index remapping, no optimizer-state desync, nothing downstream
can break, and the edit is trivially reversible from the original file.

By default only DYNAMIC rows are considered (`is_static <= 0.5`), because the
static anchor rows carry trusted background geometry that was never the problem.

Always run --dry-run first and read the reported percentiles.
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

SUPPRESSED_LOGIT = -20.0  # sigmoid(-20) ~ 2e-9


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None,
                    help="Where to write the filtered checkpoint. Required unless --dry-run.")
    ap.add_argument("--k", type=int, default=8,
                    help="Use distance to the k-th nearest neighbour as the isolation score.")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="Suppress rows above this percentile of kNN distance. "
                         "99.0 targets the top 1%% most isolated dynamic rows.")
    ap.add_argument("--abs-radius", type=float, default=None,
                    help="Absolute override (metres): suppress rows whose k-th NN is farther "
                         "than this. Takes precedence over --percentile when given.")
    ap.add_argument("--include-static", action="store_true",
                    help="Also consider static anchor rows. Off by default.")
    ap.add_argument("--neighbours-from-all", action="store_true", default=True,
                    help="Measure isolation against ALL primitives, not just the candidate "
                         "set, so a floater near a dense surface is correctly kept.")
    ap.add_argument("--max-opacity", type=float, default=None,
                    help="Optional extra guard: only suppress rows whose sigmoid(opacity) is "
                         "below this. Leave unset to filter purely on isolation.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and args.output is None:
        sys.exit("--output is required unless --dry-run")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "params" not in ckpt:
        sys.exit(f"Not a trigs training snapshot (no 'params'): {args.checkpoint}")
    params = ckpt["params"]

    means = params["means"].detach().cpu().numpy().astype(np.float64)
    n = means.shape[0]
    opac = torch.sigmoid(params["opacities"].detach().cpu().reshape(-1)).numpy()

    if "is_static" in params:
        is_static = params["is_static"].detach().cpu().reshape(-1).numpy() > 0.5
    else:
        is_static = np.zeros((n,), dtype=bool)

    candidate = np.ones((n,), dtype=bool) if args.include_static else ~is_static
    print(f"checkpoint : {args.checkpoint}")
    print(f"  step     : {ckpt.get('step_completed', '?')}")
    print(f"  primitives: {n:,}   static: {int(is_static.sum()):,}   "
          f"dynamic: {int((~is_static).sum()):,}")
    print(f"  candidates: {int(candidate.sum()):,}")

    # Isolation score. Query against all primitives so a dynamic splat sitting on
    # a dense static surface is not mistaken for a floater.
    ref = means if args.neighbours_from_all else means[candidate]
    tree = cKDTree(ref)
    dists, _ = tree.query(means[candidate], k=args.k + 1, workers=-1)
    knn_dist = dists[:, -1]  # k-th neighbour, self excluded by the +1

    pcts = [50, 75, 90, 95, 99, 99.5, 99.9]
    print("\nk-NN distance (k=%d) over candidates, metres:" % args.k)
    for p in pcts:
        print(f"  p{p:<5} {np.percentile(knn_dist, p):.5f}")
    print(f"  max    {knn_dist.max():.5f}")

    if args.abs_radius is not None:
        thresh = float(args.abs_radius)
        print(f"\nthreshold: {thresh:.5f} m (absolute)")
    else:
        thresh = float(np.percentile(knn_dist, args.percentile))
        print(f"\nthreshold: {thresh:.5f} m (p{args.percentile})")

    flagged_local = knn_dist > thresh
    if args.max_opacity is not None:
        flagged_local &= opac[candidate] < float(args.max_opacity)
        print(f"  extra guard: sigmoid(opacity) < {args.max_opacity}")

    flagged = np.zeros((n,), dtype=bool)
    flagged[np.flatnonzero(candidate)[flagged_local]] = True
    m = int(flagged.sum())

    print(f"\nwould suppress: {m:,} rows ({100.0 * m / n:.3f}% of all primitives)")
    if m:
        print(f"  their opacity : median {np.median(opac[flagged]):.4f}  "
              f"max {opac[flagged].max():.4f}")
        print(f"  their isolation: median {np.median(knn_dist[flagged_local]):.4f} m  "
              f"max {knn_dist[flagged_local].max():.4f} m")
        print(f"  kept rows      : median isolation "
              f"{np.median(knn_dist[~flagged_local]):.5f} m")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if m == 0:
        sys.exit("Nothing flagged; refusing to write an identical checkpoint.")

    op = params["opacities"].detach().clone()
    shape = op.shape
    flat = op.reshape(-1)
    flat[torch.from_numpy(flagged)] = SUPPRESSED_LOGIT
    params["opacities"] = flat.reshape(shape)

    ckpt.setdefault("postprocess", {})["floater_filter"] = {
        "source": str(args.checkpoint),
        "k": args.k,
        "threshold_m": thresh,
        "percentile": None if args.abs_radius is not None else args.percentile,
        "suppressed": m,
        "include_static": bool(args.include_static),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(args.output)
    print(f"\nwrote {args.output}  ({m:,} rows suppressed, shapes unchanged)")


if __name__ == "__main__":
    main()
