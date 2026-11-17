#!/usr/bin/env python3
"""Audit a selected sub-population of primitives in a trained TriGS checkpoint.

Read-only. Writes nothing except an optional --dump-indices .npy.

IMPORTANT -- why selection matters
----------------------------------
Scene density is wildly non-uniform: the subject is dense, the far background is
sparse. A GLOBAL isolation percentile therefore selects the sparsest geometry in
the scene (legitimate distant background), NOT floaters sitting near a dense
subject. Always restrict to the volume where the artifact actually is, using
--region-center/--region-radius, and to the population it belongs to, using
--only-dynamic / --only-static.

Workflow: run with --dump-indices, load those rows in the viewer, confirm the
cohort really is the artifact, then trust the comparison table. If the selection
is wrong the numbers are worthless -- verify first.

NPZ alignment
-------------
Checkpoint row i corresponds to NPZ row i only if nothing reordered the rows.
This script CHECKS that by comparing is_static between the two and refuses to
report NPZ-derived fields (source_frames, displacement) when they disagree.
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


def _arr(params, name):
    if name not in params:
        return None
    v = params[name]
    v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    return v.astype(np.float64)


def _flat(params, name):
    v = _arr(params, name)
    return None if v is None else v.reshape(v.shape[0], -1)[:, 0]


def _fmt(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{n}g}"


def _row(label, a, b, n=4):
    print(f"  {label:<34} {_fmt(a, n):>14} {_fmt(b, n):>14}")


def _cmp(label, x, mask_a, mask_b, fn, n=4):
    a = float(fn(x[mask_a])) if mask_a.any() else None
    b = float(fn(x[mask_b])) if mask_b.any() else None
    _row(label, a, b, n)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--npz", type=Path, default=None)

    # --- population selection (do this FIRST, before any isolation stats) ---
    ap.add_argument("--region-center", type=str, default=None,
                    help='"x,y,z" centre of the volume to analyse.')
    ap.add_argument("--region-radius", type=float, default=None,
                    help="Radius in metres around --region-center.")
    ap.add_argument("--only-dynamic", action="store_true",
                    help="Restrict to is_static <= 0.5 rows.")
    ap.add_argument("--only-static", action="store_true",
                    help="Restrict to is_static > 0.5 rows.")

    # --- isolation, computed WITHIN the selection -------------------------
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="Percentile of k-NN distance WITHIN the selected "
                         "population above which rows are flagged.")
    ap.add_argument("--abs-radius", type=float, default=None,
                    help="Absolute k-NN threshold in metres; overrides "
                         "--percentile.")
    ap.add_argument("--neighbors", choices=("all", "selected"), default="all",
                    help="knn criterion only: measure isolation against all "
                         "primitives or only the selected population.")
    ap.add_argument("--criterion", choices=("knn", "npz-distance"), default="knn",
                    help="'npz-distance' (RECOMMENDED, needs --npz) scores each "
                         "trained primitive by distance to the NEAREST NPZ point. "
                         "The NPZ is clean geometry, so a large distance means the "
                         "primitive is not on any real surface. Robust to row "
                         "reordering, unlike index-wise displacement. 'knn' scores "
                         "local sparsity, which also flags legitimately thin "
                         "geometry (bow, hair) and is usually the wrong tool.")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Flag the N most off-surface rows outright. Usually the "
                         "easiest knob: pick N near the artifact count you see, "
                         "then widen or narrow after viewing the dump.")
    ap.add_argument("--dump-indices", type=Path, default=None)
    ap.add_argument("--dump-npz", type=Path, default=None,
                    help="Write the flagged rows as a viewer-loadable NPZ "
                         "(positions/colors/velocities/is_static), so the cohort "
                         "can be confirmed by eye in npz-viewer.py.")
    args = ap.parse_args()

    if args.only_dynamic and args.only_static:
        sys.exit("--only-dynamic and --only-static are mutually exclusive")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "params" not in ckpt:
        sys.exit(f"Not a TriGS training snapshot (no 'params'): {args.checkpoint}")
    params = ckpt["params"]

    means = _arr(params, "means")
    if means is None:
        sys.exit("Checkpoint has no 'means'")
    n = means.shape[0]
    is_static_ck = _flat(params, "is_static")
    static_b = None if is_static_ck is None else is_static_ck > 0.5

    print(f"checkpoint : {args.checkpoint}")
    print(f"  primitives: {n:,}"
          + (f"   static: {int(static_b.sum()):,}   dynamic: {int((~static_b).sum()):,}"
             if static_b is not None else ""))

    # --- selection ---------------------------------------------------------
    sel = np.ones((n,), dtype=bool)
    if args.region_center is not None:
        if args.region_radius is None:
            sys.exit("--region-center requires --region-radius")
        c = np.array([float(t) for t in args.region_center.split(",")], dtype=np.float64)
        if c.shape != (3,):
            sys.exit('--region-center must be "x,y,z"')
        d = np.linalg.norm(means - c[None, :], axis=1)
        sel &= d <= float(args.region_radius)
        print(f"  region    : centre {c.tolist()} r={args.region_radius} "
              f"-> {int(sel.sum()):,} rows")
    if args.only_dynamic:
        if static_b is None:
            sys.exit("--only-dynamic requires an is_static param")
        sel &= ~static_b
        print(f"  only dynamic -> {int(sel.sum()):,} rows")
    if args.only_static:
        if static_b is None:
            sys.exit("--only-static requires an is_static param")
        sel &= static_b
        print(f"  only static  -> {int(sel.sum()):,} rows")

    if int(sel.sum()) < args.k + 2:
        sys.exit(f"Selection has only {int(sel.sum())} rows; nothing to analyse.")

    # --- score within the selection ---------------------------------------
    if args.criterion == "npz-distance":
        if args.npz is None:
            sys.exit("--criterion npz-distance requires --npz")
        _z = np.load(args.npz, allow_pickle=True)
        if "positions" not in _z:
            sys.exit("NPZ has no 'positions'")
        npz_pos = np.asarray(_z["positions"], dtype=np.float64)
        dd, _i = cKDTree(npz_pos).query(means[sel], k=1, workers=-1)
        knn_sel = np.asarray(dd).reshape(-1)
        label = "distance to nearest NPZ point"
        print(f"\n{label} ({len(npz_pos):,} NPZ points), m:")
    else:
        ref = means if args.neighbors == "all" else means[sel]
        dd, _i = cKDTree(ref).query(means[sel], k=args.k + 1, workers=-1)
        knn_sel = dd[:, -1]
        label = f"k-NN distance (k={args.k}, vs {args.neighbors})"
        print(f"\n{label} within selection, m:")

    for p in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{p:<5} {np.percentile(knn_sel, p):.5f}")
    print(f"  max    {knn_sel.max():.5f}")

    if args.criterion == "npz-distance":
        _med = float(np.median(knn_sel))
        if _med > 0.10:
            print(f"\n  WARNING: median distance to nearest NPZ point is {_med:.3f} m.")
            print("  Trained primitives should sit ON the init geometry, so a large")
            print("  median means the checkpoint and NPZ are in DIFFERENT COORDINATE")
            print("  FRAMES (or it is the wrong NPZ). Fix that before reading further;")
            print("  every distance below is offset by the frame mismatch.")
        else:
            print(f"\n  frames agree (median {_med:.4f} m) -- distances are meaningful.")

    sel_idx = np.flatnonzero(sel)
    flag = np.zeros((n,), dtype=bool)
    if args.top_n is not None:
        # Easiest knob to tune: take the N most off-surface rows outright.
        m = min(int(args.top_n), knn_sel.shape[0])
        order = np.argsort(knn_sel)[::-1][:m]
        flag[sel_idx[order]] = True
        thresh = float(knn_sel[order].min()) if m else float("nan")
        how = f"top-{m} (score >= {thresh:.5f} m)"
    else:
        if args.abs_radius is not None:
            thresh, how = float(args.abs_radius), "absolute"
        else:
            thresh, how = float(np.percentile(knn_sel, args.percentile)), f"p{args.percentile}"
        flag[sel_idx[knn_sel > thresh]] = True
    rest = sel & ~flag  # comparison group = rest of the SAME selection

    print(f"\nthreshold {thresh:.5f} m ({how})  ->  flagged {int(flag.sum()):,}"
          f"   rest-of-selection {int(rest.sum()):,}")

    if args.dump_indices is not None:
        np.save(args.dump_indices, np.flatnonzero(flag))
        print(f"  wrote indices -> {args.dump_indices}")

    if args.dump_npz is not None:
        # SH DC term -> linear RGB (gsplat convention: C0 = 0.2820947917738781).
        sh0 = _arr(params, "sh0")
        if sh0 is not None:
            c = sh0.reshape(n, -1)[:, :3] * 0.2820947917738781 + 0.5
        else:
            c = np.full((n, 3), 0.5)
        out = {
            "positions": means[flag].astype(np.float32),
            "colors": np.clip(c[flag], 0.0, 1.0).astype(np.float32),
            "velocities": np.zeros((int(flag.sum()), 3), np.float32),
            "has_velocity": np.zeros((int(flag.sum()),), np.bool_),
            "is_static": np.ones((int(flag.sum()),), np.float32),
            # Constant: per-row mu_t would make the viewer treat every distinct
            # timestamp as its own frame and cache tens of thousands of them.
            "times": np.zeros((int(flag.sum()),), np.float32),
        }
        np.savez(args.dump_npz, **out)
        print(f"  wrote viewable NPZ -> {args.dump_npz} "
              f"({int(flag.sum()):,} rows)")
        print("  OPEN THIS IN npz-viewer.py AND CONFIRM before trusting the table.")

    print(f"\n{'':<36}{'FLAGGED':>14}{'REST-OF-SEL':>14}")

    print("\n-- pool membership (% of group) --")
    for key in ("is_static", "dynamic_protect", "dynamic_allowed", "stage_keep"):
        v = _flat(params, key)
        if v is None:
            continue
        _cmp(f"{key}  %", (v > 0.5).astype(np.float64) * 100.0, flag, rest, np.mean)

    gen = _flat(params, "relocation_generation")
    print("\n-- relocation lineage --")
    if gen is None:
        print("  relocation_generation NOT PRESENT (not tracked in this run).")
    else:
        _cmp("generation == 0 (original) %", (gen == 0).astype(np.float64) * 100.0,
             flag, rest, np.mean)
        for lbl, fn in (("median", np.median), ("p90", lambda z: np.percentile(z, 90)),
                        ("max", np.max), ("mean", np.mean)):
            _cmp(f"generation {lbl}", gen, flag, rest, fn)

    print("\n-- temporal window --")
    mu = _flat(params, "times")
    if mu is not None:
        for lbl, fn in (("median", np.median), ("p10", lambda z: np.percentile(z, 10)),
                        ("p90", lambda z: np.percentile(z, 90))):
            _cmp(f"mu_t {lbl} (s)", mu, flag, rest, fn)
    ld = _flat(params, "log_durations")
    if ld is not None:
        s = np.exp(ld)
        for lbl, fn in (("median", np.median), ("p99", lambda z: np.percentile(z, 99))):
            _cmp(f"sigma_t {lbl} (s)", s, flag, rest, fn)

    print("\n-- opacity / scale --")
    op = _flat(params, "opacities")
    if op is not None:
        a = 1.0 / (1.0 + np.exp(-op))
        for lbl, fn in (("median", np.median), ("p10", lambda z: np.percentile(z, 10)),
                        ("p90", lambda z: np.percentile(z, 90))):
            _cmp(f"opacity {lbl}", a, flag, rest, fn)
        _cmp("opacity < 0.005 %", (a < 0.005).astype(np.float64) * 100.0,
             flag, rest, np.mean)
    sc = _arr(params, "scales")
    if sc is not None and sc.ndim == 2 and sc.shape[1] == 3:
        e = np.exp(sc)
        longest = e.max(axis=1)
        aniso = longest / np.maximum(np.median(e, axis=1), 1e-12)
        for lbl, fn in (("median", np.median), ("p99", lambda z: np.percentile(z, 99))):
            _cmp(f"longest axis {lbl} (m)", longest, flag, rest, fn)
            _cmp(f"anisotropy {lbl}", aniso, flag, rest, fn)

    print("\n-- motion --")
    for label, key in (("|nu_base| (m/s)", "zeta_base_t"),
                       ("|omega_base| (rad/s)", "zeta_base_r")):
        v = _arr(params, key)
        if v is None or v.ndim != 2:
            continue
        mag = np.linalg.norm(v, axis=1)
        for lbl, fn in (("median", np.median), ("p99", lambda z: np.percentile(z, 99))):
            _cmp(f"{label} {lbl}", mag, flag, rest, fn)
    for label, keys in (("|c1_t| bezier", ("zeta_c1_t",)), ("|c2_t| bezier", ("zeta_c2_t",))):
        v = _arr(params, keys[0])
        if v is None or v.ndim != 2:
            continue
        mag = np.linalg.norm(v, axis=1)
        _cmp(f"{label} p99", mag, flag, rest, lambda z: np.percentile(z, 99))
    anc = _arr(params, "anchors")
    if anc is not None and anc.shape == means.shape:
        lever = np.linalg.norm(means - anc, axis=1)
        for lbl, fn in (("median", np.median), ("p99", lambda z: np.percentile(z, 99))):
            _cmp(f"lever |mean-anchor| {lbl} (m)", lever, flag, rest, fn)

    # --- NPZ-derived fields, only when row alignment is verified -----------
    if args.npz is not None:
        npz = np.load(args.npz, allow_pickle=True)
        pos = np.asarray(npz["positions"], dtype=np.float64) if "positions" in npz else None
        aligned = False
        if pos is None or len(pos) != n:
            print(f"\n-- NPZ -- row count mismatch (ckpt {n:,} vs NPZ "
                  f"{0 if pos is None else len(pos):,}); NPZ fields skipped.")
        elif "is_static" in npz and static_b is not None:
            npz_static = np.asarray(npz["is_static"]).reshape(-1) > 0.5
            agree = float((npz_static == static_b).mean())
            print(f"\n-- NPZ alignment check -- is_static agreement: {100*agree:.2f}%")
            if agree > 0.999:
                aligned = True
                print("  aligned: row i corresponds between checkpoint and NPZ.")
            else:
                print("  NOT ALIGNED: rows were reordered or overwritten.")
                print("  Skipping displacement and source_frames (they would be noise).")
        else:
            print("\n-- NPZ alignment check -- no is_static in NPZ; cannot verify. Skipped.")

        if aligned:
            disp = np.linalg.norm(means - pos, axis=1)
            print("\n-- displacement from NPZ position (m) --")
            for lbl, fn in (("median", np.median), ("p90", lambda z: np.percentile(z, 90)),
                            ("max", np.max)):
                _cmp(f"displacement {lbl}", disp, flag, rest, fn)
            if "source_frames" in npz:
                sf = np.asarray(npz["source_frames"]).reshape(-1)
                print("\n-- source keyframe (flagged group) --")
                vals, counts = np.unique(sf[flag], return_counts=True)
                tot = max(int(counts.sum()), 1)
                for v, c in sorted(zip(vals.tolist(), counts.tolist()),
                                   key=lambda t: -t[1])[:10]:
                    print(f"  frame {int(v):>6} : {c:>8,}  ({100.0*c/tot:5.2f}%)")

    print("\nRead the two columns for a sharp divergence. If FLAGGED and "
          "REST-OF-SEL\nlook the same, the selection did not isolate the "
          "artifact -- change the\nregion/percentile and re-verify in the viewer.")


if __name__ == "__main__":
    main()
