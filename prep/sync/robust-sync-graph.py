#!/usr/bin/env python3
"""
Robust global sync solve from pairwise visual offset estimates. v3.

Changes vs v1:
  * Nodes are identified by camera FOLDER NAME carried through the pair summary
    CSV (falls back to the old index+1 assumption only with a loud warning).
  * A disconnected graph no longer disables outlier rejection: the robust loop
    runs on the largest connected component; everything else is reported as
    UNCONSTRAINED and written as 0.0 (or omitted with --omit-unconstrained).
  * Single-frame pairs no longer inherit maximal weight from a degenerate
    cross-frame MAD (weight is recomputed defensively here as well).
  * Per-camera diagnostics: degree, systematic bias, estimated uncertainty.
  * Circular angular synchronization handles phase modulo one frame.
  * A graph k-core rejects degree-1/2 cameras that residuals cannot validate.
  * Pair covariance is propagated through the graph Laplacian.
  * Optional split-half validation: solves odd frames vs even frames from the
    raw observation CSV and compares per-camera offsets — a direct empirical
    accuracy measurement.
  * Prints a final VERDICT + RECOMMENDATION and writes a machine-readable
    report JSON.

Output JSON is PLUS convention: actual_time = nominal_time + offset_sec[camera].
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque

import numpy as np


# ----------------------------------------------------------------------------- helpers

def mad(x, center=None):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    if center is None:
        center = float(np.median(x))
    return float(np.median(np.abs(x - center)))


def wrap_period(x, period):
    x = np.asarray(x, dtype=np.float64)
    return (x + 0.5 * period) % period - 0.5 * period


def weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 1e-12)
    cs = np.cumsum(w)
    return float(v[int(np.searchsorted(cs, 0.5 * float(cs[-1])))])


def circular_median(values, period, weights=None):
    """L1 center on the circle: the sample minimizing sum w*|wrap(x - c)|.

    A linear median is wrong for phases. When the cluster straddles the
    +/-half-period boundary the linear median lands in the empty gap, and every
    value on one side then wraps to the far end. Restricting candidate centers
    to the samples themselves is the exact minimizer of the circular L1 cost.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    cand = wrap_period(values, period)
    costs = [float(np.sum(w * np.abs(wrap_period(cand - c, period)))) for c in cand]
    return float(cand[int(np.argmin(costs))])


def unwrap_about(values, period, anchor=None):
    """Lift wrapped phases onto the line around their circular center."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values, 0.0
    c = circular_median(values, period) if anchor is None else float(anchor)
    return c + wrap_period(values - c, period), c


def center_phases(values, period):
    """Zero-center wrapped phases without letting the boundary corrupt the gauge.

    Returns (centered, center, arc_fraction). arc_fraction is the span of the
    unwrapped cluster as a fraction of the period; near 1.0 the cluster covers
    the whole circle and no unwrapping is meaningful.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values, 0.0, 0.0
    lifted, _ = unwrap_about(values, period)
    center = float(np.median(lifted))
    arc = float(lifted.max() - lifted.min()) / period if period > 0 else 0.0
    return wrap_period(values - center, period), center, arc


def fmt_key(folder):
    folder = str(folder)
    return f"{int(folder):04d}" if folder.isdigit() else folder


def components(edges, nodes):
    """Connected components over the given node set (isolated nodes are their own component)."""
    adj = defaultdict(set)
    for a, b, *_ in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    comps = []
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(comp))
    return comps


def solve_component(edges, nodes, period_sec=None):
    """Solve x[a]-x[b]=d, optionally as a circular phase problem.

    Circular mode uses a spectral angular-synchronization initializer followed
    by wrapped-residual IRLS. This prevents equivalent +/- one-frame phase
    representatives from being averaged as unrelated linear offsets.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    if m == 1:
        return {nodes[0]: 0.0}

    if period_sec is not None and period_sec > 0:
        H = np.zeros((m, m), dtype=np.complex128)
        scale = max(float(np.median([e[3] for e in edges])), 1e-12)
        for a, b, d, w in edges:
            z = (w / scale) * np.exp(1j * 2.0 * np.pi * d / period_sec)
            H[idx[a], idx[b]] += z
            H[idx[b], idx[a]] += np.conjugate(z)
        _, vecs = np.linalg.eigh(H)
        x_arr = np.angle(vecs[:, -1]) * period_sec / (2.0 * np.pi)

        for _ in range(20):
            rows, rhs, wts = [], [], []
            for a, b, d, w in edges:
                r = float(wrap_period(x_arr[idx[a]] - x_arr[idx[b]] - d, period_sec))
                huber = 0.10 * period_sec
                robust = 1.0 if abs(r) <= huber else huber / max(abs(r), 1e-12)
                row = np.zeros(m, dtype=np.float64)
                row[idx[a]] = 1.0
                row[idx[b]] = -1.0
                rows.append(row)
                rhs.append(-r)
                wts.append(math.sqrt(max(w * robust, 1e-12)))
            rows.append(np.ones(m, dtype=np.float64))
            rhs.append(0.0)
            wts.append(1e6)
            A = np.stack(rows)
            y = np.asarray(rhs, dtype=np.float64)
            sw = np.asarray(wts, dtype=np.float64)
            correction, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
            x_arr = wrap_period(x_arr + correction, period_sec)
            if float(np.max(np.abs(correction))) < 1e-8:
                break
        return {n: float(x_arr[idx[n]]) for n in nodes}

    rows, rhs, wts = [], [], []
    for a, b, d, w in edges:
        row = np.zeros(m, dtype=np.float64)
        row[idx[a]] = 1.0
        row[idx[b]] = -1.0
        rows.append(row)
        rhs.append(d)
        wts.append(math.sqrt(max(w, 1e-9)))
    rows.append(np.ones(m, dtype=np.float64))
    rhs.append(0.0)
    wts.append(1e6)
    A = np.stack(rows)
    y = np.asarray(rhs, dtype=np.float64)
    sw = np.asarray(wts, dtype=np.float64)
    x, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return {n: float(x[idx[n]]) for n in nodes}


def residuals(edges, x, period_sec=None):
    out = []
    for a, b, d_sec, w in edges:
        pred = x[a] - x[b]
        err_sec = pred - d_sec
        if period_sec is not None and period_sec > 0:
            err_sec = float(wrap_period(err_sec, period_sec))
        err_ms = 1000.0 * err_sec
        out.append((abs(err_ms), err_ms, a, b, 1000.0 * d_sec, 1000.0 * pred, w))
    return out


def k_core(edges, nodes, min_degree):
    """Return nodes supported by at least min_degree mutually surviving edges."""
    active = set(nodes)
    if min_degree <= 0:
        return sorted(active)
    while True:
        degree = defaultdict(int)
        for a, b, *_ in edges:
            if a in active and b in active:
                degree[a] += 1
                degree[b] += 1
        drop = {n for n in active if degree[n] < min_degree}
        if not drop:
            return sorted(active)
        active -= drop
        if not active:
            return []


def graph_uncertainty_ms(edges, nodes):
    """Mean-zero node covariance from the weighted graph Laplacian."""
    if len(nodes) < 2:
        return {n: float("inf") for n in nodes}
    idx = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for a, b, _, w in edges:
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    cov = np.linalg.pinv(L, rcond=1e-12)
    return {n: float(math.sqrt(max(cov[idx[n], idx[n]], 0.0)) * 1000.0) for n in nodes}


# ----------------------------------------------------------------------------- IO

def read_summary(path, min_pair_frames, min_total_observations,
                 max_pair_mad_frames=0.30, max_pair_sigma_ms=1.5, fps=60.0):
    edges = []
    skipped_frames = skipped_obs = skipped_scatter = skipped_sigma = 0
    used_fallback_naming = False
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                n_frames = int(float(r.get("n_frames", 1)))
                n_total = int(float(r.get("n_total", 0)))
            except ValueError:
                continue
            if n_frames < min_pair_frames:
                skipped_frames += 1
                continue
            if min_total_observations and n_total < min_total_observations:
                skipped_obs += 1
                continue
            mad_f = float(r.get("mad_frames_across_frames", 0.25))
            p80_f = float(r.get("p80_deviation_frames", mad_f) or mad_f)
            if max_pair_mad_frames > 0 and max(mad_f, p80_f) > max_pair_mad_frames:
                skipped_scatter += 1
                continue

            fa = (r.get("folder_a") or "").strip()
            fb = (r.get("folder_b") or "").strip()
            if not fa or not fb:
                # Legacy summary without folder names. This assumes camera index i
                # corresponds to folder (i+1) zero-padded, which is NOT verified.
                used_fallback_naming = True
                fa = f"{int(r['cam_a']) + 1:04d}"
                fb = f"{int(r['cam_b']) + 1:04d}"
            a, b = fmt_key(fa), fmt_key(fb)

            d_sec = float(r["delta_sec_a_minus_b"])
            sigma_ms_text = (r.get("pair_sigma_ms") or "").strip()
            if sigma_ms_text:
                sigma_ms = float(sigma_ms_text)
            else:
                # Legacy summaries: estimate uncertainty from independent
                # temporal windows, never from the dense correspondence count.
                sigma_ms = 1000.0 / fps * max(1.4826 * mad_f / math.sqrt(max(n_frames, 1)), 0.02)
            if not np.isfinite(sigma_ms) or (max_pair_sigma_ms > 0 and sigma_ms > max_pair_sigma_ms):
                skipped_sigma += 1
                continue
            sigma_sec = max(sigma_ms / 1000.0, 1e-6)
            w = 1.0 / (sigma_sec * sigma_sec)
            edges.append((a, b, d_sec, max(w, 1e-9)))
    if used_fallback_naming:
        print("[warn] pair summary has no folder_a/folder_b columns; assuming folder == index+1 "
              "(zero-padded). Regenerate the summary with the updated estimator to remove this assumption.")
    return edges, skipped_frames, skipped_obs, skipped_scatter, skipped_sigma


def read_reference_folders(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [fmt_key(str(c["folder"])) for c in data["cameras"]]


def read_observations(path, fps, min_n_used=10):
    """Raw rows -> {(a,b): {frame: (delta_frames, independent-window weight)}}."""
    per_pair = defaultdict(dict)
    frames = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("delta_b_frames") or "").strip()
            if not d:
                continue  # error row
            try:
                delta = float(d)
                n_used = int(float(r.get("n_used", 0) or 0))
                frame = int(float(r["frame"]))
            except ValueError:
                continue
            if n_used < min_n_used or not np.isfinite(delta):
                continue
            try:
                phase_se = float(r.get("phase_se_frames", "") or 0.10)
            except ValueError:
                phase_se = 0.10
            window_weight = min(1.0 / max(phase_se, 0.02) ** 2, 2500.0)
            fa = (r.get("folder_a") or "").strip()
            fb = (r.get("folder_b") or "").strip()
            if not fa or not fb:
                fa = f"{int(r['cam_a']) + 1:04d}"
                fb = f"{int(r['cam_b']) + 1:04d}"
            key = (fmt_key(fa), fmt_key(fb))
            frames.add(frame)
            prev = per_pair[key].get(frame)
            if prev is None or window_weight > prev[1]:
                per_pair[key][frame] = (delta, window_weight)
    return per_pair, sorted(frames)


# ----------------------------------------------------------------------------- validation

def split_half_solve(per_pair, all_frames, fps, active_pairs, solved_nodes, period_sec):
    """Solve odd-index vs even-index frame halves on the final edge set; return per-cam |diff|/2 in ms."""
    if len(all_frames) < 4:
        return None, "too few frames for split-half validation"
    half_frames = [set(all_frames[0::2]), set(all_frames[1::2])]
    half_edges = []
    for hf in half_frames:
        edges = []
        for (a, b) in active_pairs:
            obs = per_pair.get((a, b), {})
            vals = [(v[0], v[1]) for f, v in obs.items() if f in hf]
            if not vals:
                continue
            deltas = np.asarray([v[0] for v in vals])
            ns = np.asarray([v[1] for v in vals], dtype=np.float64)
            # Unwrap each half around its circular L1 center.
            per_frames = period_sec * fps
            center = circular_median(deltas, per_frames, ns)
            unwrapped = center + wrap_period(deltas - center, per_frames)
            med = weighted_median(unwrapped, ns)
            edges.append((a, b, med / fps, float(np.sum(ns))))
        half_edges.append(edges)

    # Solve each half on the largest common component of the solved node set.
    half_nodes = []
    for edges in half_edges:
        comps = components(edges, solved_nodes)
        comps = [c for c in comps if len(c) > 1]
        if not comps:
            return None, "a split half has no connected pairs"
        half_nodes.append(set(max(comps, key=len)))
    common = sorted(half_nodes[0] & half_nodes[1])
    if len(common) < 3:
        return None, "split halves share too few cameras"

    xs = []
    for edges in half_edges:
        e_in = [e for e in edges if e[0] in common and e[1] in common]
        comps = components(e_in, common)
        if len(comps) > 1:
            # restrict to largest component of the intersection graph
            biggest = set(max(comps, key=len))
            common = sorted(set(common) & biggest)
            if len(common) < 3:
                return None, "split halves disconnect on the common cameras"
            e_in = [e for e in e_in if e[0] in common and e[1] in common]
        xs.append(solve_component(e_in, common, period_sec))

    diffs = wrap_period(np.asarray([xs[0][n] - xs[1][n] for n in common], dtype=np.float64), period_sec)
    # Align gauges on the circle: a linear median here lands in the empty gap
    # whenever the two halves' gauge difference sits near +/-half a period.
    diffs, _, _ = center_phases(diffs, period_sec)
    unc_ms = {n: abs(d) * 0.5 * 1000.0 for n, d in zip(common, diffs)}
    return unc_ms, None


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Robust sync graph solve with validation (v2).")
    ap.add_argument("--pair-summary", required=True)
    ap.add_argument("--pair-obs-csv", default=None,
                    help="Raw per-(frame,pair) observation CSV for split-half validation.")
    ap.add_argument("--reference-json", default=None,
                    help="reference_export.json; used to enumerate ALL cameras so missing ones are named.")
    ap.add_argument("--fps", type=float, default=60.0, help="Used for verdict thresholds and split-half.")
    ap.add_argument("--out-json", default="sync_robust.json")
    ap.add_argument("--report-json", default="sync_robust_report.json")
    ap.add_argument("--drop-frames", type=float, default=0.12,
                    help="Outlier-edge threshold in frames (0.12 = 2.0 ms at 60 fps, 4.0 ms at 30 fps).")
    ap.add_argument("--drop-ms", type=float, default=None,
                    help="Absolute override for --drop-frames.")
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--min-pair-frames", type=int, default=2)
    ap.add_argument("--min-total-observations", type=int, default=0)
    ap.add_argument("--max-pair-mad-frames", type=float, default=0.30)
    ap.add_argument("--max-pair-sigma-frames", type=float, default=0.09,
                    help="Skip edges with larger uncertainty, in frames (0.09 = 1.5 ms at 60 fps, 3.0 ms at 30 fps).")
    ap.add_argument("--max-pair-sigma-ms", type=float, default=None,
                    help="Absolute override for --max-pair-sigma-frames.")
    ap.add_argument("--min-camera-degree", type=int, default=3,
                    help="Exclude cameras outside this graph k-core instead of fitting unsupported bridges.")
    ap.add_argument("--min-solved-cameras", type=int, default=3,
                    help="Floor on the solved set when dropping an outlier would demote cameras. "
                         "Below this the outlier is retained and the fit is reported as unreliable.")
    ap.add_argument("--phase-period-frames", type=float, default=1.0,
                    help="Solve offsets modulo this many frames.")
    ap.add_argument("--omit-unconstrained", action="store_true",
                    help="Omit unsolved cameras from the output JSON instead of writing 0.0.")
    args = ap.parse_args()
    # Measurement noise is in frames, so the gates scale with the frame period
    # unless pinned in ms.
    if args.drop_ms is None:
        args.drop_ms = args.drop_frames * 1000.0 / args.fps
    if args.max_pair_sigma_ms is None:
        args.max_pair_sigma_ms = args.max_pair_sigma_frames * 1000.0 / args.fps
    print(f"gates at {args.fps:g} fps: drop {args.drop_ms:.3f} ms, max pair sigma {args.max_pair_sigma_ms:.3f} ms")

    period_sec = args.phase_period_frames / args.fps
    edges, skipped_frames, skipped_obs, skipped_scatter, skipped_sigma = read_summary(
        args.pair_summary, args.min_pair_frames, args.min_total_observations,
        args.max_pair_mad_frames, args.max_pair_sigma_ms, args.fps)
    print(f"initial edges: {len(edges)} (skipped: {skipped_frames} for n_frames, "
          f"{skipped_obs} for n_total, {skipped_scatter} for pair MAD, {skipped_sigma} for uncertainty)")
    if not edges:
        sys.exit("No usable edges — nothing to solve.")

    if args.reference_json:
        all_nodes = read_reference_folders(args.reference_json)
    else:
        all_nodes = sorted({n for e in edges for n in e[:2]})
        print("[warn] no --reference-json given; cameras with zero pair estimates cannot be detected by name.")

    edge_nodes = {n for e in edges for n in e[:2]}
    unknown = edge_nodes - set(all_nodes)
    if unknown:
        print(f"[warn] edges reference cameras not in reference json: {sorted(unknown)}")
        all_nodes = sorted(set(all_nodes) | unknown)

    deg = defaultdict(int)
    for a, b, *_ in edges:
        deg[a] += 1
        deg[b] += 1
    weak = [n for n in all_nodes if deg[n] < args.min_camera_degree]
    if weak:
        print(f"weak cameras (degree < {args.min_camera_degree}):")
        for n in weak:
            print(f"  {n}: degree={deg[n]}")

    # A bridge or leaf can always be fit with zero residual, so residual checks
    # cannot validate it. Restrict solving to the requested graph k-core first.
    core_nodes = k_core(edges, all_nodes, args.min_camera_degree)
    weakly_supported = sorted(set(all_nodes) - set(core_nodes))
    if not core_nodes:
        sys.exit(f"No cameras survive min degree {args.min_camera_degree}; add frames/pairs or lower the threshold.")
    core_edges = [e for e in edges if e[0] in core_nodes and e[1] in core_nodes]

    # --- pick the largest connected component; everything else is unconstrained.
    comps = components(core_edges, core_nodes)
    comps.sort(key=len, reverse=True)
    solved_nodes = comps[0]
    unconstrained = sorted(set(weakly_supported) | {n for c in comps[1:] for n in c})
    if unconstrained:
        print(f"[warn] {len(unconstrained)} camera(s) NOT connected to the main sync graph: {unconstrained}")
        print("       Their offsets are NOT measured. Add frames/pairs covering these cameras.")
    solved_set = set(solved_nodes)
    active = [e for e in core_edges if e[0] in solved_set and e[1] in solved_set]

    # --- robust loop: greedy worst-edge dropping with disconnect guard.
    #
    # When the k-core sits exactly at the degree floor, no edge can be removed
    # without demoting one of its endpoints. The old loop treated that as "stop
    # and keep the outlier", which silently retained arbitrarily bad edges AND
    # reported 0% dropped, skipping the grade downgrade. Retaining a known
    # outlier is never the safer choice: demote the cameras it was propping up
    # and report them as unconstrained instead.
    removed = []
    demoted = []
    x = None
    for it in range(args.max_iters):
        x = solve_component(active, solved_nodes, period_sec)
        res = residuals(active, x, period_sec)
        errs = np.asarray([r[1] for r in res])
        ws = np.asarray([r[6] for r in res])
        rms = float(np.sqrt(np.mean(errs ** 2)))
        wrms = float(np.sqrt(np.sum(ws * errs ** 2) / np.sum(ws)))
        print(f"iter {it}: edges={len(active)} cams={len(solved_nodes)} RMS={rms:.3f}ms "
              f"weighted={wrms:.3f}ms median={float(np.median(errs)):+.3f}ms MAD={mad(errs):.3f}ms")
        ordered = sorted(res, key=lambda r: r[0], reverse=True)
        if ordered[0][0] <= args.drop_ms:
            break

        # Preferred: a removal that leaves the solved node set intact.
        chosen_drop = None
        for candidate in ordered:
            if candidate[0] <= args.drop_ms:
                break
            _, err_ms, a, b, obs_ms, pred_ms, w = candidate
            trial = [e for e in active if not (e[0] == a and e[1] == b)]
            if len(components(trial, solved_nodes)) > 1:
                continue
            trial_degree = defaultdict(int)
            for ta, tb, *_ in trial:
                trial_degree[ta] += 1
                trial_degree[tb] += 1
            if min(trial_degree[n] for n in solved_nodes) < args.min_camera_degree:
                continue
            chosen_drop = (candidate, trial, solved_nodes, [])
            break

        # Fallback: drop the worst edge anyway and re-derive which cameras are
        # still supported. Cameras that fall out were resting on that outlier.
        if chosen_drop is None:
            _, err_ms, a, b, obs_ms, pred_ms, w = ordered[0]
            trial = [e for e in active if not (e[0] == a and e[1] == b)]
            kept = k_core(trial, solved_nodes, args.min_camera_degree)
            sub = [e for e in trial if e[0] in set(kept) and e[1] in set(kept)]
            comps = [c for c in components(sub, kept) if len(c) > 1]
            new_nodes = max(comps, key=len) if comps else []
            if len(new_nodes) < args.min_solved_cameras:
                print(f"not dropping {a}-{b} (err={err_ms:+.3f}ms): removing it would leave only "
                      f"{len(new_nodes)} supported camera(s), below --min-solved-cameras "
                      f"{args.min_solved_cameras}. Outlier RETAINED — treat the fit as unreliable.")
                break
            lost = sorted(set(solved_nodes) - set(new_nodes))
            keep = set(new_nodes)
            chosen_drop = (ordered[0],
                           [e for e in sub if e[0] in keep and e[1] in keep],
                           new_nodes, lost)

        (_, err_ms, a, b, obs_ms, pred_ms, w), trial, new_nodes, lost = chosen_drop
        print(f"  drop {a}-{b}: err={err_ms:+.3f}ms obs={obs_ms:+.3f}ms pred={pred_ms:+.3f}ms")
        if lost:
            print(f"    -> demoting {len(lost)} camera(s) that rested on it: {lost}")
            demoted.extend(lost)
        active = trial
        solved_nodes = list(new_nodes)
        solved_set = set(solved_nodes)
        removed.append({"cam_a": a, "cam_b": b, "err_ms": err_ms, "obs_ms": obs_ms,
                        "pred_ms": pred_ms, "demoted": lost})

    if demoted:
        unconstrained = sorted(set(unconstrained) | set(demoted))

    x = solve_component(active, solved_nodes, period_sec)
    res = residuals(active, x, period_sec)
    surviving_outliers = [r for r in res if r[0] > args.drop_ms]

    if removed:
        print("\nremoved edges:")
        for r in removed:
            print(f"  {r['cam_a']}-{r['cam_b']}: err={r['err_ms']:+.3f}ms obs={r['obs_ms']:+.3f}ms pred={r['pred_ms']:+.3f}ms")

    print("\nfinal worst 20 edges:")
    for _, err_ms, a, b, obs_ms, pred_ms, w in sorted(res, reverse=True)[:20]:
        print(f"  {a}-{b}: err={err_ms:+.3f}ms obs={obs_ms:+.3f}ms pred={pred_ms:+.3f}ms")

    # --- per-camera diagnostics.
    cam_signed = defaultdict(list)   # residual oriented so + means this camera's offset sits high
    for _, err_ms, a, b, *_ in res:
        cam_signed[a].append(+err_ms)
        cam_signed[b].append(-err_ms)
    deg_final = {n: len(cam_signed.get(n, [])) for n in solved_nodes}
    propagated_sigma = graph_uncertainty_ms(active, solved_nodes)
    cam_stats = {}
    for n in solved_nodes:
        r = np.asarray(cam_signed.get(n, []), dtype=np.float64)
        if len(r) == 0:
            cam_stats[n] = {"degree": 0, "bias_ms": float("nan"), "sigma_ms": float("nan")}
            continue
        bias = float(np.mean(r))
        rmse = float(np.sqrt(np.mean(r ** 2)))
        cam_stats[n] = {
            "degree": int(len(r)),
            "bias_ms": bias,                               # systematic disagreement with its edges
            # Covariance from the weighted graph Laplacian. Unlike residual
            # RMSE, this remains large for weak/bridge-like topology.
            "sigma_ms": propagated_sigma.get(n, float("nan")),
            "residual_rmse_ms": rmse,
        }

    # --- split-half empirical validation.
    split_unc = None
    split_note = "not run (no --pair-obs-csv)"
    if args.pair_obs_csv:
        try:
            per_pair, frames_list = read_observations(args.pair_obs_csv, args.fps)
            active_pairs = {(e[0], e[1]) for e in active}
            split_unc, err = split_half_solve(
                per_pair, frames_list, args.fps, active_pairs, solved_nodes, period_sec)
            split_note = err or f"ok ({len(frames_list)} frames split odd/even)"
        except Exception as e:  # validation must never kill the solve
            split_note = f"failed: {e}"
    if split_unc is None and args.pair_obs_csv:
        print(f"[warn] split-half validation unavailable: {split_note}")

    # --- write output JSON (PLUS convention), circular-median-centered.
    #
    # The gauge matters here even though a global time shift is harmless on its
    # own: unconstrained cameras are emitted as exactly 0.0, so any nonzero
    # center becomes a real relative shift between the solved and zero-filled
    # groups. A linear median cannot deliver a zero center when the phase
    # cluster straddles the wrap boundary.
    vals = np.asarray([x[n] for n in solved_nodes], dtype=np.float64)
    centered, center, arc_fraction = center_phases(vals, period_sec)
    out = {n: float(v) for n, v in zip(solved_nodes, centered)}
    # Verify with the same estimator the centering used, not circular_median:
    # circular_median is restricted to sample points and cannot hit zero for an
    # even count, which would false-alarm on every even-sized camera set.
    _lifted, _ = unwrap_about(np.asarray(list(out.values())), period_sec)
    residual_center_ms = 1000.0 * float(np.median(_lifted))
    if abs(residual_center_ms) > 1e-6:
        print(f"[warn] output gauge is off-center by {residual_center_ms:+.4f} ms after centering.")
    if arc_fraction > 0.8:
        print(f"[warn] solved phases span {arc_fraction:.0%} of one frame period — the cluster wraps "
              "around the circle, so unwrapping (and therefore the gauge) is ambiguous. "
              "Integer-frame assignment cannot be recovered from phase alone.")
    if not args.omit_unconstrained:
        for n in unconstrained:
            out[n] = 0.0
    out = {k: out[k] for k in sorted(out)}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=4)

    # ------------------------------------------------------------------ verdict
    T_ms = 1000.0 / args.fps
    errs = np.asarray([r[1] for r in res], dtype=np.float64)
    edge_rms = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("nan")
    edge_mad = mad(errs)

    biases = {n: abs(s["bias_ms"]) for n, s in cam_stats.items() if np.isfinite(s["bias_ms"])}
    sigmas = {n: s["sigma_ms"] for n, s in cam_stats.items() if np.isfinite(s["sigma_ms"])}
    worst_bias_cam = max(biases, key=biases.get) if biases else None
    worst_sigma_cam = max(sigmas, key=sigmas.get) if sigmas else None
    metrics = []
    if worst_bias_cam:
        metrics.append(biases[worst_bias_cam])
    if worst_sigma_cam:
        metrics.append(sigmas[worst_sigma_cam])
    worst_split_cam = None
    if split_unc:
        worst_split_cam = max(split_unc, key=split_unc.get)
        metrics.append(split_unc[worst_split_cam])
    score_ms = max(metrics) if metrics else float("nan")

    drop_frac = len(removed) / max(len(active) + len(removed), 1)
    grades = [(0.05, "EXCELLENT"), (0.10, "GOOD"), (0.25, "MARGINAL"), (float("inf"), "POOR")]
    grade = next(g for thr, g in grades if score_ms <= thr * T_ms) if np.isfinite(score_ms) else "UNKNOWN"
    base_grade = grade

    # How far the score sits from the next grade up. A hard threshold turns a
    # 0.1% miss into a different word, which reads as a quality gap it is not:
    # a run 2 us over the GOOD line and one 1.6 ms over it both print MARGINAL.
    grade_margin_ms = None
    borderline = False
    if np.isfinite(score_ms):
        _gi = [g for _, g in grades].index(base_grade)
        if _gi > 0:
            grade_margin_ms = float(score_ms - grades[_gi - 1][0] * T_ms)
            borderline = bool(0.0 < grade_margin_ms <= 0.05 * grades[_gi - 1][0] * T_ms)
    notes = []
    order = [g for _, g in grades]

    def downgrade(reason):
        nonlocal grade
        grade = order[min(order.index(grade) + 1, len(order) - 1)]
        notes.append(reason)

    if drop_frac > 0.25:
        downgrade(f"{drop_frac:.0%} of edges were dropped as outliers — the pairwise estimates "
                  "themselves are noisy; downgraded one level.")
    if surviving_outliers:
        worst = max(r[0] for r in surviving_outliers)
        downgrade(f"{len(surviving_outliers)} edge(s) still exceed the {args.drop_ms:.1f} ms outlier "
                  f"threshold (worst {worst:.3f} ms) and could not be removed without collapsing the "
                  "graph; the fit is contaminated by them. Downgraded one level.")
    if demoted:
        notes.append(f"{len(sorted(set(demoted)))} camera(s) were demoted to unconstrained because "
                     f"their support included a dropped outlier: {sorted(set(demoted))}")

    # Coverage cap. The per-camera score only describes cameras that survived,
    # so aggressive pruning makes the grade look BETTER while the delivered sync
    # gets worse — unmeasured cameras ship as 0.0. Grade the delivered sync.
    coverage = len(solved_nodes) / max(len(all_nodes), 1)
    cap = "POOR" if coverage < 0.25 else ("MARGINAL" if coverage < 0.50 else None)
    if cap and order.index(grade) < order.index(cap):
        notes.append(f"grade capped at {cap}: only {len(solved_nodes)}/{len(all_nodes)} "
                     f"({coverage:.0%}) of cameras are measured, and the score above describes "
                     "only those. The rest ship unmeasured.")
        grade = cap
    min_deg = min(deg_final.values()) if deg_final else 0
    if min_deg < args.min_camera_degree:
        notes.append(f"some cameras have degree < {args.min_camera_degree} (min={min_deg}); "
                     "their offsets rest on very few constraints.")
    if unconstrained:
        disposition = "omitted from output" if args.omit_unconstrained else "written as 0.0"
        notes.append(f"{len(unconstrained)} camera(s) are unconstrained and were {disposition} — "
                     f"do NOT trust them: {unconstrained}")

    failure_modes = []
    if unconstrained:
        failure_modes.append("graph_coverage")
    if split_unc and max(split_unc.values()) > 0.10 * T_ms:
        failure_modes.append("temporal_measurement_instability")
    if drop_frac > 0.15 or edge_mad > 0.05 * T_ms or surviving_outliers:
        failure_modes.append("pairwise_inconsistency")

    recommendation = {
        "EXCELLENT": "Safe to use for 4DGS point generation as-is. Residual sync error is negligible relative to the frame period.",
        "GOOD": "Usable for 4DGS init. Expect at most very minor artifacts on the fastest motion; letting the trainer refine offsets from this baseline should fully converge.",
        "MARGINAL": "Usable only as an untrusted starting point. Do not automatically add more low-motion frames: they cannot make phase observable and may worsen the graph. Add data only if it contains stronger smooth motion; otherwise use an external sync cue or retain the timecode-only prior.",
        "POOR": "Do not ship. The solved offsets disagree with their own measurements. Re-run only with a stronger motion/sync event or corrected calibration; otherwise use an external sync cue or retain the timecode-only prior.",
        "UNKNOWN": "Could not compute quality metrics.",
    }[grade]

    print("\n" + "=" * 22 + " SYNC QUALITY ASSESSMENT " + "=" * 22)
    print(f"frame period:              {T_ms:.3f} ms  ({args.fps:g} fps)")
    print(f"cameras solved:            {len(solved_nodes)}/{len(all_nodes)}"
          + (f"  (unconstrained: {', '.join(unconstrained)})" if unconstrained else ""))
    print(f"edges used / dropped:      {len(active)} / {len(removed)}  ({drop_frac:.1%} dropped)")
    print(f"edge residual RMS / MAD:   {edge_rms:.3f} / {edge_mad:.3f} ms")
    if surviving_outliers:
        print(f"UNREMOVED outliers:        {len(surviving_outliers)} edge(s) > {args.drop_ms:.1f} ms "
              f"(worst {max(r[0] for r in surviving_outliers):.3f} ms) — still in the fit")
    if demoted:
        print(f"cameras demoted:           {len(set(demoted))}  {sorted(set(demoted))}")
    print(f"output gauge center:       {1000*center:+.3f} ms (residual {residual_center_ms:+.4f} ms), "
          f"cluster spans {arc_fraction:.0%} of a frame")
    if worst_bias_cam:
        print(f"worst per-cam bias:        {biases[worst_bias_cam]:.3f} ms  (cam {worst_bias_cam})")
    if worst_sigma_cam:
        print(f"worst per-cam uncertainty: {sigmas[worst_sigma_cam]:.3f} ms  (cam {worst_sigma_cam})")
    if split_unc:
        med_split = float(np.median(list(split_unc.values())))
        print(f"split-half repeatability:  median {med_split:.3f} ms, worst {split_unc[worst_split_cam]:.3f} ms (cam {worst_split_cam})")
    else:
        print(f"split-half validation:     {split_note}")
    if np.isfinite(score_ms):
        print(f"\nVERDICT: {grade} — worst-case per-camera error ~{score_ms:.2f} ms ({score_ms / T_ms:.3f} frames)")
        if grade_margin_ms is not None:
            _gi = [g for _, g in grades].index(base_grade)
            nxt = grades[_gi - 1][1]
            if borderline:
                print(f"         BORDERLINE: missed {nxt} by {grade_margin_ms:+.4f} ms "
                      f"({grade_margin_ms / T_ms:+.5f} frames) — this is a threshold artifact, "
                      f"not a meaningful quality gap. Judge it as ~{nxt}.")
            else:
                print(f"         (missed {nxt} by {grade_margin_ms:+.3f} ms)")
    else:
        print(f"\nVERDICT: {grade}")
    print(f"RECOMMENDATION: {recommendation}")
    for n in notes:
        print(f"NOTE: {n}")

    # cameras worth a second look: bias notably above the population
    if biases:
        b_med = float(np.median(list(biases.values())))
        b_mad = mad(list(biases.values()))
        flagged = sorted((n for n, v in biases.items() if v > max(2.0 * args.drop_ms / 2, b_med + 4 * b_mad)),
                         key=lambda n: -biases[n])
        if flagged:
            print("cameras with systematic bias (possible CALIBRATION error masquerading as timing):")
            for n in flagged[:10]:
                print(f"  {n}: bias={cam_stats[n]['bias_ms']:+.3f} ms  degree={cam_stats[n]['degree']}")
    print("=" * 69)

    report = {
        "fps": args.fps,
        "phase_period_frames": args.phase_period_frames,
        "frame_period_ms": T_ms,
        "cameras_total": len(all_nodes),
        "cameras_solved": len(solved_nodes),
        "coverage_fraction": coverage,
        "unconstrained_cameras": unconstrained,
        "edges_initial": len(edges),
        "edges_skipped": {
            "few_frames": skipped_frames,
            "few_observations": skipped_obs,
            "high_pair_mad": skipped_scatter,
            "high_pair_uncertainty": skipped_sigma,
        },
        "min_camera_degree": args.min_camera_degree,
        "edges_used": len(active),
        "edges_removed": removed,
        "cameras_demoted": sorted(set(demoted)),
        "edges_over_drop_threshold": [
            {"cam_a": r[2], "cam_b": r[3], "err_ms": r[1], "obs_ms": r[4], "pred_ms": r[5]}
            for r in sorted(surviving_outliers, key=lambda r: -r[0])
        ],
        "output_gauge": {
            "center_ms": 1000.0 * center,
            "residual_center_ms": residual_center_ms,
            "arc_fraction_of_period": arc_fraction,
        },
        "edge_residual_rms_ms": edge_rms,
        "edge_residual_mad_ms": edge_mad,
        "per_camera": {n: cam_stats[n] for n in solved_nodes},
        "split_half": {"note": split_note, "uncertainty_ms": split_unc},
        "verdict": grade,
        "verdict_before_downgrades": base_grade,
        "grade_margin_ms": grade_margin_ms,
        "borderline": borderline,
        "failure_modes": failure_modes,
        "worst_case_error_ms": score_ms if np.isfinite(score_ms) else None,
        "recommendation": recommendation,
        "notes": notes,
        "convention": "PLUS: actual_time = nominal_time + offset_sec[camera]",
        "offsets_sec": out,
    }
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nwrote {args.out_json} and {args.report_json}")
    print(f"offset range: {1000 * min(out.values()):+.3f}..{1000 * max(out.values()):+.3f} ms")


if __name__ == "__main__":
    main()
