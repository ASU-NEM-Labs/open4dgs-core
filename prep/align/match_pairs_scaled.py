#!/usr/bin/env python3
"""Scale-normalised RoMa matching for cross-zoom camera pairs.

Why this exists
---------------
Measured on example capture: RoMa acceptance collapses as the scale mismatch
between two views grows.

    scale ratio   proposed   accepted   rate    median fit ratio
      1.0-1.3x       189        117      62%         0.587
      1.6-2.0x        68         33      49%         0.271
      2.0-2.5x        70         27      39%         0.129
      2.5-3.0x        30          3      10%         0.089
      3.0-6.0x        23          2       9%         0.089

Those high-ratio pairs are exactly the wide-to-telephoto pairs the narrow-FOV
cameras need. This rig has a 3.4x focal spread, and per-camera pose error is
governed by how many accepted pairs a camera has -- so the cameras carrying the
fine detail are starved by a matcher limitation, not by geometry.

The fix is to stop asking the matcher to bridge the scale gap. Both images are
cropped to the region they actually share, predicted from the current solve, and
resized to a common resolution. The matcher then sees roughly 1:1 scale, and
matches are mapped back to original pixel coordinates afterwards.

Why crops go through disk
-------------------------
Both RoMa backends call ``self._model.match(str(path_a), str(path_b))`` -- they
read from the path and use the passed arrays only for their shape. A cropped
array cannot be handed in directly, so crops are written to temporary files and
the array passed alongside is the same crop, keeping the shape bookkeeping
consistent.

Effect on precision
-------------------
RoMa's localisation error is roughly a constant fraction of image width -- 0.86 px
measured on these 3840 px frames, about 0.022%. Cropping does not degrade that,
it improves it on the wide side: a 1000 px shared region resized to 1600 gives
~0.36 px in crop space, which maps back to ~0.22 px in the wide camera's original
pixels. The telephoto, whose crop is downscaled, lands near 0.77 px -- comparable
to what full-frame matching already achieves. So this buys pairs without paying
for them in observation noise.

Output
------
Writes its own cache directory and a report in the same schema as
dynamic_calibration_report.json, so referee_third_view.py and the bundle
adjustment read it unchanged. It never writes into the existing cache or report.
Pass --base-report to emit a merged report containing both sets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referee_third_view import load_reference  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pairs", required=True, help="pairs JSON from select_pairs.py")
    p.add_argument("--reference", required=True,
                   help="Solve providing intrinsics/distortion for the epipolar filter")
    p.add_argument("--images-root", default="reference_images")
    p.add_argument("--image-name", default="000001.jpg")
    p.add_argument("--out-cache-dir", default="dynamic_calibration_cache_scaled")
    p.add_argument("--out-report", required=True)
    p.add_argument("--base-report", default=None,
                   help="Merge these matches into the output report as well")
    p.add_argument("--min-scale-ratio", type=float, default=1.8,
                   help="Only process pairs at or above this mismatch; below it plain "
                        "full-frame matching already works")
    p.add_argument("--max-scale-ratio", type=float, default=2.4,
                   help="Skip pairs above this mismatch. Measured on example capture: "
                        "acceptance is 86%% below 2.2x and 5%% above 2.4x, and the cliff is "
                        "NOT a framing artifact (aspect mismatch separates at AUC 0.476, "
                        "i.e. not at all). Resampling cannot restore detail the wide camera "
                        "never resolved, so pairs past this are wasted GPU time.")
    p.add_argument("--pad-frac", type=float, default=0.12,
                   help="Expand each crop box by this fraction, to tolerate error in "
                        "the predicted overlap")
    p.add_argument("--target-long-side", type=int, default=0,
                   help="Long side both crops are resized to. 0 = auto: the LARGER of the "
                        "two crops, so the smaller is upscaled to match and neither is "
                        "downscaled. A fixed value below the larger crop throws away "
                        "resolution the full-frame path already had, which is worse than "
                        "not cropping at all.")
    p.add_argument("--max-long-side", type=int, default=3840,
                   help="Cap for the auto target, to bound matcher memory")
    p.add_argument("--min-crop-px", type=int, default=160)
    # RoMa settings, matching solve_4dv_dynamic_calibration.py defaults
    p.add_argument("--roma-backend", default="v2", choices=("v1", "v2"))
    p.add_argument("--roma-model", default="outdoor")
    p.add_argument("--roma-v2-setting", default="precise")
    p.add_argument("--roma-v2-mask-oversample", type=float, default=2.0)
    p.add_argument("--roma-v2-spatial-grid", type=int, default=8)
    p.add_argument("--roma-v2-spatial-fraction", type=float, default=0.5)
    p.add_argument("--roma-num-matches", type=int, default=6000)
    p.add_argument("--roma-min-certainty", type=float, default=0.05)
    p.add_argument("--pair-ransac-px", type=float, default=1.25)
    p.add_argument("--min-pair-inliers", type=int, default=100)
    p.add_argument("--min-pair-fit-ratio", type=float, default=0.30)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_prep_module():
    """Import prep-roma-trigs-memory.py through the solver's own loader.

    Reused rather than reimplemented: that loader registers the module in
    sys.modules before executing it, which @dataclass requires in order to
    resolve cls.__module__, and it puts the module's directory on sys.path so the
    sibling ``from roma_backends import ...`` resolves.
    """
    from solve_4dv_dynamic_calibration import _load_prep_module
    return _load_prep_module()


def pad_box(box, W, H, frac, min_px):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0 -= w * frac
    x1 += w * frac
    y0 -= h * frac
    y1 += h * frac
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if x1 - x0 < min_px:
        x0, x1 = cx - min_px / 2, cx + min_px / 2
    if y1 - y0 < min_px:
        y0, y1 = cy - min_px / 2, cy + min_px / 2
    x0 = int(max(0, np.floor(x0)))
    y0 = int(max(0, np.floor(y0)))
    x1 = int(min(W, np.ceil(x1)))
    y1 = int(min(H, np.ceil(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def crop_resize(img, box, target_long):
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    s = target_long / float(max(h, w))
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    out = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return out, (x0, y0, w / float(nw), h / float(nh))


def undistort(points, cam):
    k, d = cam["k"], cam["d"]
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=np.float64)
    if d.size == 0 or np.abs(d).max() < 1e-12:
        return np.asarray(points, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, d, P=K).reshape(-1, 2)


def fit_fundamental_inliers(pa, pb, thr):
    if len(pa) < 8:
        return np.zeros(len(pa), dtype=bool)
    try:
        _, mask = cv2.findFundamentalMat(
            np.asarray(pa, np.float32), np.asarray(pb, np.float32),
            method=cv2.USAC_MAGSAC, ransacReprojThreshold=float(thr),
            confidence=0.999, maxIters=10000)
    except cv2.error:
        return np.zeros(len(pa), dtype=bool)
    if mask is None:
        return np.zeros(len(pa), dtype=bool)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if len(mask) != len(pa):
        return np.zeros(len(pa), dtype=bool)
    return mask


def main() -> int:
    args = parse_args()
    root = args.data_dir
    rpath = args.reference if os.path.isabs(args.reference) else os.path.join(root, args.reference)
    cams = load_reference(rpath, True)

    with open(args.pairs if os.path.isabs(args.pairs) else os.path.join(root, args.pairs)) as f:
        pairs_doc = json.load(f)
    todo = [p for p in pairs_doc["pairs"]
            if args.min_scale_ratio <= p["scale_ratio"] <= args.max_scale_ratio]
    if args.max_pairs > 0:
        todo = todo[: args.max_pairs]
    print(f"{len(todo)} pairs in scale band "
          f"{args.min_scale_ratio}-{args.max_scale_ratio}x "
          f"(of {len(pairs_doc['pairs'])} proposed)")
    if not todo:
        return 0

    cache_dir = os.path.join(root, args.out_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    prep = load_prep_module()
    model = prep.create_roma_backend(
        backend=args.roma_backend, device=args.device, model=args.roma_model,
        v2_setting=args.roma_v2_setting, symmetric=True, upsample_preds=True,
        v2_mask_oversample=args.roma_v2_mask_oversample,
        v2_spatial_grid=args.roma_v2_spatial_grid,
        v2_spatial_fraction=args.roma_v2_spatial_fraction,
    )

    rows, n_acc = [], 0
    tmp = tempfile.mkdtemp(prefix="trigs-scaled-crop-")
    for i, pr in enumerate(todo, 1):
        a, b = pr["folder_a"], pr["folder_b"]
        pa_path = os.path.join(root, args.images_root, a, args.image_name)
        pb_path = os.path.join(root, args.images_root, b, args.image_name)
        out_npz = os.path.join(cache_dir, f"{a}_{b}_scaled.npz")
        rel_npz = os.path.relpath(out_npz, root)

        if os.path.exists(out_npz) and not args.overwrite:
            d = np.load(out_npz)
            pts_a, pts_b, cert = d["points_a"], d["points_b"], d["certainty"]
        else:
            img_a = cv2.cvtColor(cv2.imread(pa_path), cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(cv2.imread(pb_path), cv2.COLOR_BGR2RGB)
            H, W = img_a.shape[:2]
            box_a = pad_box(pr["crop_a"], W, H, args.pad_frac, args.min_crop_px)
            box_b = pad_box(pr["crop_b"], W, H, args.pad_frac, args.min_crop_px)
            if box_a is None or box_b is None:
                print(f"[{i}/{len(todo)}] {a}<->{b}: degenerate crop, skipped")
                continue
            if args.target_long_side > 0:
                target = args.target_long_side
            else:
                target = min(args.max_long_side,
                             max(box_a[2] - box_a[0], box_a[3] - box_a[1],
                                 box_b[2] - box_b[0], box_b[3] - box_b[1]))
            ca, ta = crop_resize(img_a, box_a, target)
            cb, tb = crop_resize(img_b, box_b, target)
            print(f"      crops {box_a[2]-box_a[0]}x{box_a[3]-box_a[1]} "
                  f"/ {box_b[2]-box_b[0]}x{box_b[3]-box_b[1]} -> target {target} "
                  f"(resample a x{1/ta[2]:.2f}, b x{1/tb[2]:.2f})", flush=True)
            fa = os.path.join(tmp, f"{a}_{b}_a.png")
            fb = os.path.join(tmp, f"{a}_{b}_b.png")
            cv2.imwrite(fa, cv2.cvtColor(ca, cv2.COLOR_RGB2BGR))
            cv2.imwrite(fb, cv2.cvtColor(cb, cv2.COLOR_RGB2BGR))
            ka, kb, cert = prep.run_roma_pair(
                model, Path(fa), Path(fb), ca, cb,
                args.roma_num_matches, args.roma_min_certainty,
                args.pair_ransac_px, 8, args.device,
                use_fundamental_ransac=False,
            )
            if len(ka) == 0:
                print(f"[{i}/{len(todo)}] {a}<->{b}: no matches")
                continue
            # crop pixel -> original pixel
            pts_a = np.stack([ka[:, 0] * ta[2] + ta[0], ka[:, 1] * ta[3] + ta[1]], 1)
            pts_b = np.stack([kb[:, 0] * tb[2] + tb[0], kb[:, 1] * tb[3] + tb[1]], 1)
            pts_a = pts_a.astype(np.float32)
            pts_b = pts_b.astype(np.float32)
            np.savez_compressed(out_npz, points_a=pts_a, points_b=pts_b, certainty=cert)

        ua = undistort(pts_a, cams[a])
        ub = undistort(pts_b, cams[b])
        inl = fit_fundamental_inliers(ua, ub, args.pair_ransac_px)
        n_in = int(inl.sum())
        ratio = float(n_in / max(len(pts_a), 1))
        accepted = bool(n_in >= args.min_pair_inliers and ratio >= args.min_pair_fit_ratio)
        n_acc += int(accepted)
        rows.append({
            "frame": 1, "folder_a": a, "folder_b": b,
            "raw_matches": int(len(pts_a)), "fitted_inliers": n_in,
            "fitted_ratio": round(ratio, 4), "accepted": accepted,
            "scale_ratio": pr["scale_ratio"], "cache": rel_npz,
            "scaled_crop": True,
        })
        print(f"[{i}/{len(todo)}] {a}<->{b} scale={pr['scale_ratio']:.2f}: "
              f"matches={len(pts_a)} fit={n_in} ({100 * ratio:.1f}%) "
              f"{'accepted' if accepted else 'rejected'}", flush=True)

    print(f"\naccepted {n_acc}/{len(rows)} scaled pairs "
          f"({100 * n_acc / max(len(rows), 1):.0f}%)")
    print("  for reference, full-frame matching accepted 10-39% of these same pairs")

    out = {"status": "scaled_crop_matching", "matches": rows}
    if args.base_report:
        bp = args.base_report if os.path.isabs(args.base_report) else os.path.join(root, args.base_report)
        with open(bp) as f:
            base = json.load(f)
        have = {(r["folder_a"], r["folder_b"]) for r in rows if r["accepted"]}
        kept = [m for m in base["matches"]
                if not (m.get("accepted") and (m["folder_a"], m["folder_b"]) in have)]
        out["matches"] = kept + rows
        out["merged_from"] = os.path.basename(bp)
        tot = sum(1 for m in out["matches"] if m.get("accepted"))
        print(f"merged with {os.path.basename(bp)}: {tot} accepted edges total")

    op = args.out_report if os.path.isabs(args.out_report) else os.path.join(root, args.out_report)
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
