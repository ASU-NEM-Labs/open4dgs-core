"""Build per-camera 3D subject-volume gates for mask generation.

A clean-plate matte answers "did this pixel change?", which is the wrong
question for anything outside the capture volume: crew in the control area,
light stands and cables all change legitimately and get matted in. This tool
answers the other question geometrically. A cylinder is fitted around the
subject, projected into every calibrated camera, and everything outside the
projected silhouette is background by construction.

The gate is a cone, not a depth test. It removes background *beside* the
subject and never background *behind* them, and on a tightly framed camera the
volume fills the frame and the gate is inert. Run --preview first and read the
coverage stats before wiring it into a mask run.

The volume is fitted automatically:
  * scene up   = direction orthogonal to every camera's image-right axis
                 (mean-of-camera-up is biased by tilt on a PTZ rig)
  * aim centre = least-squares intersection of the optical axes
  * axis/extent= robust core of the sparse points3D, when the model has any

Any of --center / --radius / --z-min / --z-max overrides the corresponding
fitted value.

Output:
  output_dir/<camera>.png    255 inside the volume, 0 outside
  output_dir/volume.json     the fitted volume, for reproducibility
  output_dir/preview.jpg     with --preview

Usage:
    # 1. fit and eyeball, no gates written
    python build_volume_gate.py \
        --sparse /path/to/sparse/0 --image-dir /path/to/images \
        --preview 8 --dry-run

    # 2. score the gate against masks you already have
    python build_volume_gate.py \
        --sparse /path/to/sparse/0 --masks-dir /path/to/masks --dry-run

    # 3. write the gates
    python build_volume_gate.py \
        --sparse /path/to/sparse/0 --output-dir /path/to/gate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# COLMAP model reading
# =============================================================================

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_colmap_text(sparse_dir):
    """Read cameras.txt / images.txt / points3D.txt."""
    cameras = {}
    with open(sparse_dir / "cameras.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cameras[int(p[0])] = {
                "model": p[1],
                "width": int(p[2]),
                "height": int(p[3]),
                "params": np.array([float(v) for v in p[4:]]),
            }

    images = []
    with open(sparse_dir / "images.txt") as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        images.append({
            "R": qvec2rotmat([float(v) for v in p[1:5]]),
            "t": np.array([float(v) for v in p[5:8]]),
            "camera_id": int(p[8]),
            "name": p[9],
        })

    points = []
    pts_path = sparse_dir / "points3D.txt"
    if pts_path.exists():
        with open(pts_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                points.append([float(p[1]), float(p[2]), float(p[3])])

    return cameras, images, np.array(points).reshape(-1, 3)


def read_colmap_binary(sparse_dir):
    """Read a .bin model via pycolmap."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    cameras = {}
    for cam_id, cam in rec.cameras.items():
        cameras[cam_id] = {
            "model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": np.array(cam.params),
        }

    images = []
    for img in rec.images.values():
        world_from_cam = img.cam_from_world()
        images.append({
            "R": np.array(world_from_cam.rotation.matrix()),
            "t": np.array(world_from_cam.translation),
            "camera_id": img.camera_id,
            "name": img.name,
        })
    images.sort(key=lambda im: im["name"])

    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    return cameras, images, points


def read_model(sparse_dir):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.txt").exists():
        return read_colmap_text(sparse_dir)
    if (sparse_dir / "cameras.bin").exists():
        return read_colmap_binary(sparse_dir)
    raise SystemExit(f"No cameras.txt or cameras.bin in {sparse_dir}")


def camera_name(image_name):
    """images/0000/000001.jpg -> 0000 ;  JCam000_synced.jpg -> JCam000_synced

    Mask directories are per camera, so a model whose NAME carries a frame path
    keys on the directory, and a model with one image per camera keys on the
    stem.
    """
    parts = Path(image_name).parts
    if len(parts) > 1:
        return parts[0]
    return Path(image_name).stem


# =============================================================================
# Projection
# =============================================================================

def project(points_world, R, t, cam):
    """Project world points into a camera. Returns (uv, in_front)."""
    X = (R @ points_world.T).T + t
    z = X[:, 2]
    in_front = z > 1e-6
    z_safe = np.where(in_front, z, 1.0)
    x, y = X[:, 0] / z_safe, X[:, 1] / z_safe

    model, prm = cam["model"], cam["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = prm[0], prm[1], prm[2]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = prm[0], prm[1], prm[2], prm[3]
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = prm[0], prm[1], prm[2]
        fx = fy = f
        k = prm[3:]
        r2 = x * x + y * y
        d = 1.0 + k[0] * r2 + (k[1] * r2 * r2 if len(k) > 1 else 0.0)
        x, y = x * d, y * d
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = prm[0], prm[1], prm[2], prm[3]
        k1, k2, p1, p2 = prm[4], prm[5], prm[6], prm[7]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_t = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_t = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = x_t, y_t
    else:
        raise SystemExit(f"Unsupported camera model: {model}")

    return np.stack([fx * x + cx, fy * y + cy], axis=1), in_front


# =============================================================================
# Volume fitting
# =============================================================================

def fit_scene_up(rotations):
    """Scene up = the direction orthogonal to every camera's image-right axis.

    Averaging each camera's up vector fails on a PTZ rig: a camera tilted 40
    degrees down has its up vector tilted 40 degrees too. The image-right axis
    stays perpendicular to world up under any amount of tilt, and only real
    camera roll perturbs it.
    """
    right = np.array([R.T @ np.array([1.0, 0.0, 0.0]) for R in rotations])
    _, vecs = np.linalg.eigh(right.T @ right)
    up = vecs[:, 0]

    naive = np.array([R.T @ np.array([0.0, -1.0, 0.0]) for R in rotations]).mean(axis=0)
    if up @ naive < 0:
        up = -up

    resid = np.degrees(np.arcsin(np.clip(np.abs(right @ up), 0.0, 1.0)))
    return up, resid


def fit_aim_center(centers, axes):
    """Least-squares intersection of the optical axes."""
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, axes):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        M += P
        b += P @ c
    return np.linalg.solve(M, b)


def ground_basis(up):
    seed = np.array([1.0, 0.0, 0.0])
    if abs(up @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def fit_volume_from_points(points, aim_center, up, e1, e2):
    """Robust core of the sparse cloud, as (axis_point, radius, z_min, z_max).

    Sparse models built against subject masks still carry leaked set geometry,
    so the axis is found by iteratively re-centring on the densest 60% rather
    than by a plain mean.
    """
    rel = points - aim_center
    xy = np.stack([rel @ e1, rel @ e2], axis=1)
    h = rel @ up

    c2 = np.median(xy, axis=0)
    for _ in range(12):
        r = np.linalg.norm(xy - c2, axis=1)
        keep = r < max(np.percentile(r, 60), 0.35)
        if keep.sum() < 8:
            break
        c2 = xy[keep].mean(axis=0)

    r = np.linalg.norm(xy - c2, axis=1)
    core = r < 1.0
    if core.sum() < 32:
        return None

    axis_point = aim_center + c2[0] * e1 + c2[1] * e2
    return {
        "axis_point": axis_point,
        "radius": float(np.percentile(r[core], 99)),
        "z_min": float(np.percentile(h[core], 1)),
        "z_max": float(np.percentile(h[core], 99)),
        "n_core": int(core.sum()),
        "n_total": int(len(points)),
    }


def cylinder_surface(axis_point, up, e1, e2, radius, z_min, z_max, n_theta=128, n_z=16):
    """Dense sample of the cylinder surface, for a projected convex hull."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ring = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    heights = np.linspace(z_min, z_max, n_z)
    pts = [axis_point + radius * ring + z * up for z in heights]
    pts.append(np.repeat((axis_point + z_min * up)[None], 1, axis=0))
    pts.append(np.repeat((axis_point + z_max * up)[None], 1, axis=0))
    return np.vstack(pts)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build per-camera 3D subject-volume gates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sparse", required=True,
                    help="COLMAP model dir (sparse/0), text or binary")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the gate PNGs")
    ap.add_argument("--image-dir", default=None,
                    help="Image root, for the preview overlay only")
    ap.add_argument("--masks-dir", default=None,
                    help="Score the gate against existing masks and report how "
                         "much of their foreground it would remove")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fit, report and preview without writing gates")

    ap.add_argument("--center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="Override the fitted axis point")
    ap.add_argument("--radius", type=float, default=None,
                    help="Override the fitted cylinder radius, metres")
    ap.add_argument("--z-min", type=float, default=None,
                    help="Override the fitted lower extent along up, metres")
    ap.add_argument("--z-max", type=float, default=None,
                    help="Override the fitted upper extent along up, metres")
    ap.add_argument("--margin", type=float, default=0.35,
                    help="Expand the fitted volume by this many metres. The "
                         "sparse cloud has no points on a bow or on hair, so a "
                         "fit with no margin clips them.")
    ap.add_argument("--dilate-px", type=int, default=24,
                    help="Grow the projected silhouette by this many pixels")

    ap.add_argument("--preview", type=int, default=0,
                    help="Write a contact sheet of this many cameras")
    ap.add_argument("--mask-size", default=None, metavar="WxH",
                    help="Gate resolution if it differs from the camera size")

    args = ap.parse_args()

    cameras, images, points = read_model(args.sparse)
    print(f"Model: {len(cameras)} cameras, {len(images)} images, {len(points)} points3D")
    models = sorted({c["model"] for c in cameras.values()})
    print(f"Camera models: {', '.join(models)}")

    rotations = [im["R"] for im in images]
    centers = np.array([-im["R"].T @ im["t"] for im in images])
    axes = np.array([im["R"].T @ np.array([0.0, 0.0, 1.0]) for im in images])

    up, up_resid = fit_scene_up(rotations)
    aim_center = fit_aim_center(centers, axes)
    e1, e2 = ground_basis(up)

    print(f"\nScene up: {np.round(up, 4)}")
    print(f"  camera-right orthogonality residual: median {np.median(up_resid):.2f} deg, "
          f"p90 {np.percentile(up_resid, 90):.2f}, max {up_resid.max():.2f}")
    if np.median(up_resid) > 15.0:
        print("  WARNING: large residual. Cameras are rolled, or the model is "
              "misregistered. Check the preview carefully.")
    print(f"Aim centre: {np.round(aim_center, 4)}")

    fit = fit_volume_from_points(points, aim_center, up, e1, e2) if len(points) >= 32 else None
    if fit is not None:
        print(f"\nFitted from points3D ({fit['n_core']}/{fit['n_total']} in core):")
        print(f"  radius {fit['radius']:.2f} m, extent [{fit['z_min']:.2f}, {fit['z_max']:.2f}] m")
        print(f"  axis offset from aim centre: "
              f"{np.linalg.norm(fit['axis_point'] - aim_center):.3f} m")
        axis_point = fit["axis_point"]
        radius, z_min, z_max = fit["radius"], fit["z_min"], fit["z_max"]
    else:
        print("\nNo usable points3D. Falling back to the aim centre; "
              "set --radius / --z-min / --z-max explicitly.")
        axis_point = aim_center
        radius, z_min, z_max = 1.0, -1.2, 1.2

    radius += args.margin
    z_min -= args.margin
    z_max += args.margin

    if args.center is not None:
        axis_point = np.array(args.center)
    if args.radius is not None:
        radius = args.radius
    if args.z_min is not None:
        z_min = args.z_min
    if args.z_max is not None:
        z_max = args.z_max

    print(f"\nVolume in use: radius {radius:.2f} m, extent [{z_min:.2f}, {z_max:.2f}] m "
          f"({z_max - z_min:.2f} m tall)")

    surface = cylinder_surface(axis_point, up, e1, e2, radius, z_min, z_max)

    override_wh = None
    if args.mask_size:
        w, h = args.mask_size.lower().split("x")
        override_wh = (int(w), int(h))

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    kernel = None
    if args.dilate_px > 0:
        k = args.dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    coverage = []
    degenerate = []
    scored_total = 0
    scored_outside = 0
    per_camera = []
    preview_tiles = []

    for idx, im in enumerate(images):
        cam = cameras[im["camera_id"]]
        name = camera_name(im["name"])
        W, H = (override_wh if override_wh else (cam["width"], cam["height"]))
        sx, sy = W / cam["width"], H / cam["height"]

        uv, in_front = project(surface, im["R"], im["t"], cam)

        gate = np.zeros((H, W), np.uint8)
        if not in_front.all():
            # Volume straddles the image plane; a hull is meaningless. Never
            # crop in that case.
            gate[:] = 255
            degenerate.append(name)
        else:
            pts = np.stack([uv[:, 0] * sx, uv[:, 1] * sy], axis=1).astype(np.float32)
            hull = cv2.convexHull(pts).astype(np.int32)
            cv2.fillConvexPoly(gate, hull, 255)
            if kernel is not None:
                gate = cv2.dilate(gate, kernel)

        frac = float(gate.mean() / 255.0)
        coverage.append(frac)

        if args.masks_dir:
            mpath = None
            for ext in (".png", ".jpg"):
                cand = Path(args.masks_dir) / f"{name}{ext}"
                if cand.exists():
                    mpath = cand
                    break
            if mpath is not None:
                m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    g = gate if m.shape == gate.shape else cv2.resize(
                        gate, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                    fg = m >= 128
                    a = int(fg.sum())
                    o = int((fg & (g == 0)).sum())
                    scored_total += a
                    scored_outside += o
                    per_camera.append((name, a, o, frac))

        if out_dir and not args.dry_run:
            cv2.imwrite(str(out_dir / f"{name}.png"), gate)

        if args.preview and len(preview_tiles) < args.preview:
            step = max(1, len(images) // args.preview)
            if idx % step == 0:
                tile = None
                if args.image_dir:
                    ipath = Path(args.image_dir) / im["name"]
                    if not ipath.exists():
                        alt = list((Path(args.image_dir) / name).glob("*.jpg"))
                        ipath = alt[0] if alt else None
                    if ipath is not None and Path(ipath).exists():
                        tile = cv2.imread(str(ipath))
                if tile is None:
                    tile = np.full((H, W, 3), 40, np.uint8)
                tile = cv2.resize(tile, (W, H))
                shade = tile.copy()
                shade[gate == 0] = (shade[gate == 0] * 0.25).astype(np.uint8)
                tile = shade
                cnts, _ = cv2.findContours(gate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, cnts, -1, (0, 255, 0), 8)
                cv2.putText(tile, f"{name}  {100*frac:.0f}%", (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 8)
                preview_tiles.append(cv2.resize(tile, (640, 360)))

    coverage = np.array(coverage)
    print(f"\n{'='*70}")
    print("GATE COVERAGE  (fraction of frame kept)")
    print(f"{'='*70}")
    print(f"  mean {100*coverage.mean():.1f}%   median {100*np.median(coverage):.1f}%   "
          f"min {100*coverage.min():.1f}%   max {100*coverage.max():.1f}%")
    inert = int((coverage > 0.995).sum())
    print(f"  cameras where the gate is inert (>99.5% kept): {inert}/{len(coverage)}")
    if degenerate:
        print(f"  cameras with the volume straddling the image plane "
              f"(left uncropped): {len(degenerate)}")

    if args.masks_dir and scored_total:
        print(f"\n{'='*70}")
        print("SCORED AGAINST EXISTING MASKS")
        print(f"{'='*70}")
        print(f"  foreground total   : {scored_total/1e6:.1f} Mpx")
        print(f"  outside the gate   : {scored_outside/1e6:.1f} Mpx "
              f"= {100*scored_outside/scored_total:.2f}% removed")
        per_camera.sort(key=lambda r: -(r[2] / max(r[1], 1)))
        print("\n  most affected cameras:")
        for name, a, o, frac in per_camera[:12]:
            print(f"    {name:<20} fg {a/1e6:6.2f} Mpx   removed {100*o/max(a,1):6.2f}%"
                  f"   gate keeps {100*frac:5.1f}%")
        if scored_outside / scored_total < 0.005:
            print("\n  The gate removes almost nothing on this mask set. Either the "
                  "\n  masks are already clean outside the volume, or the framing is "
                  "\n  too tight for a cone to help. Do not wire it in on this "
                  "\n  evidence alone.")

    if preview_tiles:
        while len(preview_tiles) % 4:
            preview_tiles.append(np.zeros_like(preview_tiles[0]))
        rows = [np.hstack(preview_tiles[i:i+4]) for i in range(0, len(preview_tiles), 4)]
        dest = (out_dir / "preview.jpg") if out_dir else Path("gate_preview.jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nPreview: {dest}")

    if out_dir and not args.dry_run:
        meta = {
            "axis_point": [float(v) for v in axis_point],
            "up": [float(v) for v in up],
            "radius": float(radius),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "margin": float(args.margin),
            "dilate_px": int(args.dilate_px),
            "sparse": str(args.sparse),
            "mean_coverage": float(coverage.mean()),
        }
        with open(out_dir / "volume.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Gates:   {out_dir}  ({len(images)} written)")
        print(f"Volume:  {out_dir / 'volume.json'}")
    elif args.dry_run:
        print("\nDry run: no gates written.")


if __name__ == "__main__":
    main()
