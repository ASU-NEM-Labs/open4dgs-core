#!/usr/bin/env python3
"""Scale-normalized discovery with native exact-index match export.

This module consumes provisional-structure overlap proposals, but writes only
raw keypoint-index correspondences into a copied database.  COLMAP performs the
final two-view geometric verification.  A new target observation must close a
cycle through the same provisional 3D point in at least two solved neighbors.
"""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import calibration_v2 as core


def _resized_crop(gray: np.ndarray, box, long_side: int):
    transform = core.crop_transform(box, long_side)
    x0, y0, x1, y1 = map(float, box)
    scale = float(transform[0, 0])
    size = (
        max(2, int(math.ceil((x1 - x0) * scale))),
        max(2, int(math.ceil((y1 - y0) * scale))),
    )
    crop = cv2.warpPerspective(
        gray, transform, size, flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return crop, transform


def _mutual_ratio_matches(descriptors_a, descriptors_b, ratio: float):
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    accepted_forward = {
        left.queryIdx: left.trainIdx for left, right in forward
        if left.distance < ratio * right.distance
    }
    accepted_reverse = {
        left.queryIdx: left.trainIdx for left, right in reverse
        if left.distance < ratio * right.distance
    }
    return [
        (query, train) for query, train in accepted_forward.items()
        if accepted_reverse.get(train) == query
    ]


def _patch_correlation(image_a, point_a, image_b, point_b, radius: int = 7) -> float:
    size = (2 * radius + 1, 2 * radius + 1)
    patch_a = cv2.getRectSubPix(image_a, size, tuple(map(float, point_a))).astype(np.float32)
    patch_b = cv2.getRectSubPix(image_b, size, tuple(map(float, point_b))).astype(np.float32)
    patch_a -= patch_a.mean()
    patch_b -= patch_b.mean()
    denominator = float(np.linalg.norm(patch_a) * np.linalg.norm(patch_b))
    return float(np.sum(patch_a * patch_b) / denominator) if denominator > 1e-6 else -1.0


def _nearest_exact_indices(points: np.ndarray, database_points: np.ndarray, maximum_distance: float):
    tree = cKDTree(np.asarray(database_points, dtype=np.float64))
    distances, indices = tree.query(np.asarray(points, dtype=np.float64), k=1)
    valid = distances <= maximum_distance
    return indices.astype(np.int64), distances, valid


def _refine_native_corners(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    height, width = gray.shape
    safe = (
        (values[:, 0, 0] >= 6) & (values[:, 0, 0] < width - 6)
        & (values[:, 0, 1] >= 6) & (values[:, 0, 1] < height - 6)
    )
    if np.any(safe):
        selected = values[safe].copy()
        cv2.cornerSubPix(
            gray, selected, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
        )
        values[safe] = selected
    return values[:, 0].astype(np.float64)


def discover_matches(
    database_path: Path,
    image_root: Path,
    reconstruction,
    proposals: dict,
    specs: dict | None = None,
    discovery_long_side: int = 1800,
    ratio: float = 0.72,
    minimum_patch_correlation: float = 0.25,
    native_keypoint_radius: float = 2.0,
    minimum_cycle_neighbors: int = 2,
    guided_search_radius: float = 100.0,
    guided_descriptor_ratio: float = 0.90,
):
    con = sqlite3.connect(f"file:{database_path.resolve()}?immutable=1", uri=True)
    try:
        names = {int(row[0]): str(row[1]) for row in con.execute("SELECT image_id,name FROM images")}
        ids_by_name = {name: image_id for image_id, name in names.items()}
        keypoints = core.read_keypoints(con)
        descriptors = {
            int(image_id): np.frombuffer(blob, dtype=np.uint8).reshape(int(rows), int(cols)).copy()
            for image_id, rows, cols, blob in con.execute(
                "SELECT image_id,rows,cols,data FROM descriptors"
            )
        }
        existing_raw = {}
        for pair_id, rows, blob in con.execute("SELECT pair_id,rows,data FROM matches WHERE rows>0"):
            existing_raw[int(pair_id)] = np.frombuffer(blob, dtype=np.uint32).reshape(int(rows), 2).copy()
    finally:
        con.close()

    point_for_observation = {}
    for image_id, image in reconstruction.images.items():
        for keypoint_idx, point in enumerate(image.points2D):
            if point.has_point3D():
                point_for_observation[(int(image_id), keypoint_idx)] = int(point.point3D_id)

    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.00667, edgeThreshold=10)
    # target -> target keypoint -> point3D -> list(neighbor, neighbor keypoint)
    support = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    guided_diagnostics = []

    # First exploit the allowed manifest pose prior directly. Each provisional
    # 3D observation is projected into the weak camera, and descriptor lookup
    # is restricted to a small native-resolution neighborhood. Requiring the
    # same exact target keypoint and provisional point through multiple solved
    # neighbors turns this into a third-view check, not a pose-derived match.
    if specs is not None:
        for target_folder, rows in sorted(proposals.items()):
            target_name = f"{target_folder}/000001.jpg"
            target_id = ids_by_name.get(target_name)
            spec = specs.get(target_folder)
            if target_id is None or spec is None or spec.rotation is None or spec.translation is None:
                continue
            target_tree = cKDTree(keypoints[target_id])
            nominal = core.opencv_params(spec)
            K = np.asarray([
                [nominal[0], 0.0, nominal[2]],
                [0.0, nominal[1], nominal[3]],
                [0.0, 0.0, 1.0],
            ])
            rvec = cv2.Rodrigues(spec.rotation)[0]
            for row in rows:
                neighbor_folder = row["neighbor_folder"]
                neighbor_name = f"{neighbor_folder}/000001.jpg"
                neighbor_id = ids_by_name.get(neighbor_name)
                if neighbor_id is None or neighbor_id not in reconstruction.images:
                    continue
                image = reconstruction.images[neighbor_id]
                observations = [
                    (keypoint_idx, int(point.point3D_id))
                    for keypoint_idx, point in enumerate(image.points2D)
                    if point.has_point3D()
                    and keypoint_idx < len(descriptors.get(neighbor_id, ()))
                ]
                if not observations:
                    continue
                xyz = np.asarray([
                    reconstruction.points3D[point3d_id].xyz
                    for _, point3d_id in observations
                ], dtype=np.float64)
                camera_xyz = xyz @ spec.rotation.T + spec.translation.reshape(1, 3)
                front = camera_xyz[:, 2] > 1e-9
                projected = np.full((len(xyz), 2), np.nan, dtype=np.float64)
                if np.any(front):
                    values, _ = cv2.projectPoints(
                        xyz[front], rvec, spec.translation, K, nominal[4:8]
                    )
                    projected[front] = values[:, 0]

                forward = []
                for (neighbor_keypoint, point3d_id), prediction, in_front in zip(
                    observations, projected, front
                ):
                    if not in_front or not (
                        0 <= prediction[0] < spec.width and 0 <= prediction[1] < spec.height
                    ):
                        continue
                    candidates = target_tree.query_ball_point(prediction, guided_search_radius)
                    if len(candidates) < 2:
                        continue
                    target_descriptors = descriptors[target_id][candidates].astype(np.float32)
                    neighbor_descriptor = descriptors[neighbor_id][neighbor_keypoint].astype(np.float32)
                    distances = np.linalg.norm(target_descriptors - neighbor_descriptor, axis=1)
                    order = np.argsort(distances)
                    if distances[order[0]] >= guided_descriptor_ratio * distances[order[1]]:
                        continue
                    forward.append((
                        int(candidates[int(order[0])]), point3d_id,
                        neighbor_keypoint, float(distances[order[0]]),
                    ))

                # Bidirectional closure: a target observation may accept only
                # the uniquely best projected 3D descriptor in this neighbor.
                by_target = defaultdict(list)
                for candidate in forward:
                    by_target[candidate[0]].append(candidate)
                accepted = []
                for choices in by_target.values():
                    choices.sort(key=lambda value: (value[3], value[1], value[2]))
                    if len(choices) > 1 and choices[0][3] >= (
                        guided_descriptor_ratio * choices[1][3]
                    ):
                        continue
                    accepted.append(choices[0])
                for target_keypoint, point3d_id, neighbor_keypoint, _ in accepted:
                    support[target_id][target_keypoint][point3d_id].append(
                        (neighbor_id, neighbor_keypoint)
                    )
                guided_diagnostics.append({
                    "target": target_folder,
                    "neighbor": neighbor_folder,
                    "projected_3d_observations": len(observations),
                    "forward_ratio_matches": len(forward),
                    "bidirectional_matches": len(accepted),
                })

    pair_diagnostics = []
    for target_folder, rows in sorted(proposals.items()):
        target_name = f"{target_folder}/000001.jpg"
        target_id = ids_by_name.get(target_name)
        if target_id is None:
            continue
        target_gray = cv2.imread(str(image_root / target_name), cv2.IMREAD_GRAYSCALE)
        if target_gray is None:
            continue
        for row in rows:
            neighbor_folder = row["neighbor_folder"]
            neighbor_name = f"{neighbor_folder}/000001.jpg"
            neighbor_id = ids_by_name.get(neighbor_name)
            if neighbor_id is None or neighbor_id not in reconstruction.images:
                continue
            target_box = row.get("target_crop_xyxy")
            neighbor_box = row.get("neighbor_crop_xyxy")
            if target_box is None or neighbor_box is None:
                continue
            neighbor_gray = cv2.imread(str(image_root / neighbor_name), cv2.IMREAD_GRAYSCALE)
            if neighbor_gray is None:
                continue
            crop_a, transform_a = _resized_crop(target_gray, target_box, discovery_long_side)
            crop_b, transform_b = _resized_crop(neighbor_gray, neighbor_box, discovery_long_side)
            keys_a, descriptors_a = sift.detectAndCompute(crop_a, None)
            keys_b, descriptors_b = sift.detectAndCompute(crop_b, None)
            matches = _mutual_ratio_matches(descriptors_a, descriptors_b, ratio)
            correlated = []
            for query_idx, train_idx in matches:
                point_a = keys_a[query_idx].pt
                point_b = keys_b[train_idx].pt
                if _patch_correlation(crop_a, point_a, crop_b, point_b) >= minimum_patch_correlation:
                    correlated.append((query_idx, train_idx))
            crop_points_a = np.asarray([keys_a[a].pt for a, _ in correlated], dtype=np.float64).reshape(-1, 2)
            crop_points_b = np.asarray([keys_b[b].pt for _, b in correlated], dtype=np.float64).reshape(-1, 2)
            native_a = core.transform_points(crop_points_a, np.linalg.inv(transform_a))
            native_b = core.transform_points(crop_points_b, np.linalg.inv(transform_b))
            refined_a = _refine_native_corners(target_gray, native_a)
            refined_b = _refine_native_corners(neighbor_gray, native_b)
            target_indices, target_distances, target_valid = _nearest_exact_indices(
                refined_a, keypoints[target_id], native_keypoint_radius
            ) if len(refined_a) else (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=bool))
            neighbor_indices, neighbor_distances, neighbor_valid = _nearest_exact_indices(
                refined_b, keypoints[neighbor_id], native_keypoint_radius
            ) if len(refined_b) else (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=bool))
            linked = 0
            for index in np.flatnonzero(target_valid & neighbor_valid):
                target_keypoint = int(target_indices[index])
                neighbor_keypoint = int(neighbor_indices[index])
                point3d_id = point_for_observation.get((neighbor_id, neighbor_keypoint))
                if point3d_id is None:
                    continue
                support[target_id][target_keypoint][point3d_id].append((neighbor_id, neighbor_keypoint))
                linked += 1
            pair_diagnostics.append({
                "target": target_folder,
                "neighbor": neighbor_folder,
                "discovery_matches": len(matches),
                "patch_consistent_matches": len(correlated),
                "target_native_keypoint_hits": int(np.count_nonzero(target_valid)),
                "neighbor_native_keypoint_hits": int(np.count_nonzero(neighbor_valid)),
                "bidirectional_native_keypoint_hits": int(np.count_nonzero(
                    target_valid & neighbor_valid
                )),
                "exact_native_3d_links": linked,
            })

    new_by_pair = defaultdict(list)
    cycle_landmarks = conflicts = 0
    for target_id, target_rows in support.items():
        for target_keypoint, point_choices in target_rows.items():
            qualified = [
                (point3d_id, links) for point3d_id, links in point_choices.items()
                if len({neighbor_id for neighbor_id, _ in links}) >= minimum_cycle_neighbors
            ]
            if len(qualified) != 1:
                if qualified:
                    conflicts += 1
                continue
            _, links = qualified[0]
            cycle_landmarks += 1
            for neighbor_id, neighbor_keypoint in sorted(set(links)):
                image_a, image_b = sorted((target_id, neighbor_id))
                pair_id = core.pair_id_from_image_ids(image_a, image_b)
                indices = (
                    (target_keypoint, neighbor_keypoint)
                    if image_a == target_id else (neighbor_keypoint, target_keypoint)
                )
                new_by_pair[pair_id].append(indices)

    merged_by_pair = {}
    for pair_id, new_rows in new_by_pair.items():
        previous = existing_raw.get(pair_id, np.zeros((0, 2), dtype=np.uint32))
        merged = np.vstack([previous, np.asarray(new_rows, dtype=np.uint32)])
        merged_by_pair[pair_id] = np.unique(merged, axis=0)
    report = {
        "schema_version": 1,
        "policy": {
            "descriptor_discovery": "scale-normalized crop SIFT",
            "guided_discovery": "manifest projection plus spatially gated native SIFT",
            "guided_search_radius_px": guided_search_radius,
            "guided_descriptor_ratio": guided_descriptor_ratio,
            "native_endpoint_refinement": "cornerSubPix",
            "bidirectional_closure": "mutual ratio match",
            "third_view_cycle": f"same provisional point through >= {minimum_cycle_neighbors} solved neighbors",
            "coordinate_storage": "existing native keypoint IDs only",
        },
        "candidate_pairs": len(pair_diagnostics),
        "cycle_consistent_landmarks": cycle_landmarks,
        "conflicting_landmarks_rejected": conflicts,
        "affected_pairs": len(merged_by_pair),
        "affected_pair_ids": sorted(int(pair_id) for pair_id in merged_by_pair),
        "new_match_rows": int(sum(len(rows) for rows in new_by_pair.values())),
        "pair_diagnostics": pair_diagnostics,
        "guided_pair_diagnostics": guided_diagnostics,
    }
    return merged_by_pair, names, report


def write_raw_match_list(path: Path, matches_by_pair: dict, image_names: dict) -> None:
    lines = []
    for pair_id, matches in sorted(matches_by_pair.items()):
        image_a, image_b = core.image_ids_from_pair_id(pair_id)
        lines.append(f"{image_names[image_a]} {image_names[image_b]}")
        lines.extend(f"{int(a)} {int(b)}" for a, b in matches)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def clear_pairs_for_reimport(database_path: Path, pair_ids) -> int:
    """Delete only pairs whose complete raw-match union will be reimported."""
    con = sqlite3.connect(database_path)
    pair_ids = tuple(sorted({int(pair_id) for pair_id in pair_ids}))
    try:
        con.execute("BEGIN IMMEDIATE")
        for pair_id in pair_ids:
            con.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
            con.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return len(pair_ids)


def merge_verified_pair_rows(source_database: Path, destination_database: Path, pair_ids) -> dict:
    """Union newly verified geometry with the untouched source geometry.

    COLMAP guided matching may have placed more inliers in
    ``two_view_geometries`` than exist in the raw ``matches`` row. Reimporting
    the raw union must never discard those prior verified observations.
    """
    source = sqlite3.connect(f"file:{source_database.resolve()}?immutable=1", uri=True)
    destination = sqlite3.connect(destination_database)
    columns = [
        str(row[1]) for row in destination.execute("PRAGMA table_info(two_view_geometries)")
    ]
    if not {"pair_id", "rows", "cols", "data"}.issubset(columns):
        source.close()
        destination.close()
        raise RuntimeError("unrecognized two_view_geometries schema")
    placeholders = ",".join("?" for _ in columns)
    quoted_columns = ",".join(f'"{name}"' for name in columns)
    restored = added = affected = 0

    def fetch(connection, pair_id):
        row = connection.execute(
            f"SELECT {quoted_columns} FROM two_view_geometries WHERE pair_id=?", (pair_id,)
        ).fetchone()
        return dict(zip(columns, row)) if row is not None else None

    try:
        destination.execute("BEGIN IMMEDIATE")
        for pair_id in sorted({int(value) for value in pair_ids}):
            old = fetch(source, pair_id)
            new = fetch(destination, pair_id)
            if old is None and new is None:
                continue

            def decode(row):
                if row is None or not int(row["rows"]):
                    return np.zeros((0, 2), dtype=np.uint32)
                return np.frombuffer(row["data"], dtype=np.uint32).reshape(
                    int(row["rows"]), int(row["cols"])
                )[:, :2].copy()

            old_values, new_values = decode(old), decode(new)
            union = np.unique(np.vstack([old_values, new_values]), axis=0)
            base = dict(old if old is not None else new)
            base["rows"] = int(len(union))
            base["cols"] = 2
            base["data"] = union.astype(np.uint32).tobytes()
            destination.execute(
                f"INSERT OR REPLACE INTO two_view_geometries ({quoted_columns}) "
                f"VALUES ({placeholders})",
                tuple(base[name] for name in columns),
            )
            restored += len(old_values)
            added += len(union) - len(old_values)
            affected += 1
        destination.commit()
    except Exception:
        destination.rollback()
        raise
    finally:
        source.close()
        destination.close()
    return {
        "pairs_merged": affected,
        "prior_verified_rows_preserved": restored,
        "new_verified_rows_added": added,
    }


def prepare_targeted_database(
    source_database: Path,
    destination_database: Path,
    match_list_path: Path,
    image_root: Path,
    sparse_path: Path,
    proposal_path: Path,
    specs: dict | None = None,
    **kwargs,
) -> dict:
    import pycolmap

    if destination_database.exists() or match_list_path.exists():
        raise FileExistsError("refusing to overwrite targeted matching artifacts")
    reconstruction = pycolmap.Reconstruction(sparse_path)
    proposal_document = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposals = proposal_document["targeted_pair_proposals"]
    matches, names, report = discover_matches(
        source_database, image_root, reconstruction, proposals, specs=specs, **kwargs
    )
    shutil.copyfile(source_database, destination_database)
    # matches_importer skips a pair if raw matches already exist.  Each export
    # above contains the union of the original and newly discovered rows, so
    # remove exactly the affected pairs from the copied database before asking
    # COLMAP to import and geometrically re-verify that union.
    report["replaced_existing_pair_rows"] = clear_pairs_for_reimport(
        destination_database, matches
    )
    write_raw_match_list(match_list_path, matches, names)
    return report
