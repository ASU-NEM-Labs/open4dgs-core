#!/usr/bin/env python3
"""Geometry and data primitives for the clean-plate calibration-v2 pipeline.

This module deliberately has no dependency on any previous camera solve.  It
operates on original images, a fresh COLMAP database, exact SIFT keypoint IDs,
and immutable landmark splits.  The companion ``run_calibration_v2.py`` file
provides the command-line orchestration.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np


MAX_IMAGE_ID = 2_147_483_647
SPLIT_NAMES = ("fit", "validation", "test")


def atomic_write_json(path: os.PathLike | str, value: object) -> None:
    """Write JSON without exposing a partially-written report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: os.PathLike | str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_reference_images(root: os.PathLike | str) -> list[Path]:
    root = Path(root)
    images = sorted(root.glob("*/000001.jpg"))
    if not images:
        images = sorted(root.glob("*/*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return images


def input_manifest(
    reference_root: os.PathLike | str,
    named_inputs: Mapping[str, os.PathLike | str],
) -> dict:
    """Hash only declared original inputs; derived runs never enter provenance."""
    reference_root = Path(reference_root).resolve()
    images = discover_reference_images(reference_root)
    rows = []
    for path in images:
        rows.append({
            "path": path.relative_to(reference_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    declared = {}
    for name, raw_path in sorted(named_inputs.items()):
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        declared[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(row["sha256"].encode("ascii"))
    for name, row in sorted(declared.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(row["sha256"].encode("ascii"))
    return {
        "schema_version": 1,
        "reference_root": str(reference_root),
        "reference_image_count": len(rows),
        "reference_images": rows,
        "declared_inputs": declared,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def file_set_manifest(
    paths: Sequence[os.PathLike | str],
    root: os.PathLike | str,
) -> dict:
    """Hash an exact declared file set, including explicit missing entries."""
    root = Path(root).resolve()
    rows = []
    aggregate = hashlib.sha256()
    for raw_path in sorted({Path(path).resolve() for path in paths}):
        try:
            label = raw_path.relative_to(root).as_posix()
        except ValueError:
            label = str(raw_path)
        row = {"path": label, "exists": raw_path.is_file()}
        if raw_path.is_file():
            row.update({"bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)})
        rows.append(row)
        aggregate.update(label.encode("utf-8"))
        aggregate.update((row.get("sha256") or "MISSING").encode("ascii"))
    return {
        "schema_version": 1,
        "root": str(root),
        "file_count": len(rows),
        "present_file_count": sum(row["exists"] for row in rows),
        "files": rows,
        "aggregate_sha256": aggregate.hexdigest(),
    }


@dataclass(frozen=True)
class CameraSpec:
    folder: str
    width: int
    height: int
    model: str
    params: tuple[float, ...]
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None


def load_nominal_reference(path: os.PathLike | str) -> dict[str, CameraSpec]:
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    result = {}
    for row in document["cameras"]:
        folder = str(row["folder"])
        params = tuple(float(value) for value in row["params"])
        model = str(row.get("camera_model_name", "PINHOLE"))
        minimum_params = {
            "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
            "RADIAL": 5, "OPENCV": 8,
        }.get(model)
        if minimum_params is None or len(params) < minimum_params:
            raise ValueError(f"{folder}: unsupported or incomplete nominal camera model {model}")
        result[folder] = CameraSpec(
            folder=folder,
            width=int(row["width"]),
            height=int(row["height"]),
            model=model,
            params=params,
            rotation=np.asarray(row["R"], dtype=np.float64) if "R" in row else None,
            translation=np.asarray(row["t"], dtype=np.float64) if "t" in row else None,
        )
    if len(result) != len(document["cameras"]):
        raise ValueError("duplicate camera folders in nominal reference")
    return result


def folder_from_image_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    return Path(normalized).stem


def opencv_params(spec: CameraSpec) -> np.ndarray:
    p = spec.params
    if spec.model == "SIMPLE_PINHOLE":
        values = [p[0], p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0]
    elif spec.model == "PINHOLE":
        values = [p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0, 0.0]
    elif spec.model == "SIMPLE_RADIAL":
        values = [p[0], p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0]
    elif spec.model == "RADIAL":
        values = [p[0], p[0], p[1], p[2], p[3], p[4], 0.0, 0.0]
    elif spec.model == "OPENCV":
        values = list(p[:8])
    else:
        raise ValueError(f"unsupported nominal camera model {spec.model}")
    return np.asarray(values, dtype=np.float64)


def patch_database_intrinsics(
    database_path: os.PathLike | str,
    specs: Mapping[str, CameraSpec],
) -> dict:
    """Install one OPENCV camera per image using only nominal metadata."""
    con = sqlite3.connect(database_path)
    try:
        images = list(con.execute("SELECT image_id,name,camera_id FROM images ORDER BY image_id"))
        camera_to_folder: dict[int, str] = {}
        for image_id, name, camera_id in images:
            folder = folder_from_image_name(str(name))
            if folder not in specs:
                raise KeyError(f"database image {name!r} has no nominal camera")
            previous = camera_to_folder.setdefault(int(camera_id), folder)
            if previous != folder:
                raise ValueError(
                    f"camera_id {camera_id} is shared by {previous} and {folder}; "
                    "feature extraction must use single_camera_per_image"
                )
            spec = specs[folder]
            con.execute(
                "UPDATE cameras SET model=4,width=?,height=?,params=?,prior_focal_length=1 "
                "WHERE camera_id=?",
                (spec.width, spec.height, opencv_params(spec).tobytes(), int(camera_id)),
            )
        con.commit()
        return {"images": len(images), "cameras": len(camera_to_folder), "model": "OPENCV"}
    finally:
        con.close()


def augment_apriltag_features(
    database_path: os.PathLike | str,
    reference_root: os.PathLike | str,
) -> dict:
    """Append sub-pixel AprilTag corners as ordinary matched feature rows.

    Tag identity determines only which imaged corner may correspond across
    views.  No tag size, planarity, board layout, or world geometry is used.
    Deterministic descriptors let the normal matcher/geometric verifier decide
    whether those corner correspondences are compatible with the scene.
    """
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        return {"status": "opencv_apriltag_unavailable", "detections": 0, "corners_appended": 0}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    reference_root = Path(reference_root)
    con = sqlite3.connect(database_path)
    detections = corners_appended = 0
    per_camera = {}
    try:
        rows = list(con.execute("SELECT image_id,name FROM images ORDER BY image_id"))
        for image_id, name in rows:
            image_path = reference_root / str(name)
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cannot read {image_path}")
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or not len(ids):
                per_camera[folder_from_image_name(str(name))] = {"tags": 0, "corners": 0}
                continue
            existing_row = con.execute(
                "SELECT rows,cols,data FROM keypoints WHERE image_id=?", (int(image_id),)
            ).fetchone()
            descriptor_row = con.execute(
                "SELECT rows,cols,data FROM descriptors WHERE image_id=?", (int(image_id),)
            ).fetchone()
            if existing_row is None or descriptor_row is None:
                raise RuntimeError(f"features missing for database image {name}")
            keypoint_rows, keypoint_cols, keypoint_blob = existing_row
            descriptor_rows, descriptor_cols, descriptor_blob = descriptor_row
            keypoint_values = np.frombuffer(keypoint_blob, dtype=np.float32).reshape(
                int(keypoint_rows), int(keypoint_cols)
            ).copy()
            descriptor_values = np.frombuffer(descriptor_blob, dtype=np.uint8).reshape(
                int(descriptor_rows), int(descriptor_cols)
            ).copy()
            if int(descriptor_cols) != 128:
                raise ValueError("AprilTag augmentation requires 128-column SIFT descriptors")
            new_keypoints = []
            new_descriptors = []
            for marker_corners, marker_id in zip(corners, ids[:, 0]):
                refined = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 1, 2)
                cv2.cornerSubPix(
                    gray, refined, (5, 5), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
                )
                for corner_index, point in enumerate(refined[:, 0]):
                    keypoint = np.zeros(int(keypoint_cols), dtype=np.float32)
                    keypoint[:2] = point
                    if int(keypoint_cols) == 4:
                        keypoint[2:] = (1.0, 0.0)
                    elif int(keypoint_cols) == 6:
                        keypoint[2:] = (1.0, 0.0, 0.0, 1.0)
                    seed = f"apriltag36h11:{int(marker_id)}:{corner_index}".encode("ascii")
                    descriptor = np.frombuffer(hashlib.shake_256(seed).digest(128), dtype=np.uint8)
                    new_keypoints.append(keypoint)
                    new_descriptors.append(descriptor)
                detections += 1
            if new_keypoints:
                keypoint_values = np.vstack([keypoint_values, np.asarray(new_keypoints)])
                descriptor_values = np.vstack([descriptor_values, np.asarray(new_descriptors)])
                con.execute(
                    "UPDATE keypoints SET rows=?,data=? WHERE image_id=?",
                    (len(keypoint_values), keypoint_values.astype(np.float32).tobytes(), int(image_id)),
                )
                con.execute(
                    "UPDATE descriptors SET rows=?,data=? WHERE image_id=?",
                    (len(descriptor_values), descriptor_values.astype(np.uint8).tobytes(), int(image_id)),
                )
                corners_appended += len(new_keypoints)
            per_camera[folder_from_image_name(str(name))] = {
                "tags": len(ids), "corners": len(new_keypoints),
            }
        con.commit()
    finally:
        con.close()
    return {
        "status": "completed",
        "dictionary": "APRILTAG_36h11",
        "geometry_assumed": False,
        "detections": detections,
        "corners_appended": corners_appended,
        "per_camera": per_camera,
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, item: int) -> int:
        parent = self.parent
        if item not in parent:
            parent[item] = item
            self.rank[item] = 0
            return item
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def encode_observation(image_id: int, keypoint_idx: int) -> int:
    return (int(image_id) << 32) | int(keypoint_idx)


def decode_observation(value: int) -> tuple[int, int]:
    return value >> 32, value & 0xFFFFFFFF


def image_ids_from_pair_id(pair_id: int) -> tuple[int, int]:
    return int(pair_id // MAX_IMAGE_ID), int(pair_id % MAX_IMAGE_ID)


def pair_id_from_image_ids(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return int(image_id1) * MAX_IMAGE_ID + int(image_id2)


@dataclass
class ExactTrack:
    track_id: str
    observations: tuple[tuple[int, int], ...]
    support_edges: tuple[tuple[int, int, int, int], ...]
    split: str = ""


def _track_digest(observations: Sequence[tuple[int, int]]) -> str:
    digest = hashlib.blake2b(digest_size=12, person=b"calib-v2-track")
    for image_id, keypoint_idx in observations:
        digest.update(int(image_id).to_bytes(4, "little", signed=False))
        digest.update(int(keypoint_idx).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def read_keypoints(con: sqlite3.Connection) -> dict[int, np.ndarray]:
    result = {}
    for image_id, rows, cols, blob in con.execute(
        "SELECT image_id,rows,cols,data FROM keypoints"
    ):
        if blob is None or rows == 0:
            result[int(image_id)] = np.zeros((0, 2), dtype=np.float32)
            continue
        data = np.frombuffer(blob, dtype=np.float32).reshape(int(rows), int(cols))
        result[int(image_id)] = data[:, :2].copy()
    return result


def build_exact_tracks(
    database_path: os.PathLike | str,
    min_views: int = 3,
    require_cycle: bool = True,
) -> tuple[list[ExactTrack], dict]:
    """Build tracks from verified SIFT indices without pixel quantization.

    ``require_cycle`` requires every observation to have support from at least
    two other observations. A chain A-B-C with no A-C verification is not a
    cycle-consistent landmark and is excluded from calibration-v2. An entire
    component is rejected if it contains two observations from one camera;
    ambiguous components are never split heuristically.
    """
    con = sqlite3.connect(f"file:{Path(database_path).resolve()}?immutable=1", uri=True)
    try:
        uf = UnionFind()
        edges: list[tuple[int, int, int, int]] = []
        verified_pairs = verified_matches = 0
        for pair_id, rows, blob in con.execute(
            "SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0"
        ):
            image_a, image_b = image_ids_from_pair_id(int(pair_id))
            values = np.frombuffer(blob, dtype=np.uint32).reshape(int(rows), 2)
            verified_pairs += 1
            verified_matches += len(values)
            for keypoint_a, keypoint_b in values:
                a = encode_observation(image_a, int(keypoint_a))
                b = encode_observation(image_b, int(keypoint_b))
                uf.union(a, b)
                edges.append((image_a, int(keypoint_a), image_b, int(keypoint_b)))

        connected_groups: dict[int, list[int]] = defaultdict(list)
        for node in uf.parent:
            connected_groups[uf.find(node)].append(node)
        edge_groups: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for edge in edges:
            edge_groups[uf.find(encode_observation(edge[0], edge[1]))].append(edge)

        tracks: list[ExactTrack] = []
        duplicate_camera = short = acyclic = 0
        for root, nodes in connected_groups.items():
            observations = tuple(sorted(decode_observation(node) for node in nodes))
            image_ids = [value[0] for value in observations]
            if len(set(image_ids)) != len(image_ids):
                duplicate_camera += 1
                continue
            if len(observations) < min_views:
                short += 1
                continue
            support = tuple(sorted(set(edge_groups[root])))
            if require_cycle:
                degree = defaultdict(int)
                for ia, ka, ib, kb in support:
                    degree[(ia, ka)] += 1
                    degree[(ib, kb)] += 1
                if any(degree[observation] < 2 for observation in observations):
                    acyclic += 1
                    continue
            tracks.append(ExactTrack(_track_digest(observations), observations, support))
        tracks.sort(key=lambda track: track.track_id)
        report = {
            "verified_pairs": verified_pairs,
            "verified_matches": verified_matches,
            "connected_components": len(connected_groups),
            "accepted_tracks": len(tracks),
            "duplicate_camera_components": duplicate_camera,
            "short_components": short,
            "non_cycle_components": acyclic,
            "mean_track_length": (
                float(np.mean([len(track.observations) for track in tracks])) if tracks else 0.0
            ),
        }
        return tracks, report
    finally:
        con.close()


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{text}".encode("utf-8"), digest_size=8, person=b"calib-v2-split"
    ).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


def _radial_bin(track: ExactTrack, keypoints: Mapping[int, np.ndarray], image_sizes) -> int:
    radii = []
    for image_id, keypoint_idx in track.observations:
        width, height = image_sizes[image_id]
        point = keypoints[image_id][keypoint_idx]
        dx = (float(point[0]) - (width - 1) / 2) / max(width / 2, 1)
        dy = (float(point[1]) - (height - 1) / 2) / max(height / 2, 1)
        radii.append(math.hypot(dx, dy) / math.sqrt(2.0))
    radius = float(np.median(radii)) if radii else 0.0
    return min(2, int(radius * 3.0))


def assign_track_splits(
    tracks: Sequence[ExactTrack],
    keypoints: Mapping[int, np.ndarray],
    image_sizes: Mapping[int, tuple[int, int]],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260822,
    min_validation_per_camera: int = 50,
    min_test_per_camera: int = 100,
) -> dict:
    """Deterministically stratify whole landmarks without moving them later.

    The minimum counts are readiness gates, not targets to obtain by stealing
    landmarks from the fit split.  This keeps the requested global 70/15/15
    allocation honest and makes weak-camera evidence deficits explicit.
    """
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    if any(value < 0 for value in fractions):
        raise ValueError("split fractions must be non-negative")
    strata: dict[tuple[int, int], list[ExactTrack]] = defaultdict(list)
    for track in tracks:
        length_bin = min(len(track.observations), 6)
        strata[(length_bin, _radial_bin(track, keypoints, image_sizes))].append(track)

    for group in strata.values():
        group.sort(key=lambda value: (_stable_unit_interval(value.track_id, seed), value.track_id))
        n = len(group)
        n_validation = int(round(n * fractions[1]))
        n_test = int(round(n * fractions[2]))
        if n_validation + n_test > n:
            n_test = max(0, n - n_validation)
        for index, track in enumerate(group):
            if index < n_validation:
                track.split = "validation"
            elif index < n_validation + n_test:
                track.split = "test"
            else:
                track.split = "fit"

    def per_camera(split: str) -> dict[int, int]:
        counts = defaultdict(int)
        for track in tracks:
            if track.split != split:
                continue
            for image_id, _ in track.observations:
                counts[image_id] += 1
        return counts

    counts = {name: sum(track.split == name for track in tracks) for name in SPLIT_NAMES}
    observations = {
        name: sum(len(track.observations) for track in tracks if track.split == name)
        for name in SPLIT_NAMES
    }
    camera_counts = {}
    for name in SPLIT_NAMES:
        observed = per_camera(name)
        camera_counts[name] = {
            str(image_id): int(observed.get(image_id, 0)) for image_id in sorted(image_sizes)
        }
    validation_deficits = [
        str(image_id) for image_id in sorted(image_sizes)
        if camera_counts["validation"][str(image_id)] < min_validation_per_camera
    ]
    test_deficits = [
        str(image_id) for image_id in sorted(image_sizes)
        if camera_counts["test"][str(image_id)] < min_test_per_camera
    ]
    return {
        "seed": seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions)),
        "tracks": counts,
        "observations": observations,
        "per_camera_observations": camera_counts,
        "minimums": {
            "validation_per_camera": min_validation_per_camera,
            "test_per_camera": min_test_per_camera,
        },
        "validation_deficit_image_ids": validation_deficits,
        "test_deficit_image_ids": test_deficits,
        "ready_for_model_selection": not validation_deficits,
        "ready_for_final_test": not test_deficits,
    }


def save_tracks(
    path: os.PathLike | str,
    tracks: Sequence[ExactTrack],
    keypoints: Mapping[int, np.ndarray],
    image_names: Mapping[int, str],
    image_sizes: Mapping[int, tuple[int, int]],
    metadata: Mapping[str, object],
) -> None:
    offsets = [0]
    image_ids: list[int] = []
    keypoint_ids: list[int] = []
    xy: list[tuple[float, float]] = []
    track_ids: list[str] = []
    splits: list[int] = []
    split_to_index = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for track in tracks:
        track_ids.append(track.track_id)
        splits.append(split_to_index[track.split])
        for image_id, keypoint_id in track.observations:
            image_ids.append(image_id)
            keypoint_ids.append(keypoint_id)
            point = keypoints[image_id][keypoint_id]
            xy.append((float(point[0]), float(point[1])))
        offsets.append(len(image_ids))
    document = dict(metadata)
    document.update({
        "schema_version": 1,
        "split_names": list(SPLIT_NAMES),
        "image_names": {str(key): value for key, value in sorted(image_names.items())},
        "image_sizes": {str(key): list(value) for key, value in sorted(image_sizes.items())},
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int32),
        keypoint_ids=np.asarray(keypoint_ids, dtype=np.int32),
        xy=np.asarray(xy, dtype=np.float64),
        track_ids=np.asarray(track_ids, dtype="U24"),
        splits=np.asarray(splits, dtype=np.uint8),
        metadata=np.asarray(json.dumps(document, sort_keys=True)),
    )


@dataclass(frozen=True)
class TrackStore:
    offsets: np.ndarray
    image_ids: np.ndarray
    keypoint_ids: np.ndarray
    xy: np.ndarray
    track_ids: np.ndarray
    splits: np.ndarray
    metadata: dict

    def iter_split(self, split: str) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        split_index = self.metadata["split_names"].index(split)
        for index in np.flatnonzero(self.splits == split_index):
            start, end = int(self.offsets[index]), int(self.offsets[index + 1])
            yield str(self.track_ids[index]), self.image_ids[start:end], self.xy[start:end]


def load_tracks(path: os.PathLike | str) -> TrackStore:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    return TrackStore(
        offsets=data["offsets"], image_ids=data["image_ids"],
        keypoint_ids=data["keypoint_ids"], xy=data["xy"],
        track_ids=data["track_ids"], splits=data["splits"], metadata=metadata,
    )


def filter_database_to_fit_tracks(
    source_database: os.PathLike | str,
    destination_database: os.PathLike | str,
    tracks: Sequence[ExactTrack],
) -> dict:
    """Copy a COLMAP database and remove every non-fit correspondence."""
    destination_database = Path(destination_database)
    destination_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_database, destination_database)
    fit_nodes: dict[int, str] = {}
    for track in tracks:
        for image_id, keypoint_idx in track.observations:
            fit_nodes[encode_observation(image_id, keypoint_idx)] = track.split

    con = sqlite3.connect(destination_database)
    kept_verified = kept_raw = 0
    try:
        for table in ("two_view_geometries", "matches"):
            rows = list(con.execute(f"SELECT pair_id,rows,data FROM {table} WHERE rows>0"))
            for pair_id, count, blob in rows:
                image_a, image_b = image_ids_from_pair_id(int(pair_id))
                values = np.frombuffer(blob, dtype=np.uint32).reshape(int(count), 2)
                keep = []
                for index, (keypoint_a, keypoint_b) in enumerate(values):
                    left = fit_nodes.get(encode_observation(image_a, int(keypoint_a)))
                    right = fit_nodes.get(encode_observation(image_b, int(keypoint_b)))
                    keep.append(left == "fit" and right == "fit")
                filtered = values[np.asarray(keep, dtype=bool)]
                if len(filtered):
                    con.execute(
                        f"UPDATE {table} SET rows=?,cols=2,data=? WHERE pair_id=?",
                        (len(filtered), filtered.astype(np.uint32).tobytes(), int(pair_id)),
                    )
                else:
                    con.execute(f"DELETE FROM {table} WHERE pair_id=?", (int(pair_id),))
                if table == "two_view_geometries":
                    kept_verified += len(filtered)
                else:
                    kept_raw += len(filtered)
        con.commit()
    finally:
        con.close()
    return {"verified_fit_matches": kept_verified, "raw_fit_matches": kept_raw}


def filter_database_to_topology_fit(
    source_database: os.PathLike | str,
    destination_database: os.PathLike | str,
    tracks: Sequence[ExactTrack],
) -> dict:
    """Keep all topology evidence except observations sealed for holdout.

    Rejected ambiguous and short components are useful for COLMAP's disposable
    core mapper, which has its own conflict handling.  They never enter the
    definitive fit-only triangulation or BA.  Removing every edge touching a
    validation/test observation guarantees the topology initializer cannot see
    sealed landmark membership.
    """
    destination_database = Path(destination_database)
    destination_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_database, destination_database)
    holdout_nodes = {
        encode_observation(image_id, keypoint_idx)
        for track in tracks
        if track.split in {"validation", "test"}
        for image_id, keypoint_idx in track.observations
    }
    con = sqlite3.connect(destination_database)
    kept_verified = kept_raw = removed_verified = removed_raw = 0
    try:
        for table in ("two_view_geometries", "matches"):
            rows = list(con.execute(f"SELECT pair_id,rows,data FROM {table} WHERE rows>0"))
            for pair_id, count, blob in rows:
                image_a, image_b = image_ids_from_pair_id(int(pair_id))
                values = np.frombuffer(blob, dtype=np.uint32).reshape(int(count), 2)
                keep = np.asarray([
                    encode_observation(image_a, int(keypoint_a)) not in holdout_nodes
                    and encode_observation(image_b, int(keypoint_b)) not in holdout_nodes
                    for keypoint_a, keypoint_b in values
                ], dtype=bool)
                filtered = values[keep]
                if len(filtered):
                    con.execute(
                        f"UPDATE {table} SET rows=?,cols=2,data=? WHERE pair_id=?",
                        (len(filtered), filtered.astype(np.uint32).tobytes(), int(pair_id)),
                    )
                else:
                    con.execute(f"DELETE FROM {table} WHERE pair_id=?", (int(pair_id),))
                if table == "two_view_geometries":
                    kept_verified += len(filtered)
                    removed_verified += len(values) - len(filtered)
                else:
                    kept_raw += len(filtered)
                    removed_raw += len(values) - len(filtered)
        con.commit()
    finally:
        con.close()
    return {
        "verified_topology_fit_matches": kept_verified,
        "raw_topology_fit_matches": kept_raw,
        "removed_holdout_verified_matches": removed_verified,
        "removed_holdout_raw_matches": removed_raw,
    }


def database_summary(database_path: os.PathLike | str) -> dict:
    con = sqlite3.connect(f"file:{Path(database_path).resolve()}?immutable=1", uri=True)
    try:
        images = int(con.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        keypoints = int(con.execute("SELECT COALESCE(SUM(rows),0) FROM keypoints").fetchone()[0])
        pairs, matches = con.execute(
            "SELECT COUNT(*),COALESCE(SUM(rows),0) FROM two_view_geometries WHERE rows>0"
        ).fetchone()
        degrees = defaultdict(int)
        for pair_id, in con.execute("SELECT pair_id FROM two_view_geometries WHERE rows>0"):
            a, b = image_ids_from_pair_id(int(pair_id))
            degrees[a] += 1
            degrees[b] += 1
        degree_values = sorted(degrees.values())
        return {
            "images": images,
            "keypoints": keypoints,
            "verified_pairs": int(pairs),
            "verified_matches": int(matches),
            "degree_min": min(degree_values, default=0),
            "degree_median": float(np.median(degree_values)) if degree_values else 0.0,
            "degree_max": max(degree_values, default=0),
        }
    finally:
        con.close()


def audit_resection_candidates(
    reconstruction,
    database_path: os.PathLike | str,
    specs: Mapping[str, CameraSpec],
    min_inliers: int = 20,
    min_independent_neighbors: int = 3,
    min_inliers_per_neighbor: int = 5,
    min_quadrants: int = 3,
    min_hull_fraction: float = 0.02,
    max_reprojection_error: float = 2.0,
) -> dict:
    """Independently audit strict 2D--3D PnP eligibility for missing cameras.

    Correspondences are transferred only through exact verified SIFT indices
    connected to an existing reconstruction point.  Conflicting target
    keypoints or duplicate 3D points are discarded instead of heuristically
    merged.  This function is read-only: it never registers a camera.
    """
    registered = {int(image_id) for image_id in reconstruction.images}
    point_for_observation: dict[tuple[int, int], int] = {}
    for image_id, image in reconstruction.images.items():
        for keypoint_idx, point2d in enumerate(image.points2D):
            if point2d.has_point3D():
                point_for_observation[(int(image_id), keypoint_idx)] = int(point2d.point3D_id)

    con = sqlite3.connect(f"file:{Path(database_path).resolve()}?immutable=1", uri=True)
    try:
        names = {int(row[0]): str(row[1]) for row in con.execute("SELECT image_id,name FROM images")}
        keypoints = read_keypoints(con)
        # target -> target keypoint -> point3D -> supporting registered images
        links: dict[int, dict[int, dict[int, set[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        for pair_id, rows, blob in con.execute(
            "SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0"
        ):
            image_a, image_b = image_ids_from_pair_id(int(pair_id))
            a_registered, b_registered = image_a in registered, image_b in registered
            if a_registered == b_registered:
                continue
            target_id, neighbor_id = (
                (image_b, image_a) if a_registered else (image_a, image_b)
            )
            values = np.frombuffer(blob, dtype=np.uint32).reshape(int(rows), 2)
            for keypoint_a, keypoint_b in values:
                if target_id == image_a:
                    target_keypoint, neighbor_keypoint = int(keypoint_a), int(keypoint_b)
                else:
                    target_keypoint, neighbor_keypoint = int(keypoint_b), int(keypoint_a)
                point3d_id = point_for_observation.get((neighbor_id, neighbor_keypoint))
                if point3d_id is not None:
                    links[target_id][target_keypoint][point3d_id].add(neighbor_id)
    finally:
        con.close()

    per_camera = {}
    for image_id in sorted(set(names) - registered):
        folder = folder_from_image_name(names[image_id])
        spec = specs[folder]
        accepted_links = []
        conflicts = 0
        for keypoint_idx, choices in links.get(image_id, {}).items():
            if len(choices) != 1:
                conflicts += 1
                continue
            point3d_id, neighbors = next(iter(choices.items()))
            accepted_links.append((keypoint_idx, point3d_id, neighbors))
        # A single 3D point may not vote through multiple target features.
        by_point: dict[int, list[tuple[int, int, set[int]]]] = defaultdict(list)
        for row in accepted_links:
            by_point[row[1]].append(row)
        unique_links = []
        for choices in by_point.values():
            if len(choices) == 1:
                unique_links.append(choices[0])
            else:
                conflicts += len(choices)

        object_points = np.asarray([
            reconstruction.points3D[point3d_id].xyz
            for _, point3d_id, _ in unique_links
        ], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray([
            keypoints[image_id][keypoint_idx]
            for keypoint_idx, _, _ in unique_links
        ], dtype=np.float64).reshape(-1, 2)
        nominal = opencv_params(spec)
        K = np.asarray([
            [nominal[0], 0.0, nominal[2]],
            [0.0, nominal[1], nominal[3]],
            [0.0, 0.0, 1.0],
        ])
        inlier_indices = np.zeros(0, dtype=np.int32)
        rvec = tvec = None
        pnp_error = None
        if len(object_points) >= max(6, min_inliers):
            success, raw_inliers = False, None
            for method in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
                try:
                    success, rvec, tvec, raw_inliers = cv2.solvePnPRansac(
                        object_points, image_points, K, nominal[4:8],
                        iterationsCount=10000,
                        reprojectionError=max_reprojection_error,
                        confidence=0.999,
                        flags=method,
                    )
                except cv2.error as error:
                    pnp_error = str(error).splitlines()[0]
                    success, rvec, tvec, raw_inliers = False, None, None, None
                if success:
                    break
            if success and raw_inliers is not None:
                inlier_indices = raw_inliers[:, 0].astype(np.int32)
                if len(inlier_indices) >= 6:
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(
                            object_points[inlier_indices], image_points[inlier_indices],
                            K, nominal[4:8], rvec, tvec,
                        )
                    except cv2.error as error:
                        pnp_error = str(error).splitlines()[0]
            else:
                rvec = tvec = None
        inlier_points = image_points[inlier_indices]
        neighbor_counts: dict[int, int] = defaultdict(int)
        for index in inlier_indices:
            for neighbor_id in unique_links[int(index)][2]:
                neighbor_counts[neighbor_id] += 1
        independent = [
            neighbor_id for neighbor_id, count in neighbor_counts.items()
            if count >= min_inliers_per_neighbor
        ]
        quadrants = set()
        for u, v in inlier_points:
            quadrants.add(int(u >= spec.width / 2) + 2 * int(v >= spec.height / 2))
        hull_fraction = 0.0
        if len(inlier_points) >= 3:
            hull_fraction = float(cv2.contourArea(cv2.convexHull(
                inlier_points.astype(np.float32)
            ))) / float(spec.width * spec.height)
        reasons = []
        if len(inlier_indices) < min_inliers:
            reasons.append("fewer_than_20_geometric_inliers")
        if len(independent) < min_independent_neighbors:
            reasons.append("fewer_than_3_independent_solved_neighbors")
        if len(quadrants) < min_quadrants or hull_fraction < min_hull_fraction:
            reasons.append("insufficient_image_coverage")
        per_camera[folder] = {
            "image_id": image_id,
            "candidate_correspondences": len(unique_links),
            "conflicting_correspondences_rejected": conflicts,
            "pnp_inliers": len(inlier_indices),
            "pnp_diagnostic": pnp_error,
            "independent_neighbor_count": len(independent),
            "independent_neighbor_image_ids": sorted(independent),
            "neighbor_inlier_counts": {
                str(key): value for key, value in sorted(neighbor_counts.items())
            },
            "quadrants_covered": sorted(quadrants),
            "convex_hull_fraction": hull_fraction,
            "eligible_for_registration": not reasons,
            "reasons": reasons,
            "R": cv2.Rodrigues(rvec)[0].tolist() if rvec is not None else None,
            "t": np.asarray(tvec).reshape(3).tolist() if tvec is not None else None,
        }
    eligible = [folder for folder, row in per_camera.items() if row["eligible_for_registration"]]
    return {
        "schema_version": 1,
        "policy": {
            "minimum_geometric_inliers": min_inliers,
            "minimum_independent_solved_neighbors": min_independent_neighbors,
            "minimum_inliers_per_independent_neighbor": min_inliers_per_neighbor,
            "minimum_quadrants": min_quadrants,
            "minimum_convex_hull_fraction": min_hull_fraction,
            "maximum_reprojection_error_px": max_reprojection_error,
            "read_only": True,
        },
        "registered_camera_count": len(registered),
        "missing_camera_count": len(per_camera),
        "eligible_camera_count": len(eligible),
        "eligible_cameras": eligible,
        "targeted_matching_cameras": [folder for folder in per_camera if folder not in eligible],
        "per_camera": per_camera,
    }


def propose_structure_overlap_pairs(
    reconstruction,
    specs: Mapping[str, CameraSpec],
    target_folders: Sequence[str],
    pairs_per_target: int = 8,
) -> dict[str, list[dict]]:
    """Rank solved neighbors by provisional-structure/frustum overlap.

    This deliberately does not use Euclidean camera-centre distance.  The
    aligned provisional model supplies the structure and solved-neighbor
    visibility; the manifest pose is used only to propose the weak camera's
    frustum.
    """
    target_set = set(target_folders)
    result: dict[str, list[dict]] = {}
    for target in sorted(target_set):
        spec = specs[target]
        if spec.rotation is None or spec.translation is None:
            result[target] = []
            continue
        nominal = opencv_params(spec)
        K = np.asarray([
            [nominal[0], 0.0, nominal[2]], [0.0, nominal[1], nominal[3]], [0.0, 0.0, 1.0]
        ])
        ranked = []
        for image_id, image in reconstruction.images.items():
            folder = folder_from_image_name(image.name)
            if folder == target:
                continue
            point_ids = sorted({
                int(point.point3D_id) for point in image.points2D if point.has_point3D()
            })
            if not point_ids:
                continue
            xyz = np.asarray([reconstruction.points3D[key].xyz for key in point_ids])
            camera_points = xyz @ spec.rotation.T + spec.translation.reshape(1, 3)
            front = camera_points[:, 2] > 1e-9
            projected = np.full((len(xyz), 2), np.nan)
            if np.any(front):
                rvec = cv2.Rodrigues(spec.rotation)[0]
                pixels, _ = cv2.projectPoints(
                    xyz[front], rvec, spec.translation, K, nominal[4:8]
                )
                projected[front] = pixels[:, 0]
            inside = front & (projected[:, 0] >= 0) & (projected[:, 0] < spec.width) & \
                (projected[:, 1] >= 0) & (projected[:, 1] < spec.height)
            count = int(np.count_nonzero(inside))
            if count:
                visible = projected[inside]
                visible_point_ids = [point_ids[index] for index in np.flatnonzero(inside)]
                neighbor_xy_by_point = {
                    int(point.point3D_id): np.asarray(point.xy, dtype=np.float64)
                    for point in image.points2D if point.has_point3D()
                }
                neighbor_visible = np.asarray([
                    neighbor_xy_by_point[point_id] for point_id in visible_point_ids
                    if point_id in neighbor_xy_by_point
                ], dtype=np.float64).reshape(-1, 2)

                def padded_box(points, width, height):
                    minimum = points.min(axis=0)
                    maximum = points.max(axis=0)
                    padding = np.maximum((maximum - minimum) * 0.15, 32.0)
                    return [
                        float(max(0.0, minimum[0] - padding[0])),
                        float(max(0.0, minimum[1] - padding[1])),
                        float(min(width - 1.0, maximum[0] + padding[0])),
                        float(min(height - 1.0, maximum[1] + padding[1])),
                    ]

                hull = 0.0
                if len(visible) >= 3:
                    hull = float(cv2.contourArea(cv2.convexHull(
                        visible.astype(np.float32)
                    ))) / float(spec.width * spec.height)
                ranked.append({
                    "neighbor_folder": folder,
                    "neighbor_image_id": int(image_id),
                    "shared_provisional_points": count,
                    "target_hull_fraction": hull,
                    "target_crop_xyxy": padded_box(visible, spec.width, spec.height),
                    "neighbor_crop_xyxy": padded_box(
                        neighbor_visible,
                        int(reconstruction.cameras[image.camera_id].width),
                        int(reconstruction.cameras[image.camera_id].height),
                    ) if len(neighbor_visible) else None,
                })
        ranked.sort(
            key=lambda row: (row["shared_provisional_points"], row["target_hull_fraction"]),
            reverse=True,
        )
        result[target] = ranked[:pairs_per_target]
    return result


def crop_transform(box: Sequence[float], target_long_side: int) -> np.ndarray:
    """Homogeneous transform from source pixels to resized crop pixels."""
    x0, y0, x1, y1 = (float(value) for value in box)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0 or target_long_side <= 0:
        raise ValueError("invalid crop or target size")
    scale = target_long_side / max(width, height)
    return np.asarray([
        [scale, 0.0, -scale * x0],
        [0.0, scale, -scale * y0],
        [0.0, 0.0, 1.0],
    ])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def distortion_grid_invertible(
    params: Sequence[float], width: int, height: int, grid_size: int = 21
) -> bool:
    """Numerically require a locally orientation-preserving OPENCV warp."""
    values = list(float(value) for value in params)
    values.extend([0.0] * (8 - len(values)))
    fx, fy, cx, cy, k1, k2, p1, p2 = values[:8]
    if fx <= 0 or fy <= 0 or not np.all(np.isfinite(values[:8])):
        return False
    xs = np.linspace(0, width - 1, grid_size)
    ys = np.linspace(0, height - 1, grid_size)
    eps = 1e-6

    def distort(x, y):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        return np.asarray([
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        ])

    for u in xs:
        for v in ys:
            x, y = (u - cx) / fx, (v - cy) / fy
            origin = distort(x, y)
            dx = (distort(x + eps, y) - origin) / eps
            dy = (distort(x, y + eps) - origin) / eps
            determinant = dx[0] * dy[1] - dx[1] * dy[0]
            if not np.isfinite(determinant) or determinant <= 1e-4:
                return False
    return True


def camera_model_name(camera) -> str:
    model = camera.model
    return str(model.name) if hasattr(model, "name") else str(model).split(".")[-1]


def set_camera_opencv(camera, params: Sequence[float]) -> None:
    import pycolmap

    if camera_model_name(camera) != "OPENCV":
        model_id = getattr(getattr(pycolmap, "CameraModelId", object), "OPENCV", None)
        camera.model = model_id if model_id is not None else "OPENCV"
    camera.params = np.asarray(params, dtype=np.float64)


def create_opencv_camera(camera_id: int, width: int, height: int, params: Sequence[float]):
    import pycolmap

    params = np.asarray(params, dtype=np.float64)
    if hasattr(pycolmap.Camera, "create_from_model_name"):
        camera = pycolmap.Camera.create_from_model_name(
            camera_id, "OPENCV", float(params[0]), width, height
        )
    elif hasattr(pycolmap.Camera, "create"):
        camera = pycolmap.Camera.create(camera_id, "OPENCV", float(params[0]), width, height)
    else:
        camera = pycolmap.Camera(
            camera_id=camera_id, model="OPENCV", width=width, height=height,
            params=params,
        )
    set_camera_opencv(camera, params)
    return camera


def build_nominal_reconstruction(database_path: os.PathLike | str, specs: Mapping[str, CameraSpec]):
    """Create a registered 96-camera model from the declared nominal prior."""
    import pycolmap

    con = sqlite3.connect(f"file:{Path(database_path).resolve()}?immutable=1", uri=True)
    try:
        rows = list(con.execute(
            "SELECT image_id,name,camera_id FROM images ORDER BY image_id"
        ))
    finally:
        con.close()
    reconstruction = pycolmap.Reconstruction()
    missing = []
    for image_id, name, camera_id in rows:
        folder = folder_from_image_name(str(name))
        spec = specs.get(folder)
        if spec is None or spec.rotation is None or spec.translation is None:
            missing.append(folder)
            continue
        camera = create_opencv_camera(
            int(camera_id), spec.width, spec.height, opencv_params(spec)
        )
        reconstruction.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(
            name=str(name), camera_id=int(camera_id), image_id=int(image_id)
        )
        reconstruction.add_image_with_trivial_frame(image)
        reconstruction.images[int(image_id)].frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.asarray(spec.rotation, dtype=np.float64)),
            np.asarray(spec.translation, dtype=np.float64),
        )
    if missing:
        raise RuntimeError(f"nominal prior lacks complete poses for: {sorted(set(missing))}")
    for frame_id in list(reconstruction.frames):
        reconstruction.register_frame(frame_id)
    if reconstruction.num_images() != len(rows):
        raise RuntimeError(
            f"nominal seed registered {reconstruction.num_images()} of {len(rows)} database images"
        )
    return reconstruction


def transplant_registered_poses(source, destination) -> dict:
    """Replace nominal destination poses with a clean-plate reconstructed core."""
    import pycolmap

    destination_by_name = {
        image.name: image for image in destination.images.values()
    }
    transplanted = []
    missing = []
    for source_image in source.images.values():
        destination_image = destination_by_name.get(source_image.name)
        if destination_image is None:
            missing.append(source_image.name)
            continue
        pose = image_cam_from_world(source_image)
        destination_image.frame.rig_from_world = pycolmap.Rigid3d(
            pose.rotation, np.asarray(pose.translation, dtype=np.float64)
        )
        transplanted.append(folder_from_image_name(source_image.name))
    if missing:
        raise RuntimeError(
            f"clean-plate core contains images absent from the fit database: {missing}"
        )
    return {
        "transplanted_camera_count": len(transplanted),
        "transplanted_folders": sorted(transplanted),
    }


def image_cam_from_world(image):
    pose = image.cam_from_world
    return pose() if callable(pose) else pose


def reconstruction_num_observations(reconstruction) -> int:
    for name in ("compute_num_observations", "num_observations"):
        if hasattr(reconstruction, name):
            value = getattr(reconstruction, name)
            return int(value() if callable(value) else value)
    return int(sum(image.num_points3D for image in reconstruction.images.values()))


def _reconstruction_float_metric(reconstruction, names: Sequence[str]) -> float:
    for name in names:
        if hasattr(reconstruction, name):
            value = getattr(reconstruction, name)
            return float(value() if callable(value) else value)
    raise RuntimeError(f"pycolmap reconstruction lacks metric methods {tuple(names)}")


def camera_as_opencv(camera) -> tuple[np.ndarray, np.ndarray]:
    """Return K and OpenCV [k1,k2,p1,p2] for common COLMAP models."""
    name = camera_model_name(camera)
    p = np.asarray(camera.params, dtype=np.float64)
    if name == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1:3]
        distortion = np.zeros(4)
    elif name == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        distortion = np.zeros(4)
    elif name == "SIMPLE_RADIAL":
        fx = fy = p[0]
        cx, cy, k1 = p[1:4]
        distortion = np.asarray([k1, 0.0, 0.0, 0.0])
    elif name == "RADIAL":
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1:5]
        distortion = np.asarray([k1, k2, 0.0, 0.0])
    elif name == "OPENCV":
        fx, fy, cx, cy = p[:4]
        distortion = p[4:8].copy()
    else:
        raise ValueError(f"calibration-v2 does not validate camera model {name}")
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, distortion


def undistort_pixels(points: np.ndarray, camera) -> np.ndarray:
    K, distortion = camera_as_opencv(camera)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, K, distortion).reshape(-1, 2)


def project_opencv(
    points3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera,
) -> tuple[np.ndarray, np.ndarray]:
    K, distortion = camera_as_opencv(camera)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    camera_points = points3d @ rotation.T + np.asarray(translation).reshape(1, 3)
    valid = camera_points[:, 2] > 1e-9
    projected = np.full((len(points3d), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        rvec = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0]
        image, _ = cv2.projectPoints(
            points3d[valid], rvec, np.asarray(translation, dtype=np.float64), K, distortion
        )
        projected[valid] = image[:, 0]
    return projected, valid


def triangulate_normalized(
    observations: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    """DLT from ``(R,t,normalised_xy)`` observations."""
    rows = []
    for rotation, translation, point in observations:
        projection = np.column_stack([rotation, np.asarray(translation).reshape(3)])
        x, y = np.asarray(point, dtype=np.float64)
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    if len(rows) < 4:
        return None
    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    homogeneous = vt[-1]
    if not np.all(np.isfinite(homogeneous)) or abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def _reconstruction_views(reconstruction) -> dict[int, tuple[object, np.ndarray, np.ndarray, str]]:
    result = {}
    for image_id, image in reconstruction.images.items():
        pose = image_cam_from_world(image)
        result[int(image_id)] = (
            reconstruction.cameras[image.camera_id],
            np.asarray(pose.rotation.matrix(), dtype=np.float64),
            np.asarray(pose.translation, dtype=np.float64),
            folder_from_image_name(image.name),
        )
    return result


def validate_reconstruction(
    reconstruction,
    store: TrackStore,
    split: str,
    min_source_views: int = 2,
    median_gate_px: float = 1.0,
    p90_gate_px: float = 2.0,
    min_observations: int = 100,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Candidate-independent leave-one-view-out reprojection validation.

    For every target observation, the point is triangulated from other views in
    the same held-out landmark.  The target observation never participates in
    triangulation, and no residual-dependent filtering is performed.
    """
    if split not in {"validation", "test"}:
        raise ValueError("only immutable validation or test splits may be scored")
    views = _reconstruction_views(reconstruction)
    errors: dict[int, list[float]] = defaultdict(list)
    points_by_camera: dict[int, list[np.ndarray]] = defaultdict(list)
    failures: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    track_ids_by_camera: dict[int, list[str]] = defaultdict(list)
    image_sizes = {
        int(key): tuple(value) for key, value in store.metadata["image_sizes"].items()
    }

    for track_id, image_ids, pixels in store.iter_split(split):
        available = [index for index, image_id in enumerate(image_ids) if int(image_id) in views]
        if len(available) < min_source_views + 1:
            for image_id in image_ids:
                failures[int(image_id)]["insufficient_registered_sources"] += 1
            continue
        normalized = {}
        for index in available:
            image_id = int(image_ids[index])
            camera = views[image_id][0]
            try:
                normalized[index] = undistort_pixels(pixels[index:index + 1], camera)[0]
            except (cv2.error, ValueError, FloatingPointError):
                normalized[index] = None
        for target_index in available:
            target_id = int(image_ids[target_index])
            sources = [index for index in available if index != target_index and normalized[index] is not None]
            if len(sources) < min_source_views:
                failures[target_id]["insufficient_undistorted_sources"] += 1
                continue
            observations = []
            for index in sources:
                camera_id = int(image_ids[index])
                _, rotation, translation, _ = views[camera_id]
                observations.append((rotation, translation, normalized[index]))
            point3d = triangulate_normalized(observations)
            if point3d is None:
                failures[target_id]["degenerate_triangulation"] += 1
                continue
            source_depths = [float((rotation @ point3d + translation)[2]) for rotation, translation, _ in observations]
            if any(depth <= 1e-9 for depth in source_depths):
                failures[target_id]["source_cheirality"] += 1
                continue
            camera, rotation, translation, _ = views[target_id]
            projected, valid = project_opencv(point3d[None], rotation, translation, camera)
            if not valid[0] or not np.all(np.isfinite(projected[0])):
                failures[target_id]["target_cheirality"] += 1
                continue
            error = float(np.linalg.norm(projected[0] - pixels[target_index]))
            errors[target_id].append(error)
            points_by_camera[target_id].append(np.asarray(pixels[target_index], dtype=np.float64))
            track_ids_by_camera[target_id].append(track_id)

    per_camera = {}
    residual_payload: dict[str, np.ndarray] = {}
    all_image_ids = sorted(image_sizes)
    for image_id in all_image_ids:
        values = np.asarray(errors.get(image_id, []), dtype=np.float64)
        points = np.asarray(points_by_camera.get(image_id, []), dtype=np.float64).reshape(-1, 2)
        width, height = image_sizes[image_id]
        quadrants = [False] * 4
        radial_bands = [False] * 3
        for u, v in points:
            quadrant = int(u >= width / 2) + 2 * int(v >= height / 2)
            quadrants[quadrant] = True
            dx = (u - (width - 1) / 2) / max(width / 2, 1)
            dy = (v - (height - 1) / 2) / max(height / 2, 1)
            radius = math.hypot(dx, dy) / math.sqrt(2.0)
            radial_bands[min(2, int(radius * 3.0))] = True
        view = views.get(image_id)
        invertible = False
        folder = folder_from_image_name(store.metadata["image_names"].get(str(image_id), str(image_id)))
        model = None
        params: list[float] = []
        if view is not None:
            camera = view[0]
            model = camera_model_name(camera)
            params = [float(value) for value in camera.params]
            try:
                K, d = camera_as_opencv(camera)
                invertible = distortion_grid_invertible(
                    [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d], width, height
                )
            except ValueError:
                invertible = False
        median = float(np.median(values)) if len(values) else None
        p90 = float(np.percentile(values, 90)) if len(values) else None
        reasons = []
        if view is None:
            reasons.append("camera_not_registered")
        if len(values) < min_observations:
            reasons.append(f"insufficient_{split}_observations")
        if not all(quadrants):
            reasons.append("insufficient_quadrant_coverage")
        if not all(radial_bands):
            reasons.append("insufficient_radial_coverage")
        if median is None or median >= median_gate_px:
            reasons.append("median_gate_failed")
        if p90 is None or p90 >= p90_gate_px:
            reasons.append("p90_gate_failed")
        if not invertible:
            reasons.append("noninvertible_projection_model")
        per_camera[folder] = {
            "image_id": image_id,
            "registered": view is not None,
            "model": model,
            "params": params,
            "scored_observations": len(values),
            "median_px": median,
            "p90_px": p90,
            "quadrants_covered": quadrants,
            "radial_bands_covered": radial_bands,
            "failures": dict(sorted(failures.get(image_id, {}).items())),
            "passed": not reasons,
            "reasons": reasons,
        }
        residual_payload[f"{folder}_errors"] = values
        residual_payload[f"{folder}_track_ids"] = np.asarray(
            track_ids_by_camera.get(image_id, []), dtype="U24"
        )
    passed = [folder for folder, row in per_camera.items() if row["passed"]]
    report = {
        "schema_version": 1,
        "split": split,
        "gates": {
            "minimum_observations": min_observations,
            "minimum_source_views": min_source_views,
            "median_px_strictly_below": median_gate_px,
            "p90_px_strictly_below": p90_gate_px,
            "all_quadrants": True,
            "all_radial_bands": True,
            "invertible_projection": True,
        },
        "camera_count": len(per_camera),
        "passed_camera_count": len(passed),
        "rig_passed": len(passed) == len(per_camera),
        "passed_cameras": passed,
        "failed_cameras": [folder for folder in per_camera if folder not in passed],
        "per_camera": per_camera,
    }
    return report, residual_payload


def reconstruction_summary(reconstruction) -> dict:
    models = defaultdict(int)
    folders = []
    for image in reconstruction.images.values():
        models[camera_model_name(reconstruction.cameras[image.camera_id])] += 1
        folders.append(folder_from_image_name(image.name))
    return {
        "registered_images": int(reconstruction.num_images()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": reconstruction_num_observations(reconstruction),
        "mean_track_length": _reconstruction_float_metric(
            reconstruction, ("compute_mean_track_length", "mean_track_length")
        ),
        "mean_reprojection_error": _reconstruction_float_metric(
            reconstruction, ("compute_mean_reprojection_error", "mean_reprojection_error")
        ),
        "camera_models": dict(sorted(models.items())),
        "registered_folders": sorted(folders),
    }


def align_reconstruction_to_nominal(reconstruction, specs: Mapping[str, CameraSpec]) -> dict:
    """Apply a robust Umeyama similarity from solved to nominal camera centres."""
    import pycolmap

    solved = []
    nominal = []
    folders = []
    for image in reconstruction.images.values():
        folder = folder_from_image_name(image.name)
        spec = specs.get(folder)
        if spec is None or spec.rotation is None or spec.translation is None:
            continue
        solved.append(np.asarray(image.projection_center(), dtype=np.float64))
        nominal.append(-spec.rotation.T @ spec.translation)
        folders.append(folder)
    if len(solved) < 3:
        raise ValueError("at least three common cameras are required for gauge alignment")
    source = np.asarray(solved)
    target = np.asarray(nominal)
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-15))
    translation = target_mean - scale * rotation @ source_mean
    before = np.linalg.norm(source - target, axis=1)
    after_points = (scale * (rotation @ source.T)).T + translation
    after = np.linalg.norm(after_points - target, axis=1)
    reconstruction.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation))
    return {
        "common_cameras": len(source),
        "folders": folders,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "before_center_error_median": float(np.median(before)),
        "after_center_error_median": float(np.median(after)),
        "after_center_error_p90": float(np.percentile(after, 90)),
    }


def export_reference(reconstruction, output_path: os.PathLike | str, source: str, provenance: dict) -> None:
    cameras = []
    for image in sorted(reconstruction.images.values(), key=lambda value: value.name):
        folder = folder_from_image_name(image.name)
        camera = reconstruction.cameras[image.camera_id]
        pose = image_cam_from_world(image)
        cameras.append({
            "name": f"{folder}/000001.jpg",
            "folder": folder,
            "image_name": "000001.jpg",
            "R": np.asarray(pose.rotation.matrix(), dtype=np.float64).tolist(),
            "t": np.asarray(pose.translation, dtype=np.float64).tolist(),
            "camera_model_name": camera_model_name(camera),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        })
    atomic_write_json(output_path, {
        "schema_version": 2,
        "calibration_source": source,
        "provenance": provenance,
        "cameras": cameras,
    })


def refine_intrinsics_blockwise(
    reconstruction,
    specs: Mapping[str, CameraSpec],
    stage: str,
    focal_log_sigma: float = 0.20,
    principal_sigma_px: float = 100.0,
    radial_sigma: float = 1.0,
    tangential_sigma: float = 0.05,
) -> dict:
    """Refine intrinsics per camera against fixed fit poses and structure.

    This block step gives calibration-v2 the radial-only stage that COLMAP's
    all-or-nothing ``refine_extra_params`` switch cannot express.  Global
    pose/structure BA should be run after the accepted block update.
    """
    from scipy.optimize import least_squares

    if stage not in {"focal", "radial", "principal", "tangential"}:
        raise ValueError("stage must be focal, radial, principal, or tangential")
    report = {}
    for image in reconstruction.images.values():
        folder = folder_from_image_name(image.name)
        spec = specs[folder]
        camera = reconstruction.cameras[image.camera_id]
        pose = image_cam_from_world(image)
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64)
        points3d, pixels = [], []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            points3d.append(reconstruction.points3D[point2d.point3D_id].xyz)
            pixels.append(point2d.xy)
        if len(points3d) < 20:
            report[folder] = {"status": "insufficient_fit_observations", "observations": len(points3d)}
            continue
        points3d = np.asarray(points3d, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        current_K, current_d = camera_as_opencv(camera)
        nominal = opencv_params(spec)
        stage_rank = {"focal": 0, "radial": 1, "principal": 2, "tangential": 3}[stage]
        initial = [math.log(current_K[0, 0]), math.log(current_K[1, 1])]
        parameter_names = ["log_fx", "log_fy"]
        lower = [math.log(max(nominal[0] * 0.25, 1.0)), math.log(max(nominal[1] * 0.25, 1.0))]
        upper = [math.log(max(nominal[0] * 4.0, 2.0)), math.log(max(nominal[1] * 4.0, 2.0))]
        if stage_rank >= 1:
            initial.extend(current_d[:2])
            parameter_names.extend(["k1", "k2"])
            lower.extend([-2.0, -2.0])
            upper.extend([2.0, 2.0])
        if stage_rank >= 2:
            initial.extend([current_K[0, 2], current_K[1, 2]])
            parameter_names.extend(["cx", "cy"])
            lower.extend([-0.25 * spec.width, -0.25 * spec.height])
            upper.extend([1.25 * spec.width, 1.25 * spec.height])
        if stage_rank >= 3:
            initial.extend(current_d[2:4])
            parameter_names.extend(["p1", "p2"])
            lower.extend([-0.5, -0.5])
            upper.extend([0.5, 0.5])
        initial = np.asarray(initial, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        initial = np.clip(initial, lower + 1e-12, upper - 1e-12)

        def unpack(values):
            fx, fy = math.exp(values[0]), math.exp(values[1])
            cx, cy = current_K[0, 2], current_K[1, 2]
            k1, k2, p1, p2 = current_d
            cursor = 2
            if stage_rank >= 1:
                k1, k2 = values[cursor:cursor + 2]
                cursor += 2
            if stage_rank >= 2:
                cx, cy = values[cursor:cursor + 2]
                cursor += 2
            if stage_rank >= 3:
                p1, p2 = values[cursor:cursor + 2]
            return np.asarray([fx, fy, cx, cy, k1, k2, p1, p2])

        def residual(values):
            params = unpack(values)
            K = np.asarray([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1.]])
            rvec = cv2.Rodrigues(rotation)[0]
            projected, _ = cv2.projectPoints(points3d, rvec, translation, K, params[4:8])
            data = (projected[:, 0] - pixels).reshape(-1)
            priors = [
                math.log(params[0] / nominal[0]) / focal_log_sigma,
                math.log(params[1] / nominal[1]) / focal_log_sigma,
            ]
            if stage_rank >= 1:
                priors.extend([params[4] / radial_sigma, params[5] / radial_sigma])
            if stage_rank >= 2:
                priors.extend([
                    (params[2] - nominal[2]) / principal_sigma_px,
                    (params[3] - nominal[3]) / principal_sigma_px,
                ])
            if stage_rank >= 3:
                priors.extend([params[6] / tangential_sigma, params[7] / tangential_sigma])
            return np.concatenate([data, np.asarray(priors)])

        before = residual(initial)[:2 * len(points3d)].reshape(-1, 2)
        result = least_squares(
            residual, initial, bounds=(lower, upper),
            loss="huber", f_scale=1.0, max_nfev=250,
            xtol=1e-11, ftol=1e-11, gtol=1e-11,
        )
        params = unpack(result.x)
        after = residual(result.x)[:2 * len(points3d)].reshape(-1, 2)
        invertible = distortion_grid_invertible(params, spec.width, spec.height)
        accepted = bool(result.success and invertible and np.all(np.isfinite(params)))
        covariance_diag = None
        condition_number = None
        if result.jac.size:
            try:
                normal_matrix = result.jac.T @ result.jac
                covariance_diag = np.diag(np.linalg.pinv(normal_matrix)).tolist()
                condition_number = float(np.linalg.cond(normal_matrix))
            except np.linalg.LinAlgError:
                covariance_diag = None
                condition_number = None
        bound_hits = [
            name for name, value, low, high in zip(parameter_names, result.x, lower, upper)
            if min(value - low, high - value) <= 1e-6 * max(high - low, 1.0)
        ]
        if accepted:
            set_camera_opencv(camera, params)
        report[folder] = {
            "status": "updated" if accepted else "rejected",
            "stage": stage,
            "observations": len(points3d),
            "success": bool(result.success),
            "message": str(result.message),
            "invertible": invertible,
            "params_before": [
                current_K[0, 0], current_K[1, 1], current_K[0, 2], current_K[1, 2],
                *current_d,
            ],
            "params_after": params.tolist(),
            "focal_ratio_from_nominal": [
                float(params[0] / nominal[0]), float(params[1] / nominal[1])
            ],
            "focal_ratio_from_previous_stage": [
                float(params[0] / current_K[0, 0]), float(params[1] / current_K[1, 1])
            ],
            "fit_median_before_px": float(np.median(np.linalg.norm(before, axis=1))),
            "fit_median_after_px": float(np.median(np.linalg.norm(after, axis=1))),
            "fit_p90_before_px": float(np.percentile(np.linalg.norm(before, axis=1), 90)),
            "fit_p90_after_px": float(np.percentile(np.linalg.norm(after, axis=1), 90)),
            "covariance_diagonal": covariance_diag,
            "normal_matrix_condition_number": condition_number,
            "parameter_bound_hits": bound_hits,
        }
    return report


def configure_bundle_adjustment_options(pycolmap, max_iterations: int = 100):
    """Build fixed-intrinsics robust BA options across pycolmap releases."""
    options = pycolmap.BundleAdjustmentOptions()
    applied = {}

    def set_field(target, names, value, label, required=True):
        for name in names:
            if hasattr(target, name):
                setattr(target, name, value)
                applied[label] = name
                return
        applied[label] = None
        if required:
            raise RuntimeError(
                f"pycolmap {type(target).__name__} lacks required BA fields {tuple(names)}"
            )

    set_field(options, ("refine_focal_length",), False, "refine_focal_length")
    set_field(options, ("refine_principal_point",), False, "refine_principal_point")
    set_field(options, ("refine_extra_params",), False, "refine_extra_params")
    set_field(
        options, ("refine_rig_from_world", "refine_extrinsics"), True,
        "refine_extrinsics",
    )
    set_field(options, ("refine_points3D",), True, "refine_points3D")
    set_field(options, ("min_track_length",), 3, "min_track_length", required=False)
    set_field(options, ("print_summary",), True, "print_summary", required=False)
    ceres = getattr(options, "ceres", options)
    loss_enum = getattr(getattr(pycolmap, "LossFunctionType", object), "HUBER", None)
    if loss_enum is None:
        raise RuntimeError("pycolmap does not expose the required Huber loss")
    set_field(ceres, ("loss_function_type",), loss_enum, "loss_function_type")
    set_field(ceres, ("loss_function_scale",), 1.0, "loss_function_scale")
    solver = getattr(ceres, "solver_options", getattr(options, "solver_options", None))
    if solver is None:
        raise RuntimeError("pycolmap BA options do not expose solver options")
    set_field(solver, ("max_num_iterations",), max_iterations, "max_num_iterations")
    if hasattr(options, "check"):
        checked = options.check()
        if checked is False:
            raise RuntimeError("pycolmap rejected the configured bundle-adjustment options")
    return options, applied


def run_pose_structure_ba(reconstruction, max_iterations: int = 100) -> None:
    import pycolmap

    options, _ = configure_bundle_adjustment_options(pycolmap, max_iterations)
    pycolmap.bundle_adjustment(reconstruction, options)


def _estimate_background_homography(
    reference_gray: np.ndarray,
    frame_gray: np.ndarray,
    frame_foreground: np.ndarray | None,
    scale: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Estimate frame->reference alignment from background SIFT features."""
    reference_small = cv2.resize(reference_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame_small = cv2.resize(frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = None
    if frame_foreground is not None:
        foreground_small = cv2.resize(
            frame_foreground, (frame_small.shape[1], frame_small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = np.where(foreground_small < 32, 255, 0).astype(np.uint8)
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01)
    key_ref, desc_ref = detector.detectAndCompute(reference_small, None)
    key_frame, desc_frame = detector.detectAndCompute(frame_small, mask)
    if desc_ref is None or desc_frame is None:
        return np.eye(3), {"status": "no_descriptors", "inliers": 0}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_frame, desc_ref, k=2)
    good = [left for left, right in matches if left.distance < 0.70 * right.distance]
    if len(good) < 12:
        return np.eye(3), {"status": "too_few_matches", "matches": len(good), "inliers": 0}
    source = np.float32([key_frame[match.queryIdx].pt for match in good])
    target = np.float32([key_ref[match.trainIdx].pt for match in good])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 1.0)
    if homography is None or inliers is None or int(inliers.sum()) < 10:
        return np.eye(3), {
            "status": "homography_failed", "matches": len(good),
            "inliers": 0 if inliers is None else int(inliers.sum()),
        }
    scale_matrix = np.diag([scale, scale, 1.0])
    full = np.linalg.inv(scale_matrix) @ homography @ scale_matrix
    return full / full[2, 2], {
        "status": "aligned", "matches": len(good), "inliers": int(inliers.sum()),
    }


def generate_static_masks(
    reference_root: os.PathLike | str,
    footage_root: os.PathLike | str,
    foreground_mask_root: os.PathLike | str,
    output_root: os.PathLike | str,
    frame_numbers: Sequence[int] = (1, 15, 30, 45, 60),
    temporal_difference_threshold: float = 35.0,
    dilation_px: int = 10,
    minimum_allowed_fraction: float = 0.15,
) -> dict:
    """Create conservative clean-plate masks from independently aligned footage.

    White pixels are eligible for SIFT extraction.  Actor masks and pixels that
    disagree persistently between the plate and footage are removed.  If the
    temporal comparison would leave too little image area, the routine falls
    back to the union of aligned foreground masks and records that decision.
    """
    reference_root = Path(reference_root)
    footage_root = Path(footage_root)
    foreground_mask_root = Path(foreground_mask_root)
    output_root = Path(output_root)
    rows = {}
    for reference_path in discover_reference_images(reference_root):
        folder = reference_path.parent.name
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise RuntimeError(f"cannot read {reference_path}")
        height, width = reference.shape
        aligned_foregrounds = []
        differences = []
        alignments = []
        for frame_number in frame_numbers:
            frame_path = footage_root / folder / f"{frame_number:06d}.jpg"
            mask_path = foreground_mask_root / folder / f"{frame_number:06d}.png"
            if not frame_path.exists() or not mask_path.exists():
                continue
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            foreground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or foreground is None:
                continue
            homography, alignment = _estimate_background_homography(reference, frame, foreground)
            alignment["frame"] = frame_number
            alignments.append(alignment)
            warped_frame = cv2.warpPerspective(
                frame, homography, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            warped_foreground = cv2.warpPerspective(
                foreground, homography, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255,
            )
            valid = cv2.warpPerspective(
                np.full(frame.shape, 255, dtype=np.uint8), homography, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            background = (warped_foreground < 32) & (valid > 0)
            if np.count_nonzero(background) > 1000:
                ref_values = reference[background].astype(np.float32)
                frame_values = warped_frame[background].astype(np.float32)
                ref_median, frame_median = np.median(ref_values), np.median(frame_values)
                ref_mad = max(float(np.median(np.abs(ref_values - ref_median))), 5.0)
                frame_mad = max(float(np.median(np.abs(frame_values - frame_median))), 5.0)
                normalized = (warped_frame.astype(np.float32) - frame_median) * (ref_mad / frame_mad) + ref_median
            else:
                normalized = warped_frame.astype(np.float32)
            difference = np.abs(reference.astype(np.float32) - normalized)
            difference[valid == 0] = 255.0
            differences.append(difference)
            aligned_foregrounds.append(warped_foreground > 32)
        if not differences:
            allowed = np.full((height, width), 255, dtype=np.uint8)
            mode = "no_footage_fallback"
        else:
            foreground_union = np.any(np.stack(aligned_foregrounds), axis=0)
            persistent_difference = np.median(np.stack(differences), axis=0) > temporal_difference_threshold
            invalid = foreground_union | persistent_difference
            if dilation_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
                )
                invalid = cv2.dilate(invalid.astype(np.uint8), kernel) > 0
            allowed = np.where(invalid, 0, 255).astype(np.uint8)
            mode = "foreground_plus_temporal_difference"
            if float(np.mean(allowed > 0)) < minimum_allowed_fraction:
                invalid = foreground_union
                if dilation_px > 0:
                    invalid = cv2.dilate(invalid.astype(np.uint8), kernel) > 0
                allowed = np.where(invalid, 0, 255).astype(np.uint8)
                mode = "foreground_only_fallback"
        # COLMAP appends ``.png`` to the complete relative image name.  For
        # ``0001/000001.jpg`` the corresponding mask is therefore
        # ``0001/000001.jpg.png``, not ``0001/000001.png``.
        destination = output_root / folder / f"{reference_path.name}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), allowed):
            raise RuntimeError(f"cannot write {destination}")
        rows[folder] = {
            "mode": mode,
            "allowed_fraction": float(np.mean(allowed > 0)),
            "sampled_frames": [row["frame"] for row in alignments],
            "alignments": alignments,
        }
    return {
        "schema_version": 1,
        "camera_count": len(rows),
        "frame_numbers": list(frame_numbers),
        "temporal_difference_threshold": temporal_difference_threshold,
        "dilation_px": dilation_px,
        "per_camera": rows,
    }


def audit_rig_stability(
    reference_root: os.PathLike | str,
    footage_root: os.PathLike | str,
    foreground_mask_root: os.PathLike | str,
    frame_numbers: Sequence[int],
    drift_threshold_px: float = 0.5,
    minimum_camera_fraction: float = 0.75,
) -> tuple[dict, dict | None]:
    """Measure coherent background drift without fitting per-frame cameras."""
    reference_root = Path(reference_root)
    footage_root = Path(footage_root)
    foreground_mask_root = Path(foreground_mask_root)
    per_camera = {}
    frame_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    homographies = {}
    for reference_path in discover_reference_images(reference_root):
        folder = reference_path.parent.name
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            continue
        height, width = reference.shape
        center = np.asarray([[(width - 1) / 2, (height - 1) / 2]], dtype=np.float64)
        camera_rows = {}
        for frame_number in frame_numbers:
            frame_path = footage_root / folder / f"{frame_number:06d}.jpg"
            mask_path = foreground_mask_root / folder / f"{frame_number:06d}.png"
            if not frame_path.exists() or not mask_path.exists():
                continue
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            foreground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or foreground is None:
                continue
            transform, diagnostics = _estimate_background_homography(reference, frame, foreground)
            if diagnostics.get("status") != "aligned":
                camera_rows[str(frame_number)] = {**diagnostics, "usable": False}
                continue
            mapped = transform_points(center, transform)[0]
            displacement = mapped - center[0]
            frame_vectors[int(frame_number)].append(displacement)
            camera_rows[str(frame_number)] = {
                **diagnostics,
                "usable": True,
                "center_displacement_px": displacement.tolist(),
                "center_drift_px": float(np.linalg.norm(displacement)),
                "frame_to_reference_homography": transform.tolist(),
            }
            homographies.setdefault(str(frame_number), {})[folder] = transform.tolist()
        per_camera[folder] = camera_rows

    per_frame = {}
    coherent_frames = []
    camera_count = max(len(per_camera), 1)
    for frame_number in frame_numbers:
        vectors = np.asarray(frame_vectors.get(int(frame_number), []), dtype=np.float64).reshape(-1, 2)
        if len(vectors):
            median_vector = np.median(vectors, axis=0)
            residuals = np.linalg.norm(vectors - median_vector, axis=1)
            magnitude = float(np.linalg.norm(median_vector))
            coherent_fraction = float(np.mean(residuals <= max(drift_threshold_px, magnitude)))
        else:
            median_vector = None
            residuals = np.zeros(0)
            magnitude = 0.0
            coherent_fraction = 0.0
        coherent = bool(
            len(vectors) >= math.ceil(minimum_camera_fraction * camera_count)
            and magnitude > drift_threshold_px
            and coherent_fraction >= minimum_camera_fraction
        )
        if coherent:
            coherent_frames.append(int(frame_number))
        per_frame[str(frame_number)] = {
            "usable_cameras": len(vectors),
            "median_displacement_px": median_vector.tolist() if median_vector is not None else None,
            "median_drift_px": magnitude,
            "residual_p90_px": float(np.percentile(residuals, 90)) if len(residuals) else None,
            "coherent_camera_fraction": coherent_fraction,
            "coherent_drift_above_threshold": coherent,
        }
    report = {
        "schema_version": 1,
        "drift_threshold_px": drift_threshold_px,
        "minimum_camera_fraction": minimum_camera_fraction,
        "camera_count": len(per_camera),
        "sampled_frames": list(frame_numbers),
        "coherent_frames": coherent_frames,
        "emit_background_homographies": bool(coherent_frames),
        "per_frame": per_frame,
        "per_camera": per_camera,
    }
    export = None
    if coherent_frames:
        export = {
            "schema_version": 1,
            "convention": "frame_pixels_to_static_reference_pixels",
            "scope": "2D background stabilization only; never camera 6-DoF",
            "frames": {
                str(frame): homographies.get(str(frame), {}) for frame in coherent_frames
            },
        }
    return report, export


def paired_bootstrap_p90_improvement(
    baseline_errors: np.ndarray,
    candidate_errors: np.ndarray,
    baseline_track_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    seed: int = 20260822,
    samples: int = 2000,
) -> dict:
    """Bootstrap candidate p90 improvement on identical held-out landmarks."""
    baseline_map = {str(key): float(value) for key, value in zip(baseline_track_ids, baseline_errors)}
    candidate_map = {str(key): float(value) for key, value in zip(candidate_track_ids, candidate_errors)}
    common = sorted(set(baseline_map) & set(candidate_map))
    if len(common) < 20:
        return {
            "common_observations": len(common), "accepted": False,
            "reason": "insufficient_paired_validation_observations",
        }
    before = np.asarray([baseline_map[key] for key in common])
    after = np.asarray([candidate_map[key] for key in common])
    rng = np.random.default_rng(seed)
    improvements = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = rng.integers(0, len(common), len(common))
        improvements[index] = (
            np.percentile(before[selection], 90) - np.percentile(after[selection], 90)
        )
    median_change = float(np.median(after) - np.median(before))
    ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])
    return {
        "common_observations": len(common),
        "baseline_median_px": float(np.median(before)),
        "candidate_median_px": float(np.median(after)),
        "baseline_p90_px": float(np.percentile(before, 90)),
        "candidate_p90_px": float(np.percentile(after, 90)),
        "median_change_px": median_change,
        "p90_improvement_px": float(np.percentile(before, 90) - np.percentile(after, 90)),
        "bootstrap_improvement_ci95_px": [float(ci_low), float(ci_high)],
        "accepted": bool(ci_low > 0.0 and median_change <= 0.05),
        "reason": (
            "positive_p90_improvement" if ci_low > 0.0 and median_change <= 0.05
            else "no_significant_safe_improvement"
        ),
    }


def estimate_time_offset_from_trajectories(
    trajectory_times: np.ndarray,
    trajectories: np.ndarray,
    observation_times: np.ndarray,
    landmark_indices: np.ndarray,
    pixels: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera,
    initial_offset: float = 0.0,
    offset_bounds: tuple[float, float] = (-0.25, 0.25),
) -> dict:
    """Recover one camera time offset against anchored continuous 3D tracks.

    ``trajectories`` is ``[time, landmark, xyz]`` in the anchored static world
    gauge.  The routine changes no camera parameter; it is the isolated timing
    primitive used by dynamic rescue after cello trajectories have been
    triangulated from already-passed static cameras.
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    trajectory_times = np.asarray(trajectory_times, dtype=np.float64)
    trajectories = np.asarray(trajectories, dtype=np.float64)
    observation_times = np.asarray(observation_times, dtype=np.float64)
    landmark_indices = np.asarray(landmark_indices, dtype=np.int64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[0] != len(trajectory_times) or trajectories.shape[2] != 3:
        raise ValueError("trajectories must have shape [time,landmark,3]")
    if len(observation_times) != len(landmark_indices) or pixels.shape != (len(observation_times), 2):
        raise ValueError("observation arrays have incompatible shapes")
    if np.any(landmark_indices < 0) or np.any(landmark_indices >= trajectories.shape[1]):
        raise ValueError("landmark index is out of range")
    if not np.all(np.diff(trajectory_times) > 0):
        raise ValueError("trajectory times must be strictly increasing")
    lower, upper = map(float, offset_bounds)
    feasible_lower = max(lower, float(trajectory_times[0] - observation_times.min()))
    feasible_upper = min(upper, float(trajectory_times[-1] - observation_times.max()))
    if feasible_lower >= feasible_upper:
        raise ValueError("offset bounds do not overlap the trajectory time domain")
    spline = CubicSpline(trajectory_times, trajectories, axis=0)

    def residual(value: np.ndarray) -> np.ndarray:
        sample_time = observation_times + float(value[0])
        all_points = spline(sample_time)
        points = all_points[np.arange(len(sample_time)), landmark_indices]
        projected, valid = project_opencv(points, rotation, translation, camera)
        delta = projected - pixels
        delta[~valid] = 1e3
        return delta.reshape(-1)

    initial = float(np.clip(initial_offset, feasible_lower, feasible_upper))
    result = least_squares(
        residual, [initial], bounds=([feasible_lower], [feasible_upper]),
        loss="huber", f_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    before = residual(np.asarray([initial])).reshape(-1, 2)
    after = residual(result.x).reshape(-1, 2)
    return {
        "success": bool(result.success),
        "offset_seconds": float(result.x[0]),
        "initial_offset_seconds": initial,
        "bounds_seconds": [feasible_lower, feasible_upper],
        "observations": len(observation_times),
        "median_before_px": float(np.median(np.linalg.norm(before, axis=1))),
        "median_after_px": float(np.median(np.linalg.norm(after, axis=1))),
        "p90_after_px": float(np.percentile(np.linalg.norm(after, axis=1), 90)),
    }


def refine_pose_and_time_from_trajectories(
    trajectory_times: np.ndarray,
    trajectories: np.ndarray,
    observation_times: np.ndarray,
    landmark_indices: np.ndarray,
    pixels: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    camera,
    initial_offset: float = 0.0,
    offset_bounds: tuple[float, float] = (-0.25, 0.25),
    rotation_prior_sigma_rad: float = 0.25,
    translation_prior_sigma: float = 1.0,
    offset_prior_sigma_sec: float = 0.05,
) -> dict:
    """Jointly refine one quarantined camera pose and continuous time offset."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    trajectory_times = np.asarray(trajectory_times, dtype=np.float64)
    trajectories = np.asarray(trajectories, dtype=np.float64)
    observation_times = np.asarray(observation_times, dtype=np.float64)
    landmark_indices = np.asarray(landmark_indices, dtype=np.int64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[0] != len(trajectory_times) or trajectories.shape[2] != 3:
        raise ValueError("trajectories must have shape [time,landmark,3]")
    if len(observation_times) != len(landmark_indices) or pixels.shape != (len(observation_times), 2):
        raise ValueError("observation arrays have incompatible shapes")
    if len(observation_times) < 20:
        raise ValueError("at least 20 dynamic observations are required")
    if not np.all(np.diff(trajectory_times) > 0):
        raise ValueError("trajectory times must be strictly increasing")
    if np.any(landmark_indices < 0) or np.any(landmark_indices >= trajectories.shape[1]):
        raise ValueError("landmark index is out of range")
    lower_offset = max(float(offset_bounds[0]), float(trajectory_times[0] - observation_times.min()))
    upper_offset = min(float(offset_bounds[1]), float(trajectory_times[-1] - observation_times.max()))
    if lower_offset >= upper_offset:
        raise ValueError("offset bounds do not overlap the trajectory time domain")
    initial_rotation = np.asarray(initial_rotation, dtype=np.float64)
    initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(3)
    initial_rvec = cv2.Rodrigues(initial_rotation)[0].reshape(3)
    spline = CubicSpline(trajectory_times, trajectories, axis=0)
    initial_offset = float(np.clip(initial_offset, lower_offset, upper_offset))
    x0 = np.concatenate([initial_rvec, initial_translation, [initial_offset]])
    lower = np.asarray([-np.inf] * 6 + [lower_offset])
    upper = np.asarray([np.inf] * 6 + [upper_offset])

    def residual(values: np.ndarray, include_priors: bool = True) -> np.ndarray:
        rotation = cv2.Rodrigues(values[:3])[0]
        sample_time = observation_times + values[6]
        sampled = spline(sample_time)
        points = sampled[np.arange(len(sample_time)), landmark_indices]
        projected, valid = project_opencv(points, rotation, values[3:6], camera)
        data = projected - pixels
        data[~valid] = 1e3
        if not include_priors:
            return data.reshape(-1)
        priors = np.concatenate([
            (values[:3] - initial_rvec) / rotation_prior_sigma_rad,
            (values[3:6] - initial_translation) / translation_prior_sigma,
            [(values[6] - initial_offset) / offset_prior_sigma_sec],
        ])
        return np.concatenate([data.reshape(-1), priors])

    result = least_squares(
        residual, x0, bounds=(lower, upper), loss="huber", f_scale=1.0,
        x_scale="jac", max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    before = residual(x0, include_priors=False).reshape(-1, 2)
    after = residual(result.x, include_priors=False).reshape(-1, 2)
    covariance = None
    condition = None
    if result.jac.size:
        normal = result.jac.T @ result.jac
        covariance = np.linalg.pinv(normal).tolist()
        condition = float(np.linalg.cond(normal))
    return {
        "success": bool(result.success),
        "observations": len(observation_times),
        "R": cv2.Rodrigues(result.x[:3])[0].tolist(),
        "t": result.x[3:6].tolist(),
        "offset_seconds": float(result.x[6]),
        "offset_bounds_seconds": [lower_offset, upper_offset],
        "median_before_px": float(np.median(np.linalg.norm(before, axis=1))),
        "median_after_px": float(np.median(np.linalg.norm(after, axis=1))),
        "p90_after_px": float(np.percentile(np.linalg.norm(after, axis=1), 90)),
        "covariance": covariance,
        "normal_matrix_condition_number": condition,
        "certifies_static_calibration": False,
    }
