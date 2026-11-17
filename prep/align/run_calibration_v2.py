#!/usr/bin/env python3
"""Fresh, isolated clean-plate calibration pipeline.

The commands are intentionally explicit.  ``prepare`` is the only command that
extracts or matches features; ``map`` sees only the fit split; ``refine`` sees
only fit plus validation; and ``validate --split test`` requires an explicit
``--unseal-test`` acknowledgement.  No command reads an earlier camera solve.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_v2 as core  # noqa: E402
import targeted_match_v2 as targeted  # noqa: E402


RUN_MARKER = "calibration-v2-run.json"


def _same_relocated_dataset(out_dir: Path, reference_root: Path, nominal: Path) -> bool:
    """Validate a mount relocation without trusting absolute path strings."""
    manifest_path = out_dir / "input-manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("reference_images", [])
    current = core.discover_reference_images(reference_root)
    if len(recorded) != len(current) or not recorded:
        return False
    current_by_name = {
        path.relative_to(reference_root).as_posix(): path for path in current
    }
    if set(current_by_name) != {row.get("path") for row in recorded}:
        return False
    # Compare every size and hash three spread-out plates. This avoids hashing
    # all 96 images merely because the same mount has a different host prefix.
    for row in recorded:
        path = current_by_name[row["path"]]
        if path.stat().st_size != row.get("bytes"):
            return False
    for index in sorted({0, len(recorded) // 2, len(recorded) - 1}):
        row = recorded[index]
        if core.sha256_file(current_by_name[row["path"]]) != row.get("sha256"):
            return False
    nominal_row = manifest.get("declared_inputs", {}).get("nominal_reference")
    return bool(nominal_row and core.sha256_file(nominal) == nominal_row.get("sha256"))


def _derived_run_artifacts(out_dir: Path) -> list[str]:
    allowed = {RUN_MARKER, "input-manifest.json"}
    return sorted(
        path.name for path in out_dir.iterdir()
        if path.name not in allowed and not path.name.startswith("._") and path.name != ".DS_Store"
    )


def parse_frames(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated frame numbers")
    return values


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")


def colmap_supported_options(executable: str, subcommand: str) -> set[str]:
    """Read CLI flags so one pipeline works with COLMAP 3.x and 4.x."""
    process = subprocess.run(
        [executable, subcommand, "-h"], capture_output=True, text=True
    )
    output = f"{process.stdout}\n{process.stderr}"
    options = set(re.findall(r"--[A-Za-z0-9_.-]+", output))
    if not options:
        raise RuntimeError(
            f"could not discover options from `{executable} {subcommand} -h`: "
            f"{output[-500:]}"
        )
    return options


def append_compatible_option(
    command: list[str], supported: set[str], candidates: tuple[str, ...], value,
    *, required: bool = True,
) -> str | None:
    for option in candidates:
        if option in supported:
            command.extend([option, str(value)])
            return option
    if required:
        raise RuntimeError(f"COLMAP supports none of the required options: {candidates}")
    return None


def _option_group(options, attribute: str, getter: str):
    if hasattr(options, attribute):
        return getattr(options, attribute)
    if hasattr(options, getter):
        return getattr(options, getter)()
    raise RuntimeError(f"pycolmap options expose neither `{attribute}` nor `{getter}()`")


def _set_option(target, names: tuple[str, ...], value, *, required: bool = True):
    for name in names:
        if hasattr(target, name):
            setattr(target, name, value)
            return name
    if required:
        raise RuntimeError(
            f"pycolmap {type(target).__name__} supports none of the required fields {names}"
        )
    return None


def configure_incremental_options(pycolmap, args):
    """Configure a fixed-intrinsics mapper across supported pycolmap APIs."""
    options = pycolmap.IncrementalPipelineOptions()
    mapper = _option_group(options, "mapper", "get_mapper")
    triangulation = _option_group(options, "triangulation", "get_triangulation")
    applied = {}

    def set_value(label, target, names, value, required=True):
        selected = _set_option(target, names, value, required=required)
        applied[label] = selected

    set_value("multiple_models", options, ("multiple_models",), False)
    set_value(
        "structure_less_registration_fallback", options,
        ("structure_less_registration_fallback",), False, required=False,
    )
    set_value("min_model_size", options, ("min_model_size",), 2)
    set_value("pipeline_random_seed", options, ("random_seed",), args.mapper_seed, required=False)
    set_value("init_min_num_inliers", mapper, ("init_min_num_inliers",), args.init_min_inliers)
    set_value("abs_pose_min_num_inliers", mapper, ("abs_pose_min_num_inliers",), args.abs_pose_min_inliers)
    set_value("abs_pose_max_error", mapper, ("abs_pose_max_error",), args.abs_pose_max_error)
    set_value("abs_pose_refine_focal_length", mapper, ("abs_pose_refine_focal_length",), False)
    set_value("abs_pose_refine_extra_params", mapper, ("abs_pose_refine_extra_params",), False)
    set_value("ba_refine_focal_length", options, ("ba_refine_focal_length",), False)
    set_value("ba_refine_principal_point", options, ("ba_refine_principal_point",), False)
    set_value("ba_refine_extra_params", options, ("ba_refine_extra_params",), False)
    set_value("filter_max_reproj_error", mapper, ("filter_max_reproj_error",), args.mapper_filter_error)
    set_value(
        "triangulation_min_angle", triangulation, ("min_angle", "tri_min_angle"),
        args.min_tri_angle, required=False,
    )
    if applied["triangulation_min_angle"] is None:
        set_value(
            "mapper_tri_min_angle", mapper, ("tri_min_angle",),
            args.min_tri_angle, required=False,
        )
    set_value("mapper_random_seed", mapper, ("random_seed",), args.mapper_seed, required=False)
    if hasattr(options, "check"):
        checked = options.check()
        if checked is False:
            raise RuntimeError("pycolmap rejected the configured incremental mapping options")
    return options, applied


def initialize_run(args) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    reference_root = data_dir / "reference_images"
    nominal = Path(args.nominal_reference).resolve() if args.nominal_reference else (
        data_dir / "reference_export-4dv-manifest.json"
    )
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    if not nominal.is_file():
        raise FileNotFoundError(nominal)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / RUN_MARKER
    identity = {
        "schema_version": 2,
        "pipeline": "clean-plate-calibration-v2",
        "data_dir": str(data_dir),
        "reference_root": str(reference_root),
        "nominal_reference": str(nominal),
        "policy": {
            "previous_solves_are_comparison_only": True,
            "test_split_is_sealed": True,
            "never_overwrite_reference_export": True,
        },
    }
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("pipeline") != identity["pipeline"]:
            raise RuntimeError(f"run marker mismatch for pipeline: {existing.get('pipeline')!r}")
        path_mismatch = any(
            existing.get(key) != identity[key]
            for key in ("data_dir", "reference_root", "nominal_reference")
        )
        if path_mismatch:
            if not _same_relocated_dataset(out_dir, reference_root, nominal):
                raise RuntimeError(
                    "run marker paths differ and the existing input hashes do not identify "
                    "this as the same relocated dataset"
                )
            aliases = list(existing.get("path_aliases", []))
            old_alias = {
                key: existing.get(key)
                for key in ("data_dir", "reference_root", "nominal_reference")
            }
            if old_alias not in aliases:
                aliases.append(old_alias)
            identity["path_aliases"] = aliases
            core.atomic_write_json(marker, identity)
            print(f"recognized relocated dataset root: {data_dir}")
    else:
        core.atomic_write_json(marker, identity)
    return data_dir, out_dir, nominal


def declared_inputs(args, data_dir: Path, nominal: Path) -> dict[str, Path]:
    result = {
        "nominal_reference": nominal,
        "timestamps": Path(args.timestamps).resolve() if args.timestamps else data_dir / "frame_timestamps.json",
    }
    if args.capture_manifest:
        result["capture_manifest"] = Path(args.capture_manifest).resolve()
    if args.snapshot_record:
        result["snapshot_record"] = Path(args.snapshot_record).resolve()
    return {key: value for key, value in result.items() if value.exists()}


def command_audit(args) -> None:
    data_dir, out_dir, nominal = initialize_run(args)
    destination = out_dir / "input-manifest.json"
    requested_inputs = declared_inputs(args, data_dir, nominal)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing_inputs = existing.get("declared_inputs", {})
        requested_hashes = {
            key: core.sha256_file(path) for key, path in requested_inputs.items()
        }
        # Optional provenance declared by an earlier audit remains sticky when
        # later commands omit its path. Adding or changing a declared input is
        # still detected; removing one requires an explicit forced rehash.
        unchanged = (
            set(requested_hashes).issubset(existing_inputs)
            and all(
                existing_inputs[key].get("sha256") == digest
                for key, digest in requested_hashes.items()
            )
        )
        if unchanged and not args.force_rehash:
            print(f"input manifest already covers the declared inputs: {destination}")
            return
        derived = _derived_run_artifacts(out_dir)
        if derived:
            raise RuntimeError(
                "refusing to change input provenance after derived work exists; "
                f"start a new run directory (derived artifacts: {derived[:5]})"
            )
        if not args.force_rehash:
            print("refreshing pristine input manifest to include the newly declared metadata")
    manifest = core.input_manifest(
        data_dir / "reference_images", requested_inputs
    )
    if manifest["reference_image_count"] != 96:
        raise RuntimeError(
            f"expected 96 reference images, found {manifest['reference_image_count']}"
        )
    core.atomic_write_json(destination, manifest)
    print(f"wrote immutable input manifest: {destination}")


def _database_images(database: Path) -> tuple[dict[int, str], dict[int, tuple[int, int]]]:
    con = sqlite3.connect(f"file:{database.resolve()}?immutable=1", uri=True)
    try:
        names = {int(i): str(name) for i, name in con.execute("SELECT image_id,name FROM images")}
        sizes = {
            int(i): (int(width), int(height))
            for i, width, height in con.execute(
                "SELECT images.image_id,cameras.width,cameras.height "
                "FROM images JOIN cameras USING(camera_id)"
            )
        }
        return names, sizes
    finally:
        con.close()


def build_tracks_and_fit_database(
    args, out_dir: Path, database: Path, *, prefix: str = ""
) -> dict:
    tracks, track_report = core.build_exact_tracks(
        database, min_views=args.min_track_views, require_cycle=not args.allow_chain_tracks
    )
    con = sqlite3.connect(f"file:{database.resolve()}?immutable=1", uri=True)
    try:
        keypoints = core.read_keypoints(con)
    finally:
        con.close()
    image_names, image_sizes = _database_images(database)
    split_report = core.assign_track_splits(
        tracks, keypoints, image_sizes,
        fractions=(args.fit_fraction, args.validation_fraction, args.test_fraction),
        seed=args.split_seed,
        min_validation_per_camera=args.min_validation_observations,
        min_test_per_camera=args.min_test_observations,
    )
    track_path = out_dir / f"{prefix}tracks-v2.npz"
    fit_database = out_dir / f"{prefix}fit-database.db"
    topology_fit_database = out_dir / f"{prefix}topology-fit-database.db"
    report_path = out_dir / f"{prefix}track-report.json"
    for path in (track_path, fit_database, topology_fit_database, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable split artifact: {path}")
    core.save_tracks(
        track_path, tracks, keypoints, image_names, image_sizes,
        {"track_construction": track_report, "split": split_report},
    )
    fit_report = core.filter_database_to_fit_tracks(database, fit_database, tracks)
    topology_fit_report = core.filter_database_to_topology_fit(
        database, topology_fit_database, tracks
    )
    result = {
        "database": core.database_summary(database),
        "source_database_sha256": core.sha256_file(database),
        "track_construction": track_report,
        "split": split_report,
        "fit_database": fit_report,
        "topology_fit_database": topology_fit_report,
        "tracks_path": str(track_path),
    }
    core.atomic_write_json(report_path, result)
    return result


def command_prepare(args) -> None:
    data_dir, out_dir, nominal = initialize_run(args)
    command_audit(args)
    specs = core.load_nominal_reference(nominal)
    native_max_image_size = max(
        max(spec.width, spec.height) for spec in specs.values()
    )
    database = out_dir / "database.db"
    if database.exists():
        raise FileExistsError(f"refusing to overwrite fresh database: {database}")
    mask_input_path = out_dir / "mask-input-manifest.json"
    masks_dir = out_dir / "static_masks"
    mask_report_path = out_dir / "static-mask-report.json"
    expected_mask_names = {
        f"{path.parent.name}/{path.name}.png"
        for path in core.discover_reference_images(data_dir / "reference_images")
    }
    existing_mask_names = {
        path.relative_to(masks_dir).as_posix()
        for path in masks_dir.glob("*/*.png")
    } if masks_dir.exists() else set()
    if mask_input_path.exists() or mask_report_path.exists() or existing_mask_names:
        if not (
            mask_input_path.exists() and mask_report_path.exists()
            and existing_mask_names == expected_mask_names
        ):
            raise RuntimeError(
                "incomplete static-mask checkpoint; preserve the run and restart in a new "
                "output directory rather than mixing mask inputs"
            )
        mask_report = json.loads(mask_report_path.read_text(encoding="utf-8"))
        print(f"reusing completed static-mask checkpoint: {masks_dir}")
    else:
        mask_inputs = []
        for reference_path in core.discover_reference_images(data_dir / "reference_images"):
            folder = reference_path.parent.name
            for frame_number in args.mask_frames:
                mask_inputs.extend([
                    data_dir / "images" / folder / f"{frame_number:06d}.jpg",
                    data_dir / "masks" / folder / f"{frame_number:06d}.png",
                ])
        core.atomic_write_json(mask_input_path, core.file_set_manifest(mask_inputs, data_dir))
        mask_report = core.generate_static_masks(
            data_dir / "reference_images", data_dir / "images", data_dir / "masks", masks_dir,
            frame_numbers=args.mask_frames,
            temporal_difference_threshold=args.temporal_difference_threshold,
            dilation_px=args.mask_dilation_px,
        )
        core.atomic_write_json(mask_report_path, mask_report)
    feature_options = colmap_supported_options(args.colmap, "feature_extractor")
    feature_command = [
        args.colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(data_dir / "reference_images"),
        "--ImageReader.mask_path", str(masks_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera_per_image", "1",
    ]
    append_compatible_option(
        feature_command, feature_options, ("--FeatureExtraction.type",), "SIFT", required=False
    )
    append_compatible_option(
        feature_command, feature_options,
        ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"),
        1 if args.use_gpu else 0,
    )
    append_compatible_option(
        feature_command, feature_options,
        ("--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"),
        native_max_image_size,
    )
    append_compatible_option(
        feature_command, feature_options, ("--SiftExtraction.max_num_features",),
        args.max_num_features,
    )
    append_compatible_option(
        feature_command, feature_options, ("--SiftExtraction.estimate_affine_shape",),
        1 if args.affine_sift else 0,
    )
    append_compatible_option(
        feature_command, feature_options, ("--SiftExtraction.domain_size_pooling",),
        1 if args.dsp_sift else 0,
    )
    if args.gpu_index is not None:
        append_compatible_option(
            feature_command, feature_options,
            ("--FeatureExtraction.gpu_index", "--SiftExtraction.gpu_index"), args.gpu_index,
        )
    run_command(feature_command, out_dir / "logs" / "feature-extractor.log")
    patch_report = core.patch_database_intrinsics(database, specs)
    tag_report = (
        core.augment_apriltag_features(database, data_dir / "reference_images")
        if args.apriltags else {"status": "disabled", "detections": 0, "corners_appended": 0}
    )
    core.atomic_write_json(out_dir / "apriltag-report.json", tag_report)
    matcher_options = colmap_supported_options(args.colmap, "exhaustive_matcher")
    match_command = [
        args.colmap, "exhaustive_matcher", "--database_path", str(database),
    ]
    append_compatible_option(
        match_command, matcher_options, ("--FeatureMatching.type",),
        "SIFT_BRUTEFORCE", required=False,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"),
        1 if args.use_gpu else 0,
    )
    append_compatible_option(
        match_command, matcher_options, ("--SiftMatching.cpu_brute_force_matcher",),
        0 if args.use_gpu else 1, required=False,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"), 1,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--FeatureMatching.max_num_matches", "--SiftMatching.max_num_matches"),
        args.max_num_matches,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--TwoViewGeometry.max_error", "--SiftMatching.max_error"),
        args.geometric_max_error,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--TwoViewGeometry.min_num_inliers", "--SiftMatching.min_num_inliers"),
        args.geometric_min_inliers,
    )
    append_compatible_option(
        match_command, matcher_options,
        ("--TwoViewGeometry.min_inlier_ratio", "--SiftMatching.min_inlier_ratio"),
        args.geometric_min_ratio,
    )
    if args.gpu_index is not None:
        append_compatible_option(
            match_command, matcher_options,
            ("--FeatureMatching.gpu_index", "--SiftMatching.gpu_index"), args.gpu_index,
        )
    run_command(match_command, out_dir / "logs" / "exhaustive-matcher.log")
    report = build_tracks_and_fit_database(args, out_dir, database, prefix="bootstrap-")
    report["intrinsics_patch"] = patch_report
    report["apriltags"] = tag_report
    report["mask_report"] = str(out_dir / "static-mask-report.json")
    core.atomic_write_json(out_dir / "prepare-report.json", report)
    print(json.dumps(report["database"], indent=2))


def command_tracks(args) -> None:
    _, out_dir, nominal = initialize_run(args)
    database = Path(args.database).resolve() if args.database else out_dir / "database.db"
    report = build_tracks_and_fit_database(args, out_dir, database)
    print(json.dumps(report, indent=2))


def largest_model(models: dict) -> object:
    if not models:
        raise RuntimeError("COLMAP produced no reconstruction")
    return max(
        models.values(),
        key=lambda rec: (rec.num_images(), core.reconstruction_num_observations(rec)),
    )


def command_map(args) -> None:
    import pycolmap

    data_dir, out_dir, nominal = initialize_run(args)
    prefix = "bootstrap-" if args.bootstrap else ""
    # The bootstrap is a quarantined topology scout, not a calibration
    # candidate.  It needs every freshly verified clean-plate match so it can
    # discover overlap for targeted matching.  The definitive reconstruction
    # below still consumes only the sealed fit database.
    database = out_dir / ("database.db" if args.bootstrap else "fit-database.db")
    if not database.exists():
        expected_stage = "prepare" if args.bootstrap else "tracks"
        raise FileNotFoundError(
            f"mapping database is missing: {database}; run `{expected_stage}` first"
        )
    raw_output = out_dir / f"{prefix}mapper-models"
    final_output = out_dir / "sparse-v2" / ("bootstrap-fixed" if args.bootstrap else "fit-fixed")
    # Validate the installed pycolmap API before creating a checkpoint. Older
    # versions do not expose every option available in current releases.
    options, option_report = configure_incremental_options(pycolmap, args)
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite mapper output: {final_output}")
    if raw_output.exists():
        if any(raw_output.iterdir()):
            raise FileExistsError(f"refusing to overwrite mapper output: {raw_output}")
        # A prior preflight failure may have left this directory empty.  rmdir
        # is deliberately used so this can never remove material output.
        raw_output.rmdir()
    raw_output.mkdir(parents=True)
    specs = core.load_nominal_reference(nominal)
    if args.bootstrap:
        models = pycolmap.incremental_mapping(
            database, data_dir / "reference_images", raw_output, options=options
        )
        summaries = {
            str(key): core.reconstruction_summary(value) for key, value in models.items()
        }
        reconstruction = largest_model(models)
        alignment = core.align_reconstruction_to_nominal(reconstruction, specs)
        role = "disposable_topology_scout"
    else:
        # Estimate the well-connected core from fit tracks only.  The
        # manifest supplies the gauge and fills poses outside that core; it
        # does not replace the clean-plate estimates where evidence exists.
        topology_database = out_dir / "topology-fit-database.db"
        if not topology_database.exists():
            raise FileNotFoundError(
                f"fit-only topology database is missing: {topology_database}; "
                "rerun `tracks` in a fresh calibration-v2 output directory"
            )
        fit_core_output = out_dir / "fit-core-mapper-models"
        if fit_core_output.exists():
            raise FileExistsError(
                f"refusing to overwrite fit-only core output: {fit_core_output}"
            )
        fit_core_output.mkdir(parents=True)
        core_models = pycolmap.incremental_mapping(
            topology_database,
            data_dir / "reference_images",
            fit_core_output,
            options=options,
        )
        fit_core = largest_model(core_models)
        fit_core_summaries = {
            str(key): core.reconstruction_summary(value)
            for key, value in core_models.items()
        }
        core_alignment = core.align_reconstruction_to_nominal(fit_core, specs)
        aligned_core_output = out_dir / "sparse-v2" / "fit-core-aligned"
        if aligned_core_output.exists():
            raise FileExistsError(
                f"refusing to overwrite aligned fit core: {aligned_core_output}"
            )
        aligned_core_output.mkdir(parents=True)
        fit_core.write(aligned_core_output)
        seed = core.build_nominal_reconstruction(database, specs)
        transplant = core.transplant_registered_poses(fit_core, seed)
        pycolmap.triangulate_points(
            seed, database, data_dir / "reference_images", raw_output,
            clear_points=True,
        )
        reconstruction = pycolmap.Reconstruction(raw_output)
        summaries = {
            "fit_only_core_models": fit_core_summaries,
            "fused_seed_after_triangulation": core.reconstruction_summary(reconstruction),
        }
        alignment = {
            "source": "fit_only_clean_plate_core_aligned_to_declared_nominal_gauge",
            "common_cameras": reconstruction.num_images(),
            "fit_core_alignment": core_alignment,
            "fit_core_pose_transplant": transplant,
            "fit_core_aligned_model": str(aligned_core_output),
            "topology_database": str(topology_database),
            "topology_database_sha256": core.sha256_file(topology_database),
        }
        role = "fit_only_core_plus_nominal_fill_calibration"
    final_output.mkdir(parents=True)
    reconstruction.write(final_output)
    report = {
        "models": summaries,
        "role": role,
        "source_database": str(database),
        "source_database_sha256": core.sha256_file(database),
        "source_database_summary": core.database_summary(database),
        "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
        "incremental_option_fields": option_report,
        "selected": core.reconstruction_summary(reconstruction),
        "gauge_alignment": alignment,
        "unregistered_folders": sorted(set(specs) - {
            core.folder_from_image_name(image.name) for image in reconstruction.images.values()
        }),
    }
    report_path = out_dir / ("bootstrap-mapping-report.json" if args.bootstrap else "mapping-report.json")
    core.atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))


def command_doctor(args) -> None:
    """Exercise every external interface needed after preparation."""
    import pycolmap

    _, out_dir, nominal = initialize_run(args)
    report = {
        "schema_version": 1,
        "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
        "checks": {},
        "errors": [],
    }

    def check(name, callback):
        try:
            value = callback()
            report["checks"][name] = {"passed": True, "details": value}
        except Exception as error:  # a doctor must report all failures together
            report["checks"][name] = {"passed": False, "error": repr(error)}
            report["errors"].append(f"{name}: {error}")

    check("colmap_feature_extractor_options", lambda: sorted(
        colmap_supported_options(args.colmap, "feature_extractor")
    ))
    check("colmap_exhaustive_matcher_options", lambda: sorted(
        colmap_supported_options(args.colmap, "exhaustive_matcher")
    ))
    check("colmap_matches_importer_options", lambda: sorted(
        colmap_supported_options(args.colmap, "matches_importer")
    ))
    check("incremental_mapping_options", lambda: configure_incremental_options(
        pycolmap, args
    )[1])
    check("bundle_adjustment_options", lambda: core.configure_bundle_adjustment_options(
        pycolmap, args.ba_iterations
    )[1])
    check("opencv_camera_api", lambda: {
        "model": core.camera_model_name(core.create_opencv_camera(
            1, 640, 480, [800, 800, 320, 240, 0, 0, 0, 0]
        ))
    })
    check("nominal_seed_reconstruction_api", lambda: {
        "registered_images": core.build_nominal_reconstruction(
            out_dir / "database.db", core.load_nominal_reference(nominal)
        ).num_images(),
        "triangulate_points": hasattr(pycolmap, "triangulate_points"),
    })

    def similarity_api():
        rotation = pycolmap.Rotation3d(np.eye(3))
        transform = pycolmap.Sim3d(1.0, rotation, np.zeros(3))
        reconstruction = pycolmap.Reconstruction()
        if not hasattr(reconstruction, "transform"):
            raise RuntimeError("Reconstruction.transform is unavailable")
        return {"sim3": str(transform), "transform_method": True}

    check("similarity_transform_api", similarity_api)

    def reconstruction_api():
        reconstruction = pycolmap.Reconstruction()
        required = ["write", "images", "cameras", "points3D", "num_images", "num_points3D"]
        missing = [name for name in required if not hasattr(reconstruction, name)]
        metrics = {
            name: hasattr(reconstruction, name)
            for name in (
                "compute_num_observations", "num_observations",
                "compute_mean_track_length", "mean_track_length",
                "compute_mean_reprojection_error", "mean_reprojection_error",
            )
        }
        if missing:
            raise RuntimeError(f"missing Reconstruction API: {missing}")
        if not any(metrics[name] for name in ("compute_num_observations", "num_observations")):
            raise RuntimeError("no reconstruction observation-count API")
        return {"required_methods": required, "metrics": metrics}

    check("reconstruction_api", reconstruction_api)
    bootstrap_database = out_dir / "database.db"
    check("bootstrap_database", lambda: {
        "path": str(bootstrap_database),
        "summary": core.database_summary(bootstrap_database),
        "sha256": core.sha256_file(bootstrap_database),
    } if bootstrap_database.exists() else (_ for _ in ()).throw(
        FileNotFoundError(bootstrap_database)
    ))
    report["passed"] = not report["errors"]
    destination = out_dir / "compatibility-report.json"
    core.atomic_write_json(destination, report)
    print(json.dumps({
        "passed": report["passed"],
        "pycolmap_version": report["pycolmap_version"],
        "errors": report["errors"],
        "report": str(destination),
    }, indent=2))
    if report["errors"]:
        raise RuntimeError("compatibility doctor failed; see compatibility-report.json")


def command_resection_audit(args) -> None:
    """Audit strict registration gates and propose structure-overlap pairs."""
    import pycolmap

    _, out_dir, nominal = initialize_run(args)
    database = Path(args.database).resolve() if args.database else out_dir / "database.db"
    sparse = Path(args.sparse).resolve() if args.sparse else out_dir / "sparse-v2" / "bootstrap-fixed"
    if not database.exists():
        raise FileNotFoundError(database)
    if not sparse.is_dir():
        raise FileNotFoundError(
            f"bootstrap sparse model is missing: {sparse}; run the `map --bootstrap` command first"
        )
    reconstruction = pycolmap.Reconstruction(sparse)
    specs = core.load_nominal_reference(nominal)
    report = core.audit_resection_candidates(
        reconstruction, database, specs,
        min_inliers=args.min_inliers,
        min_independent_neighbors=args.min_independent_neighbors,
        min_inliers_per_neighbor=args.min_inliers_per_neighbor,
        min_quadrants=args.min_quadrants,
        min_hull_fraction=args.min_hull_fraction,
        max_reprojection_error=args.max_reprojection_error,
    )
    proposals = core.propose_structure_overlap_pairs(
        reconstruction, specs, report["targeted_matching_cameras"],
        pairs_per_target=args.pairs_per_target,
    )
    report["targeted_pair_proposals"] = proposals
    pair_lines = []
    for target, rows in proposals.items():
        for row in rows:
            pair_lines.append(
                f"{target}/000001.jpg {row['neighbor_folder']}/000001.jpg"
            )
    core.atomic_write_json(out_dir / "resection-audit.json", report)
    (out_dir / "targeted-pairs.txt").write_text(
        "\n".join(pair_lines) + ("\n" if pair_lines else ""), encoding="utf-8"
    )
    print(json.dumps({
        "registered_camera_count": report["registered_camera_count"],
        "eligible_cameras": report["eligible_cameras"],
        "targeted_matching_cameras": report["targeted_matching_cameras"],
        "pair_proposal_count": len(pair_lines),
    }, indent=2))


def command_targeted_match(args) -> None:
    data_dir, out_dir, nominal = initialize_run(args)
    source_database = Path(args.database).resolve() if args.database else out_dir / "database.db"
    sparse = Path(args.sparse).resolve() if args.sparse else out_dir / "sparse-v2" / "bootstrap-fixed"
    proposals = Path(args.proposals).resolve() if args.proposals else out_dir / "resection-audit.json"
    destination = out_dir / "database-targeted.db"
    match_list = out_dir / "targeted-raw-matches.txt"
    report = targeted.prepare_targeted_database(
        source_database, destination, match_list,
        data_dir / "reference_images", sparse, proposals,
        specs=core.load_nominal_reference(nominal) if args.manifest_guided else None,
        discovery_long_side=args.discovery_long_side,
        ratio=args.ratio,
        minimum_patch_correlation=args.minimum_patch_correlation,
        native_keypoint_radius=args.native_keypoint_radius,
        minimum_cycle_neighbors=args.minimum_cycle_neighbors,
        guided_search_radius=args.guided_search_radius,
        guided_descriptor_ratio=args.guided_descriptor_ratio,
    )
    if report["affected_pairs"]:
        importer_options = colmap_supported_options(args.colmap, "matches_importer")
        command = [
            args.colmap, "matches_importer",
            "--database_path", str(destination),
            "--match_list_path", str(match_list),
            "--match_type", "raw",
        ]
        append_compatible_option(
            command, importer_options,
            ("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"),
            1 if args.use_gpu else 0,
        )
        append_compatible_option(
            command, importer_options, ("--SiftMatching.cpu_brute_force_matcher",),
            0 if args.use_gpu else 1, required=False,
        )
        append_compatible_option(
            command, importer_options,
            ("--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"), 1,
        )
        append_compatible_option(
            command, importer_options,
            ("--TwoViewGeometry.max_error", "--SiftMatching.max_error"),
            args.geometric_max_error,
        )
        append_compatible_option(
            command, importer_options,
            ("--TwoViewGeometry.min_num_inliers", "--SiftMatching.min_num_inliers"),
            args.geometric_min_inliers,
        )
        append_compatible_option(
            command, importer_options,
            ("--TwoViewGeometry.min_inlier_ratio", "--SiftMatching.min_inlier_ratio"),
            args.geometric_min_ratio,
        )
        if args.gpu_index is not None:
            append_compatible_option(
                command, importer_options,
                ("--FeatureMatching.gpu_index", "--SiftMatching.gpu_index"), args.gpu_index,
        )
        run_command(command, out_dir / "logs" / "targeted-geometric-verification.log")
        report["verified_geometry_merge"] = targeted.merge_verified_pair_rows(
            source_database, destination, report["affected_pair_ids"]
        )
        report["verification_status"] = "completed"
        report["verified_database"] = core.database_summary(destination)
    else:
        report["verification_status"] = "skipped_no_cycle_consistent_candidates"
    core.atomic_write_json(out_dir / "targeted-match-report.json", report)
    print(json.dumps({
        "cycle_consistent_landmarks": report["cycle_consistent_landmarks"],
        "affected_pairs": report["affected_pairs"],
        "verification_status": report["verification_status"],
    }, indent=2))


def _write_validation(out_dir: Path, label: str, report: dict, residuals: dict) -> None:
    core.atomic_write_json(out_dir / f"validation-{label}.json", report)
    np.savez_compressed(out_dir / f"validation-{label}-residuals.npz", **residuals)


def _load_residuals(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=False)
    return {key: values[key] for key in values.files}


def _stage_decisions(previous: dict, candidate: dict, seed: int) -> dict:
    folders = sorted({key[:-7] for key in previous if key.endswith("_errors")})
    result = {}
    for index, folder in enumerate(folders):
        result[folder] = core.paired_bootstrap_p90_improvement(
            previous[f"{folder}_errors"], candidate[f"{folder}_errors"],
            previous[f"{folder}_track_ids"], candidate[f"{folder}_track_ids"],
            seed=seed + index,
        )
    return result


def _copy_camera_params(source, destination, folders: set[str]) -> None:
    source_by_folder = {
        core.folder_from_image_name(image.name): source.cameras[image.camera_id]
        for image in source.images.values()
    }
    for image in destination.images.values():
        folder = core.folder_from_image_name(image.name)
        if folder not in folders:
            continue
        src = source_by_folder[folder]
        dst = destination.cameras[image.camera_id]
        if core.camera_model_name(src) != "OPENCV":
            raise RuntimeError(
                f"intrinsic rollback supports OPENCV only, got {core.camera_model_name(src)}"
            )
        core.set_camera_opencv(dst, np.asarray(src.params, dtype=np.float64).copy())


def _alternate_pose_structure_and_triangulation(
    pycolmap,
    reconstruction,
    database: Path,
    image_path: Path,
    temp_root: Path,
    *,
    rounds: int,
    ba_iterations: int,
):
    """Grow a nominal-seeded fit model without ever consulting held-out tracks."""
    history = []
    current = reconstruction
    for round_index in range(1, rounds + 1):
        before = core.reconstruction_summary(current)
        core.run_pose_structure_ba(current, ba_iterations)
        after_ba = core.reconstruction_summary(current)
        with tempfile_directory(temp_root, f"retriangulation-{round_index}") as output:
            result = pycolmap.triangulate_points(
                current,
                database,
                image_path,
                output,
                clear_points=False,
                refine_intrinsics=False,
            )
            current = result if result is not None else pycolmap.Reconstruction(output)
        after_triangulation = core.reconstruction_summary(current)
        if after_triangulation["registered_images"] != before["registered_images"]:
            raise RuntimeError(
                "fit-only retriangulation changed the registered camera set: "
                f"{before['registered_images']} -> "
                f"{after_triangulation['registered_images']}"
            )
        observation_gain = (
            after_triangulation["observations"] - before["observations"]
        )
        history.append({
            "round": round_index,
            "before": before,
            "after_bundle_adjustment": after_ba,
            "after_retriangulation": after_triangulation,
            "observation_gain": observation_gain,
        })
        if observation_gain <= 0:
            break
    core.run_pose_structure_ba(current, ba_iterations)
    return current, {
        "requested_rounds": rounds,
        "completed_rounds": len(history),
        "rounds": history,
        "final_after_bundle_adjustment": core.reconstruction_summary(current),
    }


def command_refine(args) -> None:
    import pycolmap

    data_dir, out_dir, nominal = initialize_run(args)
    tracks = core.load_tracks(out_dir / "tracks-v2.npz")
    fit_database = out_dir / "fit-database.db"
    if not fit_database.exists():
        raise FileNotFoundError(fit_database)
    input_sparse = Path(args.input_sparse).resolve() if args.input_sparse else (
        out_dir / "sparse-v2" / "fit-fixed"
    )
    accepted_dir = out_dir / "sparse-v2" / "accepted"
    if accepted_dir.exists():
        raise FileExistsError(f"refusing to overwrite {accepted_dir}")
    specs = core.load_nominal_reference(nominal)
    current = pycolmap.Reconstruction(input_sparse)
    current, alternating_report = _alternate_pose_structure_and_triangulation(
        pycolmap,
        current,
        fit_database,
        data_dir / "reference_images",
        out_dir / "tmp",
        rounds=args.retriangulation_rounds,
        ba_iterations=args.ba_iterations,
    )
    core.atomic_write_json(
        out_dir / "pose-structure-retriangulation.json", alternating_report
    )
    fixed_dir = out_dir / "sparse-v2" / "stage-fixed"
    fixed_dir.mkdir(parents=True)
    current.write(fixed_dir)
    report, residuals = core.validate_reconstruction(
        current, tracks, "validation", min_observations=args.min_validation_observations
    )
    _write_validation(out_dir, "stage-fixed", report, residuals)
    selection = {
        "stage-fixed": {
            "accepted": True,
            "summary": core.reconstruction_summary(current),
            "pose_structure_retriangulation": alternating_report,
        }
    }
    current_residuals = residuals
    current_label = "stage-fixed"

    for stage_index, stage in enumerate(("focal", "radial", "principal", "tangential"), start=1):
        with tempfile_directory(out_dir / "tmp", f"candidate-{stage}") as temp_model:
            current.write(temp_model)
            candidate = pycolmap.Reconstruction(temp_model)
        intrinsic_report = core.refine_intrinsics_blockwise(candidate, specs, stage)
        candidate_report_pre, candidate_residuals_pre = core.validate_reconstruction(
            candidate, tracks, "validation", min_observations=args.min_validation_observations
        )
        decisions = _stage_decisions(current_residuals, candidate_residuals_pre, args.bootstrap_seed + 1000 * stage_index)
        accepted_folders = {folder for folder, row in decisions.items() if row.get("accepted")}
        rejected_folders = set(decisions) - accepted_folders
        _copy_camera_params(current, candidate, rejected_folders)
        core.run_pose_structure_ba(candidate, args.ba_iterations)
        candidate_report, candidate_residuals = core.validate_reconstruction(
            candidate, tracks, "validation", min_observations=args.min_validation_observations
        )
        post_decisions = _stage_decisions(
            current_residuals, candidate_residuals, args.bootstrap_seed + 1000 * stage_index + 500
        )
        harmful = [
            folder for folder, row in post_decisions.items()
            if row.get("candidate_p90_px", np.inf) > row.get("baseline_p90_px", np.inf) + 0.05
        ]
        improved = [folder for folder, row in post_decisions.items() if row.get("accepted")]
        accept_stage = bool(accepted_folders and improved and not harmful)
        label = f"stage-{stage}"
        _write_validation(out_dir, label, candidate_report, candidate_residuals)
        core.atomic_write_json(out_dir / f"intrinsics-{label}.json", intrinsic_report)
        selection[label] = {
            "accepted": accept_stage,
            "intrinsic_blocks_accepted": sorted(accepted_folders),
            "intrinsic_blocks_rejected": sorted(rejected_folders),
            "post_ba_significant_improvements": improved,
            "post_ba_harmful_cameras": harmful,
            "paired_bootstrap": post_decisions,
            "summary": core.reconstruction_summary(candidate),
        }
        if accept_stage:
            current = candidate
            current_residuals = candidate_residuals
            current_label = label

    accepted_dir.mkdir(parents=True)
    current.write(accepted_dir)
    alignment = core.align_reconstruction_to_nominal(current, specs)
    aligned_dir = out_dir / "sparse-v2" / "final-aligned"
    aligned_dir.mkdir(parents=True)
    current.write(aligned_dir)
    provenance = json.loads((out_dir / "input-manifest.json").read_text(encoding="utf-8"))
    derived_provenance = {
        "input_aggregate_sha256": provenance["aggregate_sha256"],
        "tracks_v2_sha256": core.sha256_file(out_dir / "tracks-v2.npz"),
        "fit_database_sha256": core.sha256_file(out_dir / "fit-database.db"),
    }
    if (out_dir / "mask-input-manifest.json").exists():
        mask_inputs = json.loads(
            (out_dir / "mask-input-manifest.json").read_text(encoding="utf-8")
        )
        derived_provenance["mask_input_aggregate_sha256"] = mask_inputs["aggregate_sha256"]
    core.export_reference(
        current, out_dir / "reference_export-v2-candidate.json", "clean-plate-calibration-v2-candidate",
        {**derived_provenance, "selection": selection},
    )
    core.atomic_write_json(out_dir / "model-selection.json", {
        "schema_version": 1, "stages": selection, "final_stage": current_label,
        "final_gauge_alignment": alignment,
    })
    print(
        f"wrote {out_dir / 'reference_export-v2-candidate.json'}; "
        "reference_export-v2.json remains locked until the sealed 96/96 test passes"
    )


class tempfile_directory:
    def __init__(self, root: Path, prefix: str):
        self.root = root
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        import tempfile
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=f"{self.prefix}-", dir=self.root))
        return self.path

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.path, ignore_errors=True)


def command_validate(args) -> None:
    import pycolmap

    _, out_dir, _ = initialize_run(args)
    if args.split == "test" and not args.unseal_test:
        raise RuntimeError("the test split is sealed; pass --unseal-test only for the locked final model")
    if args.split == "test" and (out_dir / "test-split-unsealed.json").exists():
        raise RuntimeError("the sealed test has already been consumed for this run")
    sparse = Path(args.sparse).resolve() if args.sparse else out_dir / "sparse-v2" / "final-aligned"
    reconstruction = pycolmap.Reconstruction(sparse)
    store = core.load_tracks(out_dir / "tracks-v2.npz")
    report, residuals = core.validate_reconstruction(
        reconstruction, store, args.split,
        min_source_views=args.min_source_views,
        median_gate_px=args.median_gate,
        p90_gate_px=args.p90_gate,
        min_observations=args.min_test_observations if args.split == "test" else args.min_validation_observations,
    )
    split_counts = store.metadata.get("split", {}).get("per_camera_observations", {})
    folder_by_image_id = {
        str(image_id): core.folder_from_image_name(name)
        for image_id, name in (
            (int(key), value) for key, value in store.metadata["image_names"].items()
        )
    }
    model_selection_path = out_dir / "model-selection.json"
    model_selection = (
        json.loads(model_selection_path.read_text(encoding="utf-8"))
        if model_selection_path.exists() else {"stages": {}, "final_stage": "unknown"}
    )
    final_validation = {}
    final_stage = model_selection.get("final_stage")
    if final_stage and (out_dir / f"validation-{final_stage}.json").exists():
        final_validation = json.loads(
            (out_dir / f"validation-{final_stage}.json").read_text(encoding="utf-8")
        ).get("per_camera", {})
    intrinsic_reports = {}
    for stage in ("focal", "radial", "principal", "tangential"):
        path = out_dir / f"intrinsics-stage-{stage}.json"
        if path.exists():
            intrinsic_reports[stage] = json.loads(path.read_text(encoding="utf-8"))
    for folder, row in report["per_camera"].items():
        image_id = str(row["image_id"])
        row["observations_by_split"] = {
            split_name: int(split_counts.get(split_name, {}).get(image_id, 0))
            for split_name in ("fit", "validation", "test")
        }
        validation_row = final_validation.get(folder, {})
        row["validation_scored_observations"] = validation_row.get("scored_observations")
        row["validation_median_px"] = validation_row.get("median_px")
        row["validation_p90_px"] = validation_row.get("p90_px")
        choice = "fixed_manifest_intrinsics"
        uncertainty = None
        for stage in ("focal", "radial", "principal", "tangential"):
            stage_selection = model_selection.get("stages", {}).get(f"stage-{stage}", {})
            if stage_selection.get("accepted") and folder in stage_selection.get("intrinsic_blocks_accepted", []):
                choice = stage
                uncertainty = intrinsic_reports.get(stage, {}).get(folder)
        row["intrinsic_model_choice"] = choice
        if uncertainty:
            row["intrinsic_uncertainty"] = {
                "covariance_diagonal": uncertainty.get("covariance_diagonal"),
                "normal_matrix_condition_number": uncertainty.get("normal_matrix_condition_number"),
                "parameter_bound_hits": uncertainty.get("parameter_bound_hits"),
                "focal_ratio_from_nominal": uncertainty.get("focal_ratio_from_nominal"),
                "invertible": uncertainty.get("invertible"),
            }
        else:
            row["intrinsic_uncertainty"] = None
    report["model_selection_stage"] = final_stage
    report["split_count_camera_names"] = {
        key: folder_by_image_id[key] for key in sorted(folder_by_image_id, key=int)
    }
    label = f"{args.split}-final"
    candidate_export = out_dir / "reference_export-v2-candidate.json"
    final_export = out_dir / "reference_export-v2.json"
    promote = bool(args.split == "test" and report["rig_passed"] and report["camera_count"] == 96)
    if args.split == "test":
        report["accepted_reference_export"] = str(final_export) if promote else None
        if promote and not candidate_export.exists():
            raise FileNotFoundError(candidate_export)
        if promote and final_export.exists():
            raise FileExistsError(f"refusing to overwrite accepted export: {final_export}")
    _write_validation(out_dir, label, report, residuals)
    if args.split == "test":
        core.atomic_write_json(out_dir / "test-split-unsealed.json", {
            "model": str(sparse), "rig_passed": report["rig_passed"],
            "failed_cameras": report["failed_cameras"],
        })
        shutil.copyfile(out_dir / f"validation-{label}.json", out_dir / "validation-v2.json")
        if promote:
            shutil.copyfile(candidate_export, final_export)
    print(json.dumps({
        "rig_passed": report["rig_passed"],
        "passed_camera_count": report["passed_camera_count"],
        "failed_cameras": report["failed_cameras"],
    }, indent=2))


def command_dynamic_rescue_spec(args) -> None:
    data_dir, out_dir, _ = initialize_run(args)
    validation_path = Path(args.validation).resolve() if args.validation else (
        out_dir / "validation-validation-final.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    timestamps_path = Path(args.timestamps).resolve() if args.timestamps else data_dir / "frame_timestamps.json"
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
    priors = {}
    for folder in validation["failed_cameras"]:
        row = timestamps.get("cameras", {}).get(folder, {})
        priors[folder] = {
            "sync_offset_sec": row.get("sync_offset_sec"),
            "ladder": row.get("ladder"),
            "static_validation_reasons": validation["per_camera"][folder]["reasons"],
        }
    document = {
        "schema_version": 1,
        "status": "specification_only_no_camera_mutation",
        "failed_cameras": validation["failed_cameras"],
        "anchored_cameras": validation["passed_cameras"],
        "timestamp_priors": priors,
        "requirements": {
            "dynamic_tracks": "rigid cello features only",
            "camera_parameters_allowed": "failed-camera pose and time offset only",
            "promotion_gate": "must improve immutable static validation before final test",
        },
    }
    core.atomic_write_json(out_dir / "dynamic-rescue-spec.json", document)
    print(f"wrote {out_dir / 'dynamic-rescue-spec.json'}; no camera was modified")


def command_dynamic_rescue(args) -> None:
    """Fit quarantined pose/time candidates from externally tracked cello points."""
    import pycolmap

    _, out_dir, nominal = initialize_run(args)
    validation_path = Path(args.validation).resolve() if args.validation else (
        out_dir / "validation-validation-final.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    specs = core.load_nominal_reference(nominal)
    trajectories_path = Path(args.trajectories).resolve()
    data = np.load(trajectories_path, allow_pickle=False)
    times = data["trajectory_times"]
    trajectories = data["trajectories"]
    prior_offsets = {}
    rescue_spec_path = out_dir / "dynamic-rescue-spec.json"
    if rescue_spec_path.exists():
        rescue_spec = json.loads(rescue_spec_path.read_text(encoding="utf-8"))
        for folder, row in rescue_spec.get("timestamp_priors", {}).items():
            if row.get("sync_offset_sec") is not None:
                prior_offsets[folder] = float(row["sync_offset_sec"])
    candidates = {}
    sync = {}
    for index, folder in enumerate(validation["failed_cameras"]):
        required = [f"{folder}_times", f"{folder}_landmark_indices", f"{folder}_pixels"]
        missing = [key for key in required if key not in data.files]
        if missing:
            candidates[folder] = {"status": "missing_dynamic_observations", "missing_arrays": missing}
            continue
        spec = specs[folder]
        if spec.rotation is None or spec.translation is None:
            candidates[folder] = {"status": "missing_manifest_pose_initializer"}
            continue
        params = core.opencv_params(spec)
        camera = core.create_opencv_camera(index + 1, spec.width, spec.height, params)
        try:
            result = core.refine_pose_and_time_from_trajectories(
                times, trajectories,
                data[f"{folder}_times"], data[f"{folder}_landmark_indices"],
                data[f"{folder}_pixels"],
                spec.rotation, spec.translation, camera,
                initial_offset=prior_offsets.get(folder, 0.0),
                offset_bounds=(args.min_offset, args.max_offset),
                rotation_prior_sigma_rad=args.rotation_prior_sigma,
                translation_prior_sigma=args.translation_prior_sigma,
                offset_prior_sigma_sec=args.offset_prior_sigma,
            )
        except (ValueError, RuntimeError) as error:
            candidates[folder] = {"status": "optimization_failed", "error": str(error)}
            continue
        result["status"] = "candidate_only_pending_static_validation"
        candidates[folder] = result
        if result["success"]:
            sync[folder] = {
                "offset_seconds": result["offset_seconds"],
                "source": "continuous_rigid_cello_trajectory",
            }
    document = {
        "schema_version": 1,
        "trajectory_input": {
            "path": str(trajectories_path),
            "bytes": trajectories_path.stat().st_size,
            "sha256": core.sha256_file(trajectories_path),
        },
        "anchored_cameras": validation["passed_cameras"],
        "candidate_cameras": candidates,
        "promotion_gate": "apply to a candidate sparse model, then pass untouched static validation and sealed test",
        "mutated_reference_export": False,
    }
    core.atomic_write_json(out_dir / "dynamic-rescue-candidates.json", document)
    if sync:
        core.atomic_write_json(out_dir / "sync-v2.json", {
            "schema_version": 1,
            "status": "candidate_offsets_pending_static_promotion",
            "cameras": sync,
        })
    print(json.dumps({
        "candidate_count": len(candidates),
        "successful_optimizations": len(sync),
        "reference_export_mutated": False,
    }, indent=2))


def command_stability_audit(args) -> None:
    data_dir, out_dir, _ = initialize_run(args)
    report, homographies = core.audit_rig_stability(
        data_dir / "reference_images", data_dir / "images", data_dir / "masks",
        args.frames,
        drift_threshold_px=args.drift_threshold,
        minimum_camera_fraction=args.minimum_camera_fraction,
    )
    core.atomic_write_json(out_dir / "stability-audit.json", report)
    if homographies is not None:
        core.atomic_write_json(out_dir / "background-homographies-v2.json", homographies)
        print("coherent drift exceeded the gate; wrote background-homographies-v2.json")
    else:
        print("no coherent drift exceeded the gate; no homography export was written")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nominal-reference", default=None)
    parser.add_argument("--capture-manifest", default=None)
    parser.add_argument("--snapshot-record", default=None)
    parser.add_argument("--timestamps", default=None)


def add_split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-track-views", type=int, default=3)
    parser.add_argument(
        "--allow-chain-tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental: admit acyclic exact-ID components into definitive splits; "
            "disabled because this dataset's chain-only components failed held-out validation"
        ),
    )
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--min-validation-observations", type=int, default=50)
    parser.add_argument("--min-test-observations", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="hash original inputs and initialize an isolated run")
    add_common(audit)
    audit.add_argument("--force-rehash", action="store_true")
    audit.set_defaults(func=command_audit)

    doctor = sub.add_parser(
        "doctor", help="preflight all COLMAP/pycolmap interfaces before mapping"
    )
    add_common(doctor)
    doctor.add_argument("--colmap", default="colmap")
    doctor.add_argument("--init-min-inliers", type=int, default=50)
    doctor.add_argument("--abs-pose-min-inliers", type=int, default=20)
    doctor.add_argument("--abs-pose-max-error", type=float, default=2.0)
    doctor.add_argument("--mapper-filter-error", type=float, default=2.0)
    doctor.add_argument("--min-tri-angle", type=float, default=1.0)
    doctor.add_argument("--mapper-seed", type=int, default=20260822)
    doctor.add_argument("--ba-iterations", type=int, default=100)
    doctor.set_defaults(func=command_doctor)

    prepare = sub.add_parser(
        "prepare", help="native SIFT/masks plus a disposable bootstrap split"
    )
    add_common(prepare)
    add_split_options(prepare)
    prepare.add_argument("--force-rehash", action="store_true")
    prepare.add_argument("--colmap", default="colmap")
    prepare.add_argument("--use-gpu", action="store_true")
    prepare.add_argument("--gpu-index", default=None)
    prepare.add_argument("--mask-frames", type=parse_frames, default=(1, 15, 30, 45, 60))
    prepare.add_argument("--temporal-difference-threshold", type=float, default=35.0)
    prepare.add_argument("--mask-dilation-px", type=int, default=10)
    prepare.add_argument("--max-num-features", type=int, default=32768)
    prepare.add_argument("--max-num-matches", type=int, default=32768)
    prepare.add_argument("--affine-sift", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--dsp-sift", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--apriltags", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--geometric-max-error", type=float, default=2.0)
    prepare.add_argument("--geometric-min-inliers", type=int, default=15)
    prepare.add_argument("--geometric-min-ratio", type=float, default=0.10)
    prepare.set_defaults(func=command_prepare)

    tracks = sub.add_parser("tracks", help="seal final exact tracks and immutable 70/15/15 split")
    add_common(tracks)
    add_split_options(tracks)
    tracks.add_argument("--database", default=None)
    tracks.set_defaults(func=command_tracks)

    mapping = sub.add_parser(
        "map",
        help="bootstrap topology, or seed all cameras from nominal poses and triangulate fit tracks",
    )
    add_common(mapping)
    mapping.add_argument("--init-min-inliers", type=int, default=50)
    mapping.add_argument("--abs-pose-min-inliers", type=int, default=20)
    mapping.add_argument("--abs-pose-max-error", type=float, default=2.0)
    mapping.add_argument("--mapper-filter-error", type=float, default=2.0)
    mapping.add_argument("--min-tri-angle", type=float, default=1.0)
    mapping.add_argument("--mapper-seed", type=int, default=20260822)
    mapping.add_argument(
        "--bootstrap", action="store_true",
        help="use all fresh clean-plate matches for a quarantined topology scout",
    )
    mapping.set_defaults(func=command_map)

    resection = sub.add_parser(
        "resection-audit",
        help="strict exact-index 2D-3D PnP audit and structure-overlap pair proposals",
    )
    add_common(resection)
    resection.add_argument("--database", default=None)
    resection.add_argument("--sparse", default=None)
    resection.add_argument("--min-inliers", type=int, default=20)
    resection.add_argument("--min-independent-neighbors", type=int, default=3)
    resection.add_argument("--min-inliers-per-neighbor", type=int, default=5)
    resection.add_argument("--min-quadrants", type=int, default=3)
    resection.add_argument("--min-hull-fraction", type=float, default=0.02)
    resection.add_argument("--max-reprojection-error", type=float, default=2.0)
    resection.add_argument("--pairs-per-target", type=int, default=8)
    resection.set_defaults(func=command_resection_audit)

    targeted_match = sub.add_parser(
        "targeted-match",
        help="scale-normalized crop discovery with native exact-index cycle checks",
    )
    add_common(targeted_match)
    targeted_match.add_argument("--database", default=None)
    targeted_match.add_argument("--sparse", default=None)
    targeted_match.add_argument("--proposals", default=None)
    targeted_match.add_argument("--colmap", default="colmap")
    targeted_match.add_argument("--use-gpu", action="store_true")
    targeted_match.add_argument("--gpu-index", default=None)
    targeted_match.add_argument("--discovery-long-side", type=int, default=1800)
    targeted_match.add_argument("--ratio", type=float, default=0.72)
    targeted_match.add_argument("--minimum-patch-correlation", type=float, default=0.25)
    targeted_match.add_argument("--native-keypoint-radius", type=float, default=2.0)
    targeted_match.add_argument("--minimum-cycle-neighbors", type=int, default=2)
    targeted_match.add_argument(
        "--manifest-guided", action="store_true",
        help="experimental prior-projected descriptor search; disabled by default",
    )
    targeted_match.add_argument("--guided-search-radius", type=float, default=100.0)
    targeted_match.add_argument("--guided-descriptor-ratio", type=float, default=0.90)
    targeted_match.add_argument("--geometric-max-error", type=float, default=2.0)
    targeted_match.add_argument("--geometric-min-inliers", type=int, default=8)
    targeted_match.add_argument("--geometric-min-ratio", type=float, default=0.10)
    targeted_match.set_defaults(func=command_targeted_match)

    refine = sub.add_parser("refine", help="staged robust BA and validation-selected intrinsics")
    add_common(refine)
    refine.add_argument("--input-sparse", default=None)
    refine.add_argument("--ba-iterations", type=int, default=100)
    refine.add_argument(
        "--retriangulation-rounds",
        type=int,
        default=3,
        help="alternate fit-only robust BA and retriangulation before intrinsic selection",
    )
    refine.add_argument("--min-validation-observations", type=int, default=50)
    refine.add_argument("--bootstrap-seed", type=int, default=20260822)
    refine.set_defaults(func=command_refine)

    validate = sub.add_parser("validate", help="score a fixed validation or sealed test split")
    add_common(validate)
    validate.add_argument("--sparse", default=None)
    validate.add_argument("--split", choices=("validation", "test"), required=True)
    validate.add_argument("--unseal-test", action="store_true")
    validate.add_argument("--min-source-views", type=int, default=2)
    validate.add_argument("--median-gate", type=float, default=1.0)
    validate.add_argument("--p90-gate", type=float, default=2.0)
    validate.add_argument("--min-validation-observations", type=int, default=50)
    validate.add_argument("--min-test-observations", type=int, default=100)
    validate.set_defaults(func=command_validate)

    rescue = sub.add_parser(
        "dynamic-rescue-spec", help="quarantine failed cameras and timestamp priors; mutate nothing"
    )
    add_common(rescue)
    rescue.add_argument("--validation", default=None)
    rescue.set_defaults(func=command_dynamic_rescue_spec)

    dynamic = sub.add_parser(
        "dynamic-rescue",
        help="quarantined failed-camera pose/time optimization from rigid cello tracks",
    )
    add_common(dynamic)
    dynamic.add_argument("--validation", default=None)
    dynamic.add_argument("--trajectories", required=True)
    dynamic.add_argument("--min-offset", type=float, default=-0.25)
    dynamic.add_argument("--max-offset", type=float, default=0.25)
    dynamic.add_argument("--rotation-prior-sigma", type=float, default=0.25)
    dynamic.add_argument("--translation-prior-sigma", type=float, default=1.0)
    dynamic.add_argument("--offset-prior-sigma", type=float, default=0.05)
    dynamic.set_defaults(func=command_dynamic_rescue)

    stability = sub.add_parser(
        "stability-audit", help="full-rig 2D background drift audit"
    )
    add_common(stability)
    stability.add_argument("--frames", type=parse_frames, required=True)
    stability.add_argument("--drift-threshold", type=float, default=0.5)
    stability.add_argument("--minimum-camera-fraction", type=float, default=0.75)
    stability.set_defaults(func=command_stability_audit)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
