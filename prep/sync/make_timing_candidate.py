#!/usr/bin/env python3
"""Build a candidate capture-timing file (trigs.capture_timing, convention_id midpoint_v1) from a readout-aware
sync re-solve (calculate-sync --rs-mode fixed). Writes ONLY --out; an optional sync.json is read for fallbacks, never changed.

Gates for the whole candidate (pre-declared; any failure -> no candidate file, report only):
  G0  the re-solve really ran with --rs-mode fixed
  G1  re-solve verdict >= GOOD (robust graph self-consistency)
  G2  censored phase fraction <= 5%   (failed at 18.9% in the content-referenced production solve)
  G3  every pilot-measured camera (default 01 03 07 08 16 20 21) agrees with the pilot's midpoint constant within
      1.0 ms after both are expressed with the gauge camera at 0
These are supporting checks, not proof of accuracy; held-out scoring is score_timing_candidate.py.
Per-camera policy (thresholds pre-declared):
  resolved   solved by the graph, not demoted/unconstrained, degree >= 3, sigma <= 0.5 ms, split-half <= 1.0 ms
  status     measured  : pilot group camera, resolved, agrees (G3)
             weak      : in --weak-cameras (default 12 13 14), resolved
             converted : any other resolved camera (midpoint constant from the re-solve only)
             unresolved: everything else -> c = its sync.json value (0 = timecode if --sync-json is not given or
                         has no entry for it), rs = false unless --unresolved-rs (its old timing, no readout term: avoids
                         double-counting, but its supervision stays inconsistent with the other cameras)
Each camera also records whether it materially supervises the tested fast hand (hand_supervision.json, if given).
Other takes (no fast-hand pilot): omit --pilot-fit; G3 is recorded as skipped and every resolved camera is 'converted'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT_DEFAULT = ["01", "03", "07", "08", "16", "20", "21"]
WEAK_DEFAULT = ["12", "13", "14"]


def key4(fo: str) -> str:
    return f"{int(fo):04d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-dir", required=True, type=Path, help="calculate-sync --out-dir of the --rs-mode fixed run")
    ap.add_argument("--readout-ms", required=True, type=float)
    ap.add_argument("--sync-json", type=Path, default=None,
                    help="optional sync.json with values for cameras the re-solve could not place (read only). Cameras "
                         "it has no entry for, or every unplaced camera when omitted, start at 0 (timecode only). "
                         "Solved cameras always come from the re-solve, so a copy of sync_candidate.json adds nothing.")
    ap.add_argument("--pilot-fit", type=Path, default=None,
                    help="rs_fit.json with M1 midpoint constants (optional timing analysis). Optional: without it G3 is "
                         "SKIPPED (recorded as not run) and no camera can be 'measured'.")
    ap.add_argument("--hand-supervision", type=Path, default=None,
                    help="hand_supervision.json (example capture: /path/to/hand_supervision.json); metadata only")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--gauge", default="auto",
                    help="camera whose constant is 0 (the solve only fixes differences). 'auto' = sync.json's zero "
                         "camera if the re-solve resolved it, else the resolved camera with the highest graph degree. "
                         "Any choice gives the same relative timing; unresolved cameras are re-referenced to it.")
    ap.add_argument("--pilot-cameras", nargs="+", default=PILOT_DEFAULT)
    ap.add_argument("--weak-cameras", nargs="*", default=WEAK_DEFAULT,
                    help="cameras to label 'weak' (example capture default 12 13 14; pass none for other takes)")
    ap.add_argument("--waive-gate", action="append", default=[],
                    help="write the candidate even though this named gate failed (e.g. G2_censoring). The gate stays "
                         "recorded as FAIL and the candidate is marked experimental. Repeat for several gates.")
    ap.add_argument("--unresolved-rs", action="store_true",
                    help="apply the readout term to unresolved (fallback) cameras too. Use when the fallback sync.json "
                         "values are timecode zeros or came from a readout-aware solve (no double counting); leave off "
                         "when they came from an old readout-unaware solve (example capture).")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a candidate")

    pipe = json.loads((args.resolve_dir / "sync_pipeline_report.json").read_text())
    robust = json.loads((args.resolve_dir / "sync_robust_report.json").read_text())
    cand = json.loads((args.resolve_dir / "sync_candidate.json").read_text())   # PLUS seconds, midpoint-referenced
    sync = json.loads(args.sync_json.read_text()) if args.sync_json is not None else {}
    pilot = json.loads(args.pilot_fit.read_text())["M1"]["c_ms"] if args.pilot_fit is not None else {}
    if args.pilot_fit is None:
        args.pilot_cameras = []
    hand = (json.loads(args.hand_supervision.read_text())
            if args.hand_supervision is not None and args.hand_supervision.exists() else {})

    vr = pipe["phase_measurement"]["visual_report"]
    rs_mode = (vr.get("rolling_shutter") or {}).get("mode")
    censored = float(pipe["phase_measurement"]["censored_phase_fraction"])
    verdict = robust["verdict"]
    per = robust["per_camera"]; split = robust.get("split_half", {}).get("uncertainty_ms", {})
    demoted = set(robust.get("cameras_demoted", [])) | set(robust.get("unconstrained_cameras", []))

    def is_resolved(k):
        st = per.get(k, {}); sig = st.get("sigma_ms"); sh = split.get(k)
        return (k in cand and k not in demoted and st.get("degree", 0) >= 3 and sig is not None and sig <= 0.5
                and (sh is None or sh <= 1.0))

    if str(args.gauge).lower() == "auto":
        zero = [k for k, v in sync.items() if abs(float(v)) < 1e-12 and is_resolved(k)]
        pool = zero or [k for k in sorted(cand) if is_resolved(k)]
        if not pool:
            raise SystemExit("no resolved camera to use as gauge")
        args.gauge = max(pool, key=lambda k: (per.get(k, {}).get("degree", 0), k))[-2:]
        print(f"gauge: camera {args.gauge} ({'sync.json zero camera' if zero else 'highest-degree resolved camera'})")
    if key4(args.gauge) not in cand:
        raise SystemExit(f"gauge camera {args.gauge} is not in the re-solve ({args.resolve_dir}/sync_candidate.json)")
    g_ms = 1000.0 * float(cand[key4(args.gauge)])
    c_ms = {k[-2:]: 1000.0 * float(v) - g_ms for k, v in cand.items()}   # gauge camera at 0
    # Fallback (unresolved) cameras use sync.json, re-referenced to the same gauge camera.
    sync_g_ms = 1000.0 * float(sync.get(key4(args.gauge), 0.0))

    agree = {fo: (c_ms.get(fo), pilot.get(fo), None if fo not in c_ms or fo not in pilot else c_ms[fo] - pilot[fo])
             for fo in args.pilot_cameras}
    gates = dict(
        G0_readout_aware_run=dict(value=rs_mode, passed=rs_mode == "fixed"),
        G1_verdict=dict(value=verdict, passed=verdict in ("GOOD", "EXCELLENT")),
        G2_censoring=dict(value=censored, limit=0.05, passed=censored <= 0.05),
        G3_pilot_agreement=(dict(value={k: v[2] for k, v in agree.items()}, limit_ms=1.0,
                                 passed=all(v[2] is not None and abs(v[2]) <= 1.0 for v in agree.values()))
                            if args.pilot_fit is not None else
                            dict(value=None, passed=None, skipped="no --pilot-fit: pilot agreement not checked")),
    )
    all_pass = all(g["passed"] for g in gates.values() if g["passed"] is not None)
    unknown = [g for g in args.waive_gate if g not in gates]
    if unknown:
        raise SystemExit(f"unknown gate(s) {unknown}; gates are {list(gates)}")
    for must in ("G0_readout_aware_run", "G3_pilot_agreement"):
        if must in args.waive_gate:
            raise SystemExit(f"{must} cannot be waived")
    failed = [k for k, g in gates.items() if g["passed"] is False]
    waived = [k for k in failed if k in args.waive_gate]
    waiver = None
    if waived:
        waiver = dict(waived_gates=waived, gates_remain_failed=True, experimental=True, trusted=False,
                      thresholds_unchanged=True)
    proceed = all_pass or (set(failed) == set(waived))

    # Every camera the solve saw, not just sync.json's keys: a sync.json that omits cameras (e.g. a copied
    # sync_candidate.json, which drops unconstrained cameras) would otherwise silently drop them from the timing file.
    all_cams = (set(sync) | set(cand) | set(per) | set(robust.get("unconstrained_cameras", []))
                | set(robust.get("cameras_demoted", [])) | set(vr.get("offsets_plus_sec", {})))
    timecode_only = sorted(k for k in all_cams if not is_resolved(k) and k not in sync)
    if timecode_only:
        print(f"unresolved camera(s) {[k[-2:] for k in timecode_only]} have no sync.json value: starting at 0 (timecode)")
    cams = {}
    for k in sorted(all_cams):
        fo = k[-2:]
        st = per.get(k, {})
        sig = st.get("sigma_ms"); sh = split.get(k); deg = st.get("degree", 0)
        resolved = is_resolved(k)
        if resolved and fo in args.pilot_cameras and agree[fo][2] is not None and abs(agree[fo][2]) <= 1.0:
            status = "measured"
        elif resolved and fo in args.weak_cameras:
            status = "weak"
        elif resolved:
            status = "converted"
        else:
            status = "unresolved"
        entry = dict(status=status, rs=status != "unresolved" or bool(args.unresolved_rs),
                     c_ms=(round(c_ms[fo], 4) if status != "unresolved"
                           else round(1000.0 * float(sync[k]) - sync_g_ms, 4) if k in sync else 0.0),
                     c_uncertainty_ms=None if not resolved else round(max(v for v in (sig, sh) if v is not None), 3),
                     resolve=dict(degree=deg, sigma_ms=sig, split_half_ms=sh, bias_ms=st.get("bias_ms")),
                     pilot_c_ms=pilot.get(fo), sync_json_ms=round(1000.0 * float(sync[k]), 4) if k in sync else None,
                     supervises_tested_hand=hand.get(fo, {}).get("materially_supervises_tested_hand"),
                     hand_frames_86_96=hand.get(fo, {}).get("hand_frames_86_96"))
        if status == "unresolved":
            base = "sync.json timing" if k in sync else "no sync.json value: timecode (0)"
            entry["note"] = (f"{base} + readout term (--unresolved-rs)" if args.unresolved_rs else
                             f"{base}, readout term off: no double-counting, but inconsistent supervision")
        cams[k] = entry

    summary = dict(gates=gates, all_gates_passed=all_pass, waiver=waiver,
                   statuses={s: sorted(k[-2:] for k, v in cams.items() if v["status"] == s)
                             for s in ("measured", "weak", "converted", "unresolved")},
                   unresolved_supervising_hand=sorted(k[-2:] for k, v in cams.items()
                                                      if v["status"] == "unresolved" and v["supervises_tested_hand"]))
    report_path = args.out.with_suffix(".report.json")
    if report_path.exists():
        raise SystemExit(f"{report_path} exists; refusing to overwrite")
    json.dump(dict(summary=summary, cameras=cams), open(report_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if not proceed:
        print(f"GATES FAILED: no candidate written (report: {report_path})")
        return
    if waiver:
        print(f"WAIVER RECORDED for {waived}: writing an EXPERIMENTAL, UNTRUSTED candidate; gates stay FAIL")
    out = dict(convention_id="midpoint_v1", trusted=bool(all_pass), experimental=bool(waiver), waiver=waiver,
               convention="t = t_nominal + c_k + readout_s*(sensor_row/H - 0.5); c_k midpoint constant, PLUS; "
                          "readout_s > 0: lower rows later",
               readout_ms=float(args.readout_ms), readout_axis="row", fps=float(args.fps), gauge_camera=key4(args.gauge),
               source=dict(resolve_dir=str(args.resolve_dir), pilot_fit=None if args.pilot_fit is None else str(args.pilot_fit),
                           sync_json=None if args.sync_json is None else str(args.sync_json),
                           gates=gates),
               cameras={k: {kk: vv for kk, vv in v.items()} for k, v in cams.items()})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"candidate written: {args.out}")


if __name__ == "__main__":
    main()
