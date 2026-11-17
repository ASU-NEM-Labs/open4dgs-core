#!/usr/bin/env python3
"""
NPZ Keyframe Viewer - Web-based point cloud debugger for FreeTimeGsVanilla

Loads a keyframe NPZ file (output of combine_frames_fast_keyframes.py) and
serves an interactive 3D point cloud viewer at http://localhost:<port>

Features:
  - Frame-by-frame playback with scrubber
  - 3D point cloud rendered with Three.js (WebGL)
  - Velocity vector overlay (arrows or line segments)
  - Per-frame statistics panel
  - Color modes: RGB, time heatmap, velocity magnitude heatmap, has_velocity mask, static/dynamic split
  - Orbit/pan/zoom camera controls
  - Adjustable stage-capsule and detail-sphere previews with copyable ROMA arguments
  - Auto-play with configurable speed

Usage:
    python npz-viewer.py --npz /path/to/keyframes.npz --port 8765
    python npz-viewer.py --npz /path/to/keyframes.npz --port 8765 --max-points 50000
"""

import argparse
import gzip
import http.server
import json
import math
import os
import struct
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# NPZ Loading & Per-Frame Slicing
# ─────────────────────────────────────────────────────────────────────────────

class NPZFrameStore:
    """Loads the flat NPZ and slices it into per-frame views."""

    def __init__(self, npz_path: str, max_points_per_frame: int = 80000):
        self.path = npz_path
        self.max_points = max_points_per_frame

        print(f"[NPZFrameStore] Loading {npz_path} ...")
        data = np.load(npz_path, allow_pickle=False)

        # Required fields
        self.positions   = data["positions"].astype(np.float32)     # [N, 3]
        self.times       = data["times"].astype(np.float32).ravel()  # [N]

        # Optional fields with safe fallbacks
        self.velocities  = data["velocities"].astype(np.float32)  if "velocities"  in data else np.zeros_like(self.positions)
        self.colors      = data["colors"].astype(np.float32)       if "colors"      in data else np.ones((len(self.positions), 3), dtype=np.float32) * 0.5
        self.durations   = data["durations"].astype(np.float32).ravel() if "durations" in data else np.ones(len(self.positions), dtype=np.float32) * 0.1
        self.has_velocity = data["has_velocity"].astype(bool)      if "has_velocity" in data else np.ones(len(self.positions), dtype=bool)
        self.is_static    = data["is_static"].astype(bool).ravel() if "is_static" in data else np.zeros(len(self.positions), dtype=bool)
        self.dynamic_protect = data["dynamic_protect"].astype(bool).ravel() if "dynamic_protect" in data else np.zeros(len(self.positions), dtype=bool)
        self.dynamic_allowed = data["dynamic_allowed"].astype(bool).ravel() if "dynamic_allowed" in data else np.ones(len(self.positions), dtype=bool)
        self.stage_keep = data["stage_keep"].astype(bool).ravel() if "stage_keep" in data else np.ones(len(self.positions), dtype=bool)
        
        # Load integer source frame identifiers to group sub-frame points correctly
        self.source_frames = data["source_frames"].astype(np.int32).ravel() if "source_frames" in data else None

        # Defensive row-alignment guard for new hybrid NPZ fields.
        n_rows = len(self.positions)
        check_dict = {
            "times": self.times,
            "velocities": self.velocities,
            "colors": self.colors,
            "durations": self.durations,
            "has_velocity": self.has_velocity,
            "is_static": self.is_static,
            "dynamic_protect": self.dynamic_protect,
            "dynamic_allowed": self.dynamic_allowed,
            "stage_keep": self.stage_keep,
        }
        if self.source_frames is not None:
            check_dict["source_frames"] = self.source_frames
            
        for name, arr in check_dict.items():
            if len(arr) != n_rows:
                raise ValueError(f"NPZ row mismatch: positions={n_rows}, {name}={len(arr)}")

        # Normalise colors to [0,1]
        if self.colors.max() > 1.01:
            self.colors = self.colors / 255.0
        self.colors = np.clip(self.colors, 0.0, 1.0)

        # Metadata
        self.frame_start   = int(data["frame_start"])   if "frame_start"   in data else 0
        self.frame_end     = int(data["frame_end"])     if "frame_end"     in data else 0
        self.keyframe_step = int(data["keyframe_step"]) if "keyframe_step" in data else 1
        self.n_keyframes   = int(data["n_keyframes"])   if "n_keyframes"   in data else 1

        self.total_frames  = self.frame_end - self.frame_start + 1

        self._static_indices_all = np.where(self.is_static)[0]
        dynamic_time_mask = ~self.is_static
        
        # Group points by integer source_frame instead of floating-point sub-frame time
        if self.source_frames is not None:
            unique_frames = np.unique(self.source_frames[dynamic_time_mask]) if dynamic_time_mask.any() else np.unique(self.source_frames)
            self.frame_source_ids = sorted(unique_frames.tolist())
            denom = max(self.frame_end - self.frame_start, 1)
            self.frame_times = [float(f - self.frame_start) / denom for f in self.frame_source_ids]
        else:
            unique_t = np.unique(self.times[dynamic_time_mask]) if dynamic_time_mask.any() else np.unique(self.times)
            self.frame_times = sorted(unique_t.tolist())
            self.frame_source_ids = None

        self.n_frames = len(self.frame_times)

        print(f"[NPZFrameStore] Indexing {len(self.positions):,} points across {self.n_frames} logical frames ...")
        print(
            f"[NPZFrameStore] Split: static={int(self.is_static.sum()):,}, "
            f"dynamic={int((~self.is_static).sum()):,}, "
            f"protected={int(self.dynamic_protect.sum()):,}, "
            f"dynamic_allowed={int(self.dynamic_allowed.sum()):,}, "
            f"stage_keep={int(self.stage_keep.sum()):,}"
        )
        
        self._frame_indices = {}
        for frame_i, t in enumerate(self.frame_times):
            if self.frame_source_ids is not None:
                f_id = self.frame_source_ids[frame_i]
                dyn_idx = np.where((self.source_frames == f_id) & (~self.is_static))[0]
            else:
                dyn_idx = np.where((self.times == t) & (~self.is_static))[0]
                
            static_idx = self._static_indices_all

            if len(static_idx) + len(dyn_idx) > self.max_points:
                rng = np.random.default_rng(42 + frame_i)
                # Keep a balanced view under small --max-points settings.
                if len(static_idx) > 0 and len(dyn_idx) > 0:
                    static_budget = min(len(static_idx), max(1, self.max_points // 2))
                    dyn_budget = max(0, self.max_points - static_budget)
                else:
                    static_budget = min(len(static_idx), self.max_points)
                    dyn_budget = min(len(dyn_idx), self.max_points - static_budget)

                if len(static_idx) > static_budget:
                    static_idx = rng.choice(static_idx, static_budget, replace=False)
                if len(dyn_idx) > dyn_budget:
                    dyn_idx = rng.choice(dyn_idx, dyn_budget, replace=False)

            idx = np.concatenate([static_idx, dyn_idx]).astype(np.int64, copy=False)
            self._frame_indices[frame_i] = idx

        # Scene-wide stats for normalisation
        self.scene_center = np.median(self.positions, axis=0)
        self.scene_scale  = np.percentile(
            np.linalg.norm(self.positions - self.scene_center, axis=1), 90
        )
        valid_mags = np.linalg.norm(self.velocities[self.has_velocity], axis=1)
        self.vel_max = float(np.percentile(valid_mags, 95)) if len(valid_mags) > 0 else 1.0

        print("[NPZFrameStore] Computing velocity coherence per frame...")
        from scipy.spatial import cKDTree
        self._frame_coherence = {}
        K_NEIGHBORS = 16

        for frame_i, idx in self._frame_indices.items():
            pos_frame = self.positions[idx]
            vel_frame = self.velocities[idx]
            hv_frame = self.has_velocity[idx]
            coherence = np.zeros(len(idx), dtype=np.float32)

            valid_mask = hv_frame & (np.linalg.norm(vel_frame, axis=1) > 1e-9)
            if valid_mask.sum() < K_NEIGHBORS + 1:
                self._frame_coherence[frame_i] = coherence
                continue

            valid_pos = pos_frame[valid_mask]
            valid_vel = vel_frame[valid_mask]
            valid_vel_norm = valid_vel / (np.linalg.norm(valid_vel, axis=1, keepdims=True) + 1e-9)

            tree = cKDTree(valid_pos)
            _, neighbors = tree.query(valid_pos, k=K_NEIGHBORS + 1)  # +1 to skip self
            neighbor_mean = valid_vel[neighbors[:, 1:]].mean(axis=1)
            neighbor_mean_norm = neighbor_mean / (np.linalg.norm(neighbor_mean, axis=1, keepdims=True) + 1e-9)

            coh_valid = np.sum(valid_vel_norm * neighbor_mean_norm, axis=1)
            valid_idx_in_frame = np.where(valid_mask)[0]
            coherence[valid_idx_in_frame] = coh_valid
            self._frame_coherence[frame_i] = coherence

        print("[NPZFrameStore] Coherence ready.")
        print(f"[NPZFrameStore] Ready.")
        print(f"  Frames: {self.n_frames}  |  Total pts: {len(self.positions):,}")
        print(f"  Scene center: {self.scene_center}")
        print(f"  Scene scale (p90 dist): {self.scene_scale:.4f}")
        print(f"  Vel max: {self.vel_max:.6f}")

    def get_frame_data(self, frame_index: int) -> dict:
        """Return serialisable dict for frame_index (0-based)."""
        frame_index = max(0, min(frame_index, self.n_frames - 1))
        t = self.frame_times[frame_index]
        idx = self._frame_indices[frame_index]

        pos  = self.positions[idx]
        vel  = self.velocities[idx]
        col  = self.colors[idx]
        hv   = self.has_velocity[idx]
        dur  = self.durations[idx]
        st   = self.is_static[idx]
        prot = self.dynamic_protect[idx]
        dyn_allowed = self.dynamic_allowed[idx]
        st_keep = self.stage_keep[idx]

        vel_mags = np.linalg.norm(vel, axis=1)

        # Pack flat float32 arrays → list (JSON-safe via base64 later)
        return {
            "frame_index": frame_index,
            "n_frames": self.n_frames,
            "t_normalized": float(t),
            "n_points": len(idx),
            "n_valid_velocity": int(hv.sum()),
            "n_static": int(st.sum()),
            "n_dynamic": int((~st).sum()),
            "n_dynamic_protect": int(prot.sum()),
            "n_dynamic_allowed": int(dyn_allowed.sum()),
            "n_stage_keep": int(st_keep.sum()),
            "vel_mean": float(vel_mags[hv].mean()) if hv.any() else 0.0,
            "vel_max": float(vel_mags.max()) if len(vel_mags) else 0.0,
            "vel_max_global": self.vel_max,
            "scene_center": self.scene_center.tolist(),
            "scene_scale": float(self.scene_scale),
            # Float32 arrays encoded as hex strings for compact transfer
            "positions":  pos.astype(np.float32).tobytes().hex(),
            "velocities": vel.astype(np.float32).tobytes().hex(),
            "colors":     col.astype(np.float32).tobytes().hex(),
            "has_velocity": hv.astype(np.uint8).tobytes().hex(),
            "is_static": st.astype(np.uint8).tobytes().hex(),
            "dynamic_protect": prot.astype(np.uint8).tobytes().hex(),
            "dynamic_allowed": dyn_allowed.astype(np.uint8).tobytes().hex(),
            "stage_keep": st_keep.astype(np.uint8).tobytes().hex(),
            "coherence": self._frame_coherence[frame_index].astype(np.float32).tobytes().hex(),
        }

    def get_summary(self) -> dict:
        return {
            "npz_path": str(self.path),
            "n_frames": self.n_frames,
            "n_points_total": int(len(self.positions)),
            "n_static_total": int(self.is_static.sum()),
            "n_dynamic_total": int((~self.is_static).sum()),
            "n_dynamic_protect_total": int(self.dynamic_protect.sum()),
            "n_dynamic_allowed_total": int(self.dynamic_allowed.sum()),
            "n_stage_keep_total": int(self.stage_keep.sum()),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "keyframe_step": self.keyframe_step,
            "n_keyframes": self.n_keyframes,
            "total_frames": self.total_frames,
            "scene_center": self.scene_center.tolist(),
            "scene_scale": float(self.scene_scale),
            "vel_max": self.vel_max,
            "frame_times": self.frame_times,
        }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────────────────────────────────────

STORE: NPZFrameStore = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NPZ Keyframe Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:      #0a0c0f;
    --panel:   #111318;
    --border:  #1e2330;
    --accent:  #00e5ff;
    --accent2: #ff6b35;
    --accent3: #7cfc6a;
    --text:    #c8d0e0;
    --muted:   #4a5568;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  #topbar {
    height: 42px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 16px;
    gap: 20px;
    flex-shrink: 0;
    z-index: 10;
  }
  #topbar .logo {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
  }
  #npz-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }

  #main { display: flex; flex: 1; overflow: hidden; }

  #sidebar {
    width: 264px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex-shrink: 0;
  }

  .section { border-bottom: 1px solid var(--border); padding: 13px 15px; }
  .section-title {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 9px;
  }

  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; }
  .stat-item label {
    font-family: var(--mono); font-size: 9px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; display: block; margin-bottom: 1px;
  }
  .stat-item .val { font-family: var(--mono); font-size: 12px; color: var(--accent); font-weight: 500; }
  .stat-item .val.green  { color: var(--accent3); }
  .stat-item .val.orange { color: var(--accent2); }

  .ctrl-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
  .ctrl-row:last-child { margin-bottom: 0; }
  .ctrl-row label { font-family: var(--mono); font-size: 10px; color: var(--muted); min-width: 46px; }

  input[type=range]    { flex: 1; accent-color: var(--accent); cursor: pointer; }
  input[type=checkbox] { accent-color: var(--accent); cursor: pointer; }
  input[type=number] {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 10px;
    padding: 3px 5px;
    border-radius: 3px;
    width: 72px;
  }
  input[type=text] {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 9px;
    padding: 4px 5px;
    border-radius: 3px;
    width: 100%;
  }
  input[type=color] {
    width: 26px; height: 20px;
    border: 1px solid var(--border); border-radius: 3px;
    cursor: pointer; padding: 1px 2px; background: var(--bg); flex-shrink: 0;
  }
  select {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    font-family: var(--mono); font-size: 10px; padding: 3px 6px;
    border-radius: 3px; cursor: pointer; flex: 1;
  }
  select:focus { outline: 1px solid var(--accent); }

  .btn-row { display: flex; gap: 5px; flex-wrap: wrap; }
  button {
    background: transparent; border: 1px solid var(--border); color: var(--text);
    font-family: var(--mono); font-size: 10px; padding: 4px 9px;
    border-radius: 3px; cursor: pointer; transition: border-color 0.15s, color 0.15s;
  }
  button:hover  { border-color: var(--accent); color: var(--accent); }
  button.active { border-color: var(--accent); color: var(--accent); background: rgba(0,229,255,0.06); }
  button.sm     { font-size: 9px; padding: 3px 7px; }

  /* Background swatches */
  .bg-presets { display: flex; gap: 5px; margin-top: 7px; flex-wrap: wrap; }
  .bg-swatch {
    width: 22px; height: 22px; border-radius: 3px;
    border: 1px solid var(--border); cursor: pointer;
    transition: border-color 0.15s, transform 0.1s; flex-shrink: 0;
  }
  .bg-swatch:hover  { border-color: var(--accent); transform: scale(1.15); }
  .bg-swatch.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }

  /* Rotation readouts */
  .rot-val { font-family:var(--mono); font-size:10px; color:var(--accent2); min-width:38px; text-align:right; flex-shrink:0; }
  .num-val { font-family:var(--mono); font-size:10px; color:var(--accent);  min-width:30px; text-align:right; flex-shrink:0; }
  #vel-scale-val { font-family:var(--mono); font-size:10px; color:var(--accent2); min-width:36px; text-align:right; }

  /* Canvas */
  #canvas-wrap { flex: 1; position: relative; overflow: hidden; }
  #canvas-wrap canvas { display:block; width:100%!important; height:100%!important; }

  /* Scrubber overlay */
  #scrubber-bar {
    position: absolute; bottom:0; left:0; right:0; height:54px;
    background: linear-gradient(transparent, rgba(10,12,15,0.93));
    display: flex; align-items: center; padding: 0 18px; gap: 10px;
  }
  #frame-display { font-family:var(--mono); font-size:11px; color:var(--accent); min-width:88px; flex-shrink:0; }
  #scrubber { flex:1; accent-color:var(--accent); cursor:pointer; height:4px; }
  #t-display { font-family:var(--mono); font-size:10px; color:var(--muted); min-width:54px; text-align:right; flex-shrink:0; }
  .play-btn { font-size:13px; padding:3px 8px; border-color:var(--muted); }

  /* Loading */
  #loading {
    position:absolute; inset:0; background:rgba(10,12,15,0.86);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:12px; z-index:100; font-family:var(--mono); font-size:11px;
    color:var(--accent); letter-spacing:0.1em; transition:opacity 0.3s;
  }
  #loading.hidden { opacity:0; pointer-events:none; }
  .spinner {
    width:26px; height:26px; border:2px solid var(--border);
    border-top-color:var(--accent); border-radius:50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Velocity legend */
  #vel-legend { display:none; margin-top:7px; height:7px; border-radius:2px;
    background: linear-gradient(to right, #05080c, #ff6b35); }
  #vel-legend.visible { display:block; }
  .legend-labels { display:flex; justify-content:space-between;
    font-family:var(--mono); font-size:9px; color:var(--muted); margin-top:2px; }
</style>
</head>
<body>

<div id="topbar">
  <span class="logo">▸ NPZ Viewer</span>
  <span id="npz-label">Loading...</span>
</div>

<div id="main">
  <div id="sidebar">

    <div class="section">
      <div class="section-title">Dataset</div>
      <div class="stat-grid">
        <div class="stat-item"><label>Keyframes</label><span class="val" id="s-nframes">—</span></div>
        <div class="stat-item"><label>Total pts</label><span class="val" id="s-npts">—</span></div>
        <div class="stat-item"><label>Seq frames</label><span class="val" id="s-seqlen">—</span></div>
        <div class="stat-item"><label>KF step</label><span class="val" id="s-kfstep">—</span></div>
        <div class="stat-item"><label>Static</label><span class="val" id="s-static">—</span></div>
        <div class="stat-item"><label>Dynamic</label><span class="val" id="s-dynamic">—</span></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Frame Stats</div>
      <div class="stat-grid">
        <div class="stat-item"><label>Points</label><span class="val green" id="f-npts">—</span></div>
        <div class="stat-item"><label>Valid vel</label><span class="val green" id="f-vvel">—</span></div>
        <div class="stat-item"><label>t norm</label><span class="val" id="f-tnorm">—</span></div>
        <div class="stat-item"><label>vel mean</label><span class="val orange" id="f-vmean">—</span></div>
        <div class="stat-item"><label>vel max</label><span class="val orange" id="f-vmax">—</span></div>
        <div class="stat-item"><label>Static</label><span class="val" id="f-static">—</span></div>
        <div class="stat-item"><label>Dynamic</label><span class="val" id="f-dynamic">—</span></div>
        <div class="stat-item"><label>Protected</label><span class="val orange" id="f-protected">—</span></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Color Mode</div>
      <div class="ctrl-row" style="margin-bottom:0">
        <select id="color-mode">
          <option value="rgb">RGB (captured)</option>
          <option value="time">Time heatmap</option>
          <option value="velmag">Velocity magnitude</option>
          <option value="hasvel">Has velocity mask</option>
          <option value="split">Static / Dynamic split</option>
          <option value="protected">Dynamic protected mask</option>
          <option value="dynamic_allowed">Dynamic allow saved mask</option>
          <option value="stage_keep">Stage keep saved mask</option>
          <option value="roi_split">ROI split saved</option>
          <option value="capsule">Stage capsule preview</option>
          <option value="detail_sphere">Detail sphere preview</option>
          <option value="coherence">Velocity coherence</option>
        </select>
      </div>
      <div id="vel-legend"></div>
      <div class="legend-labels" id="vel-legend-labels" style="display:none">
        <span>0</span><span id="vel-legend-max">—</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Stage Capsule Preview</div>

      <div class="ctrl-row">
        <label>Target</label>
        <select id="capsule-target">
          <option value="protect">Protect capsule</option>
          <option value="dynamic_allow">Dynamic allow capsule</option>
          <option value="stage_keep">Stage keep capsule</option>
        </select>
      </div>

      <div class="ctrl-row">
        <label>Show</label>
        <input type="checkbox" id="capsule-show">
        <button class="sm" id="capsule-use-scene-center">Use scene center</button>
      </div>

      <div class="ctrl-row">
        <button class="sm" id="capsule-use-scene-up">Use current visual up</button>
      </div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:5px;letter-spacing:0.08em">CENTER</div>
      <div class="ctrl-row">
        <label>Cx</label>
        <input type="number" id="capsule-cx" step="0.05" value="0.00">
        <label style="min-width:18px">Cy</label>
        <input type="number" id="capsule-cy" step="0.05" value="0.00">
      </div>
      <div class="ctrl-row">
        <label>Cz</label>
        <input type="number" id="capsule-cz" step="0.05" value="0.00">
      </div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin:8px 0 5px;letter-spacing:0.08em">SHAPE</div>
      <div class="ctrl-row">
        <label>Length</label>
        <input type="number" id="capsule-length" step="0.05" value="3.20">
        <label style="min-width:18px">Radius</label>
        <input type="number" id="capsule-radius" step="0.05" value="0.70">
      </div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin:8px 0 5px;letter-spacing:0.08em">CAPSULE ROTATION</div>
      <div class="ctrl-row">
        <label>Rx</label>
        <input type="number" id="capsule-rx" step="1" value="0">
        <label style="min-width:18px">Ry</label>
        <input type="number" id="capsule-ry" step="1" value="0">
      </div>
      <div class="ctrl-row">
        <label>Rz</label>
        <input type="number" id="capsule-rz" step="1" value="0">
      </div>

      <div class="btn-row" style="margin-top:6px">
        <button class="sm" id="capsule-copy-args">Copy ROMA args</button>
      </div>

      <input type="text" id="capsule-args" readonly value="">
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:6px;line-height:1.45">
        Rotate the scene until the actor looks upright, click “Use current visual up,” then tune center/length/radius.
        Color mode “Stage capsule preview”: magenta = inside capsule, cyan = outside.
      </div>
    </div>

    <div class="section">
      <div class="section-title">Detail Sphere Preview</div>

      <div class="ctrl-row">
        <label>Show</label>
        <input type="checkbox" id="detail-sphere-show">
        <button class="sm" id="detail-sphere-use-scene-center">Use scene center</button>
      </div>

      <div class="btn-row" style="margin-bottom:8px">
        <button class="sm" id="detail-sphere-use-capsule-a">Use capsule A</button>
        <button class="sm" id="detail-sphere-use-capsule-b">Use capsule B</button>
      </div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:5px;letter-spacing:0.08em">CENTER</div>
      <div class="ctrl-row">
        <label>Cx</label>
        <input type="number" id="detail-sphere-cx" step="0.05" value="0.00">
        <label style="min-width:18px">Cy</label>
        <input type="number" id="detail-sphere-cy" step="0.05" value="0.00">
      </div>
      <div class="ctrl-row">
        <label>Cz</label>
        <input type="number" id="detail-sphere-cz" step="0.05" value="0.00">
      </div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin:8px 0 5px;letter-spacing:0.08em">SAMPLING</div>
      <div class="ctrl-row">
        <label>Radius</label>
        <input type="number" id="detail-sphere-radius" min="0.001" step="0.05" value="0.40">
        <label style="min-width:42px">Fraction</label>
        <input type="number" id="detail-sphere-fraction" min="0.01" max="0.99" step="0.05" value="0.25">
      </div>
      <div class="ctrl-row">
        <label>Shown inside</label>
        <span class="num-val" id="detail-sphere-count">0 / 0</span>
      </div>

      <div class="btn-row" style="margin-top:6px">
        <button class="sm" id="detail-sphere-copy-args">Copy ROMA args</button>
      </div>

      <input type="text" id="detail-sphere-args" readonly value="">
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:6px;line-height:1.45">
        The sphere is evaluated in raw COLMAP/world coordinates. Color mode “Detail sphere preview”:
        yellow = inside the reserved region, dim blue = outside. Fraction is the share of
        <code>--target-points-per-frame</code> reserved for it.
      </div>
    </div>

    <div class="section">
      <div class="section-title">Viewport Background</div>
      <div class="ctrl-row" style="margin-bottom:0">
        <label>Custom</label>
        <input type="color" id="bg-color-picker" value="#0a0c0f">
        <span style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-left:4px">or pick preset:</span>
      </div>
      <div class="bg-presets">
        <div class="bg-swatch active" data-color="#0a0c0f" style="background:#0a0c0f" title="Near black"></div>
        <div class="bg-swatch" data-color="#1c1c1c" style="background:#1c1c1c" title="Dark grey"></div>
        <div class="bg-swatch" data-color="#444"    style="background:#444"    title="Mid grey"></div>
        <div class="bg-swatch" data-color="#999"    style="background:#999"    title="Light grey"></div>
        <div class="bg-swatch" data-color="#ddd"    style="background:#ddd"    title="Off white"></div>
        <div class="bg-swatch" data-color="#ffffff" style="background:#fff;border-color:#555" title="White"></div>
        <div class="bg-swatch" data-color="#0d1b2a" style="background:#0d1b2a" title="Navy"></div>
        <div class="bg-swatch" data-color="#1a0a2e" style="background:#1a0a2e" title="Purple"></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Scene Rotation</div>
      <div style="margin-bottom:8px">
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:5px;letter-spacing:0.08em">ORBIT UP DIRECTION</div>
        <div class="btn-row">
          <button id="btn-reset-up" title="Use the camera's current up as the orbit pole — same as nerfview Reset Up">Reset Up ↑</button>
        </div>
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:6px;line-height:1.5">
          Tilt the view so the scene looks right, then press Reset Up to lock that as the orbit axis.
        </div>
      </div>
      <div class="ctrl-row">
        <label>X</label>
        <input type="range" id="rot-x" min="-180" max="180" step="1" value="0">
        <span class="rot-val" id="rot-x-val">0°</span>
      </div>
      <div class="ctrl-row">
        <label>Y</label>
        <input type="range" id="rot-y" min="-180" max="180" step="1" value="0">
        <span class="rot-val" id="rot-y-val">0°</span>
      </div>
      <div class="ctrl-row">
        <label>Z</label>
        <input type="range" id="rot-z" min="-180" max="180" step="1" value="0">
        <span class="rot-val" id="rot-z-val">0°</span>
      </div>
      <div class="btn-row" style="margin-top:6px">
        <button id="btn-rot-reset">Reset rotation</button>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Velocity Arrows</div>
      <div class="ctrl-row">
        <label>Show</label>
        <input type="checkbox" id="show-vel">
        <label style="min-width:unset;margin-left:4px">Valid only</label>
        <input type="checkbox" id="vel-valid-only" checked>
      </div>
      <div class="ctrl-row">
        <label>Scale</label>
        <input type="range" id="vel-scale" min="0.5" max="200" step="0.5" value="20">
        <span id="vel-scale-val">20×</span>
      </div>
      <div class="ctrl-row">
        <label>Max</label>
        <input type="range" id="vel-max-arrows" min="100" max="5000" step="100" value="1000">
        <span id="vel-max-val" style="font-family:var(--mono);font-size:10px;color:var(--accent2);min-width:36px;text-align:right">1k</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Point Size</div>
      <div class="ctrl-row" style="margin-bottom:0">
        <label>Size</label>
        <input type="range" id="pt-size" min="0.5" max="20" step="0.25" value="1.5">
        <span class="num-val" id="pt-size-val">1.5</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Playback</div>
      <div class="ctrl-row">
        <label>FPS</label>
        <input type="range" id="play-speed" min="1" max="30" step="1" value="10">
        <span class="num-val" id="play-speed-val">10</span>
      </div>
      <div class="btn-row">
        <button id="btn-prev">◀◀</button>
        <button id="btn-play" class="play-btn">▶ Play</button>
        <button id="btn-next">▶▶</button>
      </div>
      <div id="prefetch-row" style="margin-top:9px;transition:opacity 1s ease">
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:4px;letter-spacing:0.08em">CACHING FRAMES</div>
        <div style="height:3px;background:var(--border);border-radius:2px;overflow:hidden">
          <div id="prefetch-bar" style="height:100%;width:0%;background:var(--accent);transition:width 0.2s;border-radius:2px"></div>
        </div>
        <div id="prefetch-pct" style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:3px">0 / 0 frames cached</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Camera</div>
      <div class="btn-row">
        <button id="btn-reset-cam">Reset</button>
        <button id="btn-top-cam">Top</button>
        <button id="btn-front-cam">Front</button>
      </div>
    </div>

  </div><div id="canvas-wrap">
    <div id="loading"><div class="spinner"></div><span class="loading-msg">LOADING...</span></div>
    <div id="scrubber-bar">
      <button class="play-btn" id="sb-prev">◀</button>
      <button class="play-btn" id="sb-play">▶</button>
      <button class="play-btn" id="sb-next">▶</button>
      <span id="frame-display">Frame 0 / 0</span>
      <input type="range" id="scrubber" min="0" max="0" step="1" value="0">
      <span id="t-display">t=0.000</span>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Globals ──────────────────────────────────────────
let scene, camera, renderer, sceneGroup;
let pointCloud = null, velLines = null, stageCapsule = null, detailSphere = null;
let summary = null, frameData = null;
let currentFrame = 0, isPlaying = false, playInterval = null;
let sceneRotDeg = { x: 0, y: 0, z: 0 };

// ── Three.js init ─────────────────────────────────────
function initThree() {
  const wrap = document.getElementById('canvas-wrap');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c0f);

  // sceneGroup receives all point cloud geometry so rotation is applied
  // to the data independently of the camera — full 360° orbit stays intact.
  sceneGroup = new THREE.Group();
  scene.add(sceneGroup);

  camera = new THREE.PerspectiveCamera(55, wrap.clientWidth / wrap.clientHeight, 0.001, 1000);
  camera.position.set(0, 0.5, 2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  wrap.insertBefore(renderer.domElement, wrap.firstChild);

  // Small world-space axes only (no grid)
  scene.add(new THREE.AxesHelper(0.15));

  setupOrbitControls();
  window.addEventListener('resize', onResize);
  animate();
}

// ── Scene rotation ────────────────────────────────────
function applySceneRotation() {
  const d = Math.PI / 180;
  sceneGroup.rotation.set(sceneRotDeg.x * d, sceneRotDeg.y * d, sceneRotDeg.z * d);
}

// ── Orbit controls ────────────────────────────────────
// General-up-vector spherical orbit. Instead of always using world Y as
// the pole, we maintain an explicit `upVec` (unit vector). The camera
// position is computed in an orthonormal frame {right, upVec, forward}
// built fresh each update() call from theta/phi relative to upVec.
//
// resetUp(): reads the camera's current screen-up column from its world
// matrix and makes that the new pole — exactly like nerfview "Reset Up".
// The camera does NOT jump; we re-decompose the current offset into
// new theta/phi relative to the new pole so the view is continuous.
function setupOrbitControls() {
  let isDragging = false, isPanning = false;
  let lastX = 0, lastY = 0;
  let theta = 0.3, phi = 1.1, radius = 2.0;
  let target = new THREE.Vector3();

  // The orbit pole — default world Y. Reset Up replaces this.
  let upVec = new THREE.Vector3(0, 1, 0);

  // Build an orthonormal basis {right, up, fwd} where up = upVec.
  // We choose an arbitrary non-parallel seed to get right, then re-derive.
  function makeBasis() {
    const up = upVec.clone().normalize();
    // Pick a seed direction not parallel to up
    const seed = Math.abs(up.y) < 0.9
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
    const right = new THREE.Vector3().crossVectors(seed, up).normalize();
    const fwd   = new THREE.Vector3().crossVectors(up, right).normalize();
    return { up, right, fwd };
  }

  function update() {
    const { up, right, fwd } = makeBasis();
    // Spherical coords in the {right, up, fwd} frame:
    //   phi=0   → directly above  (along up)
    //   phi=π/2 → equator
    //   phi=π   → directly below
    const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
    const sinTh  = Math.sin(theta), cosTh = Math.cos(theta);

    // offset = radius * (sinPhi*(cosTh*fwd + sinTh*right) + cosPhi*up)
    const offset = new THREE.Vector3()
      .addScaledVector(fwd,   sinPhi * cosTh)
      .addScaledVector(right, sinPhi * sinTh)
      .addScaledVector(up,    cosPhi)
      .multiplyScalar(radius);

    camera.position.copy(target).add(offset);
    camera.up.copy(up);
    camera.lookAt(target);
    camera.updateMatrixWorld();
  }
  update();

  // resetUp: make the camera's current screen-up the new orbit pole.
  // Re-decompose the current camera→target offset into (theta, phi)
  // in the new basis so the camera doesn't jump.
  window.resetOrbitUp = () => {
    // New pole = camera's current screen-up (column 1 of world matrix)
    const newUp = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
    upVec.copy(newUp);

    // Re-decompose current offset into new spherical coords
    const offset = new THREE.Vector3().subVectors(camera.position, target);
    const len = offset.length();
    if (len < 1e-6) return;
    radius = len;

    const { up, right, fwd } = makeBasis();
    // Project offset onto new basis
    const ou = offset.dot(up)    / radius;   // cos(phi)
    const of = offset.dot(fwd)   / radius;
    const or_ = offset.dot(right) / radius;

    phi   = Math.acos(Math.max(-1, Math.min(1, ou)));
    theta = Math.atan2(or_, of);

    update();
  };

  const el = renderer.domElement;
  el.addEventListener('mousedown', e => {
    if (e.button === 0) isDragging = true;
    if (e.button === 1 || e.button === 2) isPanning = true;
    lastX = e.clientX; lastY = e.clientY;
    e.preventDefault();
  });
  window.addEventListener('mouseup', () => { isDragging = false; isPanning = false; });
  window.addEventListener('mousemove', e => {
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (isDragging) {
      theta -= dx * 0.005;
      phi = Math.max(0.001, Math.min(Math.PI - 0.001, phi + dy * 0.005));
      update();
    }
    if (isPanning) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
      const up    = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
      const s = radius * 0.0012;
      target.addScaledVector(right, -dx * s);
      target.addScaledVector(up,     dy * s);
      update();
    }
  });
  el.addEventListener('wheel', e => {
    radius = Math.max(0.02, radius * (1 + e.deltaY * 0.001));
    update(); e.preventDefault();
  }, { passive: false });
  el.addEventListener('contextmenu', e => e.preventDefault());

  window.resetCamera = (cx, cy, cz, tx, ty, tz) => {
    // Full camera reset also resets upVec to world Y
    upVec.set(0, 1, 0);
    target.set(tx??0, ty??0, tz??0);
    const ddx=cx-(tx??0), ddy=cy-(ty??0), ddz=cz-(tz??0);
    radius = Math.sqrt(ddx*ddx+ddy*ddy+ddz*ddz) || 2;
    // Decompose into new basis (which is just world Y basis after reset)
    const { up, right, fwd } = makeBasis();
    const len = radius;
    const ou = ddy/len, of = ddz/len, or_ = ddx/len;
    phi   = Math.acos(Math.max(-1, Math.min(1, ou)));
    theta = Math.atan2(or_, of);
    update();
  };
}

// ── Hex helpers ───────────────────────────────────────
function hexToFloat32(h) {
  const b = new Uint8Array(h.length/2);
  for (let i=0;i<b.length;i++) b[i]=parseInt(h.substr(i*2,2),16);
  return new Float32Array(b.buffer);
}
function hexToUint8(h) {
  const b = new Uint8Array(h.length/2);
  for (let i=0;i<b.length;i++) b[i]=parseInt(h.substr(i*2,2),16);
  return b;
}

// ── Stage capsule preview ─────────────────────────────
function degToRad(v) {
  return parseFloat(v) * Math.PI / 180.0;
}

function radToDeg(v) {
  return v * 180.0 / Math.PI;
}

function getCapsuleParams() {
  const cx = parseFloat(document.getElementById('capsule-cx').value);
  const cy = parseFloat(document.getElementById('capsule-cy').value);
  const cz = parseFloat(document.getElementById('capsule-cz').value);

  const length = Math.max(0.001, parseFloat(document.getElementById('capsule-length').value));
  const r = Math.max(0.001, parseFloat(document.getElementById('capsule-radius').value));

  const rx = degToRad(document.getElementById('capsule-rx').value);
  const ry = degToRad(document.getElementById('capsule-ry').value);
  const rz = degToRad(document.getElementById('capsule-rz').value);

  // Local capsule axis is +Y. Rotation controls rotate that axis in raw COLMAP/world space.
  const euler = new THREE.Euler(rx, ry, rz, 'XYZ');
  const axis = new THREE.Vector3(0, 1, 0).applyEuler(euler).normalize();

  const half = length * 0.5;

  const ax = cx - axis.x * half;
  const ay = cy - axis.y * half;
  const az = cz - axis.z * half;

  const bx = cx + axis.x * half;
  const by = cy + axis.y * half;
  const bz = cz + axis.z * half;

  return {
    cx, cy, cz,
    length,
    r,
    rx, ry, rz,
    axis,
    ax, ay, az,
    bx, by, bz,
  };
}

function pointInsideStageCapsule(px, py, pz, c) {
  // Same math as ROMA _stage_capsule_protect_mask:
  // distance from point P to segment A-B <= radius.
  const abx = c.bx - c.ax;
  const aby = c.by - c.ay;
  const abz = c.bz - c.az;

  const apx = px - c.ax;
  const apy = py - c.ay;
  const apz = pz - c.az;

  const abLen2 = abx*abx + aby*aby + abz*abz;

  if (abLen2 <= 1e-12) {
    const dx = px - c.ax;
    const dy = py - c.ay;
    const dz = pz - c.az;
    return (dx*dx + dy*dy + dz*dz) <= c.r * c.r;
  }

  let t = (apx*abx + apy*aby + apz*abz) / abLen2;
  t = Math.max(0.0, Math.min(1.0, t));

  const qx = c.ax + t * abx;
  const qy = c.ay + t * aby;
  const qz = c.az + t * abz;

  const dx = px - qx;
  const dy = py - qy;
  const dz = pz - qz;

  return (dx*dx + dy*dy + dz*dz) <= c.r * c.r;
}

function disposeObject3D(obj) {
  if (!obj) return;
  obj.traverse(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  });
}

function updateCapsuleArgsText() {
  const c = getCapsuleParams();
  const target = document.getElementById('capsule-target')?.value || 'protect';

  let txt = '';

  if (target === 'protect') {
    txt =
      `--stage-capsule-protect ` +
      `--stage-capsule-a="${c.ax.toFixed(3)},${c.ay.toFixed(3)},${c.az.toFixed(3)}" ` +
      `--stage-capsule-b="${c.bx.toFixed(3)},${c.by.toFixed(3)},${c.bz.toFixed(3)}" ` +
      `--stage-capsule-radius ${c.r.toFixed(3)}`;
  } else if (target === 'dynamic_allow') {
    txt =
      `--dynamic-allow-capsule ` +
      `--dynamic-allow-capsule-a="${c.ax.toFixed(3)},${c.ay.toFixed(3)},${c.az.toFixed(3)}" ` +
      `--dynamic-allow-capsule-b="${c.bx.toFixed(3)},${c.by.toFixed(3)},${c.bz.toFixed(3)}" ` +
      `--dynamic-allow-capsule-radius ${c.r.toFixed(3)}`;
  } else if (target === 'stage_keep') {
    txt =
      `--stage-keep-capsule ` +
      `--stage-keep-capsule-a="${c.ax.toFixed(3)},${c.ay.toFixed(3)},${c.az.toFixed(3)}" ` +
      `--stage-keep-capsule-b="${c.bx.toFixed(3)},${c.by.toFixed(3)},${c.bz.toFixed(3)}" ` +
      `--stage-keep-capsule-radius ${c.r.toFixed(3)}`;
  }

  const out = document.getElementById('capsule-args');
  if (out) out.value = txt;

  return txt;
}

function updateStageCapsule() {
  if (!sceneGroup) return;

  if (stageCapsule) {
    sceneGroup.remove(stageCapsule);
    disposeObject3D(stageCapsule);
    stageCapsule = null;
  }

  updateCapsuleArgsText();

  const show = document.getElementById('capsule-show')?.checked;
  if (!show) return;

  const c = getCapsuleParams();

  const a = new THREE.Vector3(c.ax, c.ay, c.az);
  const b = new THREE.Vector3(c.bx, c.by, c.bz);

  const ab = new THREE.Vector3().subVectors(b, a);
  const height = ab.length();

  if (height <= 1e-6) return;

  const dir = ab.clone().normalize();
  const mid = a.clone().add(b).multiplyScalar(0.5);

  const group = new THREE.Group();
  group.name = 'stage-capsule-preview';

  const mat = new THREE.MeshBasicMaterial({
    color: 0xff33ff,
    wireframe: true,
    transparent: true,
    opacity: 0.65,
    depthTest: true,
  });

  const lineMat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
    depthTest: true,
  });

  // CylinderGeometry is Y-axis aligned by default.
  // Rotate it so local Y points from A to B.
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

  const cylGeo = new THREE.CylinderGeometry(c.r, c.r, height, 32, 1, true);
  const cyl = new THREE.Mesh(cylGeo, mat);
  cyl.position.copy(mid);
  cyl.quaternion.copy(q);
  group.add(cyl);

  const sphereGeo = new THREE.SphereGeometry(c.r, 24, 12);

  const s0 = new THREE.Mesh(sphereGeo, mat);
  s0.position.copy(a);
  group.add(s0);

  const s1 = new THREE.Mesh(sphereGeo.clone(), mat.clone());
  s1.position.copy(b);
  group.add(s1);

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute(
    'position',
    new THREE.BufferAttribute(
      new Float32Array([
        c.ax, c.ay, c.az,
        c.bx, c.by, c.bz,
      ]),
      3
    )
  );
  const line = new THREE.Line(lineGeo, lineMat);
  group.add(line);

  stageCapsule = group;
  sceneGroup.add(stageCapsule);
}

function setCapsuleEulerFromAxis(axis) {
  const src = new THREE.Vector3(0, 1, 0);
  const dst = axis.clone().normalize();

  const q = new THREE.Quaternion();
  q.setFromUnitVectors(src, dst);

  const e = new THREE.Euler();
  e.setFromQuaternion(q, 'XYZ');

  document.getElementById('capsule-rx').value = radToDeg(e.x).toFixed(2);
  document.getElementById('capsule-ry').value = radToDeg(e.y).toFixed(2);
  document.getElementById('capsule-rz').value = radToDeg(e.z).toFixed(2);
}

function useCurrentVisualUpAsCapsuleAxis() {
  // sceneGroup.rotation maps raw COLMAP/world coordinates -> displayed coordinates.
  // We want the raw-world axis that appears as visual +Y after scene rotation:
  // raw_axis = inverse(scene_rotation) * visual_Y.
  sceneGroup.updateMatrixWorld(true);

  const qScene = sceneGroup.quaternion.clone();
  const qInv = qScene.invert();

  const rawAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(qInv).normalize();

  setCapsuleEulerFromAxis(rawAxis);
  refreshCapsulePreviewColors();
}

function refreshCapsulePreviewColors() {
  updateStageCapsule();

  if (
    frameData &&
    pointCloud &&
    document.getElementById('color-mode').value === 'capsule'
  ) {
    pointCloud.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(buildColors(frameData, 'capsule'), 3)
    );
    pointCloud.geometry.attributes.color.needsUpdate = true;
  }
}

// ── Detail sphere preview ────────────────────────────
function finiteNumberFromInput(id, fallback) {
  const value = parseFloat(document.getElementById(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function getDetailSphereParams() {
  const cx = finiteNumberFromInput('detail-sphere-cx', 0.0);
  const cy = finiteNumberFromInput('detail-sphere-cy', 0.0);
  const cz = finiteNumberFromInput('detail-sphere-cz', 0.0);
  const r = Math.max(0.001, finiteNumberFromInput('detail-sphere-radius', 0.4));
  const fraction = Math.max(
    0.01,
    Math.min(0.99, finiteNumberFromInput('detail-sphere-fraction', 0.25))
  );
  return { cx, cy, cz, r, fraction };
}

function pointInsideDetailSphere(px, py, pz, s) {
  const dx = px - s.cx;
  const dy = py - s.cy;
  const dz = pz - s.cz;
  return (dx*dx + dy*dy + dz*dz) <= s.r * s.r;
}

function updateDetailSphereArgsText() {
  const s = getDetailSphereParams();
  const txt =
    `--detail-sphere-center="${s.cx.toFixed(3)},${s.cy.toFixed(3)},${s.cz.toFixed(3)}" ` +
    `--detail-sphere-radius ${s.r.toFixed(3)} ` +
    `--detail-sphere-target-fraction ${s.fraction.toFixed(3)} ` +
    `--detail-sphere-camera-pairs ` +
    `--detail-sphere-crop-matching`;
  const out = document.getElementById('detail-sphere-args');
  if (out) out.value = txt;
  return txt;
}

function updateDetailSpherePointCount() {
  const out = document.getElementById('detail-sphere-count');
  if (!out) return;
  if (!frameData) {
    out.textContent = '0 / 0';
    return;
  }

  const pos = hexToFloat32(frameData.positions);
  const s = getDetailSphereParams();
  let inside = 0;
  for (let i = 0; i < frameData.n_points; i++) {
    if (pointInsideDetailSphere(pos[i*3], pos[i*3+1], pos[i*3+2], s)) inside++;
  }
  out.textContent = `${inside.toLocaleString()} / ${frameData.n_points.toLocaleString()}`;
}

function updateDetailSphere() {
  if (!sceneGroup) return;

  if (detailSphere) {
    sceneGroup.remove(detailSphere);
    disposeObject3D(detailSphere);
    detailSphere = null;
  }

  updateDetailSphereArgsText();
  updateDetailSpherePointCount();

  const show = document.getElementById('detail-sphere-show')?.checked;
  if (!show) return;

  const s = getDetailSphereParams();
  const geometry = new THREE.SphereGeometry(s.r, 32, 20);
  const material = new THREE.MeshBasicMaterial({
    color: 0xffd84d,
    wireframe: true,
    transparent: true,
    opacity: 0.8,
    depthTest: true,
  });
  detailSphere = new THREE.Mesh(geometry, material);
  detailSphere.name = 'detail-sphere-preview';
  detailSphere.position.set(s.cx, s.cy, s.cz);
  sceneGroup.add(detailSphere);
}

function refreshDetailSpherePreviewColors() {
  updateDetailSphere();

  if (
    frameData &&
    pointCloud &&
    document.getElementById('color-mode').value === 'detail_sphere'
  ) {
    pointCloud.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(buildColors(frameData, 'detail_sphere'), 3)
    );
    pointCloud.geometry.attributes.color.needsUpdate = true;
  }
}
// ── Color modes ───────────────────────────────────────
function heatmap(t) {
  return [Math.max(0,Math.min(1,2*t)), Math.max(0,Math.min(1,2*t<1?2*t:2-2*t)), Math.max(0,Math.min(1,1-2*t))];
}

function buildColors(fd, mode) {
  const n=fd.n_points, buf=new Float32Array(n*3);
  if (mode==='rgb') {
    buf.set(hexToFloat32(fd.colors));
  } else if (mode==='time') {
    const [r,g,b]=heatmap(fd.t_normalized);
    for (let i=0;i<n;i++){buf[i*3]=r;buf[i*3+1]=g;buf[i*3+2]=b;}
  } else if (mode==='velmag') {
    const vel=hexToFloat32(fd.velocities), gmax=fd.vel_max_global||1e-6;
    for (let i=0;i<n;i++){
      const vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
      const [r,g,b]=heatmap(Math.min(1,Math.sqrt(vx*vx+vy*vy+vz*vz)/gmax));
      buf[i*3]=r;buf[i*3+1]=g;buf[i*3+2]=b;
    }
    document.getElementById('vel-legend').classList.add('visible');
    document.getElementById('vel-legend-labels').style.display='flex';
    document.getElementById('vel-legend-max').textContent=gmax.toExponential(2);
  } else if (mode==='hasvel') {
    const hv=hexToUint8(fd.has_velocity);
    for (let i=0;i<n;i++){
      if(hv[i]){buf[i*3]=0.47;buf[i*3+1]=0.98;buf[i*3+2]=0.42;}
      else{buf[i*3]=0.8;buf[i*3+1]=0.2;buf[i*3+2]=0.1;}
    }
  } else if (mode==='split') {
    const st=hexToUint8(fd.is_static), prot=hexToUint8(fd.dynamic_protect);
    for (let i=0;i<n;i++){
      if (st[i]) {                 // static/persistent 3D pool: cyan-blue
        buf[i*3]=0.05; buf[i*3+1]=0.72; buf[i*3+2]=1.00;
      } else if (prot[i]) {        // center-protected dynamic: magenta
        buf[i*3]=1.00; buf[i*3+1]=0.10; buf[i*3+2]=0.95;
      } else {                     // normal dynamic 4D pool: orange
        buf[i*3]=1.00; buf[i*3+1]=0.48; buf[i*3+2]=0.10;
      }
    }
  } else if (mode==='capsule') {
    const pos = hexToFloat32(fd.positions);
    const c = getCapsuleParams();

    for (let i = 0; i < n; i++) {
      const x = pos[i*3];
      const y = pos[i*3 + 1];
      const z = pos[i*3 + 2];

      if (pointInsideStageCapsule(x, y, z, c)) {
        // Inside capsule: would be protected dynamic.
        buf[i*3] = 1.00;
        buf[i*3+1] = 0.10;
        buf[i*3+2] = 0.95;
      } else {
        // Outside capsule: dim cyan.
        buf[i*3] = 0.05;
        buf[i*3+1] = 0.45;
        buf[i*3+2] = 0.65;
      }
    }
  } else if (mode==='detail_sphere') {
    const pos = hexToFloat32(fd.positions);
    const s = getDetailSphereParams();

    for (let i = 0; i < n; i++) {
      const x = pos[i*3];
      const y = pos[i*3 + 1];
      const z = pos[i*3 + 2];

      if (pointInsideDetailSphere(x, y, z, s)) {
        // Inside the detail budget: bright yellow/orange.
        buf[i*3] = 1.00;
        buf[i*3+1] = 0.72;
        buf[i*3+2] = 0.08;
      } else {
        // Outside the reserved detail region: dim blue.
        buf[i*3] = 0.04;
        buf[i*3+1] = 0.25;
        buf[i*3+2] = 0.42;
      }
    }
  } else if (mode==='protected') {
    const prot=hexToUint8(fd.dynamic_protect), st=hexToUint8(fd.is_static);
    for (let i=0;i<n;i++){
      if (prot[i]) {               // protected dynamic rows
        buf[i*3]=1.00; buf[i*3+1]=0.10; buf[i*3+2]=0.95;
      } else if (st[i]) {          // static rows dim cyan
        buf[i*3]=0.05; buf[i*3+1]=0.30; buf[i*3+2]=0.42;
      } else {                     // other dynamic rows dim grey/orange
        buf[i*3]=0.28; buf[i*3+1]=0.18; buf[i*3+2]=0.10;
      }
    }
  } else if (mode==='dynamic_allowed') {
    const da = hexToUint8(fd.dynamic_allowed);
    const st = hexToUint8(fd.is_static);

    for (let i=0;i<n;i++){
      if (da[i]) {
        // inside dynamic-allow capsule
        buf[i*3]=1.00; buf[i*3+1]=0.10; buf[i*3+2]=0.95;
      } else if (st[i]) {
        // outside dynamic-allow, forced static
        buf[i*3]=0.05; buf[i*3+1]=0.65; buf[i*3+2]=1.00;
      } else {
        // should be rare: outside allow but still dynamic
        buf[i*3]=1.00; buf[i*3+1]=0.00; buf[i*3+2]=0.00;
      }
    }

  } else if (mode==='stage_keep') {
    const sk = hexToUint8(fd.stage_keep);

    for (let i=0;i<n;i++){
      if (sk[i]) {
        buf[i*3]=0.20; buf[i*3+1]=1.00; buf[i*3+2]=0.20;
      } else {
        buf[i*3]=1.00; buf[i*3+1]=0.00; buf[i*3+2]=0.00;
      }
    }

  } else if (mode==='roi_split') {
    const prot = hexToUint8(fd.dynamic_protect);
    const da = hexToUint8(fd.dynamic_allowed);
    const st = hexToUint8(fd.is_static);

    for (let i=0;i<n;i++){
      if (prot[i]) {
        // protected actor/guitar: hot magenta
        buf[i*3]=1.00; buf[i*3+1]=0.05; buf[i*3+2]=1.00;
      } else if (da[i] && !st[i]) {
        // allowed dynamic but not protected: orange
        buf[i*3]=1.00; buf[i*3+1]=0.45; buf[i*3+2]=0.05;
      } else if (!da[i] && st[i]) {
        // outside dynamic allow, forced static: cyan
        buf[i*3]=0.05; buf[i*3+1]=0.75; buf[i*3+2]=1.00;
      } else if (st[i]) {
        // normal static
        buf[i*3]=0.10; buf[i*3+1]=0.35; buf[i*3+2]=0.75;
      } else {
        // suspicious row
        buf[i*3]=1.00; buf[i*3+1]=0.00; buf[i*3+2]=0.00;
      }
    }
  } else if (mode === 'coherence') {
    const coh = hexToFloat32(fd.coherence);
    const hv  = hexToUint8(fd.has_velocity);
    for (let i = 0; i < n; i++) {
      if (!hv[i]) {
        // No velocity — dim grey so they don't dominate
        buf[i*3] = 0.12; buf[i*3+1] = 0.12; buf[i*3+2] = 0.14;
      } else {
        // coh in [-1, 1] → c in [0, 1]: green = coherent, red = incoherent
        const c = (coh[i] + 1) * 0.5;
        buf[i*3]   = 1.0 - c;   // red high when incoherent
        buf[i*3+1] = c;          // green high when coherent
        buf[i*3+2] = 0.15;
      }
    }
  }
  if (mode!=='velmag'){
    document.getElementById('vel-legend').classList.remove('visible');
    document.getElementById('vel-legend-labels').style.display='none';
  }
  return buf;
}

// renderFrame is an alias for displayFrame (used by color-mode/vel toggles)

function renderVelocity(fd) {
  const scale=parseFloat(document.getElementById('vel-scale').value);
  const maxA =parseInt(document.getElementById('vel-max-arrows').value);
  const vOnly=document.getElementById('vel-valid-only').checked;
  const n=fd.n_points;
  const pos=hexToFloat32(fd.positions), vel=hexToFloat32(fd.velocities), hv=hexToUint8(fd.has_velocity);

  let el=[];
  for (let i=0;i<n;i++){
    if(vOnly&&!hv[i]) continue;
    const vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
    if(Math.sqrt(vx*vx+vy*vy+vz*vz)<1e-9) continue;
    el.push(i);
  }
  if(el.length>maxA){const s=Math.floor(el.length/maxA);el=el.filter((_,j)=>j%s===0).slice(0,maxA);}

  const lv=new Float32Array(el.length*6), lc=new Float32Array(el.length*6);
  for(let j=0;j<el.length;j++){
    const i=el[j];
    const sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
    const vx=vel[i*3]*scale,vy=vel[i*3+1]*scale,vz=vel[i*3+2]*scale;
    lv[j*6]=sx;   lv[j*6+1]=sy;   lv[j*6+2]=sz;
    lv[j*6+3]=sx+vx;lv[j*6+4]=sy+vy;lv[j*6+5]=sz+vz;
    lc[j*6]=0;lc[j*6+1]=0.9;lc[j*6+2]=1;
    lc[j*6+3]=1;lc[j*6+4]=0.42;lc[j*6+5]=0.21;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(lv,3));
  geo.setAttribute('color',   new THREE.BufferAttribute(lc,3));
  velLines=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({vertexColors:true}));
  sceneGroup.add(velLines);
}

// ── Frame cache & prefetch ────────────────────────────
// All frames are fetched from the server once and stored in frameCache[].
// Playback then just calls displayFrame(idx) with no network calls,
// which is fast enough for smooth animation.
const frameCache = {};          // idx → parsed frame data object
let prefetchDone = false;
let prefetchTotal = 0;
let prefetchLoaded = 0;

// Show/hide the loading overlay with optional message
function showLoading(msg) {
  const el = document.getElementById('loading');
  el.classList.remove('hidden');
  const txt = el.querySelector('.loading-msg');
  if (txt && msg) txt.textContent = msg;
}
function hideLoading() { document.getElementById('loading').classList.add('hidden'); }

// Update the prefetch progress bar
function updatePrefetchBar(loaded, total) {
  const bar = document.getElementById('prefetch-bar');
  const pct = document.getElementById('prefetch-pct');
  if (!bar) return;
  const p = total > 0 ? Math.round(100 * loaded / total) : 0;
  bar.style.width = p + '%';
  if (pct) pct.textContent = `${loaded} / ${total} frames cached`;
}

// Fetch a single frame and store in cache
async function fetchFrame(idx) {
  if (frameCache[idx]) return frameCache[idx];
  const fd = await (await fetch(`/api/frame/${idx}`)).json();
  frameCache[idx] = fd;
  return fd;
}

// Background prefetch of all frames after the first one is shown.
// Fetches sequentially so we don't hammer the server.
async function prefetchAll(nFrames) {
  prefetchTotal = nFrames;
  prefetchLoaded = Object.keys(frameCache).length;
  updatePrefetchBar(prefetchLoaded, prefetchTotal);

  for (let i = 0; i < nFrames; i++) {
    if (!frameCache[i]) {
      try { await fetchFrame(i); } catch(e) { console.warn('prefetch err frame', i, e); }
    }
    prefetchLoaded = Object.keys(frameCache).length;
    updatePrefetchBar(prefetchLoaded, prefetchTotal);
  }
  prefetchDone = true;
  // Hide the prefetch bar once done
  const row = document.getElementById('prefetch-row');
  if (row) row.style.opacity = '0';
}

// Display a frame that is already in the cache (synchronous, no fetch)
function displayFrame(idx) {
  const fd = frameCache[idx];
  if (!fd) return;           // not cached yet — skip during playback
  frameData = fd;
  currentFrame = idx;

  const mode = document.getElementById('color-mode').value;
  if (pointCloud) { sceneGroup.remove(pointCloud); pointCloud.geometry.dispose(); pointCloud=null; }
  if (velLines)   { sceneGroup.remove(velLines);   velLines.geometry.dispose();   velLines=null; }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(hexToFloat32(fd.positions), 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(buildColors(fd, mode), 3));
  pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({
    size: parseFloat(document.getElementById('pt-size').value) * 0.003,
    vertexColors: true, sizeAttenuation: true,
  }));
  sceneGroup.add(pointCloud);
  updateStageCapsule();
  updateDetailSphere();
  if (document.getElementById('show-vel').checked) renderVelocity(fd);

  document.getElementById('f-npts').textContent  = fd.n_points.toLocaleString();
  document.getElementById('f-vvel').textContent  = fd.n_valid_velocity.toLocaleString();
  document.getElementById('f-tnorm').textContent = fd.t_normalized.toFixed(4);
  document.getElementById('f-vmean').textContent = fd.vel_mean.toExponential(2);
  document.getElementById('f-vmax').textContent  = fd.vel_max.toExponential(2);
  document.getElementById('f-static').textContent = (fd.n_static ?? 0).toLocaleString();
  document.getElementById('f-dynamic').textContent = (fd.n_dynamic ?? 0).toLocaleString();
  document.getElementById('f-protected').textContent = (fd.n_dynamic_protect ?? 0).toLocaleString();
  document.getElementById('frame-display').textContent = `Frame ${fd.frame_index+1} / ${fd.n_frames}`;
  document.getElementById('t-display').textContent     = `t=${fd.t_normalized.toFixed(3)}`;
  document.getElementById('scrubber').value = fd.frame_index;
}

// Load a frame: use cache if available, otherwise fetch then display
async function loadFrame(idx) {
  if (frameCache[idx]) {
    displayFrame(idx);
    return;
  }
  showLoading('LOADING FRAME...');
  try {
    await fetchFrame(idx);
    displayFrame(idx);
  } catch(e) { console.error(e); }
  hideLoading();
}

async function loadSummary() {
  summary = await (await fetch('/api/summary')).json();
  document.getElementById('npz-label').textContent = summary.npz_path;
  document.getElementById('s-nframes').textContent = summary.n_frames;
  document.getElementById('s-npts').textContent    = (summary.n_points_total/1e6).toFixed(2)+'M';
  document.getElementById('s-seqlen').textContent  = summary.total_frames;
  document.getElementById('s-kfstep').textContent  = summary.keyframe_step;
  document.getElementById('s-static').textContent = (summary.n_static_total ?? 0).toLocaleString();
  document.getElementById('s-dynamic').textContent = (summary.n_dynamic_total ?? summary.n_points_total).toLocaleString();
  document.getElementById('scrubber').max = summary.n_frames-1;
  const sc=summary.scene_scale, cx=summary.scene_center;

  // Keep the endpoint defaults unless manually edited.
  // The capsule is arbitrary A→B now, not forced to scene-center X/Z.
  // Initialize capsule center to scene median. Rotation/length/radius remain user-controlled.
  document.getElementById('capsule-cx').value = Number(cx[0]).toFixed(3);
  document.getElementById('capsule-cy').value = Number(cx[1]).toFixed(3);
  document.getElementById('capsule-cz').value = Number(cx[2]).toFixed(3);
  updateCapsuleArgsText();

  document.getElementById('detail-sphere-cx').value = Number(cx[0]).toFixed(3);
  document.getElementById('detail-sphere-cy').value = Number(cx[1]).toFixed(3);
  document.getElementById('detail-sphere-cz').value = Number(cx[2]).toFixed(3);
  updateDetailSphereArgsText();

  resetCamera(cx[0], cx[1]+sc*0.5, cx[2]+sc*2, cx[0], cx[1], cx[2]);
}

// ── Render loop ───────────────────────────────────────
function animate(){ requestAnimationFrame(animate); renderer.render(scene,camera); }
function onResize(){
  const w=document.getElementById('canvas-wrap');
  camera.aspect=w.clientWidth/w.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(w.clientWidth,w.clientHeight);
}

// ── Playback — rAF-based so it never outruns the renderer ────────────────────
// Uses requestAnimationFrame with a manual time accumulator rather than
// setInterval + fetch, so each tick is a synchronous displayFrame() call.
let playLastTime = 0;

function playTick(now) {
  if (!isPlaying) return;
  const fps = parseInt(document.getElementById('play-speed').value);
  const interval = 1000 / fps;
  if (now - playLastTime >= interval) {
    const next = (currentFrame + 1) % (summary ? summary.n_frames : 1);
    // Only advance if the next frame is already cached; otherwise wait
    if (frameCache[next]) {
      displayFrame(next);
      playLastTime = now;
    } else {
      // Frame not cached yet — stall and try again next tick
      playLastTime = now;
    }
  }
  requestAnimationFrame(playTick);
}

function startPlay(){
  if(isPlaying) return;
  isPlaying=true;
  playLastTime = performance.now();
  document.getElementById('btn-play').textContent='⏸ Pause';
  document.getElementById('sb-play').textContent='⏸';
  requestAnimationFrame(playTick);
}
function stopPlay(){
  isPlaying=false;
  document.getElementById('btn-play').textContent='▶ Play';
  document.getElementById('sb-play').textContent='▶';
}
function togglePlay(){ isPlaying?stopPlay():startPlay(); }

// ── UI wiring ─────────────────────────────────────────
document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('sb-play').addEventListener('click', togglePlay);
document.getElementById('btn-prev').addEventListener('click',()=>{ stopPlay(); loadFrame(Math.max(0,currentFrame-1)); });
document.getElementById('btn-next').addEventListener('click',()=>{ stopPlay(); loadFrame(Math.min((summary?.n_frames??1)-1,currentFrame+1)); });
document.getElementById('sb-prev').addEventListener('click', ()=>{ stopPlay(); loadFrame(Math.max(0,currentFrame-1)); });
document.getElementById('sb-next').addEventListener('click', ()=>{ stopPlay(); loadFrame(Math.min((summary?.n_frames??1)-1,currentFrame+1)); });
document.getElementById('scrubber').addEventListener('input',e=>{ stopPlay(); loadFrame(parseInt(e.target.value)); });

document.getElementById('color-mode').addEventListener('change',   ()=>{ if(frameData) displayFrame(currentFrame); });
document.getElementById('show-vel').addEventListener('change',     ()=>{ if(frameData) displayFrame(currentFrame); });
document.getElementById('vel-valid-only').addEventListener('change',()=>{ if(frameData) displayFrame(currentFrame); });
document.getElementById('vel-scale').addEventListener('input',e=>{
  document.getElementById('vel-scale-val').textContent=e.target.value+'×';
  if(frameData&&document.getElementById('show-vel').checked) displayFrame(currentFrame);
});
document.getElementById('vel-max-arrows').addEventListener('input',e=>{
  const v=parseInt(e.target.value);
  document.getElementById('vel-max-val').textContent=v>=1000?(v/1000).toFixed(1)+'k':v;
  if(frameData&&document.getElementById('show-vel').checked) displayFrame(currentFrame);
});
document.getElementById('pt-size').addEventListener('input',e=>{
  document.getElementById('pt-size-val').textContent=e.target.value;
  if(pointCloud) pointCloud.material.size=parseFloat(e.target.value)*0.003;
});
// Stage capsule controls
[
  'capsule-target',
  'capsule-show',
  'capsule-cx',
  'capsule-cy',
  'capsule-cz',
  'capsule-length',
  'capsule-radius',
  'capsule-rx',
  'capsule-ry',
  'capsule-rz',
].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('input', refreshCapsulePreviewColors);
  el.addEventListener('change', refreshCapsulePreviewColors);
});

document.getElementById('capsule-copy-args').addEventListener('click', async () => {
  const txt = updateCapsuleArgsText();
  try {
    await navigator.clipboard.writeText(txt);
  } catch (e) {
    const out = document.getElementById('capsule-args');
    out.focus();
    out.select();
  }
});

document.getElementById('capsule-use-scene-center').addEventListener('click', () => {
  if (!summary) return;

  document.getElementById('capsule-cx').value = Number(summary.scene_center[0]).toFixed(3);
  document.getElementById('capsule-cy').value = Number(summary.scene_center[1]).toFixed(3);
  document.getElementById('capsule-cz').value = Number(summary.scene_center[2]).toFixed(3);

  refreshCapsulePreviewColors();
});

document.getElementById('capsule-use-scene-up').addEventListener('click', () => {
  useCurrentVisualUpAsCapsuleAxis();
});

// Detail sphere controls
[
  'detail-sphere-show',
  'detail-sphere-cx',
  'detail-sphere-cy',
  'detail-sphere-cz',
  'detail-sphere-radius',
  'detail-sphere-fraction',
].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('input', refreshDetailSpherePreviewColors);
  el.addEventListener('change', refreshDetailSpherePreviewColors);
});

document.getElementById('detail-sphere-copy-args').addEventListener('click', async () => {
  const txt = updateDetailSphereArgsText();
  try {
    await navigator.clipboard.writeText(txt);
  } catch (e) {
    const out = document.getElementById('detail-sphere-args');
    out.focus();
    out.select();
  }
});

document.getElementById('detail-sphere-use-scene-center').addEventListener('click', () => {
  if (!summary) return;
  document.getElementById('detail-sphere-cx').value = Number(summary.scene_center[0]).toFixed(3);
  document.getElementById('detail-sphere-cy').value = Number(summary.scene_center[1]).toFixed(3);
  document.getElementById('detail-sphere-cz').value = Number(summary.scene_center[2]).toFixed(3);
  refreshDetailSpherePreviewColors();
});

function useCapsuleEndpointForDetailSphere(which) {
  const c = getCapsuleParams();
  const prefix = which === 'b' ? 'b' : 'a';
  document.getElementById('detail-sphere-cx').value = Number(c[`${prefix}x`]).toFixed(3);
  document.getElementById('detail-sphere-cy').value = Number(c[`${prefix}y`]).toFixed(3);
  document.getElementById('detail-sphere-cz').value = Number(c[`${prefix}z`]).toFixed(3);
  refreshDetailSpherePreviewColors();
}

document.getElementById('detail-sphere-use-capsule-a').addEventListener('click', () => {
  useCapsuleEndpointForDetailSphere('a');
});
document.getElementById('detail-sphere-use-capsule-b').addEventListener('click', () => {
  useCapsuleEndpointForDetailSphere('b');
});

document.getElementById('play-speed').addEventListener('input',e=>{
  document.getElementById('play-speed-val').textContent=e.target.value;
  if(isPlaying){stopPlay();startPlay();}
});

// Camera presets
document.getElementById('btn-reset-cam').addEventListener('click',()=>{
  if(!summary) return;
  const sc=summary.scene_scale,cx=summary.scene_center;
  resetCamera(cx[0],cx[1]+sc*0.5,cx[2]+sc*2,cx[0],cx[1],cx[2]);
});
document.getElementById('btn-top-cam').addEventListener('click',()=>{
  if(!summary) return;
  const sc=summary.scene_scale,cx=summary.scene_center;
  resetCamera(cx[0],cx[1]+sc*3,cx[2]+0.001,cx[0],cx[1],cx[2]);
});
document.getElementById('btn-front-cam').addEventListener('click',()=>{
  if(!summary) return;
  const sc=summary.scene_scale,cx=summary.scene_center;
  resetCamera(cx[0],cx[1],cx[2]+sc*3,cx[0],cx[1],cx[2]);
});

// Background color
function setBackground(hex){
  scene.background=new THREE.Color(hex);
  document.getElementById('bg-color-picker').value=hex;
  document.querySelectorAll('.bg-swatch').forEach(s=>s.classList.toggle('active',s.dataset.color===hex));
}
document.getElementById('bg-color-picker').addEventListener('input',e=>{
  scene.background=new THREE.Color(e.target.value);
  document.querySelectorAll('.bg-swatch').forEach(s=>s.classList.remove('active'));
});
document.querySelectorAll('.bg-swatch').forEach(s=>s.addEventListener('click',()=>setBackground(s.dataset.color)));

// Scene rotation sliders
['x','y','z'].forEach(a=>{
  document.getElementById(`rot-${a}`).addEventListener('input',e=>{
    sceneRotDeg[a]=parseInt(e.target.value);
    document.getElementById(`rot-${a}-val`).textContent=sceneRotDeg[a]+'°';
    applySceneRotation();
  });
});

// Reset Up — reads camera's current screen-up and locks it as orbit pole
document.getElementById('btn-reset-up').addEventListener('click', () => {
  window.resetOrbitUp();
});
document.getElementById('btn-rot-reset').addEventListener('click', () => {
  // Reset scene rotation sliders to zero
  sceneRotDeg = { x: 0, y: 0, z: 0 };
  ['x','y','z'].forEach(a => {
    document.getElementById(`rot-${a}`).value = 0;
    document.getElementById(`rot-${a}-val`).textContent = '0°';
  });
  applySceneRotation();
});

// Keyboard
window.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){stopPlay();loadFrame(Math.min((summary?.n_frames??1)-1,currentFrame+1));}
  if(e.key==='ArrowLeft') {stopPlay();loadFrame(Math.max(0,currentFrame-1));}
  if(e.key===' ')         {e.preventDefault();togglePlay();}
});

// Boot
(async()=>{
  initThree();
  showLoading('LOADING...');
  await loadSummary();
  await loadFrame(0);
  hideLoading();
  // Kick off background prefetch of all frames
  prefetchAll(summary.n_frames);
})();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress default access log spam
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode())

        elif self.path == "/api/summary":
            data = json.dumps(STORE.get_summary()).encode()
            self._send(200, "application/json", data)

        elif self.path.startswith("/api/frame/"):
            try:
                idx = int(self.path.split("/")[-1])
                fd = STORE.get_frame_data(idx)
                data = json.dumps(fd).encode()
                self._send(200, "application/json", data)
            except Exception as e:
                self._send(500, "text/plain", str(e).encode())

        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Web-based point cloud viewer for FreeTimeGsVanilla keyframe NPZ files"
    )
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to keyframe NPZ file"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="HTTP port (default: 8765)"
    )
    parser.add_argument(
        "--max-points", type=int, default=1400000,
        help="Max points per frame to send to browser (default: %(default)s)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open browser"
    )
    args = parser.parse_args()

    global STORE
    STORE = NPZFrameStore(args.npz, max_points_per_frame=args.max_points)

    url = f"http://localhost:{args.port}"
    print(f"\n{'='*60}")
    print(f"  NPZ Viewer running at {url}")
    print(f"  NPZ: {args.npz}")
    print(f"  Frames: {STORE.n_frames}  |  Max pts/frame: {args.max_points:,}")
    print(f"  Keyboard: ← → to step frames, SPACE to play/pause")
    print(f"{'='*60}\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    server = http.server.HTTPServer(("", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
