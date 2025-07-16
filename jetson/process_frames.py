#!/usr/bin/env python3
"""
process_frames.py

Load and process thermal frames for a given camera run,
with optional saving of debug plots to a specified directory.

Functions:
 - load_post_frames: safely list, timestamp-read, and load next-N .npy frames after a trigger frame
 - process_frame_array: run Hough-based analysis on a NumPy thermal array, returning temperatures
 - process_frames: orchestrate load + process for a camera serial, optionally saving debug plots
"""
import os
import glob
import csv
import time
import logging
import functools
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from filelock import FileLock

# Processor primitives (import from your processor module)
from jetson.utils.processor import (
    HoughBundler,
    cluster_by_angle,
    sort_line_cluster,
    offset_line,
    mean_between_lines,
)

from jetson.utils.frame_utils import (

    catch_errors,
    load_post_frames
)


PLOT_ENABLED  = True
PLOT_SAVE_DIR = None

def set_plotting(on: bool):
    global PLOT_ENABLED
    PLOT_ENABLED = on

def set_plot_dir(path: str):
    global PLOT_SAVE_DIR
    PLOT_SAVE_DIR = path
    logger.debug(f"Plot save directory set to: {PLOT_SAVE_DIR}")



# ─── Logger Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger('process_frames')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─── Configuration ────────────────────────────────────────────────────────────
PLOT_ENABLED = True
PLOT_SAVE_DIR = None

# ─── Error‐handling decorator ──────────────────────────────────────────────────
SKIPPED_FRAMES = 0

# ─── Core processing ───────────────────────────────────────────────────────────
@catch_errors(default=(np.nan, np.nan))
def process_frame_array(
    thermal: np.ndarray,
    frame_id: int = None,
    serial: str = None,
    N_offset: int = 6
) -> Tuple[float, float]:
    """
    Run Hough-based analysis on a raw thermal array.
    Returns (prev_layer_temp, recent_layer_temp) in °C.
    Optionally saves a debug plot if PLOT_ENABLED and PLOT_SAVE_DIR are set.
    """
    # Normalize & convert
    norm = cv2.normalize(thermal, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    thermal_conv = (thermal / 100.0) - 273.15
    bundler = HoughBundler(min_distance=2, min_angle=5)

    # 1) Detect lines
    edges = cv2.Canny(norm, 80, 150, apertureSize=3)
    if edges.sum() == 0:
        raise RuntimeError("No edges found")
    h_full = bundler.process_lines(
        cv2.HoughLinesP(edges, 1, np.pi/180,
                        threshold=10, minLineLength=20, maxLineGap=5)
    )
    vertical_lines, prev_bead = [], []
    if h_full is not None:
        for x1,y1,x2,y2 in h_full[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 80 < ang < 100:
                vertical_lines.append(((x1,y1),(x2,y2)))
            if 40 < ang < 50:
                prev_bead.append(((x1,y1),(x2,y2)))

    # 2) Estimate nozzle position
    xs = [pt[0] for seg in vertical_lines for pt in seg]
    ys = [pt[1] for seg in vertical_lines for pt in seg]
    nx = min(xs) if xs else norm.shape[1]//2
    ny = max(ys) if ys else norm.shape[0]//2

    # 3) Crop & detect prev-layer
    edges_crop = edges[:, :nx]
    if edges_crop.sum() == 0:
        raise RuntimeError("No edges in cropped region")
    h_crop = bundler.process_lines(
        cv2.HoughLinesP(edges_crop, 1, np.pi/180,
                        threshold=10, minLineLength=20, maxLineGap=5)
    )
    segments = h_crop[:,0] if h_crop is not None else np.empty((0,4), int)
    clusters = cluster_by_angle(segments)
    if not clusters:
        return np.nan, np.nan
    x1,y1,x2,y2 = sort_line_cluster(clusters)
    p1o,p2o = offset_line((x1,y1),(x2,y2),1)
    p3o,p4o = offset_line(p1o,p2o,N_offset)
    prev_temp = mean_between_lines(thermal_conv, p1o,p2o,p3o,p4o)

    # 4) Detect recent-layer
    seg_depo = np.array([[x1,y1,x2,y2] for ((x1,y1),(x2,y2)) in prev_bead], dtype=int)
    seg_depo = seg_depo[seg_depo[:,1] > ny]
    clusters2 = cluster_by_angle(seg_depo)
    if not clusters2:
        recent_temp = np.nan
    else:
        all_segs = [s for cl in clusters2 for s in cl['segments']]
        x1,y1,x2,y2 = max(all_segs, key=lambda s: max(s[1], s[3]))
        q1o,q2o = offset_line((x1,y1),(x2,y2),1)
        q3o,q4o = offset_line(q1o,q2o,N_offset)
        recent_temp = mean_between_lines(thermal_conv, q1o,q2o,q3o,q4o)
    
    


    # 5) Optional debug plot save
    if PLOT_ENABLED and PLOT_SAVE_DIR:
        fig = plt.figure(figsize=(6,6))
        vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, p1o,p2o, (0,255,0), 1)
        cv2.line(vis, p3o,p4o, (0,255,0), 1)
        if not np.isnan(recent_temp):
            cv2.line(vis, q1o,q2o, (0,0,255), 1)
            cv2.line(vis, q3o,q4o, (0,0,255), 1)
        plt.imshow(vis[..., ::-1], cmap='inferno')
        plt.axis('off'); plt.tight_layout()
        os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
        fname = f"frame_{serial or 'unknown'}_{frame_id or datetime.now().strftime('%H%M%S%f')}.png"
        fig.savefig(os.path.join(PLOT_SAVE_DIR, fname))
        plt.close(fig)
     

    return prev_temp, recent_temp

# ─── Batch processing ─────────────────────────────────────────────────────────
def process_frames(
    base_dir: str,
    serial: str,
    trigger_frame: int,
    N: int = 10,
    N_offset: int = 6,
    plot_dir: str = None
) -> Tuple[List[float], List[float]]:
    """
    Load and process the next N frames after trigger_frame.
    Optionally save debug plots to plot_dir.
    Returns lists of prev_layer_temps and recent_layer_temps.
    """
    # Determine plot directory
    if plot_dir:
        set_plotting(True)
        set_plot_dir(plot_dir)
    else:
        default_dir = os.path.join(base_dir, serial, 'plots', str(trigger_frame))
        set_plotting(True)
        set_plot_dir(default_dir)

    frames_info = load_post_frames(base_dir, serial, trigger_frame, N)
    prev_temps, recent_temps = [], []
    logger.info(f"Processing {len(frames_info)} frames for serial {serial}")
    for info in frames_info:
        prev, recent = process_frame_array(
            info['array'], frame_id=info['frame_id'], serial=serial, N_offset=N_offset
        )
        prev_temps.append(prev)
    
        recent_temps.append(recent)
    
    logger.info(f"Finished processing frames. Prev temps: {np.nanmean(prev_temps)}, Recent temps: {np.nanmean(recent_temps)}")
    os.makedirs(f'{PLOT_SAVE_DIR}/temp_arrays', exist_ok=True)
    if prev_temps:
        
        np.save(os.path.join(f'{PLOT_SAVE_DIR}/temp_arrays', f"{trigger_frame}_prev_region.npy"), prev_temps)
    if recent_temps:
        np.save(os.path.join(f'{PLOT_SAVE_DIR}/temp_arrays', f"{trigger_frame}_recent_region.npy"), recent_temps)
    
    return prev_temps, recent_temps
