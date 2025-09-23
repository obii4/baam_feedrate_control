#!/usr/bin/env python3
"""
process_frames.py - Thermal frame processing for BAAM layer temperature extraction.

This module loads and analyzes thermal camera frames to extract temperature data
from printed layers. It supports two processing modes:
1. Normal: Hough transform-based line detection for automatic layer identification
2. Simple: Direct pixel sampling at predefined coordinates for debugging

The module identifies two regions of interest:
- Previous layer: The just-completed layer (primary control signal)
- Recent layer: The currently printing layer 

Functions:
    load_post_frames: Safely load N frames after a trigger timestamp
    process_frame_array: Hough-based temperature extraction from thermal array
    process_frame_array_simple: Pixel-based temperature extraction
    process_frames: Orchestrate batch processing with optional visualization

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0

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

# ─── PIXEL COORDINATES FOR SIMPLE MODE ─────────────────────────────────────────
# Manually calibrated pixel coordinates for temperature sampling
# These correspond to known layer positions in the camera's field of view
# Used for debugging, will change, determine if pixels are correct if used again.

PIXEL_COORDINATES = {
    'prev_layer_pixels': [
        (46, 73),
        (39, 77),
        (29, 82),
        (24, 83),
    ],
    'recent_layer_pixels': [
        (120, 73),
        (115, 78),
        (110, 83),
        (104, 88),
        (190, 240),
    ]
}

PLOT_ENABLED  = False
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
PLOT_ENABLED = False
PLOT_SAVE_DIR = None

# ─── Error‐handling decorator ──────────────────────────────────────────────────
SKIPPED_FRAMES = 0

# ─── Core processing ───────────────────────────────────────────────────────────
@catch_errors(default=(np.nan, np.nan))
def process_frame_array_simple(
    thermal: np.ndarray,
    frame_id: int = None,
    serial: str = None,
    N_offset: int = 6,
    simple: bool = False  # EXISTING PARAMETER
) -> Tuple[float, float]:
    """
    Extract layer temperatures using either Hough transform or pixel sampling.
    
    Processes a single thermal frame to extract temperature measurements from
    the previous (completed) layer and recent (currently printing) layer.
    
    Processing modes:
        - Normal (simple=False): Uses Hough line detection to automatically
          identify layer boundaries and sample between detected lines
        - Simple (simple=True): Samples predefined pixel coordinates for
          faster processing when camera position is fixed
    
    Args:
        thermal: Raw thermal array from FLIR camera (in Kelvin * 100)
        frame_id: Frame number for identification in plots
        serial: Camera serial number for multi-camera systems
        N_offset: Pixel offset for sampling region width (normal mode only)
        simple: If True, use pixel coordinates; if False, use Hough detection
        
    Returns:
        Tuple of (prev_layer_temp, recent_layer_temp) in Celsius
        Returns (np.nan, np.nan) if processing fails
        
    Note:
        Raw thermal values are in format: Kelvin * 100 (e.g., 29315 = 20°C)
        Conversion: Celsius = (raw / 100.0) - 273.15
    """
    # Convert to Celsius
    thermal_conv = (thermal / 100.0) - 273.15
    
    # ─── SIMPLE MODE: Use pixel coordinates ─────────────────────────────────
    if simple:
        w, h = thermal.shape
        
        # Get pixel coordinates (absolute)
        prev_pixels = PIXEL_COORDINATES['prev_layer_pixels']
        recent_pixels = PIXEL_COORDINATES['recent_layer_pixels']
        
        # Ensure coordinates are within bounds and sample temperatures
        prev_temps = []
        for x, y in prev_pixels:
            if 0 <= x < w and 0 <= y < h:
                prev_temps.append(thermal_conv[y, x])
        
        recent_temps = []
        for x, y in recent_pixels:
            if 0 <= x < w and 0 <= y < h:
                recent_temps.append(thermal_conv[x, y])
        
        # Use mean temperature
        prev_temp = np.mean(prev_temps) if prev_temps else np.nan
        recent_temp = np.mean(recent_temps) if recent_temps else np.nan
        
        # Optional: Save debug plot in simple mode
        if PLOT_ENABLED and PLOT_SAVE_DIR:
            fig = plt.figure(figsize=(6, 6))
            
            # Normalize for visualization
            norm = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            
            # Draw circles at pixel locations
            for x, y in prev_pixels:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)  # Green circles
            for x, y in recent_pixels:
                cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)  # Red circles
            
            # Add text labels
            cv2.putText(vis, f"Prev: {prev_temp:.1f}C", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis, f"Recent: {recent_temp:.1f}C", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            plt.imshow(vis[..., ::-1], cmap='inferno')
            plt.title(f"Simple Mode - Frame {frame_id}")
            plt.axis('off')
            plt.tight_layout()
            
            os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
            fname = f"frame_simple_{serial or 'unknown'}_{frame_id or datetime.now().strftime('%H%M%S%f')}.png"
            fig.savefig(os.path.join(PLOT_SAVE_DIR, fname))
            plt.close(fig)
        
        return prev_temp, recent_temp
    
    # ─── NORMAL MODE:  ─────────────────────

    norm = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bundler = HoughBundler(min_distance=2, min_angle=5)

    # 1) Detect lines

    # Edge detection parameters
    # These are tuned for FLIR thermal images with typical contrast
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
            if 80 < ang < 100: # Near-vertical lines indicate nozzle
                vertical_lines.append(((x1,y1),(x2,y2)))
            if 40 < ang < 50:  # Diagonal lines indicate fresh deposition
                prev_bead.append(((x1,y1),(x2,y2)))

    # 2) Estimate nozzle position
    xs = [pt[0] for seg in vertical_lines for pt in seg]
    ys = [pt[1] for seg in vertical_lines for pt in seg]
    
    # Nozzle position estimation
    # Use extrema of vertical lines to locate print head
    nx = min(xs) if xs else norm.shape[1]//2
    ny = max(ys) if ys else norm.shape[0]//2

    # 3) Crop & detect prev-layer

    # Region cropping for previous layer
    # Only analyze region to the left of nozzle (already printed)
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
    
    
    # Temperature sampling between offset lines
    # Create parallel lines N_offset pixels apart
    # Sample all pixels between lines and average
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

@catch_errors(default=(np.nan, np.nan))  
def process_frame_array(
    thermal: np.ndarray,
    frame_id: int = None,
    serial: str = None,
    N_offset: int = 6
) -> Tuple[float, float]:
    """
    Extract layer temperatures using Hough transform line detection.
    
    Implements computer vision pipeline to automatically identify printed
    layers and extract temperature measurements:
    
    Pipeline steps:
        1. Edge detection (Canny) to find layer boundaries
        2. Hough transform to detect lines at specific angles
        3. Clustering to group related line segments
        4. Temperature sampling between parallel offset lines
        
    Line angle classifications:
        - Vertical: Nozzle/extruder position
        - Previous bead: Recently deposited material
        - Horizontal: Layer boundaries
        
    Args:
        thermal: Raw thermal array in Kelvin * 100
        frame_id: Frame identifier for debugging
        serial: Camera serial number
        N_offset: Width of sampling region in pixels
        
    Returns:
        Tuple of (prev_layer_temp, recent_layer_temp) in Celsius
        
    Raises:
        RuntimeError: If no edges detected (indicates poor image quality)
        
    Note:
        Requires good thermal contrast for edge detection.
        May fail on uniform temperature fields or noisy data.
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
    plot_dir: str = None,
    simple: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Batch process thermal frames for temperature extraction.
    
    Main entry point for processing a sequence of frames after a layer
    change event. Loads frames, extracts temperatures, and optionally
    generates visualization plots.
    
    Args:
        base_dir: Root directory containing camera data (/mnt/external/run_XXX)
        serial: Camera serial number (subdirectory name)
        trigger_frame: Frame ID at layer change detection
        N: Number of frames to process after trigger
        N_offset: Pixel width for temperature sampling region
        plot_dir: Directory for saving debug plots (None to disable)
        simple: Use pixel-based (True) or Hough-based (False) processing
        
    Returns:
        Tuple containing:
            - prev_temps: List of previous layer temperatures (°C)
            - recent_temps: List of recent layer temperatures (°C)  
            - trigger_frame: Actual trigger frame used (may be adjusted)
        
    Note:
        Temperatures below 75°C are replaced with NaN as likely errors
        (ambient temperature filtering).
    """
    # Determine plot directory
    if plot_dir:
        set_plotting(True)
        set_plot_dir(plot_dir)


    frames_info, trigger_frame = load_post_frames(base_dir, serial, trigger_frame, N)
    prev_temps, recent_temps, prev_temps_simple, recent_temps_simple = [], [], [], []
    logger.info(f"Processing {len(frames_info)} frames for serial {serial}")
    for info in frames_info:
        #prev, recent = process_frame_array(
        #   info['array'], frame_id=info['frame_id'], serial=serial, N_offset=N_offset
        #)
        
        
    	
        prev, recent = process_frame_array_simple(  # Using the existing function name
            info['array'], 
            frame_id=info['frame_id'], 
            serial=serial, 
            N_offset=N_offset,
            simple=simple  # Pass the simple flag
        )
        prev_temps.append(prev)
    
        recent_temps.append(recent)
        #prev_temps_simple.append(prev)
        #recent_temps_simple.append(recent)
    
    # Temperature validation threshold
    # Filters out ambient/background pixels
    thres = 75 # collect mistakes
    prev_temps = [x if x >= thres else np.nan for x in prev_temps]
    prev_temps_simple = [x if x >= thres else np.nan for x in prev_temps_simple]
    logger.info(f"Finished processing frames. Prev temps: {np.nanmean(prev_temps)}, Recent temps: {np.nanmean(recent_temps)}")
    #logger.info(f"Finished processing frames. Prev temps: {np.nanmean(prev_temps_simple)} {prev_temps_simple}, Recent temps: {np.nanmean(recent_temps_simple)}")
    #os.makedirs(f'{PLOT_SAVE_DIR}/temp_arrays', exist_ok=True)
    
    
    #if prev_temps:
        
        #np.save(os.path.join(f'{PLOT_SAVE_DIR}/temp_arrays', f"{trigger_frame}_prev_region.npy"), prev_temps)
    #if recent_temps:
        #np.save(os.path.join(f'{PLOT_SAVE_DIR}/temp_arrays', f"{trigger_frame}_recent_region.npy"), recent_temps)
    
    return prev_temps, recent_temps, trigger_frame

