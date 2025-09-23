"""
camera_utils.py - Low-level camera control and file I/O utilities.

This module provides hardware control functions for FLIR Lepton thermal cameras
including device management, gain control via V4L2, and frame persistence.
It handles the interface between OpenCV capture and the Linux video subsystem.

Core Functionality:
    - Video device cleanup and management
    - FLIR Lepton gain mode configuration via UVC controls  
    - Directory structure creation for multi-camera systems
    - Frame saving with timestamp synchronization

Dependencies:
    - uvcdynctrl: Dynamic UVC control for Lepton parameters
    - v4l2-ctl: Video4Linux2 control interface
    - pt1.xml: PureThermal 1 UVC control definitions

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""



import os, re, signal, subprocess
from datetime import datetime
import csv
import logging

import numpy as np
import cv2

# PureThermal 1 UVC control definition file
# Contains Lepton-specific control mappings for uvcdynctrl
PT1_XML_PATH = '/home/chris/baam_ir/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml'

# Default thermal gain mode
GAIN_MODE = 1 # 1=Low gain (high temp), 2=High gain (low temp)

logger = logging.getLogger("JetsonMain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def free_video_devices():
    """Kill any process holding /dev/video* handles."""
    try:
        pid = subprocess.check_output(
            "lsof -t /dev/video* | head -n1", shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
        if pid:
            logger.info(f"Killing hung video device PID {pid}")
            os.kill(int(pid), signal.SIGTERM)
        else:
            logger.info("No video devices in use.")
    except subprocess.CalledProcessError:
        logger.info("No video devices in use.")
    except Exception as e:
        logger.warning(f"Could not free devices: {e}")


def set_gain_mode(device, mode=GAIN_MODE):
    """Load XML control definitions and set the Lepton gain mode."""
    try:
        logger.info(f"Setting gain={mode} on {device}")
        subprocess.run(
            ["uvcdynctrl", "-d", device, "-i", PT1_XML_PATH],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL  
        )
        subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", f"lep_cid_sys_gain_mode={mode}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL   
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to set gain on {device}: {e}")


def create_dirs_for_cameras(base_dir, device_serials):
    """
    Given a base path (e.g. /mnt/external/run01) and a dict
    dev→serial, create per-camera subfolders.
    """
    for serial in device_serials.values():
        path = os.path.join(base_dir, serial)
        os.makedirs(path, exist_ok=True)


def save_frame(serial, frame, local_ts, remote_ts, frame_count, base_dir):
    """
    Save a single frame as .npy and append its timestamp to CSV.
    """
    camera_dir = os.path.join(base_dir, serial)
    npy_path   = os.path.join(camera_dir, f"{frame_count}.npy")
    csv_path   = os.path.join(camera_dir, "timestamps.csv")

    np.save(npy_path, frame)
    logger.debug(f"Saved frame {frame_count} → {npy_path}")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_count, local_ts, remote_ts])
