import os, re, signal, subprocess
from datetime import datetime
import csv
import logging

import numpy as np
import cv2

# Constants moved here
PT1_XML_PATH = '/home/chris/baam_ir/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml'
GAIN_MODE    = 1

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
            ["uvcdynctrl", "-v", "-d", device, "-i", PT1_XML_PATH],
            check=True
        )
        subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", f"lep_cid_sys_gain_mode={mode}"],
            check=True
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
