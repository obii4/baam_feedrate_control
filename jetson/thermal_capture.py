#!/usr/bin/env python3
"""
thermal_capture.py

Refactored thermal image collection and streaming for BAAM LFAM.

Usage:
    from thermal_capture import ThermalCapture

The ThermalCapture class discovers available IR cameras, sets gain modes,
captures frames continuously, saves each as .npy and logs timestamps,
and streams frames over TCP to a Windows host for live display.
"""
import os
import re
import csv
import time
import socket
import pickle
import struct
import threading
import subprocess
import logging
from datetime import datetime

import cv2

from jetson.utils.camera_utils import (
    create_dirs_for_cameras, 
    save_frame, 
    free_video_devices, 
    set_gain_mode
)


class ThermalCapture:
    """
    Discover, configure, and capture thermal frames from multiple cameras.
    Each frame is saved and streamed to a Windows host over TCP.
    """
    def __init__(self, output_dir, windows_ip, port_base=12345,
                 gain_mode=1, expected_cameras=None, log_file=None):
        self.output_dir = output_dir
        self.windows_ip = windows_ip
        self.port_base  = port_base
        self.gain_mode  = gain_mode
        self.expected  = expected_cameras

        self.device_map = {}     # {"/dev/videoX": "serial"}
        self.captures   = {}     # {serial: {"cap", "frame_count", "port"}}
        self.threads    = []
        self.running    = False

        # Logger
        self.logger = logging.getLogger("ThermalCapture")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    def discover_devices(self):
        """Free video devices and map to serial numbers."""
        free_video_devices()
        devices = [f"/dev/{n}" for n in os.listdir('/dev') if re.match(r'video\d+', n)]
        serial_pat = re.compile(r'SERIAL_SHORT=(\S+)')
        for dev in devices:
            try:
                out = subprocess.check_output([
                    'udevadm', 'info', '--query=all', f'--name={dev}'
                ], stderr=subprocess.DEVNULL).decode('utf-8')
                m = serial_pat.search(out)
                if m:
                    serial = m.group(1).split('-')[0]
                    self.device_map[dev] = serial
            except Exception as e:
                self.logger.warning(f"udevadm failed for {dev}: {e}")
        count = len(self.device_map)
        if self.expected and count != self.expected:
            self.logger.warning(f"Found {count} cameras, expected {self.expected}")
        return self.device_map

    def prepare_output(self):
        """Create output directories and record the start timestamp."""
        create_dirs_for_cameras(self.output_dir, self.device_map)
        ts_path = os.path.join(self.output_dir, 'local_start_time.csv')
        with open(ts_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time'])
            writer.writerow([datetime.now().isoformat()])

    def setup_captures(self):
        """Open and configure OpenCV VideoCapture for each camera."""
        for dev, serial in self.device_map.items():
            cap = cv2.VideoCapture(dev)
            if not cap.isOpened():
                self.logger.error(f"Cannot open {dev}. Moving to next")
                continue
            set_gain_mode(dev, self.gain_mode)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            port = self.port_base + len(self.captures)
            self.captures[serial] = {'cap': cap, 'frame_count': 0, 'port': port}
        if not self.captures:
            raise RuntimeError("No working thermal cameras found.")
        
        
        self.logger.info(f"Low gain set for {len(self.captures)} devices.")
        return self.captures

    def start(self):
        """Begin capturing: discover devices, prepare dirs, spawn threads."""
        self.discover_devices()
        self.prepare_output()
        self.setup_captures()
        self.running = True
        for serial, data in self.captures.items():
            t = threading.Thread(target=self._capture_loop, args=(serial, data), daemon=True)
            self.threads.append(t)
            t.start()
        self.logger.info("Capture threads started.")

    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        for data in self.captures.values():
            data['cap'].release()
        cv2.destroyAllWindows()
        self.logger.info("All captures stopped.")

    def _capture_loop(self, serial, data):
        """Thread loop: read frames, save, and stream over TCP."""
        cap = data['cap']
        port = data['port']
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.windows_ip, port))
        except Exception as e:
            self.logger.error(f"[{serial}] Socket connect failed: {e}")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.logger.error(f"[{serial}] Frame read error.")
                continue
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            save_frame(serial, frame, ts, 'na', data['frame_count'], self.output_dir)
            try:
                payload = pickle.dumps((serial, frame))
                sock.sendall(struct.pack("!I", len(payload)) + payload)
            except Exception as e:
                self.logger.error(f"[{serial}] Send error: {e}")
                break
            data['frame_count'] += 1
            time.sleep(0)  # yield thread

        sock.close()

# No standalone execution here; imported by main orchestrator.
