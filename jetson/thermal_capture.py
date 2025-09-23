#!/usr/bin/env python3
"""
thermal_capture.py - Multi-camera thermal image acquisition and streaming system.

This module provides real-time thermal image capture from multiple FLIR cameras
for BAAM (Big Area Additive Manufacturing) process monitoring. It handles device
discovery, configuration, parallel capture, data persistence, and TCP streaming
to a Windows host for visualization.

Key Features:
    - Automatic FLIR camera discovery via udev
    - Parallel capture from multiple cameras
    - Raw thermal data preservation (.npy format)
    - Real-time streaming over TCP
    - Synchronized timestamping across cameras

Usage:
    from thermal_capture import ThermalCapture
    
    tc = ThermalCapture(
        output_dir="/mnt/external/run_001",
        windows_ip="192.168.0.151",
        port_base=12345,
        gain_mode=1,
        expected_cameras=6
    )
    tc.start()  # Begins capture threads
    # ... main processing ...
    tc.stop()   # Cleanup

# WARNING: Windows firewall must allow incoming connections on ports 12345-12350 for streaming to work

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
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
    Multi-threaded thermal camera capture and streaming system.
    
    Discovers and manages multiple FLIR thermal cameras, capturing frames
    continuously while saving to disk and streaming over TCP. Each camera
    runs in its own thread for parallel operation.
    
    Attributes:
        output_dir: Base directory for saving captured frames
        windows_ip: IP address of Windows receiver for streaming
        port_base: Starting TCP port (increments per camera)
        gain_mode: Camera gain setting (1=low, 2=high)
        expected: Expected number of cameras (for validation)
        device_map: Mapping of /dev/videoX to camera serials
        captures: Active capture sessions by serial number
        threads: List of capture threads
        running: Global capture state flag
        
    Data Organization:
        output_dir/
        ├── local_start_time.csv     # Session timestamp
        ├── 001b000d/                # Camera serial directory
        │   ├── frame_0000.npy       # Raw thermal array
        │   ├── frame_0001.npy
        │   └── timestamps.csv       # Frame timestamps
        └── 001b000e/                # Second camera
            └── ...
            
    TCP Protocol:
        - Each camera streams to port_base + camera_index
        - Message format: [4-byte length][pickled (serial, frame)]
        - Big-endian length prefix for message framing
    """
    def __init__(self, output_dir, windows_ip, port_base=12345,
                 gain_mode=1, expected_cameras=None, log_file=None):
        
        """
        Initialize thermal capture system.
        
        Sets up logging, network parameters, and prepares for camera discovery.
        Does not start capture - call start() to begin.
        
        Args:
            output_dir: Root directory for saving frames (/mnt/external/run_XXX)
            windows_ip: IP address of Windows display host
            port_base: Starting TCP port number (cameras use sequential ports)
            gain_mode: Thermal gain setting (1=low for hot objects, 2=high for ambient)
            expected_cameras: Expected number of cameras (warns if mismatch)
            log_file: Optional log file path (uses console if None)
            
        Note:
            Low gain (1) is typical for printing operations (>180 C)
            High gain (2) is for room temperature imaging
        """

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
        """
        Discover and map thermal cameras to serial numbers.
        
        Uses udevadm to query USB device information and extract serial numbers
        from FLIR cameras. Handles device busy states by calling free_video_devices()
        to release any locks.
        
        Returns:
            Dict mapping device paths to serial numbers:
            {'/dev/video0': '001b000d', '/dev/video2': '001b000e', ...}
            
        Note:
            Serial extraction pattern: 'SERIAL_SHORT=001b000d-3132-3334'
            Only uses first segment before hyphen
            
        Warning:
            Requires udev rules properly configured for FLIR cameras
            May need sudo for some systems
        """
        # Device busy handling
        free_video_devices()  # Release any stale locks before discovery
        devices = [f"/dev/{n}" for n in os.listdir('/dev') if re.match(r'video\d+', n)]

        # Extract serial from udev output
        # FLIR cameras report serial in format: SERIAL_SHORT=001b000d-3132-3334
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
        """
        Create output directory structure and record session start time.
        
        Creates a subdirectory for each discovered camera and writes a global
        timestamp file for session synchronization. This timestamp is used
        to align data from multiple cameras and the HMI system.
        
        Directory structure created:
            output_dir/
            ├── local_start_time.csv  # ISO format timestamp
            ├── 001b000d/             # Per-camera directories
            ├── 001b000e/
            └── ...
            
        Note:
            Timestamp is in local Jetson time, its time was synced with Windows device at start
        """
        create_dirs_for_cameras(self.output_dir, self.device_map)
        ts_path = os.path.join(self.output_dir, 'local_start_time.csv')
        with open(ts_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time'])
            writer.writerow([datetime.now().isoformat()])

    def setup_captures(self):
        """
        Configure OpenCV VideoCapture objects for thermal imaging.
        
        Opens each discovered camera and configures for raw thermal data:
        - Sets resolution to 160x120 (FLIR Lepton native)
        - Configures Y16 format (16-bit grayscale for thermal data)
        - Disables RGB conversion to preserve raw values
        - Applies gain mode setting via v4l2
        
        Returns:
            Dict of capture objects by serial:
            {serial: {'cap': VideoCapture, 'frame_count': int, 'port': int}}
            
        Raises:
            RuntimeError: If no cameras can be opened
            
        Camera Settings:
            - Y16 format: Raw 16-bit thermal values
            - Resolution: 160x120 pixels (Lepton 3.5)
            - No RGB conversion: Preserves temperature data
            - Gain mode: Applied via v4l2-ctl system call
        """
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
        """
        Thread worker for continuous frame capture and streaming.
        
        Runs in a dedicated thread per camera. Captures frames continuously,
        saves each to disk as .npy file, logs timestamp, and streams to Windows
        host over TCP.
        
        Args:
            serial: Camera serial number for identification
            data: Dict with 'cap' (VideoCapture), 'port', 'frame_count'
            
        TCP Protocol:
            1. Connect to windows_ip:port
            2. For each frame:
            - Pickle (serial, frame) tuple
            - Send 4-byte length prefix (network byte order)
            - Send pickled payload
            3. Close on error or stop signal
            
        Error Handling:
            - Connection failures: Log and exit thread
            - Send failures: Log and reconnect
            - Read failures: Log and continue
            
        Note:
            Uses time.sleep(0) to yield thread for other cameras
            Frame naming: frame_XXXX.npy (zero-padded to 4 digits)
        """
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
                # Network byte order (big-endian) length prefix for reliable parsing
                payload = pickle.dumps((serial, frame))
                sock.sendall(struct.pack("!I", len(payload)) + payload)
            except Exception as e:
                self.logger.error(f"[{serial}] Send error: {e}")
                break
            data['frame_count'] += 1
            time.sleep(0)  # yield thread

        sock.close()


