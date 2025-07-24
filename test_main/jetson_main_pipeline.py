#!/usr/bin/env python3

"""
jetson_main.py

Orchestrator combining:
 - WCF client for BAAM HMI
 - Thermal camera capture & streaming
 - Layer-change detection with timestamp logging
 - Frame processing and PID feedback
"""
import os
import sys
import time
import datetime
import csv
import logging
import argparse
import numpy as np

# 1) Core camera utilities
from jetson.utils.camera_utils import free_video_devices


# 2) WCF client wrapper
from jetson.wcf_client import WcfClient

# 3) ThermalCapture class for threaded I/O
from jetson.thermal_capture import ThermalCapture

# 4) Frame processing & PID
from jetson.utils.frame_utils import find_closest_frame
from jetson.process_frames import process_frames

#from jetson.pid import PIDController

# ─── Configuration Defaults ───────────────────────────────────────────────────
WINDOWS_IP    = '192.168.0.2' #'192.168.0.151'
PORT_BASE     = 12345
GAIN_MODE     = 1   #low gain = 1, high_gain = 2
EXPECTED_CAM  = 2

LAYER_KEY     = 0    # cmd_id
CONTROL_KEY   = 3    # Extra feedrate override

CAM_INTEREST = '0018004a'  #'001b000d'
N_POST_FRAMES = 30   # Number of frames to look at post layer change

PID_SETPOINT  = 205.0
PID_PARAMS    = dict(kp=1.0, ki=0.1, kd=0.01)

POLL_INTERVAL = 0.1  # seconds
LOG_FILE      = 'jetson_edge.log'
CSV_FILENAME  = 'layer_changes.csv'
CSV_FILENAME_feed  = 'feedrate_changes.csv'

# ─── Logger Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger('JetsonMain')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)


# ─── Main Orchestrator ─────────────────────────────────────────────────────────
def main():
    # CLI for run directory name
    parser = argparse.ArgumentParser(description="Jetson edge orchestrator for BAAM LFAM")
    parser.add_argument('run_dir', help='Name of directory under /mnt/external to store data')
    args = parser.parse_args()

    # Construct full output path
    base_dir = os.path.join('/mnt/external', args.run_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    log_path = os.path.join(base_dir, LOG_FILE)
    logger = logging.getLogger('JetsonMain')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Using run directory: {base_dir}")

    # 1) Prepare camera capture
    free_video_devices()
    
    
    tc = ThermalCapture(
        output_dir=base_dir,
        windows_ip=WINDOWS_IP,
        port_base=PORT_BASE,
        gain_mode=GAIN_MODE,
        expected_cameras=EXPECTED_CAM,
        log_file=log_path
    )
    tc.start()

    # 2) Prepare WCF client
    wcf = WcfClient(log_file=LOG_FILE)

    # 3) Prepare PID controller
    #pid = PIDController(**PID_PARAMS)

    # 4) Prepare layer-change AND feed-rate CSV in run dir
    csv_path = os.path.join(base_dir, CSV_FILENAME)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','old_layer','new_layer'])
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    feed_csv_path = os.path.join(base_dir, CSV_FILENAME_feed)
    if not os.path.exists(feed_csv_path):
        with open(feed_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','layer','old_feed','new_feed'])
    feed_file   = open(feed_csv_path, 'a', newline='')
    feed_writer = csv.writer(feed_file)

    last_layer = None

    # 5) Poll for layer changes and process
    while True:
        raw = wcf.get_value(LAYER_KEY)
        try:
            current_layer = int(raw)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid layer value: {raw!r}")
            time.sleep(0.5)
            continue

        if last_layer is None:
            last_layer = current_layer
            logger.info(f"Starting on layer {current_layer}")
        elif current_layer != last_layer:

            ts = datetime.datetime.now().isoformat()
            logger.info(f"Layer changed: {last_layer} → {current_layer}")
            csv_writer.writerow([ts, last_layer, current_layer])
            csv_file.flush()
            
            base = '/mnt/external/wet_run_FIN/001b000d' 

            fake_time = datetime.datetime(2025, 7, 1, 13, 15, 40)
            
            closest = find_closest_frame(f'{base_dir}/{CAM_INTEREST}', fake_time)
            
            logger.info(f"Closest frame to {fake_time.isoformat()} is {closest}")
            
            # # Process all frames for the previous layer
            # frame_ids = tc.frames_for_layer(last_layer)
            # temps = [process_frame(fid) for fid in frame_ids]
            
            base = '/mnt/external/wet_run_FIN'
            prevs, recents = process_frames(
            base_dir=base_dir,
            serial=CAM_INTEREST,
            trigger_frame=closest,
            N=N_POST_FRAMES
            )
            


            # # Compute PID correction and push to HMI
            old_feed = float(wcf.get_value(CONTROL_KEY))
            correction = wcf.set_value(CONTROL_KEY, np.random.choice(100)) #pid.update(temps, setpoint=PID_SETPOINT)
            
            #wcf.set_value(CONTROL_KEY, correction)
            
            
            logger.info(f"Feedrate changed: {old_feed} → {correction}")
            now = datetime.datetime.now().isoformat()
            feed_writer.writerow([now, last_layer, old_feed, correction])
            feed_file.flush()
           	

            last_layer = current_layer

        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main()
