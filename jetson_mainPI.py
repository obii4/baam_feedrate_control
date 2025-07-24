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
from datetime import datetime
import csv
import logging
import argparse
import numpy as np


def load_csv_with_timestamps(filepath, timestamp_format="%Y-%m-%d %H:%M:%S.%f"):
    rows = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # # Parse timestamp string to datetime
            row['timestamp'] = datetime.fromisoformat(row['timestamp']) \
                 if 'T' in row['timestamp'] else datetime.strptime(row['timestamp'], timestamp_format)
            rows.append(row)
    return rows

timestamp_list = load_csv_with_timestamps('/home/chris/baam_ir/baam_feedrate_control/output.csv')


# 0) Random helper functions
from jetson.utils.utils import create_csv_writer

# 1) Core camera utilities
from jetson.utils.camera_utils import free_video_devices

# 2) WCF client wrapper
from jetson.wcf_client_pcam import WcfClient

# 3) ThermalCapture class for threaded I/O
from jetson.thermal_capture import ThermalCapture

# 4) Frame processing
from jetson.utils.frame_utils import find_closest_frame
from jetson.process_frames import process_frames

# 5) PI controller
from jetson.controller import PITemperatureController

# ─── Configuration Defaults ───────────────────────────────────────────────────
WINDOWS_IP    = '192.168.0.2' #'192.168.0.151'
PORT_BASE     = 12345
GAIN_MODE     = 1   #low gain = 1, high_gain = 2
EXPECTED_CAM  = 2

LAYER_KEY     = 0    # cmd_id
CONTROL_KEY   = 3    # Extra feedrate override

CAM_INTEREST = '001b000d'  #'001b000d''0018004a'
N_POST_FRAMES = 30   # Number of frames to look at post layer change

TEMPERATURE_SETPOINT  = 110

predefined_cal    = {
        'K_temp': 0.675,      # °C per % feed
        'K_time': -1.305,     # s per % feed  
        'Kp_T': 41.692,        # Proportional gain
        'Ki_T': 8.338,       # Integral gain
        'T0': 112.0,          # Baseline temperature
        'y_sp': 115.0,        # Baseline layer time
        'u_initial': 140.0    # Initial feed rate
    }

CALIBRATION_LAYERS = 10
STEP_MAGNITUDE = 40

POLL_INTERVAL = 0.1  # seconds
LOG_FILE      = 'jetson_edge.log'
CSV_FILENAME  = 'layer_changes.csv'
CSV_FILENAME_feed  = 'feedrate_changes.csv'
CSV_FILENAME_control  = 'control_changes.csv'

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

    # Whether to calibrate or nor
    parser.add_argument('--calibrate', action='store_true', 
                        help='Enable calibration mode for PI controller parameters')
    
    parser.add_argument('--calibration-layers', type=int, default=10,
                        help='Number of layers to collect for calibration (default: 10)')
    
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

    # Mode-specific constants
    USE_CALIBRATION = args.calibrate
    if USE_CALIBRATION:
        CALIBRATION_LAYERS = args.calibration_layers
        logger.info(f"Calibration mode: Will perform step test after {CALIBRATION_LAYERS} layers")
        WARMUP_LAYERS = 0
    else:
        WARMUP_LAYERS = args.calibration_layers
        logger.info(f"Warmup mode: Will use predefined parameters after {WARMUP_LAYERS} layers")
        CALIBRATION_LAYERS = 0

   
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
    controller = PITemperatureController(setpoint=TEMPERATURE_SETPOINT)
    
    # 4) Prepare layer-change, feed-rate, and controller CSV in run dir

    csv_file, csv_writer = create_csv_writer(
        base_dir, 
        CSV_FILENAME, 
        ['timestamp', 'old_layer', 'new_layer']
    )

    # Feed rate CSV
    feed_file, feed_writer = create_csv_writer(
        base_dir,
        CSV_FILENAME_feed,
        ['timestamp', 'layer', 'old_feed', 'new_feed', 'mode']
    )

    # Controller data CSV
    controller_file, controller_writer = create_csv_writer(
        base_dir,
        CSV_FILENAME_control,
        ['layer', 'duration', 'temperature', 'feed_rate', 'mode']
    )
    

    last_layer = None

    
    durations = []
    temps = []
    u_history = []
    control_active = False  
    controller = None       



    layer_index = 0
    
    # 5) Poll for layer changes and process
    while True:
        raw = wcf.get_value(LAYER_KEY)
        try:
            current_layer = int(raw) # this is what baam needs int(float(raw))
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid layer value: {raw!r}")
            time.sleep(0.5)
            continue

        if last_layer is None:
            last_layer = current_layer
            layer_start_time = timestamp_list[layer_index]['timestamp'] #datetime.datetime.now().isoformat()#time.time()
            logger.info(f"Starting on layer {current_layer}")
            initial_feed = float(wcf.get_value(CONTROL_KEY))
            u_history.append(initial_feed)
            layer_index += 1
        
        elif current_layer != last_layer:
            
            if layer_index < len(timestamp_list):
                ts = timestamp_list[layer_index]['timestamp'] #datetime.datetime.now().isoformat()
                logger.info(f"Layer changed: {last_layer} → {current_layer}")
                csv_writer.writerow([ts, last_layer, current_layer])
                csv_file.flush()
            
                # Find last layer duration
                layer_duration = int((ts - layer_start_time).total_seconds())
                layer_start_time = ts
 
            # ===== TEMPERATURE PROCESSING =====
            # TODO: Uncomment and fix your temperature processing
            # For now, using a placeholder
            #avg_temp = 140.0 + np.random.normal(0, 2)  # REPLACE with actual temperature processing
            
                base = '/mnt/external/wet_run_FIN'
                closest = find_closest_frame(f'{base}/{CAM_INTEREST}', ts)
                logger.info(f"Closest frame to {ts} is {closest}")
            
                prevs, recents = process_frames(
                base_dir=base,
                serial=CAM_INTEREST,
                trigger_frame=closest,
                N=N_POST_FRAMES
                )
            
                avg_temp = np.nanmean(prevs)
                # Get current feed rate
                old_feed = float(wcf.get_value(CONTROL_KEY))
            
            else:
                logger.info(f"moving on from exp data.")
                # Get current feed rate
                old_feed = float(wcf.get_value(CONTROL_KEY))
                
         

            layer_index += 1
            # ===== CONTROL LOGIC =====
            if not control_active:
                
                # Collect data for either calibration or warmup
                durations.append(layer_duration)
                temps.append(avg_temp)
                u_history.append(old_feed)

                if USE_CALIBRATION:
                    # CALIBRATION MODE
                    logger.info(f"Calibration layer {len(durations)}/{CALIBRATION_LAYERS}: "
                            f"duration={layer_duration:.2f}s, temp={avg_temp:.2f}°C")
                    
                    # Check if ready for step test
                    if len(durations) == CALIBRATION_LAYERS // 2:
                        # Apply step change at halfway point
                        step_feed = old_feed + STEP_MAGNITUDE
                        wcf.set_value(CONTROL_KEY, float(step_feed))
                        logger.info(f"Applied step change: {old_feed} → {step_feed}")
                        new_feed = step_feed
                    else:
                        new_feed = old_feed
                    
                    # Check if calibration complete
                    if len(durations) >= CALIBRATION_LAYERS:
                        logger.info("Calibration complete. Analyzing step response...")
                        
                        
                        # chris look at nominal feed and when to choose what
                        # Create and calibrate controller
                        controller = PITemperatureController(
                            setpoint=TEMPERATURE_SETPOINT,
                            nominal_feed=100.0,
                            feed_limits=(50.0, 200.0),
                            max_feed_change=5.0,
                            temp_deadband=1.0
                        )
                        durations_real = [114, 104, 125, 63, 62, 62, 62, 63, 64, 63, 63]
                        temps_real = [113, 113, 111, 137, 138, 139, 136, 137, 138, 139, 140]
    
                        # Auto-tune from step test data
                        cal_params = controller.calibrate_from_step_test(
                            times=np.array(durations_real),
                            temps=np.array(temps_real),
                            step_magnitude=STEP_MAGNITUDE,
                            auto_tune=True  
                        )
                        
                        control_active = True
                        logger.info(f"Controller activated. Tau={cal_params['tau']:.1f}s, "
                                f"Theta={cal_params['theta']:.1f}s")
                    
                    mode = 'calibration'
                    
                else:
                    # WARMUP MODE
                    logger.info(f"Warmup layer {len(durations)}/{WARMUP_LAYERS}: "
                            f"duration={layer_duration:.2f}s, temp={avg_temp:.2f}°C")
                    
                    new_feed = old_feed  # No changes during warmup
                    
                    # Check if warmup complete
                    if len(durations) >= WARMUP_LAYERS:
                        logger.info("Warmup complete. Loading predefined parameters...")
                        
                        # chris look at nominal feed and when to choose what
                        # Create controller
                        controller = PITemperatureController(
                            setpoint=TEMPERATURE_SETPOINT,
                            nominal_feed=140.0,
                            feed_limits=(50.0, 200.0),
                            max_feed_change=10.0,
                            temp_deadband=5.0
                        )
                        
                        # Set parameters from warmup info
                        predefined_cal['T0'] = np.mean(temps)
                        predefined_cal['y_sp'] = np.mean(durations)
                        predefined_cal['u_initial'] = u_history[-1]
                        
                        results = controller.calibrate_from_step_test(
                            times=np.array(durations),
                            temps=np.array(temps),
                            use_predefined=True,
                            predefined_values=predefined_cal
                        )
                        

                        control_active = True
                        logger.info(f"Controller activated with predefined parameters: {results}, also durations: {durations}. temps: {temps}")
                    
                    mode = 'warmup'
                
            else:
                # CONTROL MODE - Controller is active
                mode = 'control'
                               
                avg_temp = controller.predict_temperature(old_feed)
                layer_duration = controller.predict_layer_time(old_feed)
                
                durations.append(layer_duration)
                temps.append(avg_temp)
                u_history.append(old_feed)
                
                # Get next feedrate
                new_feed, info = controller.compute_next_feedrate(
                    current_temp=avg_temp,
                    current_time=layer_duration
                )
                
                # Apply the new feed rate
                wcf.set_value(CONTROL_KEY, float(new_feed))
                
                # Log control info
                logger.info(f"Control: temp={avg_temp:.1f}°C, error={info['error']:.1f}°C, "
                        f"feed={new_feed:.1f}%")
                
                # Log feed rate change
                if old_feed != new_feed:
                    logger.info(f"Feedrate changed: {old_feed:.1f} → {new_feed:.1f} (mode: {mode})")
                else:
                    logger.info(f"Feedrate remained constant:{new_feed:.1f} (mode: {mode})")
                
                now = datetime.now().isoformat() 
                feed_writer.writerow([now, last_layer, old_feed, new_feed, mode])
                feed_file.flush()
            
            # Write to controller data CSV
            controller_writer.writerow([
                last_layer, layer_duration, avg_temp, new_feed, mode
            ])
            controller_file.flush()
            
            
            
            last_layer = current_layer

        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main()
