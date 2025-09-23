#!/usr/bin/env python3

"""
jetson_main.py - Main orchestrator for BAAM closed-loop feedrate control system.

This module coordinates the thermal imaging feedback control system for Big Area 
Additive Manufacturing (BAAM). It polls the Windows HMI for layer changes, captures
thermal data via FLIR cameras, and adjusts feedrate using PI control to maintain
optimal layer adhesion temperature.

The system operates in three modes:
1. Calibration: Performs step test to auto-tune PI parameters
2. Warmup: Uses predefined parameters after initial stabilization
3. Control: Active temperature regulation via feedrate adjustment

Communication Flow:
    Windows HMI (WCF) <-> Jetson (this script) <-> Thermal Cameras
                      ^                        ^
                      |                        |
                  Layer info              Temperature data
                  Feed commands            Frame capture

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""

import os
import time
from datetime import datetime
import csv
import logging
import argparse
import numpy as np
import signal

def backup_log(message, base_dir = None, level="INFO"):
    """
    Write redundant log entry to backup file for critical events.
    
    Creates a simple text log independent of the main logging system to ensure
    critical events are captured even if the main logger fails.
    
    Args:
        message: Log message to write
        base_dir: Directory path for backup log file
        level: Log level (INFO, WARNING, ERROR)
        
    Note:
        Used for layer changes, calibration events, and control activation
    """
    tstamp = datetime.now()
    backup_log_path = os.path.join(base_dir, 'backup_log.txt')
    with open(backup_log_path, 'a') as f:
        f.write(f"{tstamp} [{level}] {message}\n")


# 0) Random helper functions
from jetson.utils.utils import create_csv_writer, FeedRateLogger #, return_actual_feed

# 1) Core camera utilities
from jetson.utils.camera_utils import free_video_devices

# 2) WCF client wrapper
from jetson.wcf_client import WcfClient #jetson.wcf_client_pcam

# 3) ThermalCapture class for threaded I/O
from jetson.thermal_capture import ThermalCapture

# 4) Frame processing
from jetson.utils.frame_utils import find_closest_frame
from jetson.process_frames import process_frames

# 5) PI controller
from jetson.controller import PITemperatureController


# ─── Configuration Defaults ───────────────────────────────────────────────────
# Network Configuration
WINDOWS_IP    = '192.168.0.151' # IP address of Windows HMI machine
PORT_BASE     = 12345           # Base port for WCF communication

# Camera Configuration
GAIN_MODE     = 1               # Thermal camera gain (1=low, 2=high)
EXPECTED_CAM  = 6               # Number of cameras expected in system
CAM_INTEREST = '001b000d'       # Serial number of primary control camera  (PCAM tesing cam --> '0018004a')

# WCF Communication Keys
LAYER_KEY     = 8               # WCF key for current layer number
CONTROL_KEY   = 0 #3            # WCF key for  feedrate override command

# Proccessing Parameters
SIMPLE_PROCESS = False          # Debug usage --> Use simplified processing (no Hough transform)
N_POST_FRAMES = 30              # Frames to analyze after layer change detection
skip_N = 1                      # Initial layers to skip at start of print (thermal stabilization)

# Control Limits
FEED_LIM_LOW = 0                # Minimum feedrate override (%)
FEED_LIM_HIGH = 175             # Max feedrate override (%)
TEMPERATURE_SETPOINT  = 125     # Target temperature controller (C), this is only used for warmup mode. for calibration, Temperture setpoint is mean of num_layers_before_change temps before step change
START_FEED = 100                # #Initial feedrate override (%) --> goes to initital rate when setting feed logger

# Calibration Parameters
num_layers_after = 5            # Layers to collect after step change before calibrating controller
num_layers_before_change = 5    # Layers to analyze before step for baseline
CALIBRATION_LAYERS = 1          # Total layers for calibration sequence, this is updated from sys arg
STEP_MAGNITUDE = -20            # Step change magnitude for system ID (%)

# Predefined controller parameters (from previous tuning --> 903_cftf_exp2_fin)
predefined_cal    = {
        'K_temp': 1.0125,       # C per % feed
        'K_time': -3.6208,      # s per % feed  
        'Kp_T': 1.7196,         # Proportional gain
        'Ki_T': 0.008818342,    # Integral gain
        'T0': 124.8,            # Baseline temperature
        'y_sp': 117.2,          # Baseline layer time
        'u_initial': 50         # Initial feed rate
    }


# System Parameters
POLL_INTERVAL = 1               # HMI polling rate (seconds)
LOG_FILE      = 'jetson_edge.log'
CSV_FILENAME  = 'change_log.csv'


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
        backup_log(f"Calibration mode: Will perform step test after {CALIBRATION_LAYERS} layers", base_dir)
        WARMUP_LAYERS = 0
    else:
        WARMUP_LAYERS = args.calibration_layers
        logger.info(f"Warmup mode: Will use predefined parameters after {WARMUP_LAYERS} layers")
        backup_log(f"Warmup mode: Will use predefined parameters after {WARMUP_LAYERS} layers", base_dir)
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
    

    backup_path = os.path.join(base_dir, 'backup.txt')
    with open(backup_path, 'a') as f:
        f.write(f'session start @ {datetime.now()}\n')

    last_layer = None
    prev_layer_start_time = None
    step_applied_at_layer = None
    
    durations = []  # Layer print time in seconds
    temps = []      # Mean layer temperature in C  
    u_history = []  # Applied feedrate for each layer (%)

    control_active = False  
    controller = None    
    feed_c = None # Tracker for pre experiment changes 
    cmd_feed = -70 # to get to nom for LL
    
    # 5) Poll for layer changes and process
    while True:
        raw = wcf.get_value(LAYER_KEY)
        wcf.set_value(CONTROL_KEY, str(cmd_feed))

        try:
            current_layer = int(float(raw)) #  what baam needs -> int(float(raw)), pcam int(raw)

        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid layer value: {raw!r}")
            time.sleep(0.5)
            continue
            

        if last_layer is None:
            last_layer = current_layer
            prev_layer_start_time = datetime.now()#.isoformat()
            logger.info(f"Starting on layer {current_layer}")
            initial_feed = float(wcf.get_value(CONTROL_KEY)) #welp chris what does this return? is it 0 or no return call? idk
            
            
            # instantiate tracker logic
            FR_logger = FeedRateLogger(initial_rate=START_FEED)
            # Initialize the tracker with the current feed value from WCF
            if initial_feed != 0:  # If we got a valid feed value
                rate, cmd_feed = FR_logger.update_feed(initial_feed)
            feed_c = initial_feed  # Track the raw WCF value, not the command
            
            u_history.append(FR_logger.get_current_rate())
        
        # Layer change detection logic
        elif current_layer != last_layer:
            # Process PREVIOUS layer's thermal data (current layer just started)
            
            # WCF client recreation every 5 layers to prevent connection timeout
            if current_layer % 5 == 0:
                logger.info(f"Layer {current_layer}: Recreating WCF Client")
                wcf = WcfClient(log_file=LOG_FILE)
                
            
            current_time = datetime.now() #.isoformat()
            logger.info(f"Layer changed: {last_layer} → {current_layer}")
            backup_log(f"Layer changed: {last_layer} → {current_layer}", base_dir)
            
            # Find last layer duration
            layer_duration = int((current_time - prev_layer_start_time).total_seconds())
            
     	    # Frame timestamp matching
            # Find the thermal frame closest to when previous layer started
            # This frame will be the trigger point for temperature analysis  
            closest = find_closest_frame(f'{base_dir}/{CAM_INTEREST}', prev_layer_start_time)
            
            # Update var so that current layer info is not proccessed
            prev_layer_start_time = current_time
            
            # Temperature extraction
            # prevs: temperatures from the completed layer (what we're analyzing)
            # recents: temperatures from current layer on right side of fine
            prevs, recents, trigger_frame = process_frames(
                base_dir=base_dir,
                serial=CAM_INTEREST,
                trigger_frame=closest,
                N=N_POST_FRAMES,
                plot_dir=None,
                simple=SIMPLE_PROCESS
                )
            
            avg_temp = np.nanmean(prevs)
            
            #handling of weird cases
            if np.isnan(avg_temp):
                if len(temps) < 5:
                    logger.info("start NaN- using setpoint")
                    avg_temp = TEMPERATURE_SETPOINT
                
                else:
                    avg_temp = temps[-1]
                    logger.info("NaN replaced with prev measurement")

            # Feed rate tracking synchronization
            # Check if operator manually changed feedrate via HMI
            # Must track this to maintain accurate control stat
            feed = float(wcf.get_value(CONTROL_KEY))
            if feed != feed_c:
                rate, cmd_feed = FR_logger.update_feed(feed)
                feed_c = feed
            # Get current feed rate
            old_feed = FR_logger.get_current_rate()
			
            # ===== CONTROL LOGIC =====
            if not control_active:
                
                # Collect data for either calibration or warmup
                durations.append(layer_duration)
                temps.append(avg_temp) # remember this is jank and this goes with prev
                u_history.append(old_feed)
                
                
                

                if USE_CALIBRATION:
                    # CALIBRATION MODE
                    #logger.info(f"Calibration layer {len(durations)}/{CALIBRATION_LAYERS}: "
                    #       f"duration={layer_duration:.2f}s, temp={avg_temp:.2f}°C, feedrate={old_feed}")
                    
                    prev_duration = durations[-1] 
                    prev_feed = u_history[-1] 

                    # since we are proc prev layer, displaying that
                    logger.info(f"Calibration layer {len(durations)}/{CALIBRATION_LAYERS}: "
                           f"duration={prev_duration:.2f}s, temp={avg_temp:.2f}°C, feedrate={prev_feed}, trigger frame = {trigger_frame}")
                    
                    backup_log(f"Calibration layer {len(durations)}/{CALIBRATION_LAYERS}: "
                           f"duration={prev_duration:.2f}s, temp={avg_temp:.2f}°C, feedrate={prev_feed}, trigger frame = {trigger_frame}", base_dir)
                    
                    # Check if ready for step test
                    
                    # Calibration step test timing
                    # Apply step change with enough layers remaining to observe response
                    # Step occurs at (CALIBRATION_LAYERS - num_layers_after) to ensure
                    # we capture both steady-state and transient response
                    if len(durations) == CALIBRATION_LAYERS - num_layers_after: #CALIBRATION_LAYERS // 2:
                        step_applied_at_layer = len(durations)
                        
                        # Apply step change 10 layers from end of calibration
                        step_feed = old_feed + STEP_MAGNITUDE 
                        
                        rate, cmd_feed = FR_logger.define_feed(step_feed)
                        #logger.info(f'{type(cmd_feed)}: to send {cmd_feed}')
                        # Apply the new feed rate
                        wcf.set_value(CONTROL_KEY, str(cmd_feed)) #changed to string 
                        
                        logger.info(f"Applied @ layer {step_applied_at_layer} → step change: {old_feed} → {step_feed}")
                        backup_log(f"Applied @ layer {step_applied_at_layer} → step change: {old_feed} → {step_feed}", base_dir)
                        new_feed = step_feed
                        old_feed = step_feed
                        
                        feed_c = step_feed  # Track the actual feed rate, not the command
                    else:
                        new_feed = old_feed
                    
                    # Check if calibration complete
                    if len(durations) >= CALIBRATION_LAYERS:
                        nom_feed = u_history[step_applied_at_layer] 
                        
                        
                        # Temperature setpoint calculation for calibration
                        # Use mean of N layers before step as baseline
                        # This represents the system's natural steady-state
                        setT = np.nanmean(temps[step_applied_at_layer-num_layers_before_change:step_applied_at_layer]) #110
                        logger.info(f"Calibration complete, reached layer {len(durations)}. Analyzing step response...")
                        logger.info(f"Nominal feed set to: {nom_feed}... Setpoint temp: {setT}")
                        backup_log(f"Nominal feed set to: {nom_feed}... Setpoint temp: {setT}", base_dir)
                        
                        time_data = np.array(durations[step_applied_at_layer-num_layers_before_change:])
                        temp_data = np.array(temps[step_applied_at_layer-num_layers_before_change:])
                        
                        logger.info(f"Using time data →  {len(time_data)}: {time_data}")
                        logger.info(f"Using temp data → {len(temp_data)}: {temp_data}")


                        

                        
                        # Create and calibrate controller
                        controller = PITemperatureController(
                            setpoint=setT,
                            nominal_feed=nom_feed,
                            feed_limits=(FEED_LIM_LOW, FEED_LIM_HIGH),
                            max_feed_change=10.0,
                            temp_deadband=10.0
                        )
                        
    					# Chris - we are excluding beginning data, looking at only 10 before and 10 after step change
                        # Auto-tune from step test data
                        cal_params = controller.calibrate_from_step_test(
                            times=time_data,
                            temps=temp_data,
                            step_magnitude=STEP_MAGNITUDE,
                            auto_tune=True  
                        )
                        
                        control_active = True
                        logger.info(f"Controller activated --> {cal_params}. Tau={cal_params['tau']:.1f}s, "
                                f"Theta={cal_params['theta']:.1f}s")
                                
                                
                        with open(backup_path, 'a') as f:
                            f.write(f'CALIBRATION,{nom_feed},{setT:.1f},{cal_params}\n')
                    
                    mode = 'calibration'
                    
                else:
                    # WARMUP MODE
                    prev_duration = durations[-1] #durations[-2] if len(durations) >= 2 else durations[-1]
                    prev_feed = u_history[-1] #u_history[-2] if len(u_history) >= 2 else u_history[-1]

                     # WARMUP MODE


                    # since we are proc prev layer, displaying that
                    logger.info(f"Warmup layer {len(durations)}/{WARMUP_LAYERS}: "
                           f"duration={prev_duration:.2f}s, temp={avg_temp:.2f}°C, feedrate={prev_feed}, trigger frame = {trigger_frame}")
                    backup_log(f"Warmup layer {len(durations)}/{WARMUP_LAYERS}: "
                           f"duration={prev_duration:.2f}s, temp={avg_temp:.2f}°C, feedrate={prev_feed}, trigger frame = {trigger_frame}", base_dir)
                    new_feed = old_feed  # No changes during warmup
                    
                    # Check if warmup complete
                    if len(durations) >= WARMUP_LAYERS:
                        
                        
                        nom_feed = u_history[-1] # for testing we are using the last feed
                        logger.info("Warmup complete. Loading predefined parameters...")
                        logger.info(f"Nominal feed set to: {nom_feed}...")
                        backup_log(f"Nominal feed set to: {nom_feed}...", base_dir)
                        
                        
                        # chris look at nominal feed and when to choose what
                        # Create controller
                        controller = PITemperatureController(
                            setpoint=TEMPERATURE_SETPOINT,
                            nominal_feed=nom_feed,
                            feed_limits=(FEED_LIM_LOW, FEED_LIM_HIGH),
                            max_feed_change=5.0,
                            temp_deadband=10.0
                        )
                        
                        # exclude first few just incase issues
                        time_data = durations[skip_N:]
                        temp_data = temps[skip_N:]
                        
                        logger.info(f"Using time data →  {len(time_data)}: {time_data}")
                        logger.info(f"Using temp data → {len(temp_data)}: {temp_data}")
                        
                        # Set parameters from warmup info
                        predefined_cal['T0'] = np.mean(temp_data)
                        predefined_cal['y_sp'] = np.mean(time_data)
                        predefined_cal['u_initial'] = u_history[-1]
                        
                        results = controller.calibrate_from_step_test(
                            times=np.array(time_data),
                            temps=np.array(temp_data),
                            use_predefined=True,
                            predefined_values=predefined_cal
                        )
                        
                        control_active = True
                        logger.info(f"Controller activated with predefined parameters: {results}, also durations: {durations}. temps: {temps}")
                        backup_log(f"Controller activated with predefined parameters: {results}, also durations: {durations}. temps: {temps}", base_dir)
                        with open(backup_path, 'a') as f:
                            f.write(f'CALIBRATION,{nom_feed},{results}\n')
                    
                    mode = 'warmup'
                
            else:
                # CONTROL MODE - Controller is active
                mode = 'control'
                
                # for simulating a controller      
                #avg_temp = controller.predict_temperature(old_feed)
                #layer_duration = controller.predict_layer_time(old_feed)
                
                durations.append(layer_duration)
                temps.append(avg_temp)
                u_history.append(old_feed)
                
                # Get next feedrate
                new_feed, info = controller.compute_next_feedrate(
                    current_temp=avg_temp,
                    current_time=layer_duration
                )
                
                rate, cmd_feed = FR_logger.define_feed(new_feed)
                
                logger.info(f"command to be applied {round(cmd_feed)}: rate = {rate}")
                backup_log(f"command to be applied {round(cmd_feed)}: rate = {rate}", base_dir)
                
                # Apply the new feed rate
                wcf.set_value(CONTROL_KEY, str(round(cmd_feed))) #changed to str 

                feed_c = new_feed  # Track the actual feed rate, not the command
                
                prev_duration = durations[-1] #durations[-2] if len(durations) >= 2 else durations[-1]
                prev_feed = u_history[-1] #u_history[-2] if len(u_history) >= 2 else u_history[-1]

                    
                # Log control info
                logger.info(f"Current: temp={avg_temp:.1f}°C, duration={prev_duration:.2f}s, error={info['error']:.1f}°C, "
                        f"feed={old_feed:.1f}%, trigger frame = {trigger_frame}")
                backup_log(f"Current: temp={avg_temp:.1f}°C, duration={prev_duration:.2f}s, error={info['error']:.1f}°C, "
                        f"feed={old_feed:.1f}%, trigger frame = {trigger_frame}", base_dir)
                
                # Log feed rate change
                if old_feed != new_feed:
                    logger.info(f"Feedrate changed: {old_feed:.1f} → {new_feed:.1f} (mode: {mode})")
                    backup_log(f"Feedrate changed: {old_feed:.1f} → {new_feed:.1f} (mode: {mode})", base_dir)
                else:
                    logger.info(f"Feedrate remained constant:{new_feed:.1f} (mode: {mode})")
                    backup_log(f"Feedrate remained constant:{new_feed:.1f} (mode: {mode})", base_dir)
                now = datetime.now().isoformat() 

            
            with open(backup_path, 'a') as f:
                f.write(f'{current_time},{last_layer},{layer_duration},{avg_temp},{old_feed},{new_feed},{mode}\n')
            
            
            last_layer = current_layer

        time.sleep(POLL_INTERVAL)
        

if __name__ == '__main__':
    """
    Main orchestration loop for BAAM thermal control system.
    
    Workflow:
        1. Initialize camera capture threads and WCF communication
        2. Poll HMI for layer changes at POLL_INTERVAL
        3. On layer change:
           - Find corresponding thermal frames
           - Process frames for temperature extraction
           - Apply control logic based on current mode
           - Send feedrate adjustment to HMI
        4. Handle mode transitions (calibration -> control)
        
    Modes:
        - Calibration: Collect baseline, apply step test, auto-tune PI
        - Warmup: Stabilize system using predefined parameters  
        - Control: Active PI regulation to maintain setpoint
        
    Command Line Args:
        run_dir: Name for data storage under /mnt/external/
        --calibrate: Enable calibration mode 
        --calibration-layers: Number of layers for calibration/warmup
        
    Raises:
        KeyboardInterrupt: Graceful shutdown on Ctrl+C
        ConnectionError: If WCF service unavailable
    """
    main()
