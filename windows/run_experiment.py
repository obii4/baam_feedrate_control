"""
run_experiment.py - Main launcher for distributed BAAM thermal control experiments.

This script coordinates the complete experimental setup across Windows and Jetson
platforms. It handles receiver startup, time synchronization, remote process
launching with calibration options, and graceful shutdown of all components.

This is the primary entry point for operators to start a thermal control session.

Usage:
    python run_experiment.py run_002 --calibrate --calibration-layers 20              # Calibration mode
    python run_experiment.py run_003 --calibration-layers 20                          # Warmup mode

Typical workflow:
1. Start with calibration to determine PI parameters
2. Run production experiments with tuned parameters
3. Monitor via Windows receiver display
4. Data saved to /mnt/external/<run_dir>/ on Jetson

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""

import subprocess
import time
import signal
import sys
import argparse
from datetime import datetime


# ─── CONFIG (adjust paths as needed) ──────────────────────────────────────────
# Windows-side receiver for thermal display

RECEIVER_SCRIPT = r'C:\Users\Administrator\Documents\obrien_villez_layer\baam_exc_June\windows_multi_camera_receiver.py'

# Jetson SSH connection details
jetson_user = 'chris'              # SSH username (requires passwordless key auth)
jetson_ip = '192.168.0.5'         # Jetson network address

# Jetson environment and script paths
venv_activation = 'source /home/chris/baam_ir/.ir_6_collect/bin/activate'
sender_script_path = '/home/chris/baam_ir/baam_feedrate_control/main_jetson.py'
# ──────────────────────────────────────────────────────────────────────────────


def main():

    """
    Orchestrate distributed BAAM thermal control experiment launch.
    
    Workflow:
        1. Parse command-line arguments for run configuration
        2. Start Windows receiver for thermal visualization
        3. Synchronize Jetson clock with Windows time
        4. Launch Jetson capture/control with appropriate mode
        5. Monitor processes and handle shutdown
        
    Command Line Arguments:
        run_dir: Directory name for data storage (required)
        --calibrate: Enable PI controller calibration mode
        --calibration-layers: Number of layers for calibration/warmup
        
    Modes:
        Normal: Use predefined PI parameters after warmup
        Calibration: Perform step test to auto-tune PI gains
        
    Process Management:
        - Receiver runs as subprocess on Windows
        - Jetson script launched via SSH with nohup
        - Graceful shutdown on Ctrl+C

    """

    parser = argparse.ArgumentParser(
        description="Launch Windows receiver and remote Jetson capture"
    )
    parser.add_argument(
        'run_dir',
        help='Name of the directory under /mnt/external on the Jetson to store this run’s data'
    )

    parser.add_argument(
        '--calibrate', action='store_true',
        help='Enable calibration mode for PI controller'
    )

    parser.add_argument(
        '--calibration-layers', type=int, default=10,
        help='Enable calibration mode for PI controller'
    )
    args = parser.parse_args()

    # 1) Start the Windows receiver
    """
    Starts thermal display receiver as background process
    - shell=True: Enables path resolution through Windows shell
    - Captures PID for later termination
    - Must start before Jetson to receive connections
    """

    recv = subprocess.Popen(['python', RECEIVER_SCRIPT], shell=True)

    print(f"[INFO] Receiver started (PID {recv.pid}).")

    # 2) Small delay so the receiver can bind its sockets
    time.sleep(2)
    

    # 3) Build the remote command:
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    remote_args = [args.run_dir]
    if args.calibrate:
        remote_args.append('--calibrate')
    
    if args.calibration_layers:
        remote_args.extend(['--calibration-layers', str(args.calibration_layers)])

    remote_args_str = ' '.join(remote_args)
    """
    Builds argument string for Jetson script:
    - Always includes run_dir
    - Conditionally adds calibration flags
    - Preserves argument order and formatting
    """
    
    # 3) Build the remote command:
    c_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    ssh_command = f'"%OpenSSH%\ssh.exe" {jetson_user}@{jetson_ip} sudo date --set={c_time!r}; "{venv_activation}; "nohup setsid python3 {sender_script_path} {remote_args_str}""'

    """
    Complex command structure - BUT HAS SYNTAX ERRORS:
    1. Time synchronization via sudo date
    2. Virtual environment activation  
    3. Process detachment with nohup/setsid
    4. Python script with arguments
    """

    print(f"[INFO] Launching on Jetson (run_dir={args.run_dir})...")
    sender = subprocess.Popen(ssh_command, shell = True)

    try:
        # Wait until user Ctrl+C
        sender.wait()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Stopping both receiver and remote script...")
    finally:
        # Cleanup remote process (if still running)
        if sender.poll() is None:
            sender.terminate()
        # Signal the Windows receiver to exit
        recv.send_signal(signal.CTRL_BREAK_EVENT)
        recv.wait()
        print("[DONE] All processes terminated.")

if __name__ == '__main__':
    main()

