#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
import argparse
from datetime import datetime

# ─── CONFIG (adjust paths as needed) ──────────────────────────────────────────
RECEIVER_SCRIPT = r'C:\Users\Chris\Documents\baam_test\baam_exc_June\windows_multi_camera_receiver.py'
JETSON_USER     = 'chris'
JETSON_IP       = '192.168.0.5'
VENV_ACTIVATE   = '/home/chris/baam_ir/.ir_6_collect/bin/activate'
REMOTE_SCRIPT   = '/home/chris/baam_ir/data_collect/jetson_main.py'
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch Windows receiver and remote Jetson capture"
    )
    parser.add_argument(
        'run_dir',
        help='Name of the directory under /mnt/external on the Jetson to store this run’s data'
    )
    args = parser.parse_args()

    # 1) Start the Windows receiver
    recv = subprocess.Popen(['python', RECEIVER_SCRIPT])
    print(f"[INFO] Receiver started (PID {recv.pid}).")

    # 2) Small delay so the receiver can bind its sockets
    time.sleep(2)

    # 3) Build the remote command:
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    remote_cmd = (
        f"sudo date -s \"{now_str}\" && "
        f"bash -lc \"source {VENV_ACTIVATE} && "
        f"python3 {REMOTE_SCRIPT} {args.run_dir}\""
    )

    ssh_cmd = [
        'ssh',
        f'{JETSON_USER}@{JETSON_IP}',
        remote_cmd
    ]

    print(f"[INFO] Launching on Jetson (run_dir={args.run_dir})...")
    sender = subprocess.Popen(ssh_cmd)

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