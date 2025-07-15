# BAAM Feedrate Control

This repository contains the code and utilities for closed-loop feedrate control in BAAM LFAM. It bridges a Windows HMI via WCF and a Jetson Nano edge device capturing and processing thermal images.

## Project Structure

```
baam_feedrate_control/
├── jetson/                  # Jetson edge components
│   ├── utils/               # Helper modules
│   │   ├── camera_utils.py  # Video device management, directory & file I/O
│   │   ├── frame_utils.py     # Frame listing, timestamp loading, locking
│   │   └── processor.py       # Hough line detection, clustering, temperature logic
│   ├── thermal_capture.py   # ThermalCapture class: camera streaming & saving
│   ├── pi_controller.py          # ThermalCapture class: camera streaming & saving
│   ├── process_frames.py    # Batch frame loading & processing, debug plots
│   └── wcf_client.py              # WCF client wrapper for HMI get/set commands
├── windows/                          # Windows receiver and launcher scripts
│   └── run_experiment.py    # Launches receiver locally and Jetson main remotely
└── jetson_main.py                 # Orchestrator on Jetson: WCF polling, PI, frame processing
```

## Usage

### 1. Define a run directory

On Windows or local machine, choose a unique name for this run (e.g. `run_20250715_1430`).

### 2. Launch end‑to‑end

On Windows:

```bash
cd windows
python run_experiment.py <RUN_DIR>
# Example:
python run_experiment.py run_20250715_1430
```

This will:

- Start the multi‑camera receiver locally.
- SSH into the Jetson, sync its clock, activate the Python environment, and run `jetson_main.py <RUN_DIR>`.

### 3. Jetson Orchestration

The `jetson_main.py` script will:

- Poll the HMI layer number via WCF (`wcf_client.py`).
- On each layer change:
  - Find the closest thermal frame timestamp (`frame_utils.py`).
  - Load and process the next *N* frames (`process_frames.py`).
  - Compute mean prev-/recent-layer temperatures via Hough line analysis (`processor.py`).
  - Apply a PI correction and send it back to the HMI vis WCF.
- All data, logs, and debug plots are saved under `/mnt/external/<RUN_DIR>/`.

## Configuration

- **JetsonMain** constants at the top of `jetson_main.py`:

  - `WINDOWS_IP`, `PORT_BASE`, `LAYER_KEY`, `CONTROL_KEY`,
  - `PID_SETPOINT`, `PID_PARAMS` (kp/ki)
  - Poll interval, directories, log filenames.

- **ThermalCapture** settings in `thermal_capture.py`:

  - Camera gain mode, resolution, ports.

- **Frame processing** in `process_frames.py`:

  - Number of post-trigger frames `N`, offset `N_offset`.
  - Debug plot directory defaults to `<base>/<serial>/plots/<layer>/`.

## Logs and Outputs

All outputs go under `/mnt/external/<RUN_DIR>/` on the Jetson:

- `jetson_edge.log` — combined logs for orchestrator and WCF client.
- `layer_changes.csv` — timestamped layer transitions.
- `feedrate_changes.csv` — logged feedrate override updates.
- Per-camera directories:
  - `<serial>/` contains:
    - `.npy` thermal frames and `timestamps.csv`
    - `plots/<layer>/` debug PNGs
    - `plots/<layer>/temp_arrays/` saved temperature arrays
