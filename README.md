# BAAM Feedrate Control System

Real-time closed-loop thermal feedback control for Big Area Additive Manufacturing (BAAM) using distributed edge computing and multi-camera thermal imaging.

## Overview

### Problem Statement

Large-scale additive manufacturing faces critical challenges in maintaining consistent inter-layer adhesion temperature across extended print times. In BAAM systems, where individual layers can take minutes to complete and parts may require hundreds of layers, thermal management becomes paramount. Traditional open-loop control fails to account for:

- Heat accumulation in the part over time
- Varying cooling rates based on geometry and layer time
- Environmental factors affecting heat dissipation
- Material-specific thermal requirements for optimal bonding

This system addresses these challenges by implementing real-time thermal monitoring and adaptive feedrate control, maintaining optimal layer adhesion temperature throughout the entire build process.

### Solution Architecture

The system employs a distributed computing architecture that separates real-time control (Jetson edge device) from operator interface (Windows HMI) while maintaining synchronized operation. Six FLIR Lepton thermal cameras provide comprehensive coverage of the deposition zone, capturing thermal data at 9 Hz per camera. The control algorithm processes this data in real-time, extracting temperature measurements from the previous layer (primary control signal) and current deposition (predictive information).

A PI controller with Internal Model Control (IMC) tuning adjusts the feedrate override command based on temperature deviation from setpoint. The system supports two operational modes:
1. **Calibration Mode**: Performs automated step testing to identify process dynamics and tune controller gains
2. **Normal Mode**: Uses calibrated parameters for continuous temperature regulation

### Key Innovations

- **Layer-Synchronous Control**: Control actions triggered by actual layer completion events from the HMI, not time-based
- **Distributed Processing**: Separates visualization (Windows) from control computation (Jetson) for optimal resource utilization
- **Auto-Tuning Capability**: Built-in system identification eliminates manual tuning requirements

## Project Structure

```
baam_feedrate_control/
├── jetson/
│   ├── jetson_main.py           # Main orchestrator
│   ├── controller.py            # PI controller implementation
│   ├── thermal_capture.py       # Camera management
│   ├── process_frames.py        # Temperature extraction
│   ├── wcf_client.py           # HMI communication
│   └── utils/
│       ├── camera_utils.py     # Device management
│       ├── frame_utils.py      # Frame I/O
│       ├── processor.py        # Line detection
│       └── utils.py            # Helper functions
├── windows/
│   ├── run_experiment.py                 # Launch script
│   └── windows_multi_camera_receiver.py  # Display
└── README.md
```

## System Architecture

### Hardware Configuration

```
                              BAAM System Architecture
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Windows HMI PC                                                         │
    │  ┌────────────────┐                                                     │
    │  │  WCF Service   │  Port 8733                                          │
    │  │                │  ◄─────── GetValue(layer_number)                    │
    │  │  - Layer Info  │  ◄─────── SetValue(feedrate_override)               │
    │  │  - Feed Control│                                                     │
    │  └────────────────┘                                                     │
    │                                                                         │
    │  ┌────────────────┐                                                     │
    │  │  TCP Receiver  │  Ports 12345-12350                                  │
    │  │                │  ◄─────── Thermal Frame Streams (6x)                │
    │  │  - Display Grid│                                                     │
    │  └────────────────┘                                                     │
    └────────────┬────────────────────────────────────────────────────────────┘
                 │
                 │  Ethernet 192.168.0.x
                 │  (Isolated Network)
                 │
    ┌────────────▼────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  NVIDIA Jetson Nano (Edge Device)                                       │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │              Main Orchestrator (jetson_main.py)                 │    │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐             │    │
    │  │  │ WCF Client   │  │ PI Controller│  │   Frame    │             │    │
    │  │  │              │  │              │  │ Processing │             │    │
    │  │  │ Poll Layer   │  │ IMC Tuning   │  │            │             │    │
    │  │  │ Send Override│  │ Anti-Windup  │  │ Hough Lines│             │    │
    │  │  └──────────────┘  └──────────────┘  └────────────┘             │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │           Thermal Capture Module (6 threads)                    │    │ 
    │  │  ┌──────────────┐ ┌────────────┐ ... ┌──────────────┐           │    │
    │  │  │  Camera 0    │ │  Camera 1  │     │   Camera 5   │           │    │
    │  │  │/dev/video0   │ │/dev/video2 │     │/dev/video10  │           │    │
    │  │  │              │ │            │     │              │           │    │
    │  │  │ Thread 0     │ │ Thread 1   │     │ Thread 5     │           │    │
    │  │  └──────────────┘ └────────────┘     └──────────────┘           │    │
    │  └──────────────────────┬──────────────────────────────────────────┘    │
    │                         │                                               │
    └─────────────────────────┼───────────────────────────────────────────────┘
                              │
                              │  USB 2.0/3.0
                              │
    ┌─────────────────────────▼─────────────────────────────────────────────┐
    │  FLIR Lepton 3.5 Thermal Cameras (x6)                                 │
    │  - Resolution: 160x120 pixels                                         │
    │  - Frame Rate: 9 Hz                                                   │
    │  - Thermal Range: 0-400°C (Low Gain Mode)                             │
    │  - Interface: UVC (USB Video Class)                                   │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
```

### Software Components

#### Windows Components

**WCF Service Interface**
- Provides SOAP-based communication with BAAM HMI
- Endpoints for reading process parameters and sending control commands
- Key parameters include feedrate override, layer number, and run status

**Multi-Camera Receiver**
- TCP server accepting connections on ports 12345-12350
- Real-time thermal visualization with false-color mapping
- Grid display showing all 6 camera feeds simultaneously
- Frame rate monitoring and connection status

#### Jetson Components

**Main Orchestrator**
- Central control loop operating at 1 Hz
- Layer change detection via WCF polling
- Orchestrates frame processing and control calculations
- Manages operational modes (calibration/warmup/control)
- Data logging and error recovery

**Thermal Capture Module**
- Multi-threaded camera management (1 thread per camera)
- Raw frame acquisition at 9 Hz per camera
- Frame serialization and TCP streaming
- Timestamp synchronization across cameras
- Automatic device discovery via udev

**Frame Processing Pipeline**
- Trigger frame identification based on layer change time
- Post-trigger frame batch loading (typically 30 frames)
- Temperature extraction using Hough transform or pixel sampling
- Line detection and clustering for layer identification

**PI Temperature Controller**
- First-Order Plus Dead Time (FOPDT) process model
- Internal Model Control (IMC) tuning methodology
- Anti-windup protection and rate limiting
- Deadband implementation to reduce actuator wear

## Control System Mathematics

### Process Model

The system identifies a First-Order Plus Dead Time (FOPDT) model:

```
        K·e^(-θs)
G(s) = -----------
         τs + 1
```

Where:
- K: Process gain (°C per % feedrate)
- τ: Time constant (seconds)
- θ: Dead time (seconds)
- s: Laplace variable

### Temperature Dynamics

The discrete-time temperature model relates feedrate to temperature:

```
T[k] = T₀ + K_temp × (u[k] - u_nom)
```

Where:
- T[k]: Temperature at layer k (°C)
- T₀: Baseline temperature (°C)
- K_temp: Temperature gain (°C/%)
- u[k]: Feedrate at layer k (%)
- u_nom: Nominal feedrate (%)

### PI Control Law

The controller implements proportional-integral control with enhancements:

```
Error:    e[k] = T_sp - T[k]
P-term:   P[k] = Kp × e[k]
I-term:   I[k] = I[k-1] + Ki × e[k]
Output:   u[k] = u_nom + P[k] + I[k]
```

### IMC Tuning

Internal Model Control provides optimal gain selection:

```
Kp = τ / (K × (λ + θ))
Ki = Kp / τ
```

Where λ is the closed-loop time constant, typically chosen as λ = (0.5 to 2.0) × θ

### Control Enhancements

**Anti-Windup**: Prevents integral accumulation during saturation
- Integration stops when output hits limits
- Prevents overshoot after constraint periods

**Rate Limiting**: Constrains feedrate change per layer
- Maximum change: ±10% per layer
- Prevents mechanical stress on extruder

**Deadband**: Reduces actuator cycling
- No action if |error| < 5°C
- Extends equipment lifetime

## System Identification

### Step Test Procedure

The calibration mode performs automated system identification through step testing:

1. **Baseline Collection**: Gather steady-state data for N layers
2. **Step Application**: Apply feedrate change (typically -20%)
3. **Response Measurement**: Record temperature and time response
4. **Parameter Calculation**: Extract K, τ, and θ from response
5. **Gain Computation**: Calculate PI gains using IMC rules

### Time Constant Estimation

The system uses the 63.2% rise time method for first-order systems:
- Identify temperature change magnitude
- Find time to reach 63.2% of final value
- This time equals the process time constant τ

## Installation

### Prerequisites

- NVIDIA Jetson Nano (4GB RAM recommended)
- 6× FLIR Lepton 3.5 thermal cameras
- Windows 10/11 with .NET Framework 4.7.2+
- Python 3.8+ on both platforms
- Gigabit Ethernet network

### Jetson Setup

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-venv v4l-utils uvcdynctrl lsof
   ```

2. Create Python environment:
   ```bash
   python3 -m venv /home/chris/baam_ir/.ir_6_collect
   source /home/chris/baam_ir/.ir_6_collect/bin/activate
   pip install numpy opencv-python zeep filelock matplotlib scipy
   ```

3. Configure camera permissions:
   ```bash
   sudo usermod -a -G video $USER
   ```

### Windows Setup

1. Install Python 3.8 or later
2. Install required packages:
   ```bash
   pip install numpy opencv-python zeep matplotlib
   ```

3. Configure firewall for ports 12345-12350 and 8733

### Network Configuration

Configure static IPs:
- Windows: 192.168.0.151
- Jetson: 192.168.0.5

Set up SSH key authentication for passwordless access.

## Usage

### Basic Operation

Launch a standard control session:
```bash
python windows/run_experiment.py run_001
```

### Calibration Mode

Perform system identification and auto-tuning:
```bash
python windows/run_experiment.py calibration_001 --calibrate --calibration-layers 20
```

### Configuration Parameters

Key parameters in jetson_main.py:
- `TEMPERATURE_SETPOINT`: Target temperature (°C)
- `FEED_LIM_LOW`: Minimum feedrate override (%)
- `FEED_LIM_HIGH`: Maximum feedrate override (%)
- `N_POST_FRAMES`: Frames to analyze per layer
- `GAIN_MODE`: Camera gain setting (1=high temp, 2=low temp)

## Data Output

Each run creates a directory structure:

```
/mnt/external/{run_dir}/
├── jetson_edge.log          # Main process log
├── backup.txt               # Control decision log
├── {camera_serial}/         # Per-camera data
│   ├── timestamps.csv       # Frame timing
│   └── frame_XXXX.npy      # Thermal arrays
```

## Performance Metrics

- **Control Frequency**: 1 Hz (layer-synchronous)
- **Frame Rate**: 9 Hz per camera (54 Hz total)
- **Processing Latency**: <500ms per frame batch
- **Temperature Accuracy**: ±2°C
- **Network Bandwidth**: ~2.1 MB/s sustained
- **Typical Correction**: 10-20% feedrate adjustment per layer

## Safety Features

- Hard limits on feedrate adjustment (0-175%)
- Rate limiting prevents sudden changes
- Temperature deadband reduces actuator cycling
- Manual override always available through HMI
- Automatic fallback to nominal on sensor failure

## Troubleshooting

### Common Issues

**Camera Not Detected**
- Reset USB devices
- Check power to USB hub
- Verify udev rules

**WCF Connection Failed**
- Verify Windows firewall settings
- Check WCF service is running
- Confirm network connectivity

**Invalid Temperature Readings**
- Check camera gain mode
- Verify mounting angle
- Calibrate pixel coordinates if using simple mode


## License

[Specify your license]

## Authors

[Your name and contact information]

## Acknowledgments

- FLIR Lepton SDK and PureThermal boards
- NVIDIA Jetson platform


## Citation

- add this
}
```