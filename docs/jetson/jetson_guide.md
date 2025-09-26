# Jetson Setup Guide - BAAM Adaptive Feedrate Control System

## Overview
This guide covers the essential network and Python environment setup for the NVIDIA Jetson Nano running the BAAM control system.

## Network Configuration

### Static IP Setup

Configure the Jetson with a static IP on the same subnet as the Windows PC:

```bash
# Using NetworkManager (recommended)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.0.XXX/24
sudo nmcli con mod "Wired connection 1" ipv4.gateway 192.168.0.XXX
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con up "Wired connection 1"
```

**Network Settings:**
- Jetson IP: `192.168.0.XXX`
- Windows PC IP: `192.168.0.XXX`
- Subnet: `255.255.255.0`
- Gateway: `0.0.0.0`

### Verify Network Configuration

```bash
# Check IP address
ip addr show eth0

# Test connection to Windows PC
ping 192.168.0.XXX

# Test WCF service endpoint
curl http://192.168.0.151:XXX/Design_Time_Addresses/BaamHmi/CIBaamInterface/?singleWsdl
```

## Python Virtual Environment Setup

### 1. Install Python and venv

```bash
# Update packages
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev

# Install system dependencies
sudo apt install -y build-essential cmake
```

### 2. Create Virtual Environment

```bash
# Clone the BAAM control repository
cd 
git clone https://github.com/obii4/baam_feedrate_control
cd baam_feedrate_control

# Create virtual environment
python3 -m venv .ir_6_collect

# Activate virtual environment
source .ir_6_collect/bin/activate
```

### 3. Install Required Packages

Use requirements.txt in your project directory:

```txt
# Example... 
numpy==1.19.4
opencv-python==4.5.3.56
matplotlib==3.3.4
scipy==1.5.4
zeep==4.1.0
pyserial
filelock
```

Install with virtual environment activated:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements file
pip install -r requirements.txt
```


## USB Camera Setup

### Install Camera Control Tools

```bash
# Install v4l-utils and uvcdynctrl
sudo apt install -y v4l-utils uvcdynctrl

# Clone PureThermal UVC definitions
cd ~
git clone https://github.com/groupgets/purethermal1-uvc-capture.git

# Load Lepton-specific controls
sudo uvcdynctrl -i ~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml
```

### Configure Camera Controls

The `pt1.xml` file contains Lepton-specific control mappings required for proper thermal camera operation.

**Update path in your control script:**
```python
# In your Python control code, reference the correct path:
PT1_XML_PATH = '~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml'
```

**Make controls persistent (optional):**
```bash
# Copy XML to system location
sudo cp ~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml /usr/share/uvcdynctrl/data/

# Add to startup script
echo "uvcdynctrl -i /usr/share/uvcdynctrl/data/pt1.xml" >> ~/.bashrc
```

### Check Camera Detection

```bash
# List video devices
ls -la /dev/video*

# Check each camera with Lepton controls
v4l2-ctl --list-devices

# List available controls for a camera
v4l2-ctl -d /dev/video0 --list-ctrls-menus
```

Expected: 6 cameras on `/dev/video0`, `/dev/video2`, `/dev/video4`, `/dev/video6`, `/dev/video8`, `/dev/video10`

## Quick Verification

### Test Camera Access

Create `test_cameras.py`:

```python
import cv2
import subprocess

# Load Lepton controls first
subprocess.run(['uvcdynctrl', '-i', '~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml'])

for i in range(6):
    cap = cv2.VideoCapture(i*2)  # 0, 2, 4, 6, 8, 10
    ret, frame = cap.read()
    if ret:
        print(f"Camera {i} (/dev/video{i*2}): OK")
        # Verify thermal data (should be 16-bit for Lepton)
        print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
    else:
        print(f"Camera {i} (/dev/video{i*2}): FAILED")
    cap.release()
```

Run test:
```bash
source venv/bin/activate
python test_cameras.py
```

## Startup Checklist

1. **Network**: Verify static IP is configured
   ```bash
   ip addr show eth0 | grep 192.168.0.XXX
   ```

2. **Windows PC**: Verify connection
   ```bash
   ping -c 4 192.168.0.XXX
   ```

3. **Virtual Environment**: Activate
   ```bash
   source ~/.ir_6_collect/venv/bin/activate
   ```

4. **Cameras**: Verify all 6 detected with Lepton controls
   ```bash
   # Load Lepton controls
   uvcdynctrl -i ~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml
   
   # Check cameras
   ls -la /dev/video* | wc -l  # Should show 12 (6 cameras x 2 devices each)
   ```


## Troubleshooting

**Network Issues:**
- Verify Ethernet cable connected
- Check Windows firewall allows port 8733
- Ensure both machines on same subnet

**Python Package Issues:**
- Make sure virtual environment is activated
- Check Python version: `python --version` (should be 3.6+)
- Reinstall problem package: `pip install --force-reinstall [package]`

**Camera Issues:**
- Check USB connections
- Load Lepton controls: `uvcdynctrl -i ~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml`
- Reboot if cameras not detected
- Try `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo`
- Verify controls loaded: `v4l2-ctl -d /dev/video0 --list-ctrls-menus`
