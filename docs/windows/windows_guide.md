# Windows Setup Guide - BAAM Adaptive Feedrate Control System

## Overview
This guide covers the Windows PC setup for the BAAM Adaptive Feedrate Control System, which handles the WCF service interface and thermal data visualization. It is to be used if the experiments have not been conducted on PC before.

## System Requirements

### Minimum Requirements
- Gigabit Ethernet adapter
- Administrator privileges

## Installation Steps

### 1. Network Configuration

**Recommended Settings:**
```
Do not modify BAAM HMI IP as it will mess up the PLC. Configure Jetson IP to be on same subnet.
```

**Configuration Steps:**
1. Open Network and Sharing Center
2. Change adapter settings
3. Right-click Ethernet adapter → Properties
4. Select IPv4 → Properties
5. Identify static IP


**Firewall Rules Required:**
- Port 8733 (TCP) - WCF Service
- Ports 12345-12350 (TCP) - Thermal camera streams
- Port 22 (TCP) - SSH to Jetson (optional, for remote management)

**Adding Firewall Rules:**
1. Open Windows Defender Firewall with Advanced Security
2. Create new Inbound Rule for each port
3. Allow TCP connections
4. Apply to all network profiles

### 2. .NET Framework Installation

The WCF service requires .NET Framework 4.7.2 or higher.

**Download:**
- Download from: https://dotnet.microsoft.com/en-us/download/dotnet-framework/net472
- Choose "Offline installer" for systems without internet access
- Filename: `NDP472-KB4054530-x86-x64-AllOS-ENU.exe` (66.8 MB)

**Installation:**
1. Run the installer as Administrator
2. Follow installation prompts
3. Restart if prompted

**Verification:**
- Open Command Prompt as Administrator
- Run: `reg query "HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" /v Release`
- Value should be 461808 or higher (indicating 4.7.2+)

### 3. Python Installation

**Install Python 3.8:**
1. Navigate to the Python installation folder
2. Run `python-3.8.2-amd64.exe`
3. **IMPORTANT**: Check "Add Python to PATH"
4. Select "Install Now"
5. Verify installation: `python --version`

**Install Required Packages:**

Navigate to the folder containing `.whl` files and install in order:

```cmd
cd [path_to_wheel_files]

pip install all .whl
```

### 4. WCF Service Setup

**Test WCF Communication:**
1. Navigate to `WCFTest/BaamWCFCommunication/bin/Debug` directory
2. Run `WCFTest.exe`
3. Service end point
4. Verify available methods:
   - GetValue(commandId)
   - SetValue(commandId, value)

**Service Commands:**
- Command 0: Feedrate override (-100 to +100%)
- Command 8: Current layer number
- Command 2: Run status


### 5. SSH Access to Jetson (Optional)

For remote management of the Jetson device:

**Windows Options:**
- Use PowerShell SSH client (Windows 10 1809+): `ssh user@192.168.0.150`
- Or install PuTTY for GUI-based SSH access

## Verification Steps

### 1. Network Connectivity
```cmd
ping 192.168.0.XXX  # Jetson IP
```

### 2. WCF Service
- Launch WCFTestClient.exe
- Connect to service endpoint
- Test GetValue(8) for layer number

### 3. Python Environment
```cmd
python --version
pip list  # Verify all packages installed
```

### 4. Firewall Rules
```cmd
netstat -an | findstr "8733"
netstat -an | findstr "12345"
```

### 5. SSH Connection (Optional)
```cmd
ssh user@192.168.0.150  # Replace 'user' with Jetson username
```

## Startup Procedure

1. Ensure Jetson is powered and on network
2. Start WCF service (if not running as Windows service)
4. Verify communication with BAAM HMI
5. Begin adaptive control operation

## Troubleshooting

**WCF Connection Issues:**
- Verify .NET Framework 4.7.2+ installed
- Check firewall allows port 8733
- Use WCFTestClient.exe for diagnostics

**Python Import Errors:**
- Ensure Python added to PATH
- Verify all .whl files installed in correct order
- Check Python version compatibility (3.8)

**Network Issues:**
- Confirm static IP configuration
- Verify subnet matches Jetson
- Check Ethernet cable connection

**No Thermal Display:**
- Verify ports 12345-12350 open in firewall
- Check Jetson is streaming data
- Review receiver script for errors

**SSH Connection Failed:**
- Verify port 22 open in firewall
- Check Jetson SSH service is running
- Confirm correct username/password

