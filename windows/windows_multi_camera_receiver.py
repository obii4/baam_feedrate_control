"""
windows_multi_camera_receiver.py - Multi-camera thermal stream receiver and display.

This script receives thermal frame streams from multiple Jetson-connected FLIR
cameras over TCP, applies colormap visualization, and displays them in a grid
layout. It serves as the Windows-side monitoring interface for the BAAM thermal
control system.

Architecture:
    - One TCP socket per camera (parallel reception)
    - Thread-safe frame buffer with latest frame per camera
    - Real-time display with OpenCV
    - Automatic grid layout for multiple cameras

Network Protocol:
    - TCP streams on ports 12345-12350
    - Message format: [4-byte length][pickled (serial, frame)]
    - Big-endian length prefix

Author: Chris O'Brien   
Date Created: 09-23-25
Version: 1.0
"""


import socket
import threading
import struct
import pickle
import cv2
import numpy as np


# Network configuration
PORT_BASE = 12345     # Starting port for camera connections
CAMERA_COUNT = 6      # Expected number of cameras
# Note: Each camera uses PORT_BASE + index (12345, 12346, ...)


# Thread-safe frame storage
frames = {}                 # Dict mapping serial -> latest processed frame
lock = threading.Lock()     # Protects frames dict during read/write

def receive_camera_stream(port):
    """
    Thread worker for receiving and processing single camera stream.
    
    Establishes TCP server socket, accepts connection from Jetson,
    and continuously receives thermal frames. Applies visualization
    and updates global frame buffer.
    
    Args:
        port: TCP port number to listen on
        
    Protocol:
        1. Listen for incoming connection
        2. Receive 4-byte header with message length
        3. Receive message body (pickled data)
        4. Unpack serial number and raw frame
        5. Apply normalization and colormap
        6. Update global frames dict
        
    Frame Processing:
        - Normalize to 0-255 range for display
        - Convert to uint8
        - Apply INFERNO colormap (black->red->yellow->white)
        
    Error Handling:
        - Continues on unpacking errors (corrupted frames)
        - Exits thread on connection loss
        - Thread is daemon, will terminate with main program

    Message framing protocol

    TCP Message Structure:
        [4 bytes] Message length (network byte order, big-endian)
        [N bytes] Pickled tuple: (camera_serial, numpy_array)

    Reassembly:
        1. Accumulate data until 4-byte header complete
        2. Parse length: struct.unpack("!I", header)[0]
        3. Accumulate data until full message received
        4. Unpickle and process
        
    This handles TCP stream fragmentation correctly.

    Note:
        Blocks waiting for connection - no timeout
        One thread per camera for parallel processing
    """

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)
    print(f"[INFO] Listening on port {port}")
    conn, _ = server_socket.accept()
    print(f"[INFO] Connected on port {port}")




    # Data accumulation pattern
    data = b""  # Buffer for incomplete messages

    payload_size = struct.calcsize("!I")

    while True:
        try:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet: # Connection closed
                    return
                data += packet

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!I", packed_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                serial, frame = pickle.loads(frame_data)
                therm_conv = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
                therm_conv = np.uint8(therm_conv)
                therm_conv = cv2.applyColorMap(therm_conv, cv2.COLORMAP_INFERNO)
                # Thread-safe frame updates
                with lock:
                    frames[serial] = therm_conv
            except Exception as e:
                print(f"[ERROR] Failed to unpack frame: {e}")

        except Exception as e:
            print(f"[ERROR] Connection lost on port {port}: {e}")
            break

# Start receiver threads
for i in range(CAMERA_COUNT):
    threading.Thread(target=receive_camera_stream, args=(PORT_BASE + i,), daemon=True).start()

# Display loop
"""
Display Strategy:
    1. Lock and copy current frames (minimize lock time)
    2. Sort by serial for consistent ordering
    3. Pad with black frames if cameras missing
    4. Apply rotation correction
    5. Create horizontal grid
    6. Display with OpenCV
    
Performance Considerations:
    - Lock held briefly (just for dict copy)
    - Processing done outside lock
    - 1ms waitKey for ~1000 FPS max (practically limited by cameras)
    
User Controls:
    - 'q' key: Quit application
"""

while True:
    with lock:
        if frames:
            ordered_keys = sorted(frames.keys())
            frame_list = [frames[k] for k in ordered_keys]
            while len(frame_list) < CAMERA_COUNT:
                dummy = np.zeros_like(frame_list[0])
                frame_list.append(dummy)
            frame_list = [np.rot90(f, k=2) for f in frame_list]
            grid = np.hstack(frame_list)
            cv2.imshow("Camera Grid", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
