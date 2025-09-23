"""
frame_utils.py - Frame data management utilities for thermal capture system.

This module provides robust file I/O operations for thermal frame data,
including timestamp synchronization, frame lookup, and thread-safe access
to shared CSV files. It handles the coordination between frame capture
threads and processing threads.

Core Functionality:
    - Thread-safe timestamp CSV reading with file locking
    - Frame ID to timestamp mapping
    - Temporal frame lookup (find frame nearest to target time)
    - Robust .npy loading with retry logic
    - Batch frame loading for processing pipelines

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""


import os
import glob
import csv
import time
import logging
import functools
from datetime import datetime
from typing import List, Dict, Tuple
from datetime import timedelta

import numpy as np

from filelock import FileLock, Timeout

# ─── Logger Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger('JetsonMain')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ─── Robust loader ─────────────────────────────────────────────────────────────
def safe_load_npy(path: str, retries: int = 3, delay: float = 0.5) -> np.ndarray:
    """
    Load .npy file with retry logic for handling transient I/O errors.
    
    Implements exponential backoff for reading numpy arrays from disk,
    handling cases where files are being written concurrently by capture
    threads. Also applies 180-degree rotation for camera orientation correction.
    
    Args:
        path: Full path to .npy file
        retries: Maximum number of load attempts
        delay: Base delay between retries in seconds
        
    Returns:
        np.ndarray: Loaded and rotated thermal array
        
    Raises:
        IOError: If file cannot be loaded after all retries
        
    Implementation:
        - Attempts load up to 'retries' times
        - Waits 'delay' seconds between attempts
        - Rotates array 180° (k=2) for camera mounting orientation
        
        
    Note:
        Rotation (np.rot90 k=2) corrects for inverted camera mounting.
        Adjust k value if camera orientation changes.
    """

    for i in range(retries):
        try:
            arr = np.load(path)
            
            # Compensates for inverted camera mounting on BAAM system
            arr = np.rot90(arr, k=2)
            logger.debug(f"Loaded .npy frame: {path}")
            return arr
        except (OSError, IOError) as e:
            logger.warning(f"Retry {i+1}/{retries} loading {path}: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"Failed to load .npy after {retries} retries: {path}")
                raise

# ─── Error‐handling decorator ──────────────────────────────────────────────────
SKIPPED_FRAMES = 0

def catch_errors(default=None):
    """
    Decorator to handle exceptions gracefully and track failure rate.
    
    Wraps functions to catch all exceptions, log them, and return a
    default value instead of crashing. Maintains global counter of
    skipped frames for monitoring processing health.
    
    Args:
        default: Value to return when exception occurs (None if not specified)
        
    Returns:
        Decorated function that swallows exceptions
        
    Global Effects:
        Increments SKIPPED_FRAMES counter on each error
        
    Use Cases:
        - Batch processing where individual failures shouldn't stop pipeline
        - Real-time processing where missing data is preferable to crashes
        
    Warning:
        Can mask programming errors - use judiciously and check logs
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global SKIPPED_FRAMES
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                SKIPPED_FRAMES += 1
                logger.warning(f"Skipped {fn.__name__} due to error: {e}")
                return default
        return wrapper
    return decorator

# ─── Frame I/O ────────────────────────────────────────────────────────────────

def list_saved_frame_ids(camera_dir: str) -> List[int]:
    """
    Extract sorted list of frame IDs from saved .npy files.
    
    Scans camera directory for frame files and extracts numeric IDs
    from filenames. Used to determine available frames for processing.
    
    Args:
        camera_dir: Path to camera-specific directory
        
    Returns:
        List[int]: Sorted frame IDs in ascending order
        
    File Pattern:
        Expects files named: {frame_id}.npy
        Example: 1500.npy → frame_id = 1500
        
    Example:
        >>> ids = list_saved_frame_ids("/data/001b000d")
        >>> print(f"Frames available: {ids[0]} to {ids[-1]}")
        
    Note:
        Ignores non-numeric filenames and non-.npy files.
        Returns empty list if directory doesn't exist.
    """

    paths = glob.glob(os.path.join(camera_dir, "*.npy"))
    ids = []
    for p in paths:
        name = os.path.basename(p)
        try:
            fid = int(os.path.splitext(name)[0])
            ids.append(fid)
        except ValueError:
            continue
    ids_sorted = sorted(ids)
    logger.debug(f"Found frame IDs in {camera_dir}: {len(ids_sorted)}")
    return ids_sorted


def load_timestamps_list(camera_dir: str, frame_ids: List[int]) -> Dict[int, Tuple[str,str]]:
    """
    Load timestamps for specific frame IDs with file locking.
    
    Efficiently reads only requested frame timestamps from CSV file,
    using file locks to coordinate with concurrent writers. Optimized
    for selective loading rather than full file parsing.
    
    Args:
        camera_dir: Path to camera directory containing timestamps.csv
        frame_ids: List of frame IDs to load timestamps for
        
    Returns:
        Dict mapping frame_id to (local_timestamp, remote_timestamp)
        
    CSV Format:
        frame_id,local_timestamp,remote_timestamp
        1500,2025-01-15 10:30:45.123456,2025-01-15 10:30:45.234567
        
    Locking:
        Uses FileLock on timestamps.csv.lock for thread safety
        Prevents reading while capture thread is writing
        
    Performance:
        Stops reading once all requested IDs are found
        More efficient than load_timestamps_locked() for selective access
    """

    ts_path = os.path.join(camera_dir, "timestamps.csv")
    mapping: Dict[int, Tuple[str,str]] = {}
    if not os.path.exists(ts_path) or not frame_ids:
        return mapping
    lock = FileLock(ts_path + ".lock")
    needed = set(frame_ids)
    with lock:
        with open(ts_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                try:
                    fid = int(row[0])
                except ValueError:
                    continue
                if fid in needed:
                    mapping[fid] = (row[1], row[2])
                    if len(mapping) == len(needed):
                        break
    logger.debug(f"Loaded timestamps for IDs {frame_ids}: {list(mapping.keys())}")
    return mapping

def load_timestamps_locked(camera_dir: str) -> Dict[int, Tuple[str,str]]:
    """
    Load all timestamps from CSV with thread-safe file locking.
    
    Reads entire timestamps.csv file under lock protection, handling
    malformed rows gracefully. Used when full timestamp history is needed.
    
    Args:
        camera_dir: Path to camera directory
        
    Returns:
        Dict mapping all frame_ids to (local_ts, remote_ts) tuples
        
    Error Handling:
        - Timeout: Returns empty dict if lock unavailable after 5 seconds
        - Malformed rows: Logs warning and skips
        - Missing file: Returns empty dict with warning
        - I/O errors: Logs error and returns partial data
        
    Lock Timeout:
        5 seconds default - prevents indefinite blocking
        May need adjustment for high-contention scenarios
        
        
    Note:
        For selective loading, prefer load_timestamps_list()
        This loads entire file even if only few timestamps needed
    """
    ts_path = os.path.join(camera_dir, "timestamps.csv")
    mapping: Dict[int, Tuple[str,str]] = {}
    if not os.path.exists(ts_path):
        logger.warning("Timestamps file not found: %s", ts_path)
        return mapping

    lock = FileLock(ts_path + ".lock")
    try:
        # wait up to 5s to acquire the lock
        with lock.acquire(timeout=5):
            try:
                with open(ts_path, newline="") as f:
                    reader = csv.reader(f)
                    for lineno, row in enumerate(reader, start=1):
                        if len(row) != 3:
                            logger.warning(
                                "Skipping malformed row %d in %s: %r",
                                lineno, ts_path, row
                            )
                            continue
                        try:
                            fid = int(row[0])
                        except ValueError:
                            logger.warning(
                                "Invalid frame ID on line %d in %s: %r",
                                lineno, ts_path, row[0]
                            )
                            continue

                        # no parsing of timestamp here, leave as strings
                        mapping[fid] = (row[1], row[2])
            except (OSError, IOError) as e:
                logger.error("I/O error reading %s: %s", ts_path, e)
    except Timeout:
        logger.error("Timeout acquiring lock for %s", ts_path)
    except Exception as e:
        # catch-all to avoid breaking your pipeline
        logger.error("Unexpected error loading %s: %s", ts_path, e)

    if not mapping:
        logger.warning("No valid timestamps loaded from %s", ts_path)
    return mapping

def parse_iso(ts: str) -> datetime:
    """
    Parse timestamp string to datetime object.
    
    Converts timestamp from capture format to Python datetime for
    temporal calculations and comparisons.
    
    Args:
        ts: Timestamp string in format 'YYYY-MM-DD HH:MM:SS.ssssss'
        
    Returns:
        datetime: Parsed timestamp with microsecond precision
        
    Format:
        Expected: '2025-07-01 13:15:40.075784'
        Year-Month-Day Hour:Minute:Second.Microseconds
        
    Example:
        >>> dt = parse_iso('2025-01-15 10:30:45.123456')
        >>> print(dt.microsecond)  # 123456
        
    Raises:
        ValueError: If timestamp doesn't match expected format
        
    Note:
        Microsecond precision important for frame correlation
        Format must match exactly (no timezone info)
    """

    # matches format: 2025-07-01 13:15:40.075784
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def find_closest_frame(camera_dir: str, target_time: datetime) -> int:
    """
    Find frame ID with timestamp closest to target time.
    
    Essential function for synchronizing thermal data with layer changes.
    Searches all available frames to find best temporal match.
    
    Args:
        camera_dir: Path to camera directory with timestamps.csv
        target_time: Target datetime to match (layer change time)
        
    Returns:
        int: Frame ID of closest matching frame
        
    Raises:
        ValueError: If no timestamps available or all invalid
        
    Algorithm:
        1. Load all timestamps from CSV
        2. Parse each timestamp to datetime
        3. Calculate time difference for each frame
        4. Return frame with minimum absolute difference
        
    Optimization:
        Skips frames more than 240 seconds before target
        (Assumes layer changes don't reference very old frames)
        
    Example:
        >>> layer_change_time = datetime(2025, 1, 15, 10, 30, 45)
        >>> frame_id = find_closest_frame("/data/001b000d", layer_change_time)
        >>> # Returns frame captured nearest to 10:30:45
        
    Note:
        Critical for control loop - inaccurate matching causes wrong
        temperature readings and control errors
    """
    ts_map = load_timestamps_locked(camera_dir)
    if not ts_map:
        raise ValueError(f"No timestamps in {camera_dir}")

    best_id = None
    best_diff = None
    for fid, (local_ts, _) in ts_map.items():
        try:
            dt = parse_iso(local_ts)
        except Exception:
            continue

        if dt < target_time - timedelta(seconds=240):
            continue
        
        diff = abs((dt - target_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_id = fid
    if best_id is None:
        raise ValueError("No valid timestamps found")
    return best_id

def load_post_frames(
    base_dir: str,
    serial: str,
    trigger_frame: int,
    N: int
) -> List[Dict]:
    """
    Load N consecutive frames after trigger for temperature analysis.
    
    Primary interface for batch frame loading after layer change detection.
    Loads frame data and associated timestamps for processing pipeline.
    
    Args:
        base_dir: Root data directory (/mnt/external/run_XXX)
        serial: Camera serial number
        trigger_frame: Frame ID at layer change
        N: Number of frames to load after trigger
        
    Returns:
        Tuple of (frames_list, actual_trigger_frame) where frames_list contains:
            - frame_id: Integer frame identifier
            - local_ts: Jetson capture timestamp
            - remote_ts: Windows receiver timestamp
            - array: Rotated thermal numpy array
            
    Data Structure:
        [
            {
                'frame_id': 1501,
                'local_ts': '2025-01-15 10:30:45.123456',
                'remote_ts': '2025-01-15 10:30:45.234567',
                'array': np.ndarray(shape=(120, 160))
            },
            ...
        ]
        
    Error Recovery:
        - Missing trigger frame: Starts from frame 0 with warning
        - Load failures: Skips problematic frames, continues with rest
        - Returns fewer than N frames if insufficient data
        
        
    Note:
        Frames are loaded sequentially by ID, not by time
        Array rotation applied during load (180° for camera orientation)
    """
    cam_dir = os.path.join(base_dir, serial)
    logger.debug(f"Loading post frames from: {cam_dir}")

    # 1) List and pick the right frame IDs
    all_ids = list_saved_frame_ids(cam_dir)
    try:
        start_idx = all_ids.index(trigger_frame) + 1
    except ValueError:
        start_idx = 0
        logger.warning(f"Trigger frame {trigger_frame} not found, starting at 0")
    selected_ids = all_ids[start_idx : start_idx + N]
    logger.debug(f"Selected frame IDs: {selected_ids}")

    # 2) Load just those timestamps
    ts_map = load_timestamps_list(cam_dir, selected_ids)
    
    results = []
    for fid in selected_ids: #ids[idx:idx+N]:
        try:
            arr = safe_load_npy(os.path.join(cam_dir, f"{fid}.npy"))
            local_ts, remote_ts = ts_map.get(fid, ("", ""))
            
            if arr is not None:
            
                results.append({
                    "frame_id": fid,
                    "local_ts": local_ts,
                    "remote_ts": remote_ts,
                    "array": arr
                })
                
        except Exception as e:
            logger.warning(f"Error loading frame {fid}: {e}")
            continue
        
    logger.info(f"Loaded {len(results)} frames after trigger {trigger_frame} for serial {serial}")
    return results, trigger_frame
