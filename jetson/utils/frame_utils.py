
import os
import glob
import csv
import time
import logging
import functools
from datetime import datetime
from typing import List, Dict, Tuple

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
    """Load a .npy file with retry logic for transient I/O errors."""
    for i in range(retries):
        try:
            arr = np.load(path)
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
    """Decorator: swallow any Exception, count skips, return default."""
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
    """Return sorted integer frame IDs from camera_dir/*.npy."""
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
    """Read only specified frame_ids from timestamps.csv under lock."""
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
    Safely read camera_dir/timestamps.csv under a file lock.
    Returns mapping: frame_id -> (local_ts, remote_ts).
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
    """Parse timestamp 'YYYY-MM-DD HH:MM:SS.ssssss' to datetime."""
    # matches format: 2025-07-01 13:15:40.075784
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def find_closest_frame(camera_dir: str, target_time: datetime) -> int:
    """
    Return the frame_id whose local timestamp is closest to target_time.
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
    After `trigger_frame`, grab the next N frames for camera `serial`.
    Returns dicts with keys: frame_id, local_ts, remote_ts, array.
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
        arr = safe_load_npy(os.path.join(cam_dir, f"{fid}.npy"))
        local_ts, remote_ts = ts_map.get(fid, ("", ""))
        results.append({
            "frame_id": fid,
            "local_ts": local_ts,
            "remote_ts": remote_ts,
            "array": arr
        })
    logger.info(f"Loaded {len(results)} frames after trigger {trigger_frame} for serial {serial}")
    return results
