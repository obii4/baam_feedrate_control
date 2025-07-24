from jetson.process_frames import process_frames
import os
import numpy as np
from jetson.utils.frame_utils import find_closest_frame
from datetime import datetime

base = '/mnt/external/wet_run_FIN'
prevs, recents = process_frames(
    base_dir=base,
    serial='001b000d',
    trigger_frame=8440,
    N=30
)

# test = np.load('/home/shared_folder/wet_run_FIN/001b000d/plots/8440/temp_arrays/8468_prev_region.npy')



### test loading frames from timestamps
#base = '/mnt/external/wet_run_FIN/001b000d' 

#fake_time = datetime(2025, 7, 1, 13, 15, 40)
# print(datetime.now())
#try:
#     closest = find_closest_frame(base, fake_time)
#     print(f"Closest frame to {fake_time.isoformat()} is {closest}")
#except Exception as e:
#     print(f"Error: {e}")




