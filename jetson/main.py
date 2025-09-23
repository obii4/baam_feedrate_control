import time
import datetime
import csv
import logging
from wcf_client import WcfClient
#from capture import ThermalCapture
#from processor import process_frame
#from pid import PIDController

# Configuration
SHARED_DIR    = '/home/shared_folder/wet_run_FIN'
LAYER_KEY     = 0    # cmd id    
CONTROL_KEY   = 0    # Extra feedrate override
LOG_FILE      = 'jetson_edge.log'
CSV_PATH      = 'layer_changes.csv'
PID_SETPOINT  = 205.0  # target inter-layer temp in °C
PID_PARAMS    = dict(kp=1.0, ki=0.1, kd=0.01)
POLL_INTERVAL = 0.1  # seconds


def main():
    # ─── Logger Setup ─────────────────────────────────────────────────────────
    logger = logging.getLogger('JetsonEdge')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # ─── WCF Client ────────────────────────────────────────────────────────────
    wcf = WcfClient(log_file=LOG_FILE)

    # ─── Thermal Capture ───────────────────────────────────────────────────────
    #cap = ThermalCapture(output_dir=SHARED_DIR)
    #cap.start()  # begins camera capture in background

    # ─── PID Controller ────────────────────────────────────────────────────────
    #pid = PIDController(**PID_PARAMS)

    # ─── CSV for layer changes ─────────────────────────────────────────────────
    try:
        with open(CSV_PATH, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','old_layer','new_layer'])
    except FileExistsError:
        pass
    csv_file   = open(CSV_PATH, 'a', newline='')
    csv_writer = csv.writer(csv_file)

    # ─── Main polling loop ─────────────────────────────────────────────────────
    last_layer = None
    while True:
        raw = wcf.get_value(int(LAYER_KEY))
        try:
            current_layer = int(raw)
        except (TypeError, ValueError):
            logger.warning(f"Non-int layer value: {raw!r}")
            time.sleep(0.5)
            continue

        if last_layer is None:
            last_layer = current_layer
            logger.info(f"Starting on layer {current_layer}")
        elif current_layer != last_layer:
            # 1) Record timestamp and layers
            ts = datetime.datetime.now().isoformat()
            print(f'detected change! {current_layer}')
            logger.info(f"Layer changed: {last_layer} → {current_layer}")
            csv_writer.writerow([ts, last_layer, current_layer])
            csv_file.flush()

            # 2) Process frames from previous layer
            #frames = cap.frames_for_layer(last_layer)
            #temps  = [process_frame(fid) for fid in frames]

            # 3) Compute PID correction and send once
            #correction = pid.update(temps, setpoint=PID_SETPOINT)
            #wcf.set_value(CONTROL_KEY, correction)

            last_layer = current_layer

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
