import time
import logging
import functools
from requests import Session
from zeep import Client
from zeep.transports import Transport



WSDL_URL = "http://192.168.0.2:8000/?Wsdl"

# Map of command IDs to human‐readable names
COMMAND_NAMES = {
    0:  "Extra feedrate override",
    1:  "Extra extruder override",
    2:  "Run status",
    3:  "CNC mode",
    4:  "Program name",
    5:  "Total run time",
    6:  "Remaining run time",
    7:  "Remaining layer time",
    8:  "Current layer number",
    9:  "Total layers",
    10: "Dwell remaining",
    11: "Positions",
    12: "Extruder speed",
}

def throttle(seconds: float):
    """
    Decorator to enforce a minimum interval between calls to a method.
    """
    def decorator(fn):
        last_call = {"t": 0.0}
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            elapsed = time.time() - last_call["t"]
            if elapsed < seconds:
                time.sleep(seconds - elapsed)
            result = fn(self, *args, **kwargs)
            last_call["t"] = time.time()
            return result
        return wrapper
    return decorator

class WcfClient:
    """
    Wrapper for the BAAM HMI WCF service, with hard‐coded endpoints.
    Provides:
      - initialization using built‐in WSDL_URL and SERVICE_ADDRESS
      - logging of calls
      - throttling to 1 Hz
      - command‐ID to name mapping
    """
    def __init__(self, log_file: str = None):
        # Setup SOAP client

        
        self.client = Client(wsdl=WSDL_URL)
        # Setup logger
        self.logger = logging.getLogger("WcfClient")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    #@throttle(1.0)
    def get_value(self, command_id: int) -> str:
        name = COMMAND_NAMES.get(command_id, f"ID {command_id}")
        self.logger.info(f"GetValue({command_id}) – {name}")
        result = self.client.service.GetValue(command_id)
        self.logger.info(f"→ {result}")
        return result

    #@throttle(1.0)
    def set_value(self, command_id: int, value) -> str:
        name = COMMAND_NAMES.get(command_id, f"ID {command_id}")
        #self.logger.info(f"SetValue({command_id}, {value}) – {name}")
        self.client.service.SetValue(command_id, value)



    def get_all_status(self) -> dict:
        status = {}
        for cid in COMMAND_NAMES:
            try:
                status[cid] = self.get_value(cid)
            except Exception as e:
                self.logger.warning(f"Failed GetValue({cid}): {e}")
                status[cid] = None
        return status
        
    def close(self):
        try:
            if hasattr(self, 'client'):
                self.client.service._client.transport.session.close()
                self.logger.info("WCF Connection Closed")
        except Exception as e:
            self.logger.error(f"error closing wcf: {e}")
