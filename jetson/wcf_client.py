"""
wcf_client.py - Windows Communication Foundation (WCF) client for BAAM HMI control.

This module provides a SOAP-based interface to the BAAM Human Machine Interface (HMI)
running on Windows. It enables the Jetson edge device to read process parameters
and send control commands, particularly feedrate override adjustments based on
thermal feedback.

The WCF service exposes two main operations:
    - GetValue(command_id): Read current parameter values
    - SetValue(command_id, value): Write control commands

Communication Flow:
    Jetson (this client) <--> Windows HMI (WCF Service) <--> BAAM Machine

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""


import time
import logging
import functools
from requests import Session
from zeep import Client
from zeep.transports import Transport

# Service endpoints - must match Windows HMI configuration
WSDL_URL = "http://192.168.0.151:8733/Design_Time_Addresses/BaamHmi/CIBaamInterface/?singleWsdl"
SERVICE_ADDRESS = "http://192.168.0.151:8733/Design_Time_Addresses/BaamHmi/CIBaamInterface/"
# Note: Port 8733 is the default WCF development server port
# Production deployments may use different ports

# Command ID mapping for BAAM HMI interface
# These IDs are defined by the Windows HMI service contract
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
    Rate-limiting decorator to prevent overwhelming the WCF service.
    
    Enforces a minimum time interval between successive calls to the
    decorated method. If called too quickly, sleeps until the minimum
    interval has elapsed.
    
    Args:
        seconds: Minimum time between calls in seconds
        
    Returns:
        Decorated function with rate limiting
        
    Example:
        @throttle(0.5)  # Max 2 calls per second
        def get_value(self, cmd_id):
            ...
            
    Note:
        Uses closure to maintain last call timestamp per method.
        Thread-safe for single-threaded access only.
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
    SOAP client wrapper for BAAM HMI Windows Communication Foundation service.
    
    Provides a simplified interface to the BAAM control system, handling
    SOAP protocol details and connection management. Implements logging
    and optional rate limiting for stable communication.
    
    The client uses Zeep library for SOAP communication, which automatically
    generates Python methods from the WSDL service definition.
    
    Attributes:
        client: Zeep SOAP client instance
        logger: Configured logger for operation tracking
        
    Methods:
        get_value: Read parameter from HMI
        set_value: Send control command to HMI
        get_all_status: Batch read all parameters
        close: Clean connection shutdown
    """
    
    def __init__(self, log_file: str = None):
        # Setup SOAP client
        session   = Session()
        transport = Transport(session=session)
        self.client = Client(wsdl=WSDL_URL, transport=transport)

        # # Override endpoint address
        self.client.set_default_soapheaders([])
        self.client.service._binding_options['address'] = SERVICE_ADDRESS
        
        
        #self.client = Client(wsdl=WSDL_URL)
        # Setup logger
        self.logger = logging.getLogger("WcfClient")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    #@throttle(0.5)
    def get_value(self, command_id: int) -> str:
        name = COMMAND_NAMES.get(command_id, f"ID {command_id}")
        self.logger.info(f"GetValue({command_id}) – {name}")
        result = self.client.service.GetValue(command_id)
        return result

    #@throttle(10.0)
    def set_value(self, command_id: str, value) -> str:
        name = COMMAND_NAMES.get(command_id, f"ID {command_id}")
        #self.logger.info(f"SetValue({command_id}, {value}) – {name}")
        self.client.service.SetValue(command_id, value)

        #return result

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
