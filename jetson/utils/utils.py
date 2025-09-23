"""
utils.py - Utility functions and classes for BAAM feedrate control system.

This module provides helper utilities for CSV logging and feedrate command 
tracking. It handles the interface between desired feedrate percentages and 
the WCF command system which uses offset-based control.

Key Components:
    - CSV file creation with headers
    - FeedRateLogger: Tracks feedrate changes with offset-based commands


Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""

import os
import csv
from datetime import datetime


def create_csv_writer(base_dir: str, filename: str, headers: list):
    """
    Create or open a CSV file with headers for data logging.
    
    Creates a new CSV file with headers if it doesn't exist, or opens
    an existing file in append mode. Returns both file handle and writer
    for efficient batch writing.
    
    Args:
        base_dir: Directory path where CSV will be created
        filename: Name of the CSV file (e.g., 'layer_changes.csv')
        headers: List of column header strings
        
    Returns:
        tuple: (file_handle, csv_writer)
            - file_handle: Open file object (must be closed after use)
            - csv_writer: CSV writer object for row operations
        
    Note:
        File is opened with buffering=1 for line buffering.
        Caller is responsible for closing the file handle.
    """
    csv_path = os.path.join(base_dir, filename)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', buffering=1) as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # Open file in append mode and create writer
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    return csv_file, csv_writer
    
class FeedRateLogger:
    """
    Tracks and manages feedrate commands using offset-based control.
    
    The BAAM HMI uses an offset-based feedrate control system where commands
    represent deviations from a baseline rate (typically 100%). This class
    manages the translation between absolute feedrates and offset commands.
    
    Control Model:
        - Baseline rate: 100% (normal speed)
        - Commands are offsets: cmd = desired_rate - 100
        - Examples:
            * Feedrate 110% → Command +10
            * Feedrate 80% → Command -20
            * Feedrate 100% → Command 0
    
    Attributes:
        initial_rate: Baseline feedrate (default 100%)
        current_offset: Current offset from baseline
        command_history: List of all commands applied
        
    Example:
        --> logger = FeedRateLogger(initial_rate=100)
        --> rate, cmd = logger.define_feed(120)  # Set to 120%
        --> print(f"Rate: {rate}, WCF command: {cmd}")  # Rate: 120, WCF command: 20
        --> rate, cmd = logger.define_feed(90)   # Change to 90%
        --> print(f"Rate: {rate}, WCF command: {cmd}")  # Rate: 90, WCF command: -10
    """
    def __init__(self, initial_rate=100):
        """
        Initialize feedrate logger with baseline rate.
        
        Args:
            initial_rate: Starting feedrate percentage (typically 100)
            
        Note:
            Initial rate should match the machine's nominal speed.
            Standard BAAM operation uses 100% as baseline.
        """
        self.initial_rate = initial_rate
        self.current_offset = 0  # Track offset from initial rate
        self.command_history = []
    
    def update_feed(self, cmd_feed):
        """
        Update feedrate using raw WCF offset value.
        
        Used when reading current feedrate from HMI and synchronizing
        local state. The HMI returns offset values, not absolute rates.
        
        Args:
            cmd_feed: WCF offset command (e.g., -20 for 80% rate)
            
        Returns:
            tuple: (new_rate, cmd_feed_offset)
                - new_rate: Absolute feedrate percentage
                - cmd_feed_offset: The offset value (same as input)
                
        Example:
            >>> # HMI reports offset of +15
            >>> rate, offset = logger.update_feed(15)
            >>> print(rate)  # 115 (100 + 15)
            
        Note:
            This maintains synchronization when operator manually
            changes feedrate through HMI interface.
        """
        # Calculate the actual command needed (difference from current offset)
        actual_command = cmd_feed - self.current_offset
        
        # Update the offset
        self.current_offset = cmd_feed
        
        # Store the actual command that was applied
        self.command_history.append(actual_command)
        
        return (self.get_current_rate(), cmd_feed)  # Return the offset, not the delta
        
    def calculate_command(self, target_feed):
        """
        Calculate WCF command for target feedrate without applying.
        
        Utility method for preview/planning without state change.
        
        Args:
            target_feed: Desired feedrate percentage
            
        Returns:
            int: WCF offset command needed
            
        Example:
            >>> cmd = logger.calculate_command(150)  # What command for 150%?
            >>> print(cmd)  # 50 (150 - 100)
        """
        return target_feed - self.initial_rate
    
    def define_feed(self, new_feed):
        """
        Set feedrate to specific absolute value.
        
        Primary method for controller output. Converts desired absolute
        feedrate to WCF offset command.
        
        Args:
            new_feed: Target feedrate percentage (e.g., 125)
            
        Returns:
            tuple: (new_rate, wcf_offset_command)
                - new_rate: Confirmed feedrate (same as input)
                - wcf_offset_command: Offset to send to WCF
                
        Example:
            >>> # Controller wants 130% feedrate
            >>> rate, wcf_cmd = logger.define_feed(130)
            >>> wcf_client.set_value(0, str(wcf_cmd))  # Send +30 to HMI
            
        Implementation:
            Calculates: offset = new_feed - initial_rate
            Tracks: actual_command = new_offset - current_offset
        """
        # Calculate the offset from initial rate
        new_offset = new_feed - self.initial_rate
        
        # Calculate the actual command (difference from current offset)
        actual_command = new_offset - self.current_offset
        
        # Update the offset
        self.current_offset = new_offset
        
        # Store the actual command
        self.command_history.append(actual_command)
        
        return (new_feed, new_offset)  # Return the offset for WCF
    
    def get_current_rate(self):
        """
        Get the current feed rate.
        
        Returns:
            int: Current feed rate (initial_rate + current_offset)
        """
        return self.initial_rate + self.current_offset
    
    def get_history(self):
        """
        Get the history of all commands.
        
        Returns:
            list: List of all feed commands applied (actual deltas)
        """
        return self.command_history
    
    def reset(self):
        """
        Reset the feed rate to initial value.
        """
        self.current_offset = 0
        self.command_history = []

    

        

