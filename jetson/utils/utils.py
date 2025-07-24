import os
import csv

def create_csv_writer(base_dir: str, filename: str, headers: list):
    """
    Create a CSV file with headers and return the file handle and writer.
    
    Args:
        base_dir: Directory path where CSV will be created
        filename: Name of the CSV file
        headers: List of column header strings
        
    Returns:
        tuple: (file_handle, csv_writer)
    """
    csv_path = os.path.join(base_dir, filename)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # Open file in append mode and create writer
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    return csv_file, csv_writer

