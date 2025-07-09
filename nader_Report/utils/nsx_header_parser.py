"""
NSX Header Parser - BlackRock NSX File Header Parsing Utility

This module provides functions to parse BlackRock NSX file headers according to the 
official BlackRock NSX file specification. It extracts the Time Origin and other 
metadata directly from the binary header structure.

Based on BlackRock NSX File Specification:
- File Type ID (8 bytes): "BRSMPGRP" 
- File Spec (2 bytes): Major.Minor version
- Bytes in Headers (4 bytes): Total header size
- Label (16 bytes): Sampling group label
- Comment (256 bytes): File comment
- Period (4 bytes): Sampling period (1/30000 second units)
- Time Resolution (4 bytes): Global clock frequency
- Time Origin (16 bytes): Windows SYSTEM TIME structure (UTC)
- Channel Count (4 bytes): Number of channels
"""

import struct
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class NSXHeaderParseError(Exception):
    """Exception raised when NSX header parsing fails"""
    pass

def parse_nsx_header(file_path):
    """
    Parse BlackRock NSX file header according to official specification.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the NSX file (.ns6, .ns5, etc.)
        
    Returns:
    --------
    dict
        Dictionary containing parsed header fields:
        - file_type_id: File type identifier
        - file_spec: File specification version
        - bytes_in_headers: Total header size
        - label: Sampling group label
        - comment: File comment
        - period: Sampling period
        - sampling_rate: Calculated sampling rate (Hz)
        - time_resolution: Global clock frequency
        - time_origin: UTC datetime when recording started
        - channel_count: Number of channels
        
    Raises:
    -------
    NSXHeaderParseError
        If file cannot be read or header is invalid
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise NSXHeaderParseError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            header = {}
            
            # File Type ID (8 bytes) - char array
            file_type_id = f.read(8)
            header['file_type_id'] = file_type_id.decode('ascii', errors='ignore').rstrip('\x00')
            
            # Validate file type
            if header['file_type_id'] not in ['BRSMPGRP', 'NEURALSG', 'NEURALCD']:
                raise NSXHeaderParseError(f"Invalid file type ID: {header['file_type_id']}")
            
            # File Spec (2 bytes) - 2 x unsigned char
            file_spec = struct.unpack('<BB', f.read(2))
            header['file_spec'] = f"{file_spec[0]}.{file_spec[1]}"
            
            # Bytes in Headers (4 bytes) - unsigned int-32
            bytes_in_headers = struct.unpack('<I', f.read(4))[0]
            header['bytes_in_headers'] = bytes_in_headers
            
            # Label (16 bytes) - char array
            label = f.read(16)
            header['label'] = label.decode('ascii', errors='ignore').rstrip('\x00')
            
            # Comment (256 bytes) - char array
            comment = f.read(256)
            header['comment'] = comment.decode('ascii', errors='ignore').rstrip('\x00')
            
            # Period (4 bytes) - unsigned int-32
            period = struct.unpack('<I', f.read(4))[0]
            header['period'] = period
            
            # Calculate sampling rate
            if period > 0:
                header['sampling_rate'] = 30000.0 / period
            else:
                header['sampling_rate'] = 0.0
                logger.warning("Invalid period value (0), cannot calculate sampling rate")
            
            # Time Resolution (4 bytes) - unsigned int-32
            time_resolution = struct.unpack('<I', f.read(4))[0]
            header['time_resolution'] = time_resolution
            
            # ⭐ TIME ORIGIN (16 bytes) - Windows SYSTEM TIME structure ⭐
            time_origin_bytes = f.read(16)
            time_values = struct.unpack('<8H', time_origin_bytes)  # 8 unsigned int-16, little-endian
            
            year, month, day_of_week, day, hour, minute, second, millisecond = time_values
            
            # Convert to Python datetime (UTC)
            if year > 0 and month > 0 and day > 0:
                try:
                    time_origin = datetime(year, month, day, hour, minute, second, 
                                         millisecond * 1000, timezone.utc)
                    header['time_origin'] = time_origin
                    logger.info(f"Extracted time origin: {time_origin}")
                except ValueError as e:
                    raise NSXHeaderParseError(f"Invalid time origin values: {time_values} - {e}")
            else:
                raise NSXHeaderParseError(f"Invalid time origin values: {time_values}")
            
            # Channel Count (4 bytes) - unsigned int-32
            channel_count = struct.unpack('<I', f.read(4))[0]
            header['channel_count'] = channel_count
            
            logger.info(f"Successfully parsed NSX header: {file_path}")
            logger.info(f"  File Type: {header['file_type_id']}")
            logger.info(f"  Sampling Rate: {header['sampling_rate']:.1f} Hz")
            logger.info(f"  Channels: {header['channel_count']}")
            logger.info(f"  Time Origin: {header['time_origin']}")
            
            return header
            
    except (struct.error, UnicodeDecodeError, IOError) as e:
        raise NSXHeaderParseError(f"Error parsing NSX header: {e}")

def get_time_origin(file_path):
    """
    Extract just the time origin from an NSX file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the NSX file
        
    Returns:
    --------
    datetime
        UTC datetime when recording started
        
    Raises:
    -------
    NSXHeaderParseError
        If file cannot be read or time origin is invalid
    """
    header = parse_nsx_header(file_path)
    return header['time_origin']

def get_sampling_rate(file_path):
    """
    Extract just the sampling rate from an NSX file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the NSX file
        
    Returns:
    --------
    float
        Sampling rate in Hz
        
    Raises:
    -------
    NSXHeaderParseError
        If file cannot be read or sampling rate is invalid
    """
    header = parse_nsx_header(file_path)
    return header['sampling_rate']

def get_channel_count(file_path):
    """
    Extract just the channel count from an NSX file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the NSX file
        
    Returns:
    --------
    int
        Number of channels
        
    Raises:
    -------
    NSXHeaderParseError
        If file cannot be read or channel count is invalid
    """
    header = parse_nsx_header(file_path)
    return header['channel_count'] 