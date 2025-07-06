#!/usr/bin/env python3
"""
Debug script for neural-behavioral integration issues.

This script helps identify what's going wrong with neural data extraction.
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import neo
from neo.io import BlackrockIO
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_h5_file(h5_path: str):
    """Inspect what's actually in the H5 file."""
    logger.info(f"Inspecting H5 file: {h5_path}")
    
    if not Path(h5_path).exists():
        logger.error(f"H5 file not found: {h5_path}")
        return
    
    file_size = Path(h5_path).stat().st_size
    logger.info(f"File size: {file_size / 1024:.2f} KB")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            logger.info("H5 file contents:")
            
            # Print global attributes
            logger.info("Global attributes:")
            for key, value in f.attrs.items():
                logger.info(f"  {key}: {value}")
            
            # Print groups (trials)
            trial_count = 0
            for key in f.keys():
                if key.startswith('trial_'):
                    trial_count += 1
                    trial_group = f[key]
                    logger.info(f"\n{key}:")
                    
                    # Check if neural data exists
                    if 'neural' in trial_group:
                        neural_data = trial_group['neural']
                        logger.info(f"  Neural data shape: {neural_data.shape}")
                        logger.info(f"  Neural data size: {neural_data.size * 4 / 1024:.2f} KB")  # Assuming float32
                    else:
                        logger.warning(f"  No neural data found in {key}")
                    
                    # Check behavioral data
                    for dataset in trial_group.keys():
                        if dataset != 'neural':
                            data = trial_group[dataset]
                            logger.info(f"  {dataset} shape: {data.shape}")
                    
                    # Check attributes
                    logger.info("  Attributes:")
                    for attr_key, attr_value in trial_group.attrs.items():
                        logger.info(f"    {attr_key}: {attr_value}")
                    
                    # Only show first few trials to avoid spam
                    if trial_count >= 3:
                        logger.info(f"  ... and {len([k for k in f.keys() if k.startswith('trial_')]) - 3} more trials")
                        break
                        
    except Exception as e:
        logger.error(f"Error inspecting H5 file: {e}")

def test_neural_file_access(neural_path: str):
    """Test if we can access the neural file and read basic info."""
    logger.info(f"Testing neural file access: {neural_path}")
    
    if not Path(neural_path).exists():
        logger.error(f"Neural file not found: {neural_path}")
        return False
    
    file_size = Path(neural_path).stat().st_size / (1024**3)  # GB
    logger.info(f"Neural file size: {file_size:.2f} GB")
    
    try:
        # Try to initialize BlackrockIO
        logger.info("Initializing BlackrockIO...")
        try:
            io = BlackrockIO(filename=neural_path, lazy=True)
            logger.info("✅ BlackrockIO initialized with lazy=True")
        except TypeError:
            io = BlackrockIO(filename=neural_path)
            logger.info("✅ BlackrockIO initialized without lazy parameter")
        
        # Try to read basic info
        logger.info("Reading basic file info...")
        
        # Check if we can read a small test block
        try:
            logger.info("Attempting to read first 1 second...")
            block = io.read_block(time_slice=(0, 1))
            
            if block.segments:
                segment = block.segments[0]
                logger.info(f"Found {len(segment.analogsignals)} analog signals")
                
                total_channels = 0
                for i, signal in enumerate(segment.analogsignals):
                    logger.info(f"  Signal {i}: shape {signal.shape}, sampling_rate {signal.sampling_rate}")
                    total_channels += signal.shape[1] if signal.ndim > 1 else 1
                
                logger.info(f"Total channels: {total_channels}")
                
                # Check the time range
                if segment.analogsignals:
                    first_signal = segment.analogsignals[0]
                    t_start = first_signal.t_start
                    t_stop = first_signal.t_stop
                    logger.info(f"Time range: {t_start} to {t_stop}")
                    
                return True
            else:
                logger.warning("No segments found in neural file")
                return False
                
        except Exception as e:
            logger.error(f"Failed to read test block: {e}")
            
            # Try reading without time slice
            try:
                logger.info("Trying to read without time slice...")
                block = io.read_block()
                logger.info(f"Block has {len(block.segments)} segments")
                return True
            except Exception as e2:
                logger.error(f"Failed to read block at all: {e2}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to initialize BlackrockIO: {e}")
        return False

def test_behavioral_file_access(behavioral_path: str):
    """Test if we can access and parse the behavioral file."""
    logger.info(f"Testing behavioral file access: {behavioral_path}")
    
    if not Path(behavioral_path).exists():
        logger.error(f"Behavioral file not found: {behavioral_path}")
        return False
    
    try:
        df = pd.read_csv(behavioral_path)
        logger.info(f"Behavioral file loaded: {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['timestamp', 'trial_start', 'trial_win', 'trial_lose']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Parse timestamps
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        except Exception as e:
            logger.error(f"Failed to parse timestamps: {e}")
            return False
        
        # Check trial markers
        trial_starts = df['trial_start'].sum()
        trial_wins = df['trial_win'].sum()
        trial_loses = df['trial_lose'].sum()
        
        logger.info(f"Trial markers: {trial_starts} starts, {trial_wins} wins, {trial_loses} loses")
        
        if trial_starts == 0:
            logger.error("No trial starts found!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load behavioral file: {e}")
        return False

def test_time_alignment(neural_path: str, behavioral_path: str):
    """Test if the time alignment between neural and behavioral data makes sense."""
    logger.info("Testing time alignment...")
    
    # Load behavioral data
    try:
        df = pd.read_csv(behavioral_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        behavioral_start = df['timestamp'].min()
        behavioral_end = df['timestamp'].max()
        logger.info(f"Behavioral time range: {behavioral_start} to {behavioral_end}")
    except Exception as e:
        logger.error(f"Failed to load behavioral data: {e}")
        return False
    
    # Try to get neural timing
    try:
        try:
            io = BlackrockIO(filename=neural_path, lazy=True)
        except TypeError:
            io = BlackrockIO(filename=neural_path)
        
        # Try to get time origin
        neural_start_time = None
        try:
            if hasattr(io, 'raw_annotations') and io.raw_annotations:
                neural_start_time = io.raw_annotations.get('Time Origin', None)
            elif hasattr(io, 'header') and io.header:
                neural_start_time = io.header.get('Time Origin', None)
        except Exception as e:
            logger.warning(f"Could not access header: {e}")
        
        if neural_start_time:
            if isinstance(neural_start_time, str):
                neural_start_time = neural_start_time.rstrip('Z')
                neural_start_time = datetime.fromisoformat(neural_start_time)
            logger.info(f"Neural start time: {neural_start_time}")
            
            # Check alignment
            time_diff = (behavioral_start - neural_start_time).total_seconds()
            logger.info(f"Time difference (behavioral - neural): {time_diff:.2f} seconds")
            
            if abs(time_diff) > 3600:  # More than 1 hour difference
                logger.warning("Large time difference detected - check time alignment!")
                
        else:
            logger.warning("Could not determine neural start time")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to check neural timing: {e}")
        return False

def main():
    """Run all debugging tests."""
    logger.info("="*60)
    logger.info("NEURAL-BEHAVIORAL INTEGRATION DEBUGGING")
    logger.info("="*60)
    
    # File paths - update these to match your actual files
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"
    h5_file = r"D:\Data\ScienceCorp\trials.h5"
    
    logger.info(f"Neural file: {neural_file}")
    logger.info(f"Behavioral file: {behavioral_file}")
    logger.info(f"H5 file: {h5_file}")
    
    # Test 1: Inspect the H5 file
    logger.info("\n" + "="*40)
    logger.info("TEST 1: Inspecting H5 file")
    logger.info("="*40)
    inspect_h5_file(h5_file)
    
    # Test 2: Test neural file access
    logger.info("\n" + "="*40)
    logger.info("TEST 2: Testing neural file access")
    logger.info("="*40)
    neural_ok = test_neural_file_access(neural_file)
    
    # Test 3: Test behavioral file access
    logger.info("\n" + "="*40)
    logger.info("TEST 3: Testing behavioral file access")
    logger.info("="*40)
    behavioral_ok = test_behavioral_file_access(behavioral_file)
    
    # Test 4: Test time alignment
    logger.info("\n" + "="*40)
    logger.info("TEST 4: Testing time alignment")
    logger.info("="*40)
    if neural_ok and behavioral_ok:
        test_time_alignment(neural_file, behavioral_file)
    else:
        logger.info("Skipping time alignment test due to file access issues")
    
    logger.info("\n" + "="*60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("="*60)
    
    if not neural_ok:
        logger.error("❌ Neural file access failed")
    if not behavioral_ok:
        logger.error("❌ Behavioral file access failed")
    
    if neural_ok and behavioral_ok:
        logger.info("✅ Files accessible - check the logs above for timing and data issues")
        logger.info("💡 Common issues:")
        logger.info("   - Time alignment problems (behavioral vs neural timestamps)")
        logger.info("   - Trial windows falling outside neural recording time")
        logger.info("   - Neural data extraction failing silently")
    else:
        logger.error("❌ File access issues need to be resolved first")

if __name__ == "__main__":
    main() 