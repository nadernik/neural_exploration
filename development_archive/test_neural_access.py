#!/usr/bin/env python3
"""
Test script to confirm neural.ns6 file access and implement workarounds.
"""

import numpy as np
import neo
from neo.io import BlackrockIO
import logging
from pathlib import Path
import h5py
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neural_file_direct():
    """Test direct access to the neural file."""
    
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"
    
    logger.info(f"Testing direct neural file access: {neural_file}")
    
    # Check file exists
    if not Path(neural_file).exists():
        logger.error(f"❌ Neural file not found: {neural_file}")
        return False
    
    file_size = Path(neural_file).stat().st_size / (1024**3)  # GB
    logger.info(f"✅ Neural file size: {file_size:.2f} GB")
    
    # Test BlackrockIO initialization
    try:
        logger.info("Initializing BlackrockIO...")
        try:
            io = BlackrockIO(filename=neural_file, lazy=True)
            logger.info("✅ BlackrockIO initialized with lazy=True")
        except TypeError:
            io = BlackrockIO(filename=neural_file)
            logger.info("✅ BlackrockIO initialized (lazy not supported)")
        
        # Get file info
        logger.info("Getting file information...")
        
        # Check available methods
        available_methods = [method for method in dir(io) if not method.startswith('_')]
        logger.info(f"Available methods: {available_methods}")
        
        # Try to get header info
        try:
            if hasattr(io, 'header'):
                logger.info(f"Header available: {bool(io.header)}")
                if io.header:
                    logger.info(f"Header keys: {list(io.header.keys())}")
        except Exception as e:
            logger.warning(f"Could not access header: {e}")
        
        # Try to get raw annotations
        try:
            if hasattr(io, 'raw_annotations'):
                logger.info(f"Raw annotations available: {bool(io.raw_annotations)}")
                if io.raw_annotations:
                    logger.info(f"Raw annotation keys: {list(io.raw_annotations.keys())}")
        except Exception as e:
            logger.warning(f"Could not access raw annotations: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BlackrockIO initialization failed: {e}")
        return False

def test_neural_reading_strategies():
    """Test different strategies for reading neural data."""
    
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"
    
    logger.info("Testing neural reading strategies...")
    
    try:
        # Initialize IO
        try:
            io = BlackrockIO(filename=neural_file, lazy=True)
        except TypeError:
            io = BlackrockIO(filename=neural_file)
        
        # Strategy 1: Try time slice (we know this fails)
        logger.info("Strategy 1: Testing time_slice support...")
        try:
            block = io.read_block(time_slice=(0, 1))
            logger.info("✅ time_slice supported - this should work!")
            return True
        except Exception as e:
            logger.info(f"❌ time_slice failed: {e}")
        
        # Strategy 2: Try reading with channel selection
        logger.info("Strategy 2: Testing channel selection...")
        try:
            # Try to read just a few channels
            block = io.read_block(channel_indexes=[0, 1, 2])
            logger.info("✅ Channel selection supported!")
            
            if block.segments:
                segment = block.segments[0]
                logger.info(f"Segment has {len(segment.analogsignals)} analog signals")
                for i, signal in enumerate(segment.analogsignals):
                    logger.info(f"  Signal {i}: shape {signal.shape}, sampling_rate {signal.sampling_rate}")
            
            return True
        except Exception as e:
            logger.info(f"❌ Channel selection failed: {e}")
        
        # Strategy 3: Check if we can get file metadata without reading data
        logger.info("Strategy 3: Testing metadata access...")
        try:
            # Try to get channel information
            if hasattr(io, 'channel_count'):
                logger.info(f"Channel count: {io.channel_count}")
            
            if hasattr(io, 'get_signal_size'):
                try:
                    size = io.get_signal_size()
                    logger.info(f"Signal size: {size}")
                except Exception as e:
                    logger.info(f"get_signal_size failed: {e}")
            
            # Try to access sampling rate
            if hasattr(io, 'get_signal_sampling_rate'):
                try:
                    sr = io.get_signal_sampling_rate()
                    logger.info(f"Sampling rate: {sr}")
                except Exception as e:
                    logger.info(f"get_signal_sampling_rate failed: {e}")
                    
            return True
            
        except Exception as e:
            logger.info(f"❌ Metadata access failed: {e}")
        
        # Strategy 4: Last resort - check if we can read ANY data
        logger.info("Strategy 4: Testing minimal data read...")
        try:
            logger.warning("This may use a lot of memory - attempting to read full block...")
            # This will likely fail due to memory, but let's see what happens
            block = io.read_block()
            logger.info("✅ Full block read successful!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full block read failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Neural reading test failed: {e}")
        return False

def create_test_h5_with_neural():
    """Create a test H5 file with actual neural data to verify the approach."""
    
    logger.info("Creating test H5 file with neural data...")
    
    # Create synthetic neural data for testing
    n_channels = 96
    n_samples = 30000  # 1 second at 30kHz
    
    # Create test data
    neural_data = np.random.randn(n_channels, n_samples).astype(np.float32)
    
    # Save to H5 file
    test_file = "test_neural.h5"
    
    try:
        with h5py.File(test_file, 'w') as f:
            # Global attributes
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['neural_file'] = "test_neural.ns6"
            f.attrs['total_trials'] = 1
            f.attrs['original_sampling_rate'] = 30000
            f.attrs['final_sampling_rate'] = 1000
            
            # Create a trial
            trial_group = f.create_group('trial_1')
            trial_group.create_dataset('neural', data=neural_data, compression='gzip')
            
            # Add metadata
            trial_group.attrs['trial_number'] = 1
            trial_group.attrs['outcome'] = 'test'
            trial_group.attrs['duration'] = 1.0
        
        # Check the file size
        file_size = Path(test_file).stat().st_size / 1024  # KB
        logger.info(f"✅ Test H5 file created: {file_size:.2f} KB")
        
        # This should be much larger than 114 KB if it contains actual neural data
        expected_size = n_channels * n_samples * 4 / 1024  # 4 bytes per float32
        logger.info(f"Expected minimum size: {expected_size:.2f} KB")
        
        if file_size > expected_size * 0.5:  # Account for compression
            logger.info("✅ File size looks correct for neural data")
        else:
            logger.warning("❌ File size too small - something's wrong")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Test H5 creation failed: {e}")
        return False

def main():
    """Run all neural access tests."""
    
    logger.info("="*60)
    logger.info("NEURAL FILE ACCESS TESTING")
    logger.info("="*60)
    
    # Test 1: Direct file access
    logger.info("\n" + "="*40)
    logger.info("TEST 1: Direct neural file access")
    logger.info("="*40)
    neural_accessible = test_neural_file_direct()
    
    # Test 2: Reading strategies
    logger.info("\n" + "="*40)
    logger.info("TEST 2: Neural reading strategies")
    logger.info("="*40)
    if neural_accessible:
        reading_works = test_neural_reading_strategies()
    else:
        logger.info("Skipping reading test - file not accessible")
        reading_works = False
    
    # Test 3: H5 file creation test
    logger.info("\n" + "="*40)
    logger.info("TEST 3: H5 file creation test")
    logger.info("="*40)
    create_test_h5_with_neural()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if not neural_accessible:
        logger.error("❌ Cannot access neural file - check file path and permissions")
    elif not reading_works:
        logger.error("❌ Neural file accessible but cannot read data")
        logger.error("   This version of neo doesn't support time_slice or channel selection")
        logger.error("   The file is too large to load entirely into memory")
        logger.info("💡 SOLUTIONS:")
        logger.info("   1. Upgrade neo to a version that supports time_slice")
        logger.info("   2. Use a different neural data reading library")
        logger.info("   3. Pre-process the .ns6 file to extract smaller chunks")
        logger.info("   4. Use a machine with more RAM")
    else:
        logger.info("✅ Neural file access should work!")

if __name__ == "__main__":
    main() 