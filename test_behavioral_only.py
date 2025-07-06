#!/usr/bin/env python3
"""
Test script to debug behavioral data parsing issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from neural_behavioral_integration import NeuralBehavioralIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_behavioral_parsing():
    """Test behavioral data parsing with detailed debugging."""
    
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"
    
    logger.info("Testing behavioral data parsing...")
    
    # Load raw CSV first
    df = pd.read_csv(behavioral_file)
    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Examine raw timestamps
    logger.info("Raw timestamp examination:")
    logger.info(f"  First 5 timestamps: {df['timestamp'].head().tolist()}")
    logger.info(f"  Last 5 timestamps: {df['timestamp'].tail().tolist()}")
    logger.info(f"  Min timestamp: {df['timestamp'].min()}")
    logger.info(f"  Max timestamp: {df['timestamp'].max()}")
    logger.info(f"  Timestamp dtype: {df['timestamp'].dtype}")
    
    # Check the range
    timestamp_range = df['timestamp'].max() - df['timestamp'].min()
    logger.info(f"  Raw timestamp range: {timestamp_range}")
    
    # Now try the integrator parsing
    logger.info("\nTesting integrator parsing...")
    
    integrator = NeuralBehavioralIntegrator(
        neural_file="dummy.ns6",
        behavioral_file=behavioral_file,
        output_file="dummy.h5"
    )
    
    try:
        behavioral_data = integrator.load_behavioral_data()
        
        logger.info("Integrator parsing results:")
        logger.info(f"  Parsed timestamp range: {behavioral_data['timestamp'].min()} to {behavioral_data['timestamp'].max()}")
        logger.info(f"  Time span: {(behavioral_data['timestamp'].max() - behavioral_data['timestamp'].min()).total_seconds():.2f} seconds")
        
        # Check trial markers
        trial_starts = behavioral_data['trial_start'].sum()
        trial_wins = behavioral_data['trial_win'].sum()
        trial_loses = behavioral_data['trial_lose'].sum()
        
        logger.info(f"  Trial markers: {trial_starts} starts, {trial_wins} wins, {trial_loses} loses")
        
        # Try trial segmentation
        logger.info("\nTesting trial segmentation...")
        trials = integrator.segment_trials()
        logger.info(f"Found {len(trials)} trials")
        
        if trials:
            logger.info(f"First trial: {trials[0]['start_time']} to {trials[0]['end_time']}")
            logger.info(f"First trial duration: {trials[0]['duration']:.2f} seconds")
            
        return True
        
    except Exception as e:
        logger.error(f"Integrator parsing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def suggest_timestamp_fix():
    """Suggest how to fix the timestamp issues."""
    
    logger.info("\n" + "="*50)
    logger.info("TIMESTAMP FIX SUGGESTIONS")
    logger.info("="*50)
    
    # Load the raw data to examine
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"
    df = pd.read_csv(behavioral_file)
    
    raw_timestamps = df['timestamp'].iloc[:10].tolist()
    logger.info(f"Raw timestamp examples: {raw_timestamps}")
    
    # The timestamps look like they might be in a high-precision format
    # Let's try different interpretations
    
    logger.info("\nTrying different timestamp interpretations:")
    
    for i, ts in enumerate(raw_timestamps[:3]):
        logger.info(f"\nTimestamp {i+1}: {ts}")
        
        # Try as seconds since epoch
        try:
            dt1 = datetime.fromtimestamp(ts)
            logger.info(f"  As seconds since epoch: {dt1}")
        except:
            logger.info(f"  As seconds since epoch: FAILED")
        
        # Try as milliseconds since epoch
        try:
            dt2 = datetime.fromtimestamp(ts / 1000)
            logger.info(f"  As milliseconds since epoch: {dt2}")
        except:
            logger.info(f"  As milliseconds since epoch: FAILED")
        
        # Try as microseconds since epoch
        try:
            dt3 = datetime.fromtimestamp(ts / 1000000)
            logger.info(f"  As microseconds since epoch: {dt3}")
        except:
            logger.info(f"  As microseconds since epoch: FAILED")
    
    logger.info("\n" + "="*50)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*50)
    logger.info("1. Check the timestamp format in your data collection system")
    logger.info("2. The timestamps might be in a different epoch or timezone")
    logger.info("3. Consider if there's an offset that needs to be applied")
    logger.info("4. Verify the timestamp units (seconds, milliseconds, microseconds)")

if __name__ == "__main__":
    test_behavioral_parsing()
    suggest_timestamp_fix() 