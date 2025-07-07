#!/usr/bin/env python3
"""
Quick Spike Detection Comparison Example
========================================

This script provides a simple example of how to use the new spike detection
comparison functionality with dual raster plots.

Usage:
    python quick_spike_comparison.py

Author: Neural Exploration Team  
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.spike_detection import SpikeDetector, create_raster_comparison
from utils.data_loader import load_trial_data
import warnings
warnings.filterwarnings('ignore')

def run_quick_comparison():
    """Run a quick spike detection comparison example."""
    
    print("🚀 QUICK SPIKE DETECTION COMPARISON")
    print("=" * 45)
    
    # Configuration
    TRIAL_NUMBER = 10  # Trial with good data
    SAMPLING_RATE = 30000  # Hz
    SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32]  # Subset of channels for quick demo
    
    # Load trial data
    print(f"📊 Loading trial {TRIAL_NUMBER}...")
    trial_data = load_trial_data(TRIAL_NUMBER, SAMPLING_RATE)
    
    if trial_data is None:
        print("❌ Failed to load trial data")
        return
    
    duration = trial_data.get('duration', trial_data['neural_data'].shape[1] / SAMPLING_RATE)
    print(f"✅ Data loaded: {duration:.2f}s, {trial_data['neural_data'].shape[0]} channels")
    
    # Initialize spike detector
    print(f"\n🔬 Initializing spike detector...")
    spike_detector = SpikeDetector(sampling_rate=SAMPLING_RATE)
    
    # Create dual raster plot comparison
    print(f"\n🎯 Creating dual raster plots...")
    print(f"   Comparing: Threshold vs PyWaveClus methods")
    print(f"   Channels: {SPIKE_CHANNELS}")
    
    raster_results = create_raster_comparison(
        trial_data=trial_data,
        spike_channels=SPIKE_CHANNELS,
        spike_detector=spike_detector,
        trial_number=TRIAL_NUMBER,
        time_window=None,
        figsize=(15, 10)
    )
    
    if raster_results:
        # Show summary statistics
        thresh_total = raster_results['total_threshold_spikes']
        waveclus_total = raster_results['total_waveclus_spikes']
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   • Threshold method: {thresh_total} spikes")
        print(f"   • PyWaveClus method: {waveclus_total} spikes")
        print(f"   • Rate difference: {abs(waveclus_total - thresh_total)/duration:.1f} spikes/s")
        
        if abs(waveclus_total - thresh_total) > 0.1 * max(thresh_total, waveclus_total):
            print(f"   • Significant difference detected - consider parameter tuning")
        else:
            print(f"   • Methods show similar results")
            
        print(f"\n✅ Dual raster plots created successfully!")
        print(f"   Two raster plots displayed: one for each detection method")
        
    else:
        print(f"❌ Failed to create raster comparison")
    
    print(f"\n🎯 Done! Check the plots above to compare the two methods.")

if __name__ == "__main__":
    run_quick_comparison() 