#!/usr/bin/env python3
"""
Spike Detection Comparison Demo
===============================

This script demonstrates the comparison between traditional threshold-based
spike detection and PyWaveClus-inspired detection algorithms with dual raster plots.

Usage:
    python spike_detection_demo.py

Author: Neural Exploration Team
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.spike_detection import SpikeDetector, create_raster_comparison
from utils.data_loader import load_trial_data
from utils.diagnostics import diagnose_trial_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main demonstration function."""
    
    print("🔬 SPIKE DETECTION COMPARISON DEMO")
    print("=" * 50)
    
    # Configuration
    TRIAL_NUMBER = 10  # Trial with good movement data
    SAMPLING_RATE = 30000  # Hz
    SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
    
    # Use subset of channels for cleaner visualization
    COMPARISON_CHANNELS = SPIKE_CHANNELS[:8]  # First 8 channels
    
    print(f"📊 Configuration:")
    print(f"   Trial: {TRIAL_NUMBER}")
    print(f"   Sampling rate: {SAMPLING_RATE} Hz")
    print(f"   Channels to compare: {COMPARISON_CHANNELS}")
    
    # Step 1: Load trial data
    print(f"\n🔍 Loading trial data...")
    try:
        trial_data = load_trial_data(TRIAL_NUMBER, SAMPLING_RATE)
        
        if trial_data is None:
            print("❌ Failed to load trial data")
            return
        
        # Diagnose data quality
        diagnostics = diagnose_trial_data(trial_data, SAMPLING_RATE)
        print(f"✅ Data loaded successfully:")
        print(f"   Duration: {trial_data.get('duration', 'unknown'):.2f}s")
        print(f"   Neural channels: {trial_data['neural_data'].shape[0]}")
        print(f"   Quality: {diagnostics.get('overall_quality', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Initialize spike detector
    print(f"\n🔬 Initializing spike detector...")
    spike_detector = SpikeDetector(sampling_rate=SAMPLING_RATE)
    
    # Configure parameters for comparison
    spike_detector.threshold_params['multiplier'] = -4.0  # Standard threshold
    spike_detector.waveclus_params['threshold_multiplier'] = 4.0  # PyWaveClus threshold
    
    print(f"✅ Spike detector initialized:")
    print(f"   Threshold method: {spike_detector.threshold_params['multiplier']}x RMS")
    print(f"   PyWaveClus method: {spike_detector.waveclus_params['threshold_multiplier']}x MAD")
    
    # Step 3: Single channel detailed comparison
    print(f"\n🔍 Single channel detailed comparison...")
    
    # Pick a representative channel
    test_channel = COMPARISON_CHANNELS[0]
    
    if test_channel < trial_data['neural_data'].shape[0]:
        signal_data = trial_data['neural_data'][test_channel, :]
        
        print(f"   Analyzing channel {test_channel}:")
        print(f"   Signal duration: {len(signal_data)/SAMPLING_RATE:.2f}s")
        print(f"   Signal RMS: {np.sqrt(np.mean(signal_data**2)):.1f}μV")
        
        # Compare methods
        comparison = spike_detector.compare_methods(signal_data)
        
        # Plot detailed comparison for a 5-second window
        time_window = (2.0, 7.0)  # Show 5 seconds starting from 2s
        spike_detector.plot_detection_comparison(
            signal_data=signal_data,
            comparison_result=comparison,
            time_window=time_window,
            figsize=(15, 10)
        )
        
        print(f"   ✅ Single channel comparison complete")
    else:
        print(f"   ❌ Channel {test_channel} not available")
    
    # Step 4: Create dual raster plot comparison
    print(f"\n🎯 Creating dual raster plots...")
    
    raster_results = create_raster_comparison(
        trial_data=trial_data,
        spike_channels=COMPARISON_CHANNELS,
        spike_detector=spike_detector,
        trial_number=TRIAL_NUMBER,
        time_window=None,  # Show full trial
        figsize=(15, 12)
    )
    
    if raster_results:
        print(f"   ✅ Dual raster plots created successfully")
        
        # Step 5: Statistical comparison
        print(f"\n📊 STATISTICAL COMPARISON:")
        print("=" * 50)
        
        thresh_total = raster_results['total_threshold_spikes']
        waveclus_total = raster_results['total_waveclus_spikes']
        duration = raster_results['duration']
        
        print(f"📈 Overall Results:")
        print(f"   Trial duration: {duration:.2f}s")
        print(f"   Channels analyzed: {len(COMPARISON_CHANNELS)}")
        print(f"   Threshold method: {thresh_total} total spikes ({thresh_total/duration:.1f} spikes/s)")
        print(f"   PyWaveClus method: {waveclus_total} total spikes ({waveclus_total/duration:.1f} spikes/s)")
        
        # Per-channel breakdown
        print(f"\n🔍 Per-Channel Results:")
        print(f"   {'Channel':<8} {'Threshold':<10} {'PyWaveClus':<10} {'Ratio':<8}")
        print("-" * 40)
        
        for ch in COMPARISON_CHANNELS:
            if ch in raster_results['threshold_spikes'] and ch in raster_results['waveclus_spikes']:
                thresh_spikes = len(raster_results['threshold_spikes'][ch])
                waveclus_spikes = len(raster_results['waveclus_spikes'][ch])
                ratio = waveclus_spikes / thresh_spikes if thresh_spikes > 0 else 0
                print(f"   {ch:<8} {thresh_spikes:<10} {waveclus_spikes:<10} {ratio:<8.2f}")
        
        # Method comparison insights
        print(f"\n💡 Method Comparison Insights:")
        if waveclus_total > thresh_total:
            pct_increase = ((waveclus_total/thresh_total - 1)*100)
            print(f"   • PyWaveClus detected {waveclus_total - thresh_total} more spikes ({pct_increase:.1f}% increase)")
            print(f"   • Advanced filtering may be more sensitive to smaller spikes")
        elif thresh_total > waveclus_total:
            pct_increase = ((thresh_total/waveclus_total - 1)*100)
            print(f"   • Threshold method detected {thresh_total - waveclus_total} more spikes ({pct_increase:.1f}% increase)")
            print(f"   • PyWaveClus may be more selective due to noise rejection")
        else:
            print(f"   • Both methods detected similar numbers of spikes")
        
        rate_diff = abs(waveclus_total/duration - thresh_total/duration)
        print(f"   • Rate difference: {rate_diff:.1f} spikes/s")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        if rate_diff > 10:
            print(f"   • Large rate difference suggests parameter tuning may be needed")
        else:
            print(f"   • Similar rates suggest both methods are well-calibrated")
        
        print(f"   • Consider using PyWaveClus for higher sensitivity")
        print(f"   • Consider using threshold method for faster processing")
        print(f"   • Validate results with manual spike annotation")
    
    else:
        print(f"   ❌ Failed to create raster comparison")
    
    print(f"\n✅ Spike detection comparison demo complete!")
    print(f"🎯 Key files created:")
    print(f"   • utils/spike_detection.py - Main spike detection module")
    print(f"   • spike_detection_demo.py - This demonstration script")
    print(f"\n📚 Next steps:")
    print(f"   • Run this demo on different trials")
    print(f"   • Tune parameters for your specific data")
    print(f"   • Integrate into your analysis pipeline")

if __name__ == "__main__":
    main() 