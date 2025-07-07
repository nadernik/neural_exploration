#!/usr/bin/env python3
"""
Spike Detection Reliability Assessment Demo
==========================================

This script demonstrates how to assess PyWaveClus spike detection reliability
by visualizing raw neural data, filtered data, and detected spikes for a 
selected channel and trial.

Usage:
    python spike_detection_reliability_demo.py

Author: Neural Exploration Team
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_trial_data
from utils.diagnostics import diagnose_trial_data
from utils.visualization import plot_spike_detection_reliability
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main demonstration function."""
    
    print("🔍 SPIKE DETECTION RELIABILITY ASSESSMENT")
    print("=" * 50)
    
    # Configuration
    TRIAL_NUMBER = 10  # Trial with good movement data
    SAMPLING_RATE = 30000  # Hz
    SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
    
    # Select a channel for detailed analysis (most active channel)
    TEST_CHANNEL = 0  # Channel index to analyze
    
    print(f"📊 Configuration:")
    print(f"   Trial: {TRIAL_NUMBER}")
    print(f"   Channel: {TEST_CHANNEL}")
    print(f"   Sampling rate: {SAMPLING_RATE} Hz")
    
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
    
    # Step 2: Full trial reliability assessment
    print(f"\n🔍 Full trial reliability assessment...")
    
    plot_spike_detection_reliability(
        trial_data=trial_data,
        channel_idx=TEST_CHANNEL,
        trial_number=TRIAL_NUMBER,
        time_window=None,  # Full trial
        sampling_rate=SAMPLING_RATE,
        figsize=(15, 12)
    )
    
    # Step 3: Zoomed-in view for detailed analysis
    print(f"\n🔍 Zoomed reliability assessment (2-4 seconds)...")
    
    # Select a 2-second window for detailed view
    duration = trial_data.get('duration', trial_data['neural_data'].shape[1] / SAMPLING_RATE)
    start_time = max(2.0, duration * 0.2)  # Start at 2s or 20% through trial
    end_time = min(start_time + 2.0, duration - 0.5)  # 2-second window
    
    plot_spike_detection_reliability(
        trial_data=trial_data,
        channel_idx=TEST_CHANNEL,
        trial_number=TRIAL_NUMBER,
        time_window=(start_time, end_time),
        sampling_rate=SAMPLING_RATE,
        figsize=(15, 12)
    )
    
    # Step 4: Compare multiple channels
    print(f"\n🔍 Comparing multiple channels...")
    
    # Test several channels for comparison
    comparison_channels = SPIKE_CHANNELS[:4]  # First 4 channels
    
    for i, channel in enumerate(comparison_channels):
        if channel < trial_data['neural_data'].shape[0]:
            print(f"\n--- Channel {channel} Analysis ---")
            
            # Quick assessment with a focused time window
            plot_spike_detection_reliability(
                trial_data=trial_data,
                channel_idx=channel,
                trial_number=TRIAL_NUMBER,
                time_window=(start_time, end_time),
                sampling_rate=SAMPLING_RATE,
                figsize=(12, 8)
            )
            
            # Ask user if they want to continue
            if i < len(comparison_channels) - 1:
                response = input("\nPress Enter to continue to next channel, or 'q' to quit: ")
                if response.lower() == 'q':
                    break
    
    # Step 5: Custom analysis
    print(f"\n🎯 Custom Analysis Options:")
    print(f"   You can now run custom analyses by calling:")
    print(f"   plot_spike_detection_reliability(trial_data, channel_idx, trial_number)")
    print(f"\n   Available parameters:")
    print(f"   • channel_idx: Channel to analyze (0-{trial_data['neural_data'].shape[0]-1})")
    print(f"   • time_window: (start, end) in seconds for zoomed view")
    print(f"   • trial_number: Trial number for display")
    print(f"   • sampling_rate: Sampling rate in Hz")
    print(f"   • figsize: Figure size tuple")
    
    print(f"\n✅ Spike detection reliability assessment complete!")

def analyze_specific_channel(trial_data, channel_idx, trial_number=None, 
                           time_window=None, sampling_rate=30000):
    """
    Convenient function for analyzing a specific channel.
    
    Args:
        trial_data: Trial data dictionary
        channel_idx: Channel index to analyze
        trial_number: Trial number for display
        time_window: Optional (start, end) time window in seconds
        sampling_rate: Sampling rate in Hz
    """
    
    plot_spike_detection_reliability(
        trial_data=trial_data,
        channel_idx=channel_idx,
        trial_number=trial_number,
        time_window=time_window,
        sampling_rate=sampling_rate,
        figsize=(15, 10)
    )

if __name__ == "__main__":
    main()
    
    print(f"\n💡 Usage Tips:")
    print(f"   • Use time_window=(start, end) for detailed analysis of specific periods")
    print(f"   • Compare channels to find the most reliable spike detection")
    print(f"   • Look for consistency between PyWaveClus and threshold methods")
    print(f"   • Pay attention to signal quality metrics (SNR, noise level)")
    print(f"   • Higher SNR (>3) indicates better spike detection reliability")
    print(f"   • PyWaveClus typically detects 10-30% more spikes than threshold methods") 