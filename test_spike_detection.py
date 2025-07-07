#!/usr/bin/env python3
"""
Test Spike Detection Functionality
==================================

Simple test to verify the spike detection comparison works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.spike_detection import SpikeDetector

def test_spike_detection():
    """Test spike detection on synthetic data."""
    
    print("🧪 Testing Spike Detection Functionality")
    print("=" * 45)
    
    # Create synthetic neural signal
    fs = 30000  # sampling rate
    duration = 1.0  # 1 second
    t = np.arange(0, duration * fs) / fs
    
    # Create base signal with noise
    signal = np.random.randn(len(t)) * 10  # 10 μV noise
    
    # Add some synthetic spikes
    spike_times = [0.1, 0.3, 0.5, 0.7, 0.9]  # spike times in seconds
    spike_amplitude = -100  # μV
    spike_width = 0.001  # 1 ms
    
    for spike_time in spike_times:
        spike_idx = int(spike_time * fs)
        spike_samples = int(spike_width * fs)
        
        # Add spike waveform
        if spike_idx + spike_samples < len(signal):
            spike_waveform = spike_amplitude * np.exp(-np.linspace(0, 5, spike_samples))
            signal[spike_idx:spike_idx+spike_samples] += spike_waveform
    
    print(f"✅ Created synthetic signal:")
    print(f"   Duration: {duration}s")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   True spikes: {len(spike_times)}")
    print(f"   Signal RMS: {np.sqrt(np.mean(signal**2)):.1f} μV")
    
    # Test spike detection
    detector = SpikeDetector(sampling_rate=fs)
    
    # Compare methods
    comparison = detector.compare_methods(signal)
    
    print(f"\n📊 DETECTION RESULTS:")
    print(f"   Threshold method: {comparison['n_threshold_spikes']} spikes detected")
    print(f"   PyWaveClus method: {comparison['n_waveclus_spikes']} spikes detected")
    print(f"   True spikes: {len(spike_times)}")
    
    # Check accuracy
    thresh_accuracy = abs(comparison['n_threshold_spikes'] - len(spike_times)) / len(spike_times)
    waveclus_accuracy = abs(comparison['n_waveclus_spikes'] - len(spike_times)) / len(spike_times)
    
    print(f"\n🎯 ACCURACY:")
    print(f"   Threshold method error: {thresh_accuracy:.1%}")
    print(f"   PyWaveClus method error: {waveclus_accuracy:.1%}")
    
    if thresh_accuracy < 0.5 and waveclus_accuracy < 0.5:
        print("✅ Both methods show reasonable accuracy!")
    else:
        print("⚠️  High detection error - consider parameter tuning")
    
    # Plot comparison
    detector.plot_detection_comparison(
        signal_data=signal,
        comparison_result=comparison,
        time_window=(0, 1.0),
        figsize=(12, 8)
    )
    
    print(f"\n✅ Spike detection test completed!")
    return True

if __name__ == "__main__":
    test_spike_detection() 