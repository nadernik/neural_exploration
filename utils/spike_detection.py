"""
Spike Detection Utilities
=========================

PyWaveClus-style spike detection algorithms for neural decoding.
Implements threshold-based spike detection without clustering.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import h5py

class SpikeDetector:
    """
    PyWaveClus-style spike detection without clustering.
    
    Implements threshold-based spike detection with waveform extraction
    and basic feature computation for neural decoding.
    """
    
    def __init__(self, sampling_rate: float = 30000.0, 
                 threshold_factor: float = 5.0,
                 spike_window: Tuple[int, int] = (-10, 32),
                 good_channels: Optional[List[int]] = None):
        """
        Initialize spike detector.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz (default: 30kHz)
        threshold_factor : float
            Threshold factor for spike detection (default: 5.0)
        spike_window : tuple
            Samples before and after spike peak (default: (-10, 32))
        good_channels : list
            List of good channel indices to use
        """
        self.sampling_rate = sampling_rate
        self.threshold_factor = threshold_factor
        self.spike_window = spike_window
        self.spike_length = spike_window[1] - spike_window[0]
        
        # Default good channels from neural_feature_exploration
        self.good_channels = good_channels or [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
        
        # Initialize filters
        self._init_filters()
        
        print(f"🔍 SpikeDetector initialized:")
        print(f"  • Sampling rate: {self.sampling_rate/1000:.1f} kHz")
        print(f"  • Threshold factor: {self.threshold_factor}")
        print(f"  • Spike window: {self.spike_window}")
        print(f"  • Good channels: {len(self.good_channels)}")
    
    def _init_filters(self):
        """Initialize bandpass filters for spike detection."""
        # Bandpass filter for spike detection (300-3000 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 300.0 / nyquist
        high_freq = 3000.0 / nyquist
        
        self.bp_filter = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # High-pass filter for artifact removal (500 Hz)
        hp_freq = 500.0 / nyquist
        self.hp_filter = signal.butter(2, hp_freq, btype='high')
        
        print("🔧 Filters initialized (300-3000 Hz bandpass, 500 Hz highpass)")
    
    def preprocess_signal(self, neural_data: np.ndarray) -> np.ndarray:
        """
        Preprocess neural signals for spike detection.
        
        Parameters:
        -----------
        neural_data : np.ndarray
            Raw neural data (channels x samples)
            
        Returns:
        --------
        np.ndarray
            Preprocessed neural data
        """
        print(f"🔧 Preprocessing neural data: {neural_data.shape}")
        
        # Apply bandpass filter
        filtered_data = signal.filtfilt(self.bp_filter[0], self.bp_filter[1], neural_data, axis=1)
        
        # Apply high-pass filter to remove slow drifts
        filtered_data = signal.filtfilt(self.hp_filter[0], self.hp_filter[1], filtered_data, axis=1)
        
        print("✅ Signal preprocessing complete")
        return filtered_data
    
    def detect_spikes_channel(self, channel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect spikes in a single channel using PyWaveClus-style detection.
        
        Parameters:
        -----------
        channel_data : np.ndarray
            Single channel neural data
            
        Returns:
        --------
        tuple
            (spike_times, spike_waveforms)
        """
        # Calculate threshold using median-based noise estimation
        noise_std = np.median(np.abs(channel_data)) / 0.6745
        threshold = -self.threshold_factor * noise_std  # Negative for negative spikes
        
        # Find crossings below threshold
        crossings = np.where(channel_data < threshold)[0]
        
        if len(crossings) == 0:
            return np.array([]), np.array([]).reshape(0, self.spike_length)
        
        # Remove crossings too close to edges
        window_start = abs(self.spike_window[0])
        window_end = abs(self.spike_window[1])
        valid_crossings = crossings[(crossings >= window_start) & 
                                   (crossings < len(channel_data) - window_end)]
        
        if len(valid_crossings) == 0:
            return np.array([]), np.array([]).reshape(0, self.spike_length)
        
        # Remove consecutive crossings (keep only local minima)
        spike_times = []
        spike_waveforms = []
        
        i = 0
        while i < len(valid_crossings):
            current_crossing = valid_crossings[i]
            
            # Find the actual minimum in a small window around the crossing
            search_window = slice(max(0, current_crossing - 5), 
                                min(len(channel_data), current_crossing + 6))
            local_min_idx = np.argmin(channel_data[search_window])
            spike_time = search_window.start + local_min_idx
            
            # Skip if too close to previous spike
            if spike_times and spike_time - spike_times[-1] < 20:  # 20 samples ~ 0.67ms refractory period
                i += 1
                continue
            
            # Extract waveform
            start_idx = spike_time + self.spike_window[0]
            end_idx = spike_time + self.spike_window[1]
            
            if start_idx >= 0 and end_idx <= len(channel_data):
                waveform = channel_data[start_idx:end_idx]
                spike_times.append(spike_time)
                spike_waveforms.append(waveform)
            
            i += 1
        
        return np.array(spike_times), np.array(spike_waveforms)
    
    def detect_spikes_all_channels(self, neural_data: np.ndarray) -> Dict[int, Dict]:
        """
        Detect spikes across all good channels.
        
        Parameters:
        -----------
        neural_data : np.ndarray
            Neural data (channels x samples)
            
        Returns:
        --------
        dict
            Dictionary with channel-wise spike data
        """
        print(f"🔍 Detecting spikes across {len(self.good_channels)} channels...")
        
        # Preprocess data
        filtered_data = self.preprocess_signal(neural_data)
        
        spike_data = {}
        total_spikes = 0
        
        for channel_idx in self.good_channels:
            if channel_idx >= neural_data.shape[0]:
                continue
                
            channel_data = filtered_data[channel_idx, :]
            spike_times, spike_waveforms = self.detect_spikes_channel(channel_data)
            
            spike_data[channel_idx] = {
                'spike_times': spike_times,
                'spike_waveforms': spike_waveforms,
                'n_spikes': len(spike_times)
            }
            
            total_spikes += len(spike_times)
        
        print(f"✅ Spike detection complete: {total_spikes} spikes across {len(spike_data)} channels")
        return spike_data
    
    def compute_firing_rates(self, spike_data: Dict[int, Dict], 
                           duration: float, bin_size: float = 0.05) -> Dict[int, np.ndarray]:
        """
        Compute binned firing rates from spike times.
        
        Parameters:
        -----------
        spike_data : dict
            Spike data from detect_spikes_all_channels
        duration : float
            Total duration in seconds
        bin_size : float
            Bin size in seconds (default: 50ms)
            
        Returns:
        --------
        dict
            Dictionary with channel-wise firing rates
        """
        print(f"🔥 Computing firing rates (bin size: {bin_size*1000:.0f}ms)")
        
        n_bins = int(np.ceil(duration / bin_size))
        firing_rates = {}
        
        for channel_idx, data in spike_data.items():
            spike_times = data['spike_times'] / self.sampling_rate  # Convert to seconds
            
            # Bin spikes
            counts, _ = np.histogram(spike_times, bins=n_bins, range=(0, duration))
            
            # Convert to firing rate (spikes/second)
            rates = counts / bin_size
            firing_rates[channel_idx] = rates
        
        print(f"✅ Firing rates computed: {n_bins} bins x {len(firing_rates)} channels")
        return firing_rates
    
    def extract_features_from_h5(self, h5_file_path: str, 
                                trial_number: int) -> Dict[int, np.ndarray]:
        """
        Extract spike features from H5 file for a specific trial.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to H5 file
        trial_number : int
            Trial number to process
            
        Returns:
        --------
        dict
            Dictionary with channel-wise firing rates
        """
        print(f"🔍 Extracting spike features from trial {trial_number}")
        
        with h5py.File(h5_file_path, 'r') as f:
            trial_key = f'trial_{trial_number}'
            
            if trial_key not in f:
                raise ValueError(f"Trial {trial_number} not found in H5 file")
            
            trial_group = f[trial_key]
            
            # Get neural data - try different key names
            possible_neural_keys = ['neural_data', 'neural', 'data', 'signals']
            neural_data = None
            
            for key in possible_neural_keys:
                if key in trial_group:
                    neural_data = trial_group[key][:]
                    break
            
            if neural_data is None:
                available_keys = list(trial_group.keys())
                raise KeyError(f"No neural data found in trial {trial_number}. Available keys: {available_keys}")
            
            # Get trial duration from metadata
            metadata = trial_group.attrs
            duration = metadata.get('duration', neural_data.shape[1] / self.sampling_rate)
            
            # Detect spikes
            spike_data = self.detect_spikes_all_channels(neural_data)
            
            # Compute firing rates
            firing_rates = self.compute_firing_rates(spike_data, duration)
            
            print(f"✅ Features extracted for trial {trial_number}")
            return firing_rates
    
    def get_channel_quality_metrics(self, spike_data: Dict[int, Dict]) -> pd.DataFrame:
        """
        Compute quality metrics for each channel.
        
        Parameters:
        -----------
        spike_data : dict
            Spike data from detect_spikes_all_channels
            
        Returns:
        --------
        pd.DataFrame
            Channel quality metrics
        """
        metrics = []
        
        for channel_idx, data in spike_data.items():
            spike_waveforms = data['spike_waveforms']
            n_spikes = data['n_spikes']
            
            if n_spikes > 0:
                # Compute waveform statistics
                mean_waveform = np.mean(spike_waveforms, axis=0)
                std_waveform = np.std(spike_waveforms, axis=0)
                
                # Peak-to-peak amplitude
                peak_to_peak = np.max(mean_waveform) - np.min(mean_waveform)
                
                # SNR estimate
                noise_std = np.mean(std_waveform)
                snr = peak_to_peak / noise_std if noise_std > 0 else 0
                
                # Waveform consistency (inverse of coefficient of variation)
                consistency = 1 / (noise_std / peak_to_peak) if peak_to_peak > 0 else 0
                
            else:
                peak_to_peak = 0
                snr = 0
                consistency = 0
            
            metrics.append({
                'channel': channel_idx,
                'n_spikes': n_spikes,
                'peak_to_peak': peak_to_peak,
                'snr': snr,
                'consistency': consistency
            })
        
        return pd.DataFrame(metrics) 