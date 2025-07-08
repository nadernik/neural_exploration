"""
Neural-Behavioral Alignment Utilities
=====================================

Tools for aligning neural features with behavioral data using H5 files.
Handles time alignment and synchronization between neural and behavioral data.
"""

import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
import matplotlib.pyplot as plt

class NeuralBehavioralAligner:
    """
    Aligns neural features with behavioral data using H5 file structure.
    
    Handles time alignment, interpolation, and synchronization between
    neural firing rates and cursor velocity data.
    """
    
    def __init__(self, bin_size: float = 0.05, 
                 interpolation_method: str = 'linear'):
        """
        Initialize neural-behavioral aligner.
        
        Parameters:
        -----------
        bin_size : float
            Time bin size in seconds (default: 50ms)
        interpolation_method : str
            Method for interpolating behavioral data ('linear', 'nearest', 'cubic')
        """
        self.bin_size = bin_size
        self.interpolation_method = interpolation_method
        
        print(f"🔗 NeuralBehavioralAligner initialized:")
        print(f"  • Bin size: {self.bin_size*1000:.0f}ms")
        print(f"  • Interpolation: {self.interpolation_method}")
    
    def load_trial_data(self, h5_file_path: str, trial_number: int) -> Dict:
        """
        Load neural and behavioral data for a specific trial.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to H5 file
        trial_number : int
            Trial number to load
            
        Returns:
        --------
        dict
            Dictionary containing neural and behavioral data
        """
        print(f"📂 Loading trial {trial_number} data...")
        
        with h5py.File(h5_file_path, 'r') as f:
            trial_key = f'trial_{trial_number}'
            
            if trial_key not in f:
                raise ValueError(f"Trial {trial_number} not found in H5 file")
            
            trial_group = f[trial_key]
            
            # Load neural data - try different key names
            possible_neural_keys = ['neural_data', 'neural', 'data', 'signals']
            neural_data = None
            
            for key in possible_neural_keys:
                if key in trial_group:
                    neural_data = trial_group[key][:]
                    break
            
            if neural_data is None:
                available_keys = list(trial_group.keys())
                raise KeyError(f"No neural data found in trial {trial_number}. Available keys: {available_keys}")
            
            # Load behavioral data
            velocity_x = trial_group['velocity_x'][:]
            velocity_y = trial_group['velocity_y'][:]
            
            # Load timestamps
            behavioral_timestamps = trial_group['behavioral_timestamps'][:]
            
            # Load metadata
            metadata = dict(trial_group.attrs)
            
            trial_data = {
                'neural_data': neural_data,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'behavioral_timestamps': behavioral_timestamps,
                'metadata': metadata,
                'trial_number': trial_number
            }
            
            print(f"✅ Trial {trial_number} loaded:")
            print(f"  • Neural data: {neural_data.shape}")
            print(f"  • Velocity data: {len(velocity_x)} samples")
            print(f"  • Duration: {metadata.get('duration', 'unknown'):.2f}s")
            
            return trial_data
    
    def create_time_bins(self, duration: float) -> np.ndarray:
        """
        Create time bins for alignment.
        
        Parameters:
        -----------
        duration : float
            Total duration in seconds
            
        Returns:
        --------
        np.ndarray
            Array of time bin centers
        """
        n_bins = int(np.ceil(duration / self.bin_size))
        time_bins = np.linspace(self.bin_size/2, duration - self.bin_size/2, n_bins)
        return time_bins
    
    def align_neural_features(self, neural_firing_rates: Dict[int, np.ndarray], 
                            duration: float) -> np.ndarray:
        """
        Align neural firing rates to time bins.
        
        Parameters:
        -----------
        neural_firing_rates : dict
            Dictionary of firing rates by channel
        duration : float
            Trial duration in seconds
            
        Returns:
        --------
        np.ndarray
            Aligned neural features (n_bins x n_channels)
        """
        time_bins = self.create_time_bins(duration)
        n_bins = len(time_bins)
        
        # Get channel indices
        channel_indices = sorted(neural_firing_rates.keys())
        n_channels = len(channel_indices)
        
        # Create feature matrix
        neural_features = np.zeros((n_bins, n_channels))
        
        for i, channel_idx in enumerate(channel_indices):
            firing_rates = neural_firing_rates[channel_idx]
            
            # Ensure firing rates match time bins
            if len(firing_rates) == n_bins:
                neural_features[:, i] = firing_rates
            else:
                # Interpolate if lengths don't match
                old_time = np.linspace(0, duration, len(firing_rates))
                interp_func = interpolate.interp1d(old_time, firing_rates, 
                                                 kind=self.interpolation_method,
                                                 bounds_error=False, fill_value=0)
                neural_features[:, i] = interp_func(time_bins)
        
        return neural_features
    
    def align_behavioral_data(self, trial_data: Dict) -> np.ndarray:
        """
        Align behavioral data to time bins.
        
        Parameters:
        -----------
        trial_data : dict
            Trial data from load_trial_data
            
        Returns:
        --------
        np.ndarray
            Aligned behavioral data (n_bins x 2) for [velocity_x, velocity_y]
        """
        velocity_x = trial_data['velocity_x']
        velocity_y = trial_data['velocity_y']
        behavioral_timestamps = trial_data['behavioral_timestamps']
        
        # Convert absolute timestamps to relative time
        # Check if timestamps are absolute (Unix epoch) and convert to relative
        if behavioral_timestamps.max() > 1000000:  # Likely Unix timestamp
            # Get trial start time
            if 'start_seconds' in trial_data.get('metadata', {}):
                trial_start = trial_data['metadata']['start_seconds']
                if trial_start < 1000000:  # start_seconds is relative, use first timestamp
                    trial_start = behavioral_timestamps[0]
            else:
                trial_start = behavioral_timestamps[0]
            behavioral_time = behavioral_timestamps - trial_start
        else:
            behavioral_time = behavioral_timestamps
        
        duration = trial_data['metadata'].get('duration', 
                                             behavioral_time[-1] - behavioral_time[0])
        
        # Create time bins
        time_bins = self.create_time_bins(duration)
        
        # Interpolate behavioral data to time bins
        behavioral_data = np.zeros((len(time_bins), 2))
        
        # Interpolate velocity_x
        if len(velocity_x) > 1:
            interp_func_x = interpolate.interp1d(behavioral_time, velocity_x,
                                               kind=self.interpolation_method,
                                               bounds_error=False, fill_value=0)
            behavioral_data[:, 0] = interp_func_x(time_bins)
        
        # Interpolate velocity_y
        if len(velocity_y) > 1:
            interp_func_y = interpolate.interp1d(behavioral_time, velocity_y,
                                               kind=self.interpolation_method,
                                               bounds_error=False, fill_value=0)
            behavioral_data[:, 1] = interp_func_y(time_bins)
        
        return behavioral_data
    
    def create_aligned_dataset(self, neural_firing_rates: Dict[int, np.ndarray], 
                             trial_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create aligned neural-behavioral dataset.
        
        Parameters:
        -----------
        neural_firing_rates : dict
            Neural firing rates by channel
        trial_data : dict
            Trial data from load_trial_data
            
        Returns:
        --------
        tuple
            (neural_features, behavioral_targets) aligned arrays
        """
        duration = trial_data['metadata'].get('duration', 
                                             trial_data['behavioral_timestamps'][-1] - trial_data['behavioral_timestamps'][0])
        
        # Align neural features
        neural_features = self.align_neural_features(neural_firing_rates, duration)
        
        # Align behavioral data
        behavioral_targets = self.align_behavioral_data(trial_data)
        
        # Ensure same number of time bins
        min_bins = min(neural_features.shape[0], behavioral_targets.shape[0])
        neural_features = neural_features[:min_bins, :]
        behavioral_targets = behavioral_targets[:min_bins, :]
        
        return neural_features, behavioral_targets
    
    def process_multiple_trials(self, h5_file_path: str, 
                              trial_numbers: List[int],
                              spike_detector) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process multiple trials and create combined dataset.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to H5 file
        trial_numbers : list
            List of trial numbers to process
        spike_detector : SpikeDetector
            Spike detector instance
            
        Returns:
        --------
        tuple
            (combined_neural_features, combined_behavioral_targets)
        """
        print(f"🔄 Processing {len(trial_numbers)} trials...")
        
        all_neural_features = []
        all_behavioral_targets = []
        
        for trial_num in trial_numbers:
            try:
                # Load trial data
                trial_data = self.load_trial_data(h5_file_path, trial_num)
                
                # Extract neural features
                neural_firing_rates = spike_detector.extract_features_from_h5(h5_file_path, trial_num)
                
                # Create aligned dataset
                neural_features, behavioral_targets = self.create_aligned_dataset(
                    neural_firing_rates, trial_data)
                
                all_neural_features.append(neural_features)
                all_behavioral_targets.append(behavioral_targets)
                
                print(f"  ✅ Trial {trial_num}: {neural_features.shape[0]} time bins")
                
            except Exception as e:
                print(f"  ❌ Trial {trial_num}: {str(e)}")
                continue
        
        if not all_neural_features:
            raise ValueError("No trials were successfully processed")
        
        # Combine all trials
        combined_neural_features = np.vstack(all_neural_features)
        combined_behavioral_targets = np.vstack(all_behavioral_targets)
        
        print(f"✅ Combined dataset created:")
        print(f"  • Neural features: {combined_neural_features.shape}")
        print(f"  • Behavioral targets: {combined_behavioral_targets.shape}")
        
        return combined_neural_features, combined_behavioral_targets
    
    def visualize_alignment(self, neural_features: np.ndarray, 
                          behavioral_targets: np.ndarray,
                          channel_indices: List[int] = None,
                          max_channels: int = 5) -> plt.Figure:
        """
        Visualize neural-behavioral alignment.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features (n_bins x n_channels)
        behavioral_targets : np.ndarray
            Behavioral targets (n_bins x 2)
        channel_indices : list, optional
            Specific channels to plot
        max_channels : int
            Maximum number of channels to plot
            
        Returns:
        --------
        plt.Figure
            Figure showing alignment
        """
        n_bins, n_channels = neural_features.shape
        time_axis = np.arange(n_bins) * self.bin_size
        
        if channel_indices is None:
            channel_indices = list(range(min(max_channels, n_channels)))
        
        fig, axes = plt.subplots(len(channel_indices) + 1, 1, figsize=(12, 8))
        
        # Plot behavioral data
        axes[0].plot(time_axis, behavioral_targets[:, 0], 'b-', label='Velocity X', alpha=0.7)
        axes[0].plot(time_axis, behavioral_targets[:, 1], 'r-', label='Velocity Y', alpha=0.7)
        axes[0].set_ylabel('Velocity (units/s)')
        axes[0].set_title('Behavioral Data (Cursor Velocity)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot neural data
        for i, ch_idx in enumerate(channel_indices):
            if ch_idx < n_channels:
                axes[i+1].plot(time_axis, neural_features[:, ch_idx], 'g-', alpha=0.7)
                axes[i+1].set_ylabel('Firing Rate (Hz)')
                axes[i+1].set_title(f'Neural Channel {ch_idx}')
                axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        return fig
    
    def get_alignment_quality_metrics(self, neural_features: np.ndarray,
                                    behavioral_targets: np.ndarray) -> Dict:
        """
        Compute alignment quality metrics.
        
        Parameters:
        -----------
        neural_features : np.ndarray
            Neural features
        behavioral_targets : np.ndarray
            Behavioral targets
            
        Returns:
        --------
        dict
            Quality metrics
        """
        n_bins, n_channels = neural_features.shape
        
        # Compute correlations between neural and behavioral data
        correlations_x = []
        correlations_y = []
        
        for ch in range(n_channels):
            # Correlation with velocity_x
            corr_x = np.corrcoef(neural_features[:, ch], behavioral_targets[:, 0])[0, 1]
            correlations_x.append(corr_x if not np.isnan(corr_x) else 0)
            
            # Correlation with velocity_y
            corr_y = np.corrcoef(neural_features[:, ch], behavioral_targets[:, 1])[0, 1]
            correlations_y.append(corr_y if not np.isnan(corr_y) else 0)
        
        # Behavioral data statistics
        velocity_magnitude = np.sqrt(behavioral_targets[:, 0]**2 + behavioral_targets[:, 1]**2)
        
        metrics = {
            'n_time_bins': n_bins,
            'n_channels': n_channels,
            'bin_size_ms': self.bin_size * 1000,
            'total_duration_s': n_bins * self.bin_size,
            'neural_correlations_x': correlations_x,
            'neural_correlations_y': correlations_y,
            'max_correlation_x': np.max(np.abs(correlations_x)),
            'max_correlation_y': np.max(np.abs(correlations_y)),
            'mean_velocity_magnitude': np.mean(velocity_magnitude),
            'std_velocity_magnitude': np.std(velocity_magnitude),
            'behavioral_data_range': {
                'velocity_x': [np.min(behavioral_targets[:, 0]), np.max(behavioral_targets[:, 0])],
                'velocity_y': [np.min(behavioral_targets[:, 1]), np.max(behavioral_targets[:, 1])]
            }
        }
        
        return metrics 