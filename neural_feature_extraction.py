#!/usr/bin/env python3
"""
Neural Feature Extraction Script

Extracts various neural signal features from selected spike channels:
1. Signal Features (Spike-free): Band power (400-6000 Hz) in time bins
2. Local Field Potential (LFP): Low-pass filtered (<250 Hz) with band-specific power
3. Simple voltage features: Moving average and variance
4. Thresholded spike counts: Threshold crossings per bin

Author: Neural Exploration Assistant
"""

import numpy as np
import h5py
import os
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
H5_FILE = r'D:\Data\ScienceCorp\trials_aligned.h5'
TRIAL_NUMBER = 1  # Trial to analyze

# Fixed H5 file path
def find_h5_file():
    """Return the fixed H5 file path."""
    return H5_FILE

# Manually identified spike channels (from manual inspection)
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]

# Signal processing parameters
SAMPLING_RATE = 30000  # Hz
TIME_BIN_SIZE = 0.05   # seconds (50 ms bins)
TIME_BIN_SIZE_MS = int(TIME_BIN_SIZE * 1000)  # for display

# Filter parameters
SPIKE_BAND_LOW = 400    # Hz - lower bound for spike band
SPIKE_BAND_HIGH = 6000  # Hz - upper bound for spike band
LFP_CUTOFF = 250       # Hz - LFP low-pass cutoff
GAMMA_LOW = 30         # Hz - gamma band lower bound
GAMMA_HIGH = 100       # Hz - gamma band upper bound

# Threshold parameters
THRESHOLD_MULTIPLIER = -4  # Negative threshold multiplier (e.g., -4x RMS)

class NeuralFeatureExtractor:
    """Extract various neural signal features from selected channels."""
    
    def __init__(self, sampling_rate: int = 30000):
        self.fs = sampling_rate
        self.nyquist = self.fs / 2
        
    def load_trial_data(self, h5_file: str, trial_number: int) -> Dict:
        """Load neural data for a specific trial."""
        with h5py.File(h5_file, 'r') as f:
            # Try different trial key formats
            possible_keys = [
                f'trial_{trial_number:03d}',  # trial_001
                f'trial_{trial_number}',      # trial_1
                f'Trial_{trial_number:03d}',  # Trial_001
                f'Trial_{trial_number}',      # Trial_1
            ]
            
            trial_key = None
            for key in possible_keys:
                if key in f:
                    trial_key = key
                    break
            
            if trial_key is None:
                available_keys = list(f.keys())
                raise KeyError(f"Trial {trial_number} not found in {h5_file}. Available keys: {available_keys}")
            
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
                raise KeyError(f"No neural data found in trial. Available keys: {available_keys}")
            
            # Load metadata
            metadata = {}
            for key in trial_group.attrs.keys():
                metadata[key] = trial_group.attrs[key]
            
            # Load behavioral data if available
            behavioral_data = {}
            if 'velocity_x' in trial_group and trial_group['velocity_x'] is not None:
                behavioral_data['velocity_x'] = trial_group['velocity_x'][:]
                behavioral_data['velocity_y'] = trial_group['velocity_y'][:]
                behavioral_data['behavioral_timestamps'] = trial_group['behavioral_timestamps'][:]
            
            return {
                'neural_data': neural_data,
                'metadata': metadata,
                'behavioral_data': behavioral_data,
                'trial_number': trial_number
            }
    
    def create_time_bins(self, n_samples: int, bin_size: float) -> np.ndarray:
        """Create time bins for feature extraction."""
        samples_per_bin = int(bin_size * self.fs)
        n_bins = n_samples // samples_per_bin
        
        # Create bin edges
        bin_edges = np.arange(0, n_bins + 1) * samples_per_bin
        return bin_edges
    
    def extract_spike_band_power(self, neural_data: np.ndarray, bin_size: float) -> Dict:
        """
        Extract spike-free band power (400-6000 Hz).
        
        Args:
            neural_data: Shape (channels, samples)
            bin_size: Time bin size in seconds
            
        Returns:
            Dictionary with band power features
        """
        print(f"🔍 Extracting spike band power ({SPIKE_BAND_LOW}-{SPIKE_BAND_HIGH} Hz)...")
        
        # Design bandpass filter
        sos = signal.butter(4, [SPIKE_BAND_LOW/self.nyquist, SPIKE_BAND_HIGH/self.nyquist], 
                           btype='band', output='sos')
        
        # Create time bins
        bin_edges = self.create_time_bins(neural_data.shape[1], bin_size)
        n_bins = len(bin_edges) - 1
        
        # Initialize output arrays
        n_channels = neural_data.shape[0]
        rms_power = np.zeros((n_channels, n_bins))
        log_power = np.zeros((n_channels, n_bins))
        
        for ch_idx in range(n_channels):
            # Bandpass filter
            filtered_signal = signal.sosfilt(sos, neural_data[ch_idx])
            
            # Compute power in each time bin
            for bin_idx in range(n_bins):
                start_idx = bin_edges[bin_idx]
                end_idx = bin_edges[bin_idx + 1]
                
                bin_data = filtered_signal[start_idx:end_idx]
                
                # RMS power
                rms_power[ch_idx, bin_idx] = np.sqrt(np.mean(bin_data**2))
                
                # Log power (add small constant to avoid log(0))
                log_power[ch_idx, bin_idx] = np.log10(np.mean(bin_data**2) + 1e-12)
        
        # Create time axis for bins (center of each bin)
        time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / self.fs
        
        return {
            'rms_power': rms_power,
            'log_power': log_power,
            'time_axis': time_axis,
            'frequency_band': f'{SPIKE_BAND_LOW}-{SPIKE_BAND_HIGH} Hz',
            'bin_size_ms': TIME_BIN_SIZE_MS
        }
    
    def extract_lfp_features(self, neural_data: np.ndarray, bin_size: float) -> Dict:
        """
        Extract Local Field Potential (LFP) features.
        
        Args:
            neural_data: Shape (channels, samples)
            bin_size: Time bin size in seconds
            
        Returns:
            Dictionary with LFP features
        """
        print(f"🧠 Extracting LFP features (low-pass <{LFP_CUTOFF} Hz, gamma {GAMMA_LOW}-{GAMMA_HIGH} Hz)...")
        
        # Design filters
        sos_lfp = signal.butter(4, LFP_CUTOFF/self.nyquist, btype='low', output='sos')
        sos_gamma = signal.butter(4, [GAMMA_LOW/self.nyquist, GAMMA_HIGH/self.nyquist], 
                                 btype='band', output='sos')
        
        # Create time bins
        bin_edges = self.create_time_bins(neural_data.shape[1], bin_size)
        n_bins = len(bin_edges) - 1
        
        # Initialize output arrays
        n_channels = neural_data.shape[0]
        lfp_power = np.zeros((n_channels, n_bins))
        gamma_power = np.zeros((n_channels, n_bins))
        gamma_amplitude = np.zeros((n_channels, n_bins))
        
        for ch_idx in range(n_channels):
            # LFP filtering
            lfp_signal = signal.sosfilt(sos_lfp, neural_data[ch_idx])
            
            # Gamma filtering
            gamma_signal = signal.sosfilt(sos_gamma, neural_data[ch_idx])
            
            # Gamma amplitude envelope using Hilbert transform
            gamma_analytic = signal.hilbert(gamma_signal)
            gamma_envelope = np.abs(gamma_analytic)
            
            # Compute features in each time bin
            for bin_idx in range(n_bins):
                start_idx = bin_edges[bin_idx]
                end_idx = bin_edges[bin_idx + 1]
                
                # LFP power
                lfp_bin = lfp_signal[start_idx:end_idx]
                lfp_power[ch_idx, bin_idx] = np.mean(lfp_bin**2)
                
                # Gamma power
                gamma_bin = gamma_signal[start_idx:end_idx]
                gamma_power[ch_idx, bin_idx] = np.mean(gamma_bin**2)
                
                # Gamma amplitude
                gamma_amp_bin = gamma_envelope[start_idx:end_idx]
                gamma_amplitude[ch_idx, bin_idx] = np.mean(gamma_amp_bin)
        
        # Create time axis for bins
        time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / self.fs
        
        return {
            'lfp_power': lfp_power,
            'gamma_power': gamma_power,
            'gamma_amplitude': gamma_amplitude,
            'time_axis': time_axis,
            'lfp_cutoff': LFP_CUTOFF,
            'gamma_band': f'{GAMMA_LOW}-{GAMMA_HIGH} Hz',
            'bin_size_ms': TIME_BIN_SIZE_MS
        }
    
    def extract_voltage_features(self, neural_data: np.ndarray, bin_size: float) -> Dict:
        """
        Extract simple voltage features (moving average and variance).
        
        Args:
            neural_data: Shape (channels, samples)
            bin_size: Time bin size in seconds
            
        Returns:
            Dictionary with voltage features
        """
        print(f"⚡ Extracting voltage features (moving average and variance)...")
        
        # Create time bins
        bin_edges = self.create_time_bins(neural_data.shape[1], bin_size)
        n_bins = len(bin_edges) - 1
        
        # Initialize output arrays
        n_channels = neural_data.shape[0]
        moving_avg = np.zeros((n_channels, n_bins))
        moving_var = np.zeros((n_channels, n_bins))
        
        for ch_idx in range(n_channels):
            # Compute features in each time bin
            for bin_idx in range(n_bins):
                start_idx = bin_edges[bin_idx]
                end_idx = bin_edges[bin_idx + 1]
                
                bin_data = neural_data[ch_idx, start_idx:end_idx]
                
                # Moving average
                moving_avg[ch_idx, bin_idx] = np.mean(bin_data)
                
                # Moving variance
                moving_var[ch_idx, bin_idx] = np.var(bin_data)
        
        # Create time axis for bins
        time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / self.fs
        
        return {
            'moving_average': moving_avg,
            'moving_variance': moving_var,
            'time_axis': time_axis,
            'bin_size_ms': TIME_BIN_SIZE_MS
        }
    
    def extract_threshold_crossings(self, neural_data: np.ndarray, bin_size: float) -> Dict:
        """
        Extract thresholded spike counts (threshold crossings per bin).
        
        Args:
            neural_data: Shape (channels, samples)
            bin_size: Time bin size in seconds
            
        Returns:
            Dictionary with threshold crossing features
        """
        print(f"🎯 Extracting threshold crossings ({THRESHOLD_MULTIPLIER}x RMS)...")
        
        # Create time bins
        bin_edges = self.create_time_bins(neural_data.shape[1], bin_size)
        n_bins = len(bin_edges) - 1
        
        # Initialize output arrays
        n_channels = neural_data.shape[0]
        crossing_counts = np.zeros((n_channels, n_bins))
        thresholds = np.zeros(n_channels)
        
        for ch_idx in range(n_channels):
            # Calculate threshold (negative, based on RMS)
            rms = np.sqrt(np.mean(neural_data[ch_idx]**2))
            threshold = THRESHOLD_MULTIPLIER * rms
            thresholds[ch_idx] = threshold
            
            # Count threshold crossings in each bin
            for bin_idx in range(n_bins):
                start_idx = bin_edges[bin_idx]
                end_idx = bin_edges[bin_idx + 1]
                
                bin_data = neural_data[ch_idx, start_idx:end_idx]
                
                # Count crossings below threshold
                crossings = np.sum(bin_data < threshold)
                crossing_counts[ch_idx, bin_idx] = crossings
        
        # Create time axis for bins
        time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2 / self.fs
        
        return {
            'crossing_counts': crossing_counts,
            'thresholds': thresholds,
            'time_axis': time_axis,
            'threshold_multiplier': THRESHOLD_MULTIPLIER,
            'bin_size_ms': TIME_BIN_SIZE_MS
        }
    
    def extract_all_features(self, neural_data: np.ndarray, selected_channels: List[int], 
                           bin_size: float = 0.05) -> Dict:
        """
        Extract all neural features for selected channels.
        
        Args:
            neural_data: Shape (channels, samples)
            selected_channels: List of channel indices to analyze
            bin_size: Time bin size in seconds
            
        Returns:
            Dictionary with all extracted features
        """
        # Select only the specified channels
        selected_data = neural_data[selected_channels]
        
        print(f"🚀 Extracting features for {len(selected_channels)} channels...")
        print(f"📊 Data shape: {selected_data.shape}")
        print(f"⏱️  Time bin size: {TIME_BIN_SIZE_MS} ms")
        
        # Extract all feature types
        spike_features = self.extract_spike_band_power(selected_data, bin_size)
        print("✅ Spike band power extraction completed")
        
        lfp_features = self.extract_lfp_features(selected_data, bin_size)
        print("✅ LFP feature extraction completed")
        
        voltage_features = self.extract_voltage_features(selected_data, bin_size)
        print("✅ Voltage feature extraction completed")
        
        threshold_features = self.extract_threshold_crossings(selected_data, bin_size)
        print("✅ Threshold crossing extraction completed")
        
        return {
            'spike_band': spike_features,
            'lfp': lfp_features,
            'voltage': voltage_features,
            'threshold': threshold_features,
            'selected_channels': selected_channels,
            'sampling_rate': self.fs,
            'original_shape': neural_data.shape
        }
    
    def plot_feature_summary(self, features: Dict, trial_data: Dict):
        """Create a comprehensive visualization of extracted features."""
        selected_channels = features['selected_channels']
        n_channels = len(selected_channels)
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main title
        trial_num = trial_data['trial_number']
        duration = trial_data['neural_data'].shape[1] / self.fs
        fig.suptitle(f'Neural Feature Extraction - Trial {trial_num} ({duration:.1f}s)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Spike Band Power (RMS)
        ax1 = fig.add_subplot(gs[0, 0])
        spike_data = features['spike_band']['rms_power']
        time_axis = features['spike_band']['time_axis']
        
        for i, ch in enumerate(selected_channels[:5]):  # Show first 5 channels
            ax1.plot(time_axis, spike_data[i], label=f'Ch {ch}', alpha=0.7)
        ax1.set_title('Spike Band Power (RMS)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RMS Power')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. LFP Power
        ax2 = fig.add_subplot(gs[0, 1])
        lfp_data = features['lfp']['lfp_power']
        
        for i, ch in enumerate(selected_channels[:5]):
            ax2.plot(time_axis, lfp_data[i], label=f'Ch {ch}', alpha=0.7)
        ax2.set_title('LFP Power')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gamma Power
        ax3 = fig.add_subplot(gs[0, 2])
        gamma_data = features['lfp']['gamma_power']
        
        for i, ch in enumerate(selected_channels[:5]):
            ax3.plot(time_axis, gamma_data[i], label=f'Ch {ch}', alpha=0.7)
        ax3.set_title('Gamma Power')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Threshold Crossings
        ax4 = fig.add_subplot(gs[1, 0])
        crossing_data = features['threshold']['crossing_counts']
        
        for i, ch in enumerate(selected_channels[:5]):
            ax4.plot(time_axis, crossing_data[i], label=f'Ch {ch}', alpha=0.7)
        ax4.set_title('Threshold Crossings')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Count per bin')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Moving Variance
        ax5 = fig.add_subplot(gs[1, 1])
        var_data = features['voltage']['moving_variance']
        
        for i, ch in enumerate(selected_channels[:5]):
            ax5.plot(time_axis, var_data[i], label=f'Ch {ch}', alpha=0.7)
        ax5.set_title('Moving Variance')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Variance')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Heatmap - Spike Band Power
        ax6 = fig.add_subplot(gs[1, 2])
        im = ax6.imshow(spike_data, aspect='auto', cmap='viridis', origin='lower')
        ax6.set_title('Spike Band Power Heatmap')
        ax6.set_xlabel('Time Bins')
        ax6.set_ylabel('Channels')
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        # 7. Summary Statistics
        ax7 = fig.add_subplot(gs[2, :])
        
        # Calculate summary statistics
        spike_mean = np.mean(spike_data, axis=1)
        lfp_mean = np.mean(lfp_data, axis=1)
        gamma_mean = np.mean(gamma_data, axis=1)
        crossing_sum = np.sum(crossing_data, axis=1)
        
        x = np.arange(len(selected_channels))
        width = 0.2
        
        ax7.bar(x - 1.5*width, spike_mean, width, label='Spike Power', alpha=0.8)
        ax7.bar(x - 0.5*width, lfp_mean/np.max(lfp_mean)*np.max(spike_mean), width, 
               label='LFP Power (norm)', alpha=0.8)
        ax7.bar(x + 0.5*width, gamma_mean/np.max(gamma_mean)*np.max(spike_mean), width, 
               label='Gamma Power (norm)', alpha=0.8)
        ax7.bar(x + 1.5*width, crossing_sum/np.max(crossing_sum)*np.max(spike_mean), width, 
               label='Total Crossings (norm)', alpha=0.8)
        
        ax7.set_title('Feature Summary by Channel')
        ax7.set_xlabel('Channel Index')
        ax7.set_ylabel('Feature Value')
        ax7.set_xticks(x)
        ax7.set_xticklabels([f'Ch {ch}' for ch in selected_channels], rotation=45)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot instead of showing interactively
        output_file = f'neural_features_trial_{trial_data["trial_number"]}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Plot saved as {output_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"📊 Analyzed {n_channels} channels: {selected_channels}")
        print(f"⏱️  Time bins: {len(time_axis)} bins of {TIME_BIN_SIZE_MS} ms each")
        print(f"🎯 Spike band: {features['spike_band']['frequency_band']}")
        print(f"🧠 LFP cutoff: {features['lfp']['lfp_cutoff']} Hz")
        print(f"🌊 Gamma band: {features['lfp']['gamma_band']}")
        print(f"⚡ Threshold: {features['threshold']['threshold_multiplier']}x RMS")
        
        print(f"\n📈 Feature Statistics:")
        print(f"  • Spike RMS Power: {np.mean(spike_mean):.3f} ± {np.std(spike_mean):.3f}")
        print(f"  • LFP Power: {np.mean(lfp_mean):.3f} ± {np.std(lfp_mean):.3f}")
        print(f"  • Gamma Power: {np.mean(gamma_mean):.3f} ± {np.std(gamma_mean):.3f}")
        print(f"  • Total Crossings: {np.sum(crossing_sum):.0f} across all channels")
        
        # Find most active channels
        activity_score = spike_mean + crossing_sum/np.max(crossing_sum)*np.max(spike_mean)
        top_channels = np.argsort(activity_score)[-5:][::-1]
        print(f"\n🔥 Most active channels: {[selected_channels[i] for i in top_channels]}")


def main():
    """Main execution function."""
    print("🧠 Neural Feature Extraction Tool")
    print("="*50)
    
    # Use fixed H5 file path
    h5_file = find_h5_file()
    print(f"📁 Using H5 file: {h5_file}")
    
    # Check if file exists
    if not os.path.exists(h5_file):
        print(f"❌ Error: H5 file not found at {h5_file}")
        print("Please ensure the file exists at the specified location.")
        return
    
    # Initialize extractor
    extractor = NeuralFeatureExtractor(sampling_rate=SAMPLING_RATE)
    
    try:
        # Load trial data
        print(f"📂 Loading trial {TRIAL_NUMBER} from {h5_file}...")
        trial_data = extractor.load_trial_data(h5_file, TRIAL_NUMBER)
        
        # Extract features
        print(f"🔍 Analyzing {len(SPIKE_CHANNELS)} spike channels...")
        print(f"📊 Neural data shape: {trial_data['neural_data'].shape}")
        features = extractor.extract_all_features(
            trial_data['neural_data'], 
            SPIKE_CHANNELS, 
            TIME_BIN_SIZE
        )
        print("🔍 Feature extraction completed!")
        
        # Visualize results
        print("📊 Creating visualization...")
        extractor.plot_feature_summary(features, trial_data)
        print("📊 Visualization completed!")
        
        print("\n✅ Feature extraction completed successfully!")
        
        # Optionally save features
        save_features = input("\n💾 Save features to file? (y/n): ").lower() == 'y'
        if save_features:
            output_file = f'neural_features_trial_{TRIAL_NUMBER:03d}.npz'
            np.savez_compressed(output_file, **features)
            print(f"💾 Features saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 