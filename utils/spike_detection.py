"""
Spike Detection Module
=====================

This module implements multiple spike detection algorithms including:
1. Threshold-based detection (current method)
2. PyWaveClus-inspired detection with advanced filtering and clustering

Author: Neural Exploration Team
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SpikeDetector:
    """
    Multi-algorithm spike detection class supporting various detection methods.
    """
    
    def __init__(self, sampling_rate: int = 30000):
        """
        Initialize spike detector.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Default parameters for threshold detection
        self.threshold_params = {
            'method': 'rms',  # 'rms', 'std', 'median', 'manual'
            'multiplier': -4.0,  # threshold multiplier
            'manual_threshold': 50.0,  # manual threshold in μV
            'polarity': 'negative',  # 'positive', 'negative', 'both'
            'min_spike_interval': 0.001,  # minimum interval between spikes (1ms)
            'spike_band_low': 400,  # Hz
            'spike_band_high': 6000,  # Hz
            'filter_order': 4
        }
        
        # Default parameters for PyWaveClus detection
        self.waveclus_params = {
            'detection_method': 'neg',  # 'pos', 'neg', 'both'
            'threshold_method': 'automatic',  # 'automatic', 'manual'
            'threshold_multiplier': 4.0,  # threshold multiplier for automatic
            'manual_threshold': 50.0,  # manual threshold in μV
            'w_pre': 20,  # samples before spike peak
            'w_post': 44,  # samples after spike peak
            'ref_period': 1.5,  # refractory period in ms
            'detection_filter': True,  # apply detection filter
            'filter_type': 'elliptic',  # 'butter', 'elliptic', 'cheby1'
            'filter_order': 4,
            'low_cutoff': 300,  # Hz
            'high_cutoff': 3000,  # Hz
            'detect_fmin': 300,  # Hz - detection filter low
            'detect_fmax': 8000,  # Hz - detection filter high
            'alignment': True,  # align spikes to peak/trough
            'align_window': 5,  # samples for alignment window
            'feature_extraction': 'wavelet',  # 'wavelet', 'pca', 'both'
            'pca_components': 3,  # number of PCA components
            'wavelet_scales': 4,  # number of wavelet scales
            'clustering': True,  # perform clustering
            'cluster_method': 'hierarchical',  # 'hierarchical', 'kmeans'
            'max_clusters': 10,  # maximum number of clusters
            'min_spikes_per_cluster': 20  # minimum spikes per cluster
        }
    
    def detect_spikes_threshold(self, signal_data: np.ndarray, 
                               params: Optional[Dict] = None) -> Dict:
        """
        Threshold-based spike detection (current method).
        
        Args:
            signal_data: 1D array of neural signal
            params: Optional parameters dictionary
            
        Returns:
            Dictionary with detection results
        """
        if params is None:
            params = self.threshold_params
        
        # Apply bandpass filter
        sos = signal.butter(
            params['filter_order'], 
            [params['spike_band_low']/self.nyquist, params['spike_band_high']/self.nyquist], 
            btype='band', 
            output='sos'
        )
        filtered_signal = signal.sosfilt(sos, signal_data)
        
        # Calculate threshold
        if params['method'] == 'rms':
            threshold = params['multiplier'] * np.sqrt(np.mean(filtered_signal**2))
        elif params['method'] == 'std':
            threshold = params['multiplier'] * np.std(filtered_signal)
        elif params['method'] == 'median':
            threshold = params['multiplier'] * np.median(np.abs(filtered_signal))
        else:  # manual
            threshold = params['manual_threshold']
        
        # Find threshold crossings
        if params['polarity'] == 'negative':
            crossings = np.where((filtered_signal[:-1] >= threshold) & 
                               (filtered_signal[1:] < threshold))[0]
        elif params['polarity'] == 'positive':
            crossings = np.where((filtered_signal[:-1] <= threshold) & 
                               (filtered_signal[1:] > threshold))[0]
        else:  # both
            neg_crossings = np.where((filtered_signal[:-1] >= threshold) & 
                                   (filtered_signal[1:] < threshold))[0]
            pos_crossings = np.where((filtered_signal[:-1] <= threshold) & 
                                   (filtered_signal[1:] > threshold))[0]
            crossings = np.sort(np.concatenate([neg_crossings, pos_crossings]))
        
        # Apply refractory period
        if len(crossings) > 0:
            refractory_samples = int(params['min_spike_interval'] * self.fs)
            filtered_crossings = [crossings[0]]
            
            for crossing in crossings[1:]:
                if crossing - filtered_crossings[-1] > refractory_samples:
                    filtered_crossings.append(crossing)
            
            spike_times = np.array(filtered_crossings) / self.fs
            spike_indices = np.array(filtered_crossings)
        else:
            spike_times = np.array([])
            spike_indices = np.array([])
        
        return {
            'method': 'threshold',
            'spike_times': spike_times,
            'spike_indices': spike_indices,
            'threshold': threshold,
            'filtered_signal': filtered_signal,
            'n_spikes': len(spike_times),
            'parameters': params
        }
    
    def detect_spikes_waveclus(self, signal_data: np.ndarray, 
                              params: Optional[Dict] = None) -> Dict:
        """
        PyWaveClus-inspired spike detection with advanced filtering and clustering.
        
        Args:
            signal_data: 1D array of neural signal
            params: Optional parameters dictionary
            
        Returns:
            Dictionary with detection results
        """
        if params is None:
            params = self.waveclus_params
        
        # Apply detection filter (more aggressive than threshold method)
        if params['detection_filter']:
            if params['filter_type'] == 'elliptic':
                sos = signal.ellip(
                    params['filter_order'], 0.1, 40,
                    [params['detect_fmin']/self.nyquist, params['detect_fmax']/self.nyquist],
                    btype='band', output='sos'
                )
            elif params['filter_type'] == 'cheby1':
                sos = signal.cheby1(
                    params['filter_order'], 0.1,
                    [params['detect_fmin']/self.nyquist, params['detect_fmax']/self.nyquist],
                    btype='band', output='sos'
                )
            else:  # butter
                sos = signal.butter(
                    params['filter_order'],
                    [params['detect_fmin']/self.nyquist, params['detect_fmax']/self.nyquist],
                    btype='band', output='sos'
                )
            
            filtered_signal = signal.sosfilt(sos, signal_data)
        else:
            filtered_signal = signal_data
        
        # Calculate detection threshold (PyWaveClus style)
        if params['threshold_method'] == 'automatic':
            # Use median-based threshold (more robust than RMS)
            noise_mad = np.median(np.abs(filtered_signal)) / 0.6745
            threshold = params['threshold_multiplier'] * noise_mad
        else:
            threshold = params['manual_threshold']
        
        # Detect spikes based on polarity
        if params['detection_method'] == 'neg':
            # Negative spikes (most common)
            potential_spikes = np.where(filtered_signal < -threshold)[0]
        elif params['detection_method'] == 'pos':
            # Positive spikes
            potential_spikes = np.where(filtered_signal > threshold)[0]
        else:  # both
            potential_spikes = np.where(np.abs(filtered_signal) > threshold)[0]
        
        if len(potential_spikes) == 0:
            return {
                'method': 'waveclus',
                'spike_times': np.array([]),
                'spike_indices': np.array([]),
                'spike_waveforms': np.array([]),
                'threshold': threshold,
                'filtered_signal': filtered_signal,
                'n_spikes': 0,
                'clusters': np.array([]),
                'parameters': params
            }
        
        # Group consecutive threshold crossings and find local extrema
        spike_candidates = []
        ref_samples = int(params['ref_period'] * self.fs / 1000)  # convert ms to samples
        
        i = 0
        while i < len(potential_spikes):
            # Find the start of a potential spike event
            start_idx = potential_spikes[i]
            
            # Find the end of consecutive threshold crossings
            end_idx = start_idx
            while (i + 1 < len(potential_spikes) and 
                   potential_spikes[i + 1] - potential_spikes[i] <= ref_samples):
                i += 1
                end_idx = potential_spikes[i]
            
            # Find local extremum within this window
            window_start = max(0, start_idx - params['align_window'])
            window_end = min(len(filtered_signal), end_idx + params['align_window'])
            window_signal = filtered_signal[window_start:window_end]
            
            if params['detection_method'] == 'neg':
                # Find minimum (most negative point)
                local_extremum = np.argmin(window_signal)
            elif params['detection_method'] == 'pos':
                # Find maximum (most positive point)
                local_extremum = np.argmax(window_signal)
            else:  # both
                # Find the point with maximum absolute value
                local_extremum = np.argmax(np.abs(window_signal))
            
            spike_idx = window_start + local_extremum
            spike_candidates.append(spike_idx)
            i += 1
        
        # Apply refractory period to final candidates
        if len(spike_candidates) > 0:
            spike_candidates = np.array(spike_candidates)
            final_spikes = [spike_candidates[0]]
            
            for spike_idx in spike_candidates[1:]:
                if spike_idx - final_spikes[-1] > ref_samples:
                    final_spikes.append(spike_idx)
            
            spike_indices = np.array(final_spikes)
        else:
            spike_indices = np.array([])
        
        # Extract spike waveforms
        spike_waveforms = []
        valid_spikes = []
        
        for spike_idx in spike_indices:
            start_idx = spike_idx - params['w_pre']
            end_idx = spike_idx + params['w_post']
            
            if start_idx >= 0 and end_idx < len(filtered_signal):
                waveform = filtered_signal[start_idx:end_idx]
                spike_waveforms.append(waveform)
                valid_spikes.append(spike_idx)
        
        spike_indices = np.array(valid_spikes)
        spike_waveforms = np.array(spike_waveforms)
        spike_times = spike_indices / self.fs
        
        # Perform clustering if requested and we have enough spikes
        clusters = np.zeros(len(spike_indices), dtype=int)
        if (params['clustering'] and len(spike_waveforms) >= params['min_spikes_per_cluster']):
            try:
                clusters = self._cluster_spikes(spike_waveforms, params)
            except Exception as e:
                print(f"⚠️ Clustering failed: {e}")
                clusters = np.zeros(len(spike_indices), dtype=int)
        
        return {
            'method': 'waveclus',
            'spike_times': spike_times,
            'spike_indices': spike_indices,
            'spike_waveforms': spike_waveforms,
            'threshold': threshold,
            'filtered_signal': filtered_signal,
            'n_spikes': len(spike_times),
            'clusters': clusters,
            'parameters': params
        }
    
    def _cluster_spikes(self, spike_waveforms: np.ndarray, params: Dict) -> np.ndarray:
        """
        Cluster spike waveforms using hierarchical clustering.
        
        Args:
            spike_waveforms: Array of spike waveforms (n_spikes, n_samples)
            params: Parameters dictionary
            
        Returns:
            Cluster labels for each spike
        """
        if len(spike_waveforms) < params['min_spikes_per_cluster']:
            return np.zeros(len(spike_waveforms), dtype=int)
        
        # Feature extraction
        if params['feature_extraction'] == 'pca':
            # PCA features
            scaler = StandardScaler()
            normalized_waveforms = scaler.fit_transform(spike_waveforms)
            pca = PCA(n_components=params['pca_components'])
            features = pca.fit_transform(normalized_waveforms)
            
        elif params['feature_extraction'] == 'wavelet':
            # Simplified wavelet features (using basic statistics)
            features = []
            for waveform in spike_waveforms:
                # Basic wavelet-like features
                f1 = np.max(waveform)  # maximum
                f2 = np.min(waveform)  # minimum
                f3 = np.std(waveform)  # standard deviation
                f4 = np.mean(np.abs(np.diff(waveform)))  # mean absolute derivative
                features.append([f1, f2, f3, f4])
            features = np.array(features)
            
        else:  # both
            # Combined PCA and basic features
            scaler = StandardScaler()
            normalized_waveforms = scaler.fit_transform(spike_waveforms)
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(normalized_waveforms)
            
            basic_features = []
            for waveform in spike_waveforms:
                f1 = np.max(waveform)
                f2 = np.min(waveform)
                basic_features.append([f1, f2])
            
            features = np.hstack([pca_features, np.array(basic_features)])
        
        # Hierarchical clustering
        linkage_matrix = linkage(features, method='ward')
        
        # Determine optimal number of clusters (simplified approach)
        max_clusters = min(params['max_clusters'], len(spike_waveforms) // params['min_spikes_per_cluster'])
        
        if max_clusters > 1:
            cluster_labels = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
            
            # Filter out small clusters
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_clusters = unique_labels[counts >= params['min_spikes_per_cluster']]
            
            # Reassign small clusters to noise (cluster 0)
            for i, label in enumerate(cluster_labels):
                if label not in valid_clusters:
                    cluster_labels[i] = 0
        else:
            cluster_labels = np.ones(len(spike_waveforms), dtype=int)
        
        return cluster_labels
    
    def compare_methods(self, signal_data: np.ndarray, 
                       threshold_params: Optional[Dict] = None,
                       waveclus_params: Optional[Dict] = None) -> Dict:
        """
        Compare spike detection results between threshold and PyWaveClus methods.
        
        Args:
            signal_data: 1D array of neural signal
            threshold_params: Parameters for threshold detection
            waveclus_params: Parameters for PyWaveClus detection
            
        Returns:
            Dictionary with comparison results
        """
        # Detect spikes using both methods
        threshold_result = self.detect_spikes_threshold(signal_data, threshold_params)
        waveclus_result = self.detect_spikes_waveclus(signal_data, waveclus_params)
        
        # Calculate comparison metrics
        thresh_spikes = set(threshold_result['spike_indices'])
        waveclus_spikes = set(waveclus_result['spike_indices'])
        
        # Find matches within a tolerance window
        tolerance_samples = int(0.001 * self.fs)  # 1ms tolerance
        matches = 0
        
        for thresh_spike in thresh_spikes:
            for waveclus_spike in waveclus_spikes:
                if abs(thresh_spike - waveclus_spike) <= tolerance_samples:
                    matches += 1
                    break
        
        # Calculate metrics
        n_threshold = len(thresh_spikes)
        n_waveclus = len(waveclus_spikes)
        
        if n_threshold > 0:
            sensitivity = matches / n_threshold  # True positive rate
        else:
            sensitivity = 0
        
        if n_waveclus > 0:
            precision = matches / n_waveclus  # Positive predictive value
        else:
            precision = 0
        
        if sensitivity + precision > 0:
            f1_score = 2 * (sensitivity * precision) / (sensitivity + precision)
        else:
            f1_score = 0
        
        comparison = {
            'threshold_result': threshold_result,
            'waveclus_result': waveclus_result,
            'n_threshold_spikes': n_threshold,
            'n_waveclus_spikes': n_waveclus,
            'n_matched_spikes': matches,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1_score,
            'signal_duration': len(signal_data) / self.fs,
            'threshold_rate': n_threshold / (len(signal_data) / self.fs),
            'waveclus_rate': n_waveclus / (len(signal_data) / self.fs)
        }
        
        return comparison
    
    def plot_detection_comparison(self, signal_data: np.ndarray, 
                                 comparison_result: Dict, 
                                 time_window: Optional[Tuple[float, float]] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comparison between spike detection methods.
        
        Args:
            signal_data: Original signal data
            comparison_result: Result from compare_methods()
            time_window: Optional (start_time, end_time) in seconds
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Time axis
        time_axis = np.arange(len(signal_data)) / self.fs
        
        # Apply time window if specified
        if time_window is not None:
            start_idx = int(time_window[0] * self.fs)
            end_idx = int(time_window[1] * self.fs)
            time_axis = time_axis[start_idx:end_idx]
            signal_data = signal_data[start_idx:end_idx]
        else:
            start_idx = 0
            end_idx = len(signal_data)
        
        # Plot 1: Original signal with threshold detection
        thresh_result = comparison_result['threshold_result']
        axes[0].plot(time_axis, signal_data, 'k-', linewidth=0.5, alpha=0.7, label='Raw signal')
        axes[0].plot(time_axis, thresh_result['filtered_signal'][start_idx:end_idx], 
                    'b-', linewidth=0.8, label='Filtered signal')
        axes[0].axhline(thresh_result['threshold'], color='r', linestyle='--', 
                       label=f'Threshold ({thresh_result["threshold"]:.1f})')
        
        # Mark threshold spikes
        thresh_spikes = thresh_result['spike_indices']
        thresh_spikes = thresh_spikes[(thresh_spikes >= start_idx) & (thresh_spikes < end_idx)]
        if len(thresh_spikes) > 0:
            spike_times = thresh_spikes / self.fs
            spike_values = thresh_result['filtered_signal'][thresh_spikes]
            axes[0].scatter(spike_times, spike_values, color='red', s=30, zorder=5, 
                          label=f'Threshold spikes ({len(thresh_spikes)})')
        
        axes[0].set_title('Threshold-based Detection')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: PyWaveClus detection
        waveclus_result = comparison_result['waveclus_result']
        axes[1].plot(time_axis, signal_data, 'k-', linewidth=0.5, alpha=0.7, label='Raw signal')
        axes[1].plot(time_axis, waveclus_result['filtered_signal'][start_idx:end_idx], 
                    'g-', linewidth=0.8, label='Filtered signal')
        axes[1].axhline(waveclus_result['threshold'], color='orange', linestyle='--', 
                       label=f'Threshold ({waveclus_result["threshold"]:.1f})')
        axes[1].axhline(-waveclus_result['threshold'], color='orange', linestyle='--', alpha=0.5)
        
        # Mark PyWaveClus spikes
        waveclus_spikes = waveclus_result['spike_indices']
        waveclus_spikes = waveclus_spikes[(waveclus_spikes >= start_idx) & (waveclus_spikes < end_idx)]
        if len(waveclus_spikes) > 0:
            spike_times = waveclus_spikes / self.fs
            spike_values = waveclus_result['filtered_signal'][waveclus_spikes]
            axes[1].scatter(spike_times, spike_values, color='orange', s=30, zorder=5, 
                          label=f'PyWaveClus spikes ({len(waveclus_spikes)})')
        
        axes[1].set_title('PyWaveClus-inspired Detection')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Comparison
        axes[2].plot(time_axis, signal_data, 'k-', linewidth=0.5, alpha=0.7, label='Raw signal')
        
        # Mark both types of spikes
        if len(thresh_spikes) > 0:
            spike_times = thresh_spikes / self.fs
            spike_values = signal_data[thresh_spikes - start_idx]
            axes[2].scatter(spike_times, spike_values, color='red', s=20, alpha=0.7, 
                          label=f'Threshold ({len(thresh_spikes)})')
        
        if len(waveclus_spikes) > 0:
            spike_times = waveclus_spikes / self.fs
            spike_values = signal_data[waveclus_spikes - start_idx]
            axes[2].scatter(spike_times, spike_values, color='orange', s=20, alpha=0.7, 
                          label=f'PyWaveClus ({len(waveclus_spikes)})')
        
        axes[2].set_title('Comparison: Both Methods')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print comparison metrics
        print(f"\n📊 Spike Detection Comparison:")
        print(f"   • Threshold method: {comparison_result['n_threshold_spikes']} spikes")
        print(f"   • PyWaveClus method: {comparison_result['n_waveclus_spikes']} spikes")
        print(f"   • Matched spikes: {comparison_result['n_matched_spikes']}")
        print(f"   • Sensitivity: {comparison_result['sensitivity']:.3f}")
        print(f"   • Precision: {comparison_result['precision']:.3f}")
        print(f"   • F1-score: {comparison_result['f1_score']:.3f}")
        print(f"   • Threshold rate: {comparison_result['threshold_rate']:.1f} spikes/s")
        print(f"   • PyWaveClus rate: {comparison_result['waveclus_rate']:.1f} spikes/s")
        
        plt.show()


def create_raster_comparison(trial_data: Dict, spike_channels: List[int], 
                           spike_detector: SpikeDetector, 
                           trial_number: int = None,
                           time_window: Optional[Tuple[float, float]] = None,
                           figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create dual raster plots comparing threshold and PyWaveClus spike detection.
    
    Args:
        trial_data: Trial data dictionary containing neural data
        spike_channels: List of spike channel indices
        spike_detector: Initialized SpikeDetector instance
        trial_number: Trial number for display
        time_window: Optional (start_time, end_time) in seconds
        figsize: Figure size
    """
    
    if trial_data is None or 'neural_data' not in trial_data:
        print("❌ No neural data found in trial_data")
        return
    
    neural_data = trial_data['neural_data']
    duration = trial_data.get('duration', neural_data.shape[1] / spike_detector.fs)
    
    # Detect spikes for all channels using both methods
    print("🔍 Detecting spikes using both methods...")
    
    threshold_spikes = {}
    waveclus_spikes = {}
    
    for ch_idx in spike_channels:
        if ch_idx < neural_data.shape[0]:
            signal_data = neural_data[ch_idx, :]
            
            # Compare methods for this channel
            comparison = spike_detector.compare_methods(signal_data)
            
            threshold_spikes[ch_idx] = comparison['threshold_result']['spike_times']
            waveclus_spikes[ch_idx] = comparison['waveclus_result']['spike_times']
        else:
            threshold_spikes[ch_idx] = np.array([])
            waveclus_spikes[ch_idx] = np.array([])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Apply time window if specified
    if time_window is not None:
        xlim = time_window
    else:
        xlim = (0, duration)
    
    # Plot 1: Threshold-based raster
    ax1 = axes[0]
    threshold_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(spike_channels)))
    
    for i, ch_idx in enumerate(spike_channels):
        if ch_idx in threshold_spikes and len(threshold_spikes[ch_idx]) > 0:
            spike_times = threshold_spikes[ch_idx]
            
            # Filter spikes within time window
            if time_window is not None:
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times = spike_times[mask]
            
            if len(spike_times) > 0:
                y_positions = np.full_like(spike_times, i)
                ax1.scatter(spike_times, y_positions, c=[threshold_colors[i]], 
                          s=8, alpha=0.8, marker='|', linewidths=1)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(-0.5, len(spike_channels) - 0.5)
    ax1.set_ylabel('Channel Index')
    ax1.set_title(f'Threshold-based Spike Detection - Trial {trial_number}' if trial_number else 'Threshold-based Spike Detection')
    ax1.grid(True, alpha=0.3)
    
    # Add channel labels
    channel_labels = [f'Ch{ch}' for ch in spike_channels]
    ax1.set_yticks(range(len(spike_channels)))
    ax1.set_yticklabels(channel_labels)
    
    # Plot 2: PyWaveClus raster
    ax2 = axes[1]
    waveclus_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(spike_channels)))
    
    for i, ch_idx in enumerate(spike_channels):
        if ch_idx in waveclus_spikes and len(waveclus_spikes[ch_idx]) > 0:
            spike_times = waveclus_spikes[ch_idx]
            
            # Filter spikes within time window
            if time_window is not None:
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times = spike_times[mask]
            
            if len(spike_times) > 0:
                y_positions = np.full_like(spike_times, i)
                ax2.scatter(spike_times, y_positions, c=[waveclus_colors[i]], 
                          s=8, alpha=0.8, marker='|', linewidths=1)
    
    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.5, len(spike_channels) - 0.5)
    ax2.set_ylabel('Channel Index')
    ax2.set_xlabel('Time (s)')
    ax2.set_title(f'PyWaveClus-inspired Spike Detection - Trial {trial_number}' if trial_number else 'PyWaveClus-inspired Spike Detection')
    ax2.grid(True, alpha=0.3)
    
    # Add channel labels
    ax2.set_yticks(range(len(spike_channels)))
    ax2.set_yticklabels(channel_labels)
    
    plt.tight_layout()
    
    # Calculate and display summary statistics
    total_threshold_spikes = sum(len(spikes) for spikes in threshold_spikes.values())
    total_waveclus_spikes = sum(len(spikes) for spikes in waveclus_spikes.values())
    
    print(f"\n📊 Raster Plot Summary:")
    print(f"   • Trial {trial_number}: {duration:.2f}s duration")
    print(f"   • {len(spike_channels)} channels analyzed")
    print(f"   • Threshold method: {total_threshold_spikes} total spikes")
    print(f"   • PyWaveClus method: {total_waveclus_spikes} total spikes")
    print(f"   • Threshold rate: {total_threshold_spikes/duration:.1f} spikes/s")
    print(f"   • PyWaveClus rate: {total_waveclus_spikes/duration:.1f} spikes/s")
    
    plt.show()
    
    return {
        'threshold_spikes': threshold_spikes,
        'waveclus_spikes': waveclus_spikes,
        'total_threshold_spikes': total_threshold_spikes,
        'total_waveclus_spikes': total_waveclus_spikes,
        'duration': duration
    } 