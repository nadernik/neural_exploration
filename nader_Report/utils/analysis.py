#!/usr/bin/env python3
"""
Analysis utilities for neural feature exploration.

This module provides classes and functions for comprehensive neural signal analysis,
including feature extraction, quality assessment, and behavioral correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import neural feature extraction
from neural_feature_extraction import NeuralFeatureExtractor

# Import spike detection components
from utils.spike_detection import SpikeDetector
from utils.neural_behavioral_alignment import NeuralBehavioralAligner

class FeatureAnalyzer:
    """
    Analyzes neural features extracted from selected spike channels.
    """
    
    def __init__(self, spike_channels: List[int]):
        self.spike_channels = spike_channels
        
    def load_and_extract_features(self, trial_number: int, time_bin_size: float = 0.02, 
                                sampling_rate: int = 30000) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load trial data and extract neural features.
        
        Args:
            trial_number: Trial number to analyze
            time_bin_size: Time bin size in seconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (trial_data, features) or (None, None) if failed
        """
        # Use hardcoded H5 file path
        h5_file = r"D:\Data\ScienceCorp\trials_aligned.h5"
        
        print(f"📁 Using H5 file: {h5_file}")
        
        # Initialize extractor
        extractor = NeuralFeatureExtractor(sampling_rate=sampling_rate)
        
        # Load trial data
        print(f"📂 Loading trial {trial_number}...")
        try:
            trial_data = extractor.load_trial_data(h5_file, trial_number)
        except Exception as e:
            print(f"❌ Failed to load trial {trial_number}: {e}")
            return None, None
        
        # Extract features
        print(f"🔍 Extracting features for {len(self.spike_channels)} channels...")
        try:
            features = extractor.extract_all_features(
                trial_data['neural_data'], 
                self.spike_channels, 
                time_bin_size
            )
        except Exception as e:
            print(f"❌ Failed to extract features: {e}")
            return trial_data, None
        
        print("✅ Feature extraction complete!")
        return trial_data, features
    
    def get_feature_summary(self, features: Dict) -> Dict:
        """
        Generate a comprehensive summary of extracted features.
        
        Args:
            features: Dictionary containing extracted features
            
        Returns:
            Dictionary with feature summary statistics
        """
        if features is None:
            return {}
        
        summary = {}
        
        # Spike band features
        if 'spike_band' in features:
            spike_rms = features['spike_band']['rms_power']
            summary['spike_band'] = {
                'mean_power': np.mean(spike_rms),
                'std_power': np.std(spike_rms),
                'max_power': np.max(spike_rms),
                'min_power': np.min(spike_rms),
                'channel_means': np.mean(spike_rms, axis=1),
                'time_means': np.mean(spike_rms, axis=0)
            }
        
        # LFP features
        if 'lfp' in features:
            lfp_power = features['lfp']['lfp_power']
            gamma_power = features['lfp']['gamma_power']
            
            summary['lfp'] = {
                'mean_lfp_power': np.mean(lfp_power),
                'std_lfp_power': np.std(lfp_power),
                'mean_gamma_power': np.mean(gamma_power),
                'std_gamma_power': np.std(gamma_power),
                'channel_lfp_means': np.mean(lfp_power, axis=1),
                'channel_gamma_means': np.mean(gamma_power, axis=1)
            }
        
        # Threshold crossing features
        if 'threshold' in features:
            crossings = features['threshold']['crossing_counts']
            summary['threshold'] = {
                'total_crossings': np.sum(crossings),
                'mean_crossings_per_bin': np.mean(crossings),
                'std_crossings_per_bin': np.std(crossings),
                'channel_crossing_totals': np.sum(crossings, axis=1),
                'max_crossings_per_bin': np.max(crossings)
            }
        
        # Voltage features
        if 'voltage' in features:
            mov_avg = features['voltage']['moving_average']
            mov_var = features['voltage']['moving_variance']
            
            summary['voltage'] = {
                'mean_moving_avg': np.mean(mov_avg),
                'std_moving_avg': np.std(mov_avg),
                'mean_moving_var': np.mean(mov_var),
                'std_moving_var': np.std(mov_var)
            }
        
        return summary
    
    def find_most_active_channels(self, features: Dict, top_n: int = 5) -> List[int]:
        """
        Find the most active channels based on multiple features.
        
        Args:
            features: Dictionary containing extracted features
            top_n: Number of top channels to return
            
        Returns:
            List of most active channel indices (original channel numbers)
        """
        if features is None:
            return []
        
        # Calculate activity score based on multiple features
        n_channels = len(self.spike_channels)
        activity_scores = np.zeros(n_channels)
        
        # Spike band power contribution
        if 'spike_band' in features:
            spike_rms = features['spike_band']['rms_power']
            spike_means = np.mean(spike_rms, axis=1)
            activity_scores += spike_means / np.max(spike_means)
        
        # Threshold crossings contribution
        if 'threshold' in features:
            crossings = features['threshold']['crossing_counts']
            crossing_sums = np.sum(crossings, axis=1)
            if np.max(crossing_sums) > 0:
                activity_scores += crossing_sums / np.max(crossing_sums)
        
        # LFP power contribution
        if 'lfp' in features:
            lfp_power = features['lfp']['lfp_power']
            lfp_means = np.mean(lfp_power, axis=1)
            activity_scores += lfp_means / np.max(lfp_means)
        
        # Get top channels
        top_indices = np.argsort(activity_scores)[-top_n:][::-1]
        top_channels = [self.spike_channels[i] for i in top_indices]
        
        return top_channels
    
    def get_channel_statistics(self, features: Dict, channel_indices: List[int] = None) -> Dict:
        """
        Get detailed statistics for specific channels.
        
        Args:
            features: Dictionary containing extracted features
            channel_indices: List of channel indices to analyze (None for all)
            
        Returns:
            Dictionary with per-channel statistics
        """
        if features is None:
            return {}
        
        if channel_indices is None:
            channel_indices = list(range(len(self.spike_channels)))
        
        channel_stats = {}
        
        for i, ch_idx in enumerate(channel_indices):
            if ch_idx >= len(self.spike_channels):
                continue
                
            ch_num = self.spike_channels[ch_idx]
            stats = {'channel_number': ch_num}
            
            # Spike band statistics
            if 'spike_band' in features:
                spike_rms = features['spike_band']['rms_power'][ch_idx]
                stats['spike_band'] = {
                    'mean': np.mean(spike_rms),
                    'std': np.std(spike_rms),
                    'max': np.max(spike_rms),
                    'min': np.min(spike_rms)
                }
            
            # LFP statistics
            if 'lfp' in features:
                lfp_power = features['lfp']['lfp_power'][ch_idx]
                gamma_power = features['lfp']['gamma_power'][ch_idx]
                stats['lfp'] = {
                    'lfp_mean': np.mean(lfp_power),
                    'lfp_std': np.std(lfp_power),
                    'gamma_mean': np.mean(gamma_power),
                    'gamma_std': np.std(gamma_power)
                }
            
            # Threshold crossing statistics
            if 'threshold' in features:
                crossings = features['threshold']['crossing_counts'][ch_idx]
                stats['threshold'] = {
                    'total_crossings': np.sum(crossings),
                    'mean_per_bin': np.mean(crossings),
                    'max_per_bin': np.max(crossings)
                }
            
            # Voltage statistics
            if 'voltage' in features:
                mov_avg = features['voltage']['moving_average'][ch_idx]
                mov_var = features['voltage']['moving_variance'][ch_idx]
                stats['voltage'] = {
                    'moving_avg_mean': np.mean(mov_avg),
                    'moving_avg_std': np.std(mov_avg),
                    'moving_var_mean': np.mean(mov_var),
                    'moving_var_std': np.std(mov_var)
                }
            
            channel_stats[ch_num] = stats
        
        return channel_stats
    
    def compare_channels(self, features: Dict, channel_list: List[int]) -> Dict:
        """
        Compare features across multiple channels.
        
        Args:
            features: Dictionary containing extracted features
            channel_list: List of channel numbers to compare
            
        Returns:
            Dictionary with comparison results
        """
        if features is None:
            return {}
        
        # Find indices of requested channels
        channel_indices = []
        for ch_num in channel_list:
            try:
                idx = self.spike_channels.index(ch_num)
                channel_indices.append(idx)
            except ValueError:
                print(f"⚠️ Channel {ch_num} not in spike channels list")
        
        if not channel_indices:
            return {}
        
        comparison = {
            'channels': [self.spike_channels[i] for i in channel_indices],
            'features': {}
        }
        
        # Compare spike band power
        if 'spike_band' in features:
            spike_rms = features['spike_band']['rms_power']
            channel_means = [np.mean(spike_rms[i]) for i in channel_indices]
            channel_stds = [np.std(spike_rms[i]) for i in channel_indices]
            
            comparison['features']['spike_band'] = {
                'means': channel_means,
                'stds': channel_stds,
                'highest_mean': max(channel_means),
                'lowest_mean': min(channel_means),
                'most_active_channel': comparison['channels'][np.argmax(channel_means)]
            }
        
        # Compare threshold crossings
        if 'threshold' in features:
            crossings = features['threshold']['crossing_counts']
            crossing_totals = [np.sum(crossings[i]) for i in channel_indices]
            
            comparison['features']['threshold'] = {
                'total_crossings': crossing_totals,
                'highest_crossings': max(crossing_totals),
                'most_active_channel': comparison['channels'][np.argmax(crossing_totals)]
            }
        
        # Compare LFP features
        if 'lfp' in features:
            lfp_power = features['lfp']['lfp_power']
            gamma_power = features['lfp']['gamma_power']
            
            lfp_means = [np.mean(lfp_power[i]) for i in channel_indices]
            gamma_means = [np.mean(gamma_power[i]) for i in channel_indices]
            
            comparison['features']['lfp'] = {
                'lfp_means': lfp_means,
                'gamma_means': gamma_means,
                'highest_lfp': max(lfp_means),
                'highest_gamma': max(gamma_means)
            }
        
        return comparison


class SpikeAnalyzer:
    """
    Analyzes neural spikes using spike detection algorithms.
    """
    
    def __init__(self, spike_channels: List[int], 
                 sampling_rate: float = 30000.0,
                 threshold_factor: float = 5.0,
                 spike_window: Tuple[int, int] = (-10, 32),
                 bin_size: float = 0.05):
        """
        Initialize spike analyzer.
        
        Parameters:
        -----------
        spike_channels : list
            List of good channel indices
        sampling_rate : float
            Sampling rate in Hz
        threshold_factor : float
            Spike detection threshold multiplier
        spike_window : tuple
            Spike waveform window (start, end) in samples
        bin_size : float
            Time bin size for firing rates in seconds
        """
        self.spike_channels = spike_channels
        self.h5_file_path = r"D:\Data\ScienceCorp\trials_aligned.h5"
        
        # Initialize spike detector
        self.spike_detector = SpikeDetector(
            sampling_rate=sampling_rate,
            threshold_factor=threshold_factor,
            spike_window=spike_window,
            good_channels=spike_channels
        )
        
        # Initialize aligner
        self.aligner = NeuralBehavioralAligner(
            bin_size=bin_size,
            interpolation_method='linear'
        )
        
        print(f"🔍 SpikeAnalyzer initialized:")
        print(f"  • {len(spike_channels)} channels")
        print(f"  • {threshold_factor}x threshold")
        print(f"  • {bin_size*1000:.0f}ms bins")
    
    def analyze_trial(self, trial_number: int) -> Dict:
        """
        Perform comprehensive spike analysis of a trial.
        
        Parameters:
        -----------
        trial_number : int
            Trial number to analyze
            
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        print(f"🔍 Analyzing trial {trial_number}...")
        
        try:
            # Extract spike features
            firing_rates = self.spike_detector.extract_features_from_h5(
                self.h5_file_path, trial_number)
            
            # Load trial data
            trial_data = self.aligner.load_trial_data(
                self.h5_file_path, trial_number)
            
            # Get spike data for quality metrics
            spike_data = self.spike_detector.detect_spikes_all_channels(
                trial_data['neural_data'])
            
            # Get quality metrics
            quality_metrics = self.spike_detector.get_channel_quality_metrics(spike_data)
            
            # Find most active channels
            top_channels = quality_metrics.nlargest(10, 'n_spikes')['channel'].tolist()
            
            print(f"✅ Trial {trial_number} analysis complete!")
            
            return {
                'trial_data': trial_data,
                'firing_rates': firing_rates,
                'spike_data': spike_data,
                'quality_metrics': quality_metrics,
                'top_channels': top_channels,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_summary_statistics(self, analysis_results: Dict) -> Dict:
        """
        Generate summary statistics from analysis results.
        
        Parameters:
        -----------
        analysis_results : dict
            Results from analyze_trial
            
        Returns:
        --------
        dict
            Summary statistics
        """
        if not analysis_results['success']:
            return {}
        
        firing_rates = analysis_results['firing_rates']
        quality_metrics = analysis_results['quality_metrics']
        
        summary = {
            'total_channels': len(firing_rates),
            'total_spikes': quality_metrics['n_spikes'].sum(),
            'mean_firing_rate': np.mean([np.mean(rates) for rates in firing_rates.values()]),
            'high_quality_channels': len(quality_metrics[quality_metrics['snr'] > 3]),
            'active_channels': len(quality_metrics[quality_metrics['n_spikes'] > 50]),
            'mean_snr': quality_metrics['snr'].mean(),
            'top_channels': analysis_results['top_channels'][:5]
        }
        
        return summary
    
    def analyze_neural_behavioral_correlation(self, analysis_results: Dict) -> Dict:
        """
        Analyze correlation between neural activity and behavior.
        
        Parameters:
        -----------
        analysis_results : dict
            Results from analyze_trial
            
        Returns:
        --------
        dict
            Correlation analysis results
        """
        if not analysis_results['success']:
            return {'success': False}
        
        trial_data = analysis_results['trial_data']
        firing_rates = analysis_results['firing_rates']
        
        # Check if we have behavioral data
        if (trial_data['velocity_x'] is None or 
            trial_data['velocity_y'] is None or 
            len(trial_data['velocity_x']) == 0):
            return {'success': False, 'reason': 'No behavioral data'}
        
        try:
            # Calculate velocity magnitude
            velocity_magnitude = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
            
            # Calculate population firing rate
            duration = trial_data['metadata']['duration']
            time_bins = np.linspace(0, duration, len(list(firing_rates.values())[0]))
            
            population_rate = np.zeros(len(time_bins))
            for channel, rates in firing_rates.items():
                population_rate += rates
            
            # Align behavioral data to neural time bins
            from scipy.interpolate import interp1d
            
            if 'behavioral_timestamps' in trial_data:
                behavioral_timestamps = trial_data['behavioral_timestamps']
                
                # Convert absolute timestamps to relative time
                # Check if timestamps are absolute (Unix epoch) and convert to relative
                if behavioral_timestamps.max() > 1000000:  # Likely Unix timestamp
                    # Get trial start time
                    if 'metadata' in trial_data and 'start_seconds' in trial_data['metadata']:
                        trial_start = trial_data['metadata']['start_seconds']
                        if trial_start < 1000000:  # start_seconds is relative, use first timestamp
                            trial_start = behavioral_timestamps[0]
                    else:
                        trial_start = behavioral_timestamps[0]
                    behavioral_time = behavioral_timestamps - trial_start
                else:
                    behavioral_time = behavioral_timestamps
            else:
                behavioral_time = np.linspace(0, duration, len(velocity_magnitude))
            
            if len(velocity_magnitude) > 1:
                interp_func = interp1d(behavioral_time, velocity_magnitude, 
                                     kind='linear', bounds_error=False, fill_value=0)
                velocity_aligned = interp_func(time_bins)
                
                # Calculate correlation
                correlation = np.corrcoef(population_rate, velocity_aligned)[0, 1]
                
                # Individual channel correlations
                channel_correlations = []
                for channel in analysis_results['top_channels'][:5]:
                    if channel in firing_rates:
                        corr = np.corrcoef(firing_rates[channel], velocity_aligned)[0, 1]
                        channel_correlations.append((channel, corr))
                
                return {
                    'success': True,
                    'population_correlation': correlation,
                    'neural_peak': np.max(population_rate),
                    'velocity_peak': np.max(velocity_aligned),
                    'time_bins': len(time_bins),
                    'channel_correlations': channel_correlations
                }
            else:
                return {'success': False, 'reason': 'Insufficient behavioral data'}
                
        except Exception as e:
            return {'success': False, 'reason': str(e)}


def print_feature_summary(features: Dict, spike_channels: List[int]):
    """
    Print a formatted summary of extracted features.
    
    Args:
        features: Dictionary containing extracted features
        spike_channels: List of spike channel indices
    """
    if features is None:
        print("❌ No features to display")
        return
    
    print("=" * 60)
    print("NEURAL FEATURE SUMMARY")
    print("=" * 60)
    
    # Get feature data
    spike_rms = features['spike_band']['rms_power']
    lfp_power = features['lfp']['lfp_power']
    gamma_power = features['lfp']['gamma_power']
    crossings = features['threshold']['crossing_counts']
    
    # Calculate statistics
    print(f"📊 Feature Statistics:")
    print(f"  • Spike RMS Power: {np.mean(spike_rms):.3f} ± {np.std(spike_rms):.3f}")
    print(f"  • LFP Power: {np.mean(lfp_power):.3f} ± {np.std(lfp_power):.3f}")
    print(f"  • Gamma Power: {np.mean(gamma_power):.3f} ± {np.std(gamma_power):.3f}")
    print(f"  • Total Crossings: {np.sum(crossings):.0f}")
    
    # Find most active channels
    analyzer = FeatureAnalyzer(spike_channels)
    top_channels = analyzer.find_most_active_channels(features, top_n=5)
    print(f"\n🔥 Most active channels: {top_channels}")
    
    # Channel-wise statistics
    print(f"\n📈 Channel Statistics (first 5 channels):")
    for i, ch in enumerate(spike_channels[:5]):
        spike_mean = np.mean(spike_rms[i])
        crossing_sum = np.sum(crossings[i])
        print(f"  • Channel {ch:2d}: Spike={spike_mean:.3f}, Crossings={crossing_sum:.0f}")


def print_spike_summary(analysis_results: Dict):
    """
    Print a comprehensive summary of spike analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from SpikeAnalyzer.analyze_trial()
    """
    if not analysis_results['success']:
        print(f"❌ Analysis failed: {analysis_results.get('error', 'Unknown error')}")
        return
    
    firing_rates = analysis_results['firing_rates']
    quality_metrics = analysis_results['quality_metrics']
    top_channels = analysis_results['top_channels']
    
    print("📊 SPIKE ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Basic statistics
    print(f"🔍 Detection Results:")
    print(f"   • Total channels analyzed: {len(firing_rates)}")
    print(f"   • Total spikes detected: {quality_metrics['n_spikes'].sum()}")
    print(f"   • Mean firing rate: {np.mean([np.mean(rates) for rates in firing_rates.values()]):.1f} Hz")
    print(f"   • Active channels (>50 spikes): {len(quality_metrics[quality_metrics['n_spikes'] > 50])}")
    
    # Quality assessment
    print(f"\n⭐ Quality Assessment:")
    print(f"   • Mean SNR: {quality_metrics['snr'].mean():.2f}")
    print(f"   • High-quality channels (SNR > 3): {len(quality_metrics[quality_metrics['snr'] > 3])}")
    print(f"   • Mean peak-to-peak amplitude: {quality_metrics['peak_to_peak'].mean():.1f}")
    
    # Top channels
    print(f"\n🏆 Top 5 Most Active Channels:")
    for i, channel in enumerate(top_channels[:5]):
        if channel in firing_rates:
            mean_rate = np.mean(firing_rates[channel])
            max_rate = np.max(firing_rates[channel])
            n_spikes = quality_metrics[quality_metrics['channel'] == channel]['n_spikes'].iloc[0]
            snr = quality_metrics[quality_metrics['channel'] == channel]['snr'].iloc[0]
            print(f"   {i+1}. Channel {channel}: {n_spikes} spikes, {mean_rate:.1f} Hz (avg), {max_rate:.1f} Hz (peak), SNR: {snr:.1f}")


def analyze_behavioral_correlation(trial_data: Dict, features: Dict) -> Dict:
    """
    Analyze correlation between neural features and behavioral data.
    
    Args:
        trial_data: Dictionary containing trial data
        features: Dictionary containing extracted features
        
    Returns:
        Dictionary with correlation analysis results
    """
    correlation_results = {'success': False}
    
    # Check if behavioral data is available
    velocity_x = trial_data.get('velocity_x')
    velocity_y = trial_data.get('velocity_y')
    
    if velocity_x is None or velocity_y is None:
        print("⚠️ No behavioral data available for correlation analysis")
        return correlation_results
    
    # Check if behavioral data has movement
    if np.count_nonzero(velocity_x) == 0 and np.count_nonzero(velocity_y) == 0:
        print("⚠️ No movement detected in behavioral data")
        return correlation_results
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
    
    # Analyze neural-behavioral relationships
    if 'spike_band' in features:
        spike_rms = features['spike_band']['rms_power']
        time_axis = features['spike_band']['time_axis']
        
        # Calculate average neural activity over time
        avg_neural_activity = np.mean(spike_rms, axis=0)
        
        # Interpolate behavioral data to match neural time bins
        behavioral_time = np.linspace(0, trial_data.get('duration', 1), len(velocity_magnitude))
        
        # Simple correlation analysis (if time scales are compatible)
        if len(time_axis) > 5 and len(velocity_magnitude) > 5:
            # Resample to common time base
            min_length = min(len(time_axis), len(velocity_magnitude))
            neural_resampled = avg_neural_activity[:min_length]
            behavioral_resampled = velocity_magnitude[:min_length]
            
            # Calculate correlation
            if np.std(neural_resampled) > 0 and np.std(behavioral_resampled) > 0:
                correlation = np.corrcoef(neural_resampled, behavioral_resampled)[0, 1]
                
                correlation_results = {
                    'success': True,
                    'correlation': correlation,
                    'neural_samples': len(neural_resampled),
                    'behavioral_samples': len(behavioral_resampled),
                    'velocity_peak': np.max(velocity_magnitude),
                    'neural_activity_peak': np.max(avg_neural_activity)
                }
    
    return correlation_results


def get_trial_quality_score(trial_data: Dict, features: Dict) -> Dict:
    """
    Calculate a quality score for the trial based on multiple factors.
    
    Args:
        trial_data: Dictionary containing trial data
        features: Dictionary containing extracted features
        
    Returns:
        Dictionary with quality assessment
    """
    quality = {
        'overall_score': 0.0,
        'neural_quality': 0.0,
        'behavioral_quality': 0.0,
        'data_completeness': 0.0,
        'assessment': 'poor'
    }
    
    # Neural data quality (0-40 points)
    if trial_data.get('neural_data') is not None and features is not None:
        neural_data = trial_data['neural_data']
        
        # Check data variance (indicates real signals vs noise/artifacts)
        neural_std = np.std(neural_data)
        if neural_std > 1.0:  # Reasonable signal variance
            quality['neural_quality'] += 20
        
        # Check if features were extracted successfully
        if 'spike_band' in features and 'threshold' in features:
            quality['neural_quality'] += 20
    
    # Behavioral data quality (0-30 points)
    velocity_x = trial_data.get('velocity_x')
    velocity_y = trial_data.get('velocity_y')
    
    if velocity_x is not None and velocity_y is not None:
        quality['data_completeness'] += 15
        
        # Check for actual movement
        movement_detected = (np.count_nonzero(velocity_x) > 0 or 
                           np.count_nonzero(velocity_y) > 0)
        if movement_detected:
            quality['behavioral_quality'] += 30
            
            # Bonus for good movement range
            velocity_range = np.max(np.sqrt(velocity_x**2 + velocity_y**2))
            if velocity_range > 0.1:
                quality['behavioral_quality'] += 10
    
    # Data completeness (0-30 points)
    required_fields = ['neural_data', 'duration', 'outcome']
    present_fields = sum(1 for field in required_fields if trial_data.get(field) is not None)
    quality['data_completeness'] += (present_fields / len(required_fields)) * 15
    
    # Calculate overall score
    quality['overall_score'] = (quality['neural_quality'] + 
                              quality['behavioral_quality'] + 
                              quality['data_completeness'])
    
    # Determine assessment
    if quality['overall_score'] >= 80:
        quality['assessment'] = 'excellent'
    elif quality['overall_score'] >= 60:
        quality['assessment'] = 'good'
    elif quality['overall_score'] >= 40:
        quality['assessment'] = 'fair'
    else:
        quality['assessment'] = 'poor'
    
    return quality 