#!/usr/bin/env python3
"""
Individual Channel PSTH by Target Analysis
===========================================

This script generates smoothed average PSTH plots for each individual channel (21 neurons)
for all trials of the same target (win trials only).

Features:
- Loads all win trials from H5 data
- Groups trials by target
- Calculates smoothed average PSTH for each channel for each target
- Creates comprehensive visualization plots
- Exports results for further analysis

Usage:
    python individual_channel_psth_by_target.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
SAMPLING_RATE = 30000  # Hz
PSTH_BIN_SIZE = 0.01  # seconds (10ms bins)
GAUSSIAN_SIGMA = 0.025  # seconds (25ms smoothing)
THRESHOLD_MULTIPLIER = -4.0  # Spike detection threshold
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"

# Filter parameters
SPIKE_BAND_LOW = 400    # Hz
SPIKE_BAND_HIGH = 6000  # Hz
FILTER_ORDER = 4

class IndividualChannelPSTHAnalyzer:
    """
    Analyzes and plots individual channel PSTH for each target in win trials.
    """
    
    def __init__(self, h5_file_path: str, spike_channels: List[int]):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to the H5 file containing trial data
        spike_channels : list
            List of spike channel indices to analyze
        """
        self.h5_file_path = h5_file_path
        self.spike_channels = spike_channels
        self.trial_data = {}
        self.target_trials = defaultdict(list)
        self.psth_data = {}
        
        print(f"🧠 Individual Channel PSTH Analyzer initialized")
        print(f"   • H5 file: {h5_file_path}")
        print(f"   • Spike channels: {len(spike_channels)} channels")
        print(f"   • Channels: {spike_channels}")
        
    def load_win_trials(self) -> None:
        """Load all win trials from H5 file and group by target."""
        print(f"\n📂 Loading win trials from {self.h5_file_path}...")
        
        if not Path(self.h5_file_path).exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_file_path}")
        
        with h5py.File(self.h5_file_path, 'r') as f:
            available_trials = [key for key in f.keys() if key.startswith('trial_')]
            print(f"   • Found {len(available_trials)} total trials")
            
            win_trials = []
            for trial_key in available_trials:
                trial_group = f[trial_key]
                
                # Check if this is a win trial
                outcome = trial_group.attrs.get('outcome', 'unknown')
                if outcome == 'win':
                    trial_number = int(trial_key.split('_')[1])
                    target_index = trial_group.attrs.get('target_index', -1)
                    
                    if target_index != -1:
                        # Load neural data
                        neural_data = trial_group['neural'][:]
                        
                        # Load behavioral data if available
                        velocity_x = trial_group.get('velocity_x', None)
                        velocity_y = trial_group.get('velocity_y', None)
                        behavioral_timestamps = trial_group.get('behavioral_timestamps', None)
                        
                        if velocity_x is not None:
                            velocity_x = velocity_x[:]
                        if velocity_y is not None:
                            velocity_y = velocity_y[:]
                        if behavioral_timestamps is not None:
                            behavioral_timestamps = behavioral_timestamps[:]
                        
                        # Store trial data
                        self.trial_data[trial_number] = {
                            'neural_data': neural_data,
                            'velocity_x': velocity_x,
                            'velocity_y': velocity_y,
                            'behavioral_timestamps': behavioral_timestamps,
                            'target_index': target_index,
                            'outcome': outcome,
                            'duration': neural_data.shape[1] / SAMPLING_RATE
                        }
                        
                        # Group by target
                        self.target_trials[target_index].append(trial_number)
                        win_trials.append(trial_number)
        
        print(f"✅ Loaded {len(win_trials)} win trials")
        print(f"   • Targets found: {sorted(self.target_trials.keys())}")
        for target_idx, trials in self.target_trials.items():
            print(f"   • Target {target_idx}: {len(trials)} trials")
    
    def detect_spikes_channel(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Detect spikes in a single channel using threshold-based detection.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Raw neural signal data
            
        Returns:
        --------
        np.ndarray
            Array of spike times in seconds
        """
        # Apply bandpass filter
        nyquist = SAMPLING_RATE / 2
        low = SPIKE_BAND_LOW / nyquist
        high = SPIKE_BAND_HIGH / nyquist
        
        try:
            b, a = signal.butter(FILTER_ORDER, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
        except:
            filtered_signal = signal_data
        
        # Calculate threshold
        threshold = THRESHOLD_MULTIPLIER * np.sqrt(np.mean(filtered_signal**2))
        
        # Find threshold crossings (negative-going spikes)
        spike_indices = np.where((filtered_signal[:-1] > threshold) & 
                                (filtered_signal[1:] <= threshold))[0]
        
        # Remove duplicates within refractory period (1ms)
        if len(spike_indices) > 0:
            refractory_samples = int(0.001 * SAMPLING_RATE)  # 1ms refractory period
            clean_spikes = [spike_indices[0]]
            
            for spike_idx in spike_indices[1:]:
                if spike_idx - clean_spikes[-1] > refractory_samples:
                    clean_spikes.append(spike_idx)
            
            spike_indices = np.array(clean_spikes)
        
        # Convert to time
        spike_times = spike_indices / SAMPLING_RATE
        
        return spike_times
    
    def calculate_psth_for_target(self, target_index: int) -> Dict[int, Dict]:
        """
        Calculate PSTH for each channel for a specific target.
        
        Parameters:
        -----------
        target_index : int
            Target index to analyze
            
        Returns:
        --------
        dict
            Dictionary containing PSTH data for each channel
        """
        print(f"\n🔍 Calculating PSTH for target {target_index}...")
        
        trials = self.target_trials[target_index]
        if len(trials) == 0:
            print(f"   ❌ No trials found for target {target_index}")
            return {}
        
        # Get maximum duration across all trials for this target
        max_duration = max([self.trial_data[trial]['duration'] for trial in trials])
        
        # Create time bins
        time_bins = np.arange(0, max_duration + PSTH_BIN_SIZE, PSTH_BIN_SIZE)
        time_centers = time_bins[:-1] + PSTH_BIN_SIZE / 2
        
        channel_psth_data = {}
        
        for channel in self.spike_channels:
            print(f"   • Processing channel {channel}...")
            
            # Collect all spike times for this channel across all trials
            all_spike_times = []
            trial_psths = []
            
            for trial_num in trials:
                if trial_num not in self.trial_data:
                    continue
                    
                trial_data = self.trial_data[trial_num]
                neural_data = trial_data['neural_data']
                
                if channel >= neural_data.shape[0]:
                    continue
                
                # Detect spikes for this channel in this trial
                spike_times = self.detect_spikes_channel(neural_data[channel, :])
                
                # Only include spikes within the trial duration
                trial_duration = trial_data['duration']
                spike_times = spike_times[spike_times < trial_duration]
                
                if len(spike_times) > 0:
                    all_spike_times.extend(spike_times)
                    
                    # Calculate PSTH for this individual trial
                    trial_psth_counts, _ = np.histogram(spike_times, bins=time_bins)
                    trial_psth_rate = trial_psth_counts / PSTH_BIN_SIZE
                    trial_psths.append(trial_psth_rate)
            
            # Calculate average PSTH across all trials
            if len(trial_psths) > 0:
                # Pad trial PSTHs to same length
                max_len = max(len(psth) for psth in trial_psths)
                padded_psths = []
                for psth in trial_psths:
                    padded = np.zeros(max_len)
                    padded[:len(psth)] = psth
                    padded_psths.append(padded)
                
                # Calculate mean and SEM
                mean_psth = np.mean(padded_psths, axis=0)
                sem_psth = np.std(padded_psths, axis=0) / np.sqrt(len(padded_psths))
                
                # Apply Gaussian smoothing
                sigma_bins = GAUSSIAN_SIGMA / PSTH_BIN_SIZE
                smoothed_psth = gaussian_filter1d(mean_psth, sigma=sigma_bins)
                smoothed_sem = gaussian_filter1d(sem_psth, sigma=sigma_bins)
                
                # Trim to match time_centers length
                if len(smoothed_psth) > len(time_centers):
                    smoothed_psth = smoothed_psth[:len(time_centers)]
                    smoothed_sem = smoothed_sem[:len(time_centers)]
                
                channel_psth_data[channel] = {
                    'time_centers': time_centers[:len(smoothed_psth)],
                    'mean_psth': smoothed_psth,
                    'sem_psth': smoothed_sem,
                    'raw_psth': mean_psth[:len(smoothed_psth)],
                    'n_trials': len(trial_psths),
                    'total_spikes': len(all_spike_times),
                    'peak_rate': np.max(smoothed_psth),
                    'mean_rate': np.mean(smoothed_psth)
                }
                
                print(f"     ✓ Channel {channel}: {len(trial_psths)} trials, "
                      f"{len(all_spike_times)} spikes, peak rate: {np.max(smoothed_psth):.1f} Hz")
            else:
                print(f"     ✗ Channel {channel}: No spikes detected")
                channel_psth_data[channel] = {
                    'time_centers': time_centers,
                    'mean_psth': np.zeros(len(time_centers)),
                    'sem_psth': np.zeros(len(time_centers)),
                    'raw_psth': np.zeros(len(time_centers)),
                    'n_trials': 0,
                    'total_spikes': 0,
                    'peak_rate': 0.0,
                    'mean_rate': 0.0
                }
        
        return channel_psth_data
    
    def calculate_all_target_psth(self) -> None:
        """Calculate PSTH for all targets and channels."""
        print(f"\n🧮 Calculating PSTH for all targets...")
        
        self.psth_data = {}
        for target_index in sorted(self.target_trials.keys()):
            self.psth_data[target_index] = self.calculate_psth_for_target(target_index)
        
        print(f"✅ PSTH calculation complete for {len(self.psth_data)} targets")
    
    def plot_individual_channel_psth(self, figsize: Tuple[int, int] = (20, 25)) -> None:
        """
        Create comprehensive plots showing individual channel PSTH for each target.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        print(f"\n📊 Creating individual channel PSTH plots...")
        
        if not self.psth_data:
            print("   ❌ No PSTH data available. Run calculate_all_target_psth() first.")
            return
        
        targets = sorted(self.psth_data.keys())
        n_targets = len(targets)
        n_channels = len(self.spike_channels)
        
        # Create a large figure with subplots for each channel
        fig, axes = plt.subplots(n_channels, n_targets, figsize=figsize, 
                                sharex=True, sharey=False)
        
        # Handle single target case
        if n_targets == 1:
            axes = axes.reshape(-1, 1)
        elif n_channels == 1:
            axes = axes.reshape(1, -1)
        
        # Color map for targets
        colors = plt.cm.tab10(np.linspace(0, 1, n_targets))
        
        for ch_idx, channel in enumerate(self.spike_channels):
            for t_idx, target in enumerate(targets):
                ax = axes[ch_idx, t_idx]
                
                if target in self.psth_data and channel in self.psth_data[target]:
                    data = self.psth_data[target][channel]
                    
                    time_centers = data['time_centers']
                    mean_psth = data['mean_psth']
                    sem_psth = data['sem_psth']
                    
                    if len(time_centers) > 0 and len(mean_psth) > 0:
                        # Plot smoothed PSTH
                        ax.plot(time_centers, mean_psth, 
                               color=colors[t_idx], linewidth=2, alpha=0.8)
                        
                        # Plot confidence interval
                        ax.fill_between(time_centers, 
                                       mean_psth - sem_psth,
                                       mean_psth + sem_psth,
                                       color=colors[t_idx], alpha=0.2)
                        
                        # Set title and labels
                        if ch_idx == 0:
                            ax.set_title(f'Target {target}\n({len(self.target_trials[target])} trials)', 
                                        fontsize=12, fontweight='bold')
                        
                        # Statistics text
                        stats_text = (f'Peak: {data["peak_rate"]:.1f} Hz\n'
                                    f'Mean: {data["mean_rate"]:.1f} Hz\n'
                                    f'Spikes: {data["total_spikes"]}')
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=10, alpha=0.7)
                else:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=10, alpha=0.7)
                
                # Channel label on left side
                if t_idx == 0:
                    ax.set_ylabel(f'Ch {channel}\n(Hz)', fontsize=10, fontweight='bold')
                
                # X-axis label on bottom row
                if ch_idx == n_channels - 1:
                    ax.set_xlabel('Time (s)', fontsize=10)
                
                # Grid
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, max([max(data['time_centers']) for target_data in self.psth_data.values() 
                                   for data in target_data.values() if len(data['time_centers']) > 0]))
        
        plt.suptitle('Individual Channel PSTH by Target (Win Trials Only)\n'
                     f'Smoothed with σ={GAUSSIAN_SIGMA*1000:.0f}ms Gaussian, '
                     f'Bin size={PSTH_BIN_SIZE*1000:.0f}ms', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        
        # Save the figure
        output_path = 'individual_channel_psth_by_target.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Plot saved as: {output_path}")
        
        plt.show()
    
    def plot_target_comparison_per_channel(self, figsize: Tuple[int, int] = (20, 25)) -> None:
        """
        Create plots showing all targets overlaid for each channel.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        print(f"\n📊 Creating target comparison plots for each channel...")
        
        if not self.psth_data:
            print("   ❌ No PSTH data available. Run calculate_all_target_psth() first.")
            return
        
        targets = sorted(self.psth_data.keys())
        n_targets = len(targets)
        n_channels = len(self.spike_channels)
        
        # Create subplots for each channel
        cols = 4
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=False)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for targets
        colors = plt.cm.tab10(np.linspace(0, 1, n_targets))
        
        for ch_idx, channel in enumerate(self.spike_channels):
            row = ch_idx // cols
            col = ch_idx % cols
            ax = axes[row, col]
            
            max_rate = 0
            
            for t_idx, target in enumerate(targets):
                if target in self.psth_data and channel in self.psth_data[target]:
                    data = self.psth_data[target][channel]
                    
                    time_centers = data['time_centers']
                    mean_psth = data['mean_psth']
                    sem_psth = data['sem_psth']
                    
                    if len(time_centers) > 0 and len(mean_psth) > 0:
                        # Plot smoothed PSTH
                        ax.plot(time_centers, mean_psth, 
                               color=colors[t_idx], linewidth=2, alpha=0.8,
                               label=f'Target {target} ({len(self.target_trials[target])} trials)')
                        
                        # Plot confidence interval
                        ax.fill_between(time_centers, 
                                       mean_psth - sem_psth,
                                       mean_psth + sem_psth,
                                       color=colors[t_idx], alpha=0.2)
                        
                        max_rate = max(max_rate, np.max(mean_psth + sem_psth))
            
            # Set title and labels
            ax.set_title(f'Channel {channel} - All Targets', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Legend for first few plots
            if ch_idx < 6:  # Show legend only for first 6 channels to avoid clutter
                ax.legend(fontsize=8, loc='upper right')
            
            # Set reasonable y-limits
            if max_rate > 0:
                ax.set_ylim(0, max_rate * 1.1)
        
        # Hide unused subplots
        for idx in range(n_channels, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Target Comparison per Channel (Win Trials Only)\n'
                     f'Smoothed with σ={GAUSSIAN_SIGMA*1000:.0f}ms Gaussian, '
                     f'Bin size={PSTH_BIN_SIZE*1000:.0f}ms', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        
        # Save the figure
        output_path = 'target_comparison_per_channel.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Plot saved as: {output_path}")
        
        plt.show()
    
    def print_summary_statistics(self) -> None:
        """Print summary statistics for all targets and channels."""
        print(f"\n📊 Summary Statistics")
        print("=" * 50)
        
        if not self.psth_data:
            print("   ❌ No PSTH data available.")
            return
        
        for target in sorted(self.psth_data.keys()):
            print(f"\n🎯 Target {target} ({len(self.target_trials[target])} trials):")
            print("-" * 40)
            
            target_data = self.psth_data[target]
            active_channels = [ch for ch in self.spike_channels if target_data[ch]['total_spikes'] > 0]
            
            print(f"   • Active channels: {len(active_channels)}/{len(self.spike_channels)}")
            
            if len(active_channels) > 0:
                peak_rates = [target_data[ch]['peak_rate'] for ch in active_channels]
                mean_rates = [target_data[ch]['mean_rate'] for ch in active_channels]
                total_spikes = [target_data[ch]['total_spikes'] for ch in active_channels]
                
                print(f"   • Peak firing rates: {np.mean(peak_rates):.1f} ± {np.std(peak_rates):.1f} Hz")
                print(f"   • Mean firing rates: {np.mean(mean_rates):.1f} ± {np.std(mean_rates):.1f} Hz")
                print(f"   • Total spikes: {sum(total_spikes)}")
                print(f"   • Most active channel: {active_channels[np.argmax(peak_rates)]} "
                      f"(peak: {max(peak_rates):.1f} Hz)")
                
                # Top 5 most active channels
                sorted_indices = np.argsort(peak_rates)[::-1][:5]
                top_channels = [active_channels[i] for i in sorted_indices]
                top_rates = [peak_rates[i] for i in sorted_indices]
                
                print(f"   • Top 5 channels: {[(ch, f'{rate:.1f}Hz') for ch, rate in zip(top_channels, top_rates)]}")
    
    def export_data(self, output_path: str = 'psth_data_export.csv') -> None:
        """
        Export PSTH data to CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the CSV file
        """
        print(f"\n💾 Exporting PSTH data to {output_path}...")
        
        if not self.psth_data:
            print("   ❌ No PSTH data available.")
            return
        
        # Create a comprehensive DataFrame
        export_data = []
        
        for target in sorted(self.psth_data.keys()):
            target_data = self.psth_data[target]
            
            for channel in self.spike_channels:
                if channel in target_data:
                    data = target_data[channel]
                    
                    # Create one row per time bin
                    for i, time_center in enumerate(data['time_centers']):
                        if i < len(data['mean_psth']):
                            export_data.append({
                                'target': target,
                                'channel': channel,
                                'time_center': time_center,
                                'mean_firing_rate': data['mean_psth'][i],
                                'sem_firing_rate': data['sem_psth'][i],
                                'raw_firing_rate': data['raw_psth'][i],
                                'n_trials': data['n_trials'],
                                'total_spikes': data['total_spikes'],
                                'peak_rate': data['peak_rate'],
                                'mean_rate': data['mean_rate']
                            })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        print(f"   ✅ Exported {len(df)} rows to {output_path}")
        print(f"   • Targets: {df['target'].nunique()}")
        print(f"   • Channels: {df['channel'].nunique()}")
        print(f"   • Time points: {df['time_center'].nunique()}")

def main():
    """Main function to run the analysis."""
    print("🧠 Individual Channel PSTH by Target Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = IndividualChannelPSTHAnalyzer(H5_FILE_PATH, SPIKE_CHANNELS)
    
    try:
        # Load win trials
        analyzer.load_win_trials()
        
        if len(analyzer.target_trials) == 0:
            print("❌ No win trials found. Please check your data.")
            return
        
        # Calculate PSTH for all targets
        analyzer.calculate_all_target_psth()
        
        # Print summary statistics
        analyzer.print_summary_statistics()
        
        # Create visualizations
        analyzer.plot_individual_channel_psth()
        analyzer.plot_target_comparison_per_channel()
        
        # Export data
        analyzer.export_data()
        
        print("\n✅ Analysis complete!")
        print(f"   • Processed {len(analyzer.trial_data)} win trials")
        print(f"   • Analyzed {len(analyzer.target_trials)} targets")
        print(f"   • Evaluated {len(SPIKE_CHANNELS)} channels")
        print(f"   • Generated plots and exported data")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 