#!/usr/bin/env python3
"""
Aligned PSTH with Dynamic Time Warping
======================================

This script uses dynamic time warping (DTW) to temporally align neural responses
across trials with the same target direction (heading) and outcome, using spike
time data rather than raw voltages or binned RMS.

Key Features:
- Groups trials by unique (heading, outcome) pairs
- Converts spike trains to binned spike density functions 
- Applies DTW to align trials to a reference template
- Computes aligned and unaligned PSTHs
- Generates comprehensive visualizations
- Exports results for further analysis

Movement Onset Detection:
- Uses trial_start boolean flag to identify trial beginning
- Detects movement onset as first non-zero velocity after trial_start
- Velocity values of 0 indicate no movement (not noise)
- Aligns all neural data to this precise movement initiation time

Usage:
    python aligned_psth.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from pathlib import Path
from scipy import signal, ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Add DTW capability
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️  FastDTW not available, using custom DTW implementation")

# Import existing utilities
from utils.spike_detection import SpikeDetector
from utils.h5_data_loader import H5DataLoader

# Configuration
SPIKE_CHANNELS = [0, 1, 2]#, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
SAMPLING_RATE = 30000  # Hz
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"

# PSTH and DTW parameters
PSTH_BIN_SIZE = 0.01  # seconds (10ms bins)
ALIGN_WINDOW = (-0.5, 1.0)  # seconds relative to alignment event
GAUSSIAN_SIGMA = 0.025  # seconds (25ms smoothing)
DTW_RADIUS = 0.1  # DTW constraint radius (10% of sequence length)

class AlignedPSTHAnalyzer:
    """
    Analyzes neural responses using DTW alignment across trials with same heading and outcome.
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
        
        # Initialize data structures
        self.trial_data = {}
        self.grouped_trials = defaultdict(list)  # (heading, outcome) -> [trial_numbers]
        self.spike_times = {}  # trial -> channel -> spike_times
        self.align_times = {}  # trial -> alignment_time
        self.trial_headings = {}  # trial -> heading
        self.trial_outcomes = {}  # trial -> outcome
        
        # Results storage
        self.aligned_psths = {}  # (heading, outcome, channel) -> aligned_psth
        self.unaligned_psths = {}  # (heading, outcome, channel) -> unaligned_psth
        self.alignment_info = {}  # (heading, outcome, channel) -> alignment_metadata
        
        # Initialize utilities
        self.spike_detector = SpikeDetector(
            sampling_rate=SAMPLING_RATE,
            threshold_factor=5.0,
            spike_window=(-10, 32),
            good_channels=spike_channels
        )
        
        self.h5_loader = H5DataLoader(h5_file_path)
        
        print(f"🧠 Aligned PSTH Analyzer initialized")
        print(f"   • H5 file: {h5_file_path}")
        print(f"   • Spike channels: {len(spike_channels)} channels")
        print(f"   • Alignment window: {ALIGN_WINDOW[0]:.1f}s to {ALIGN_WINDOW[1]:.1f}s")
        print(f"   • PSTH bin size: {PSTH_BIN_SIZE*1000:.0f}ms")
        print(f"   • DTW available: {DTW_AVAILABLE}")
    
    def load_trial_data(self, min_trials_per_group: int = 3) -> None:
        """
        Load trial data and group by (heading, outcome) pairs.
        
        Parameters:
        -----------
        min_trials_per_group : int
            Minimum number of trials required per (heading, outcome) group
        """
        print(f"\n📊 Loading trial data and grouping by (heading, outcome)...")
        
        # Get trial info
        trial_info = self.h5_loader.get_trial_info()
        
        loaded_trials = 0
        for _, row in trial_info.iterrows():
            trial_number = row['trial_number']
            outcome = row['outcome']
            target_index = row['target_index']
            
            # Skip trials with unknown targets
            if target_index < 0 or target_index > 7:
                continue
                
            try:
                # Load trial data
                trial_data = self.h5_loader.load_trial_data(trial_number)
                
                # Extract neural data
                neural_data = trial_data['neural_data']
                if neural_data is None:
                    continue
                
                # Store trial data
                self.trial_data[trial_number] = trial_data
                
                # Map target index to heading (0-7 -> 0°, 45°, 90°, ..., 315°)
                heading = target_index * 45  # 8 targets in circle
                
                # Store trial metadata
                self.trial_headings[trial_number] = heading
                self.trial_outcomes[trial_number] = 1 if outcome == 'win' else 0
                
                # Group trials by (heading, outcome)
                group_key = (heading, self.trial_outcomes[trial_number])
                self.grouped_trials[group_key].append(trial_number)
                
                # Calculate alignment time (movement onset)
                align_time = self._calculate_alignment_time(trial_data)
                self.align_times[trial_number] = align_time
                
                # Extract spike times
                spike_times = self._extract_spike_times(neural_data, trial_number)
                self.spike_times[trial_number] = spike_times
                
                loaded_trials += 1
                
            except Exception as e:
                print(f"   ⚠️  Failed to load trial {trial_number}: {e}")
                continue
        
        # Filter groups with insufficient trials
        filtered_groups = {}
        for group_key, trials in self.grouped_trials.items():
            if len(trials) >= min_trials_per_group:
                filtered_groups[group_key] = trials
        
        self.grouped_trials = filtered_groups
        
        print(f"✅ Loaded {loaded_trials} trials")
        print(f"   • Valid groups: {len(self.grouped_trials)}")
        for (heading, outcome), trials in self.grouped_trials.items():
            outcome_str = 'success' if outcome == 1 else 'failure'
            print(f"   • Heading {heading}°, {outcome_str}: {len(trials)} trials")
    
    def _calculate_alignment_time(self, trial_data: Dict) -> float:
        """
        Calculate alignment time (movement onset) for a trial.
        
        Movement onset is defined as the first non-zero velocity after trial_start flag.
        
        Parameters:
        -----------
        trial_data : dict
            Trial data dictionary
            
        Returns:
        --------
        float
            Alignment time in seconds from start of trial
        """
        velocity_x = trial_data.get('velocity_x')
        velocity_y = trial_data.get('velocity_y')
        behavioral_timestamps = trial_data.get('behavioral_timestamps')
        
        if velocity_x is None or velocity_y is None or behavioral_timestamps is None:
            # Fallback to middle of trial
            return trial_data.get('duration', 1.0) / 2
        
        # Convert to numpy arrays if needed
        velocity_x = np.array(velocity_x)
        velocity_y = np.array(velocity_y)
        behavioral_timestamps = np.array(behavioral_timestamps)
        
        # Calculate speed magnitude
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Find trial start index 
        trial_start_idx = 0  # Default for H5 individual trial data
        
        # If we have trial_start flag data, use it to find the actual start
        trial_start_flag = trial_data.get('trial_start')
        if trial_start_flag is not None:
            trial_start_flag = np.array(trial_start_flag)
            start_indices = np.where(trial_start_flag == True)[0]
            if len(start_indices) > 0:
                trial_start_idx = start_indices[0]
                print(f"   📍 Using trial_start flag: movement search starts at index {trial_start_idx}")
        
        # Look for first non-zero movement after trial start
        # Start searching from trial start onwards
        post_start_speed = speed[trial_start_idx:]
        post_start_timestamps = behavioral_timestamps[trial_start_idx:]
        
        # Find first non-zero velocity (actual movement onset)
        movement_indices = np.where(post_start_speed > 0)[0]
        
        if len(movement_indices) > 0:
            # First non-zero velocity after trial start
            movement_onset_idx = movement_indices[0]
            movement_onset_time = post_start_timestamps[movement_onset_idx]
            
            # Return time relative to start of trial
            return movement_onset_time - behavioral_timestamps[0]
        
        # If no movement detected, fallback to middle of trial
        print(f"   ⚠️  No movement detected in trial, using trial midpoint")
        return trial_data.get('duration', 1.0) / 2
    
    def _extract_spike_times(self, neural_data: np.ndarray, trial_number: int) -> Dict[int, np.ndarray]:
        """
        Extract spike times from neural data.
        
        Parameters:
        -----------
        neural_data : np.ndarray
            Neural data (channels x samples)
        trial_number : int
            Trial number for context
            
        Returns:
        --------
        dict
            Dictionary mapping channel -> spike_times array
        """
        # Use existing spike detector
        spike_data = self.spike_detector.detect_spikes_all_channels(neural_data)
        
        spike_times = {}
        for channel in self.spike_channels:
            if channel in spike_data:
                # Convert spike indices to times
                spike_indices = spike_data[channel]['spike_times']
                spike_times[channel] = spike_indices / SAMPLING_RATE
            else:
                spike_times[channel] = np.array([])
        
        return spike_times
    
    def _create_spike_density_function(self, spike_times: np.ndarray, 
                                     align_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binned spike density function aligned to event.
        
        Parameters:
        -----------
        spike_times : np.ndarray
            Spike times in seconds
        align_time : float
            Alignment time in seconds
            
        Returns:
        --------
        tuple
            (time_bins, spike_density)
        """
        # Create time bins relative to alignment
        time_bins = np.arange(ALIGN_WINDOW[0], ALIGN_WINDOW[1] + PSTH_BIN_SIZE, PSTH_BIN_SIZE)
        
        # Align spike times to event
        aligned_spikes = spike_times - align_time
        
        # Only include spikes within window
        valid_spikes = aligned_spikes[(aligned_spikes >= ALIGN_WINDOW[0]) & 
                                     (aligned_spikes <= ALIGN_WINDOW[1])]
        
        # Create histogram
        spike_counts, _ = np.histogram(valid_spikes, bins=time_bins)
        
        # Convert to firing rate
        spike_density = spike_counts / PSTH_BIN_SIZE
        
        # Time centers for plotting
        time_centers = time_bins[:-1] + PSTH_BIN_SIZE / 2
        
        return time_centers, spike_density
    
    def _simple_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Simple DTW implementation as fallback.
        
        Parameters:
        -----------
        seq1, seq2 : np.ndarray
            Sequences to align
            
        Returns:
        --------
        tuple
            (path1, path2) - aligned indices
        """
        n, m = len(seq1), len(seq2)
        
        # Create distance matrix
        distance_matrix = cdist(seq1.reshape(-1, 1), seq2.reshape(-1, 1), metric='euclidean')
        
        # DTW dynamic programming
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = distance_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                             dtw_matrix[i, j-1],      # deletion
                                             dtw_matrix[i-1, j-1])    # match
        
        # Backtrack to find path
        path1, path2 = [], []
        i, j = n, m
        
        while i > 0 and j > 0:
            path1.append(i - 1)
            path2.append(j - 1)
            
            # Choose minimum cost direction
            costs = [dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]]
            min_idx = np.argmin(costs)
            
            if min_idx == 0:    # diagonal
                i -= 1
                j -= 1
            elif min_idx == 1:  # up
                i -= 1
            else:               # left
                j -= 1
        
        return path1[::-1], path2[::-1]
    
    def _plot_aligned_behavioral_kinematics(self, ax: plt.Axes, unique_groups: List[Tuple[int, int]]) -> None:
        """
        Plot time-warped average behavioral kinematics for each group.
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes to plot on
        unique_groups : list
            List of (heading, outcome) groups
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        
        for group_idx, group_key in enumerate(unique_groups):
            heading, outcome = group_key
            trials = self.grouped_trials[group_key]
            outcome_str = 'Success' if outcome == 1 else 'Failure'
            
            # Collect behavioral data for this group
            velocity_x_trials = []
            velocity_y_trials = []
            time_centers = None
            
            for trial_num in trials:
                if trial_num not in self.trial_data:
                    continue
                    
                trial_data = self.trial_data[trial_num]
                velocity_x = trial_data.get('velocity_x')
                velocity_y = trial_data.get('velocity_y')
                
                if velocity_x is None or velocity_y is None:
                    continue
                
                # Create aligned velocity data using same approach as neural data
                align_time = self.align_times[trial_num]
                behavioral_timestamps = trial_data.get('behavioral_timestamps')
                
                if behavioral_timestamps is None:
                    continue
                
                # Create time bins relative to alignment
                time_bins = np.arange(ALIGN_WINDOW[0], ALIGN_WINDOW[1] + PSTH_BIN_SIZE, PSTH_BIN_SIZE)
                
                # Align behavioral timestamps to event
                aligned_timestamps = np.array(behavioral_timestamps) - behavioral_timestamps[0] - align_time
                
                # Interpolate velocity to alignment time bins
                from scipy.interpolate import interp1d
                
                # Only use data within the alignment window
                valid_indices = (aligned_timestamps >= ALIGN_WINDOW[0]) & (aligned_timestamps <= ALIGN_WINDOW[1])
                
                if np.sum(valid_indices) < 2:  # Need at least 2 points for interpolation
                    continue
                
                valid_times = aligned_timestamps[valid_indices]
                valid_vx = np.array(velocity_x)[valid_indices]
                valid_vy = np.array(velocity_y)[valid_indices]
                
                try:
                    # Interpolate to common time grid
                    t_centers = time_bins[:-1] + PSTH_BIN_SIZE / 2
                    
                    interp_vx = interp1d(valid_times, valid_vx, kind='linear', 
                                        bounds_error=False, fill_value=0)(t_centers)
                    interp_vy = interp1d(valid_times, valid_vy, kind='linear', 
                                        bounds_error=False, fill_value=0)(t_centers)
                    
                    velocity_x_trials.append(interp_vx)
                    velocity_y_trials.append(interp_vy)
                    
                    if time_centers is None:
                        time_centers = t_centers
                        
                except:
                    continue
            
            if len(velocity_x_trials) == 0:
                continue
            
            # Convert to arrays
            velocity_x_trials = np.array(velocity_x_trials)
            velocity_y_trials = np.array(velocity_y_trials)
            
            # Apply DTW alignment to behavioral data (same as neural)
            if len(velocity_x_trials) > 1:
                # Use magnitude as reference for DTW
                magnitude_trials = np.sqrt(velocity_x_trials**2 + velocity_y_trials**2)
                reference_magnitude = np.median(magnitude_trials, axis=0)
                
                aligned_vx_trials = []
                aligned_vy_trials = []
                
                for i in range(len(velocity_x_trials)):
                    try:
                        if DTW_AVAILABLE:
                            distance, path = fastdtw(reference_magnitude, magnitude_trials[i], 
                                                   dist=lambda x, y: abs(x - y))
                            ref_indices = [p[0] for p in path]
                            trial_indices = [p[1] for p in path]
                        else:
                            ref_indices, trial_indices = self._simple_dtw(reference_magnitude, magnitude_trials[i])
                        
                        # Warp velocity components
                        warped_vx = np.interp(np.arange(len(reference_magnitude)), ref_indices, 
                                            velocity_x_trials[i][trial_indices])
                        warped_vy = np.interp(np.arange(len(reference_magnitude)), ref_indices, 
                                            velocity_y_trials[i][trial_indices])
                        
                        aligned_vx_trials.append(warped_vx)
                        aligned_vy_trials.append(warped_vy)
                        
                    except:
                        # Fallback to unaligned
                        aligned_vx_trials.append(velocity_x_trials[i])
                        aligned_vy_trials.append(velocity_y_trials[i])
                
                # Compute averages
                mean_vx = np.mean(aligned_vx_trials, axis=0)
                mean_vy = np.mean(aligned_vy_trials, axis=0)
                sem_vx = np.std(aligned_vx_trials, axis=0) / np.sqrt(len(aligned_vx_trials))
                sem_vy = np.std(aligned_vy_trials, axis=0) / np.sqrt(len(aligned_vy_trials))
            else:
                # Single trial
                mean_vx = velocity_x_trials[0]
                mean_vy = velocity_y_trials[0]
                sem_vx = np.zeros_like(mean_vx)
                sem_vy = np.zeros_like(mean_vy)
            
            # Calculate magnitude
            mean_magnitude = np.sqrt(mean_vx**2 + mean_vy**2)
            
            # Plot velocity components and magnitude
            color = colors[group_idx]
            label_base = f'{heading}° {outcome_str} ({len(trials)}T)'
            
            # Plot with offset for visibility
            offset = group_idx * 0.02
            
            ax.plot(time_centers, mean_vx + offset, color=color, linestyle='-', alpha=0.8, 
                   label=f'{label_base} - Vx')
            ax.plot(time_centers, mean_vy + offset, color=color, linestyle='--', alpha=0.8, 
                   label=f'{label_base} - Vy')
            ax.plot(time_centers, mean_magnitude + offset, color=color, linestyle='-', linewidth=2, 
                   label=f'{label_base} - Mag')
            
            # Add confidence intervals for magnitude
            ax.fill_between(time_centers, 
                           mean_magnitude - np.sqrt(sem_vx**2 + sem_vy**2) + offset,
                           mean_magnitude + np.sqrt(sem_vx**2 + sem_vy**2) + offset,
                           color=color, alpha=0.2)
        
        # Format behavioral subplot
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Movement Onset')
        ax.set_xlabel('Time from movement onset (s)')
        ax.set_ylabel('Velocity (units/s)')
        ax.set_title('Time-Warped Average Behavioral Kinematics\n'
                    'Solid: X-velocity, Dashed: Y-velocity, Bold: Magnitude', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ALIGN_WINDOW)
        
        # Legend with smaller font
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_single_condition_behavioral_kinematics(self, ax: plt.Axes, group_key: Tuple[int, int]) -> None:
        """
        Plot time-warped behavioral kinematics for a single condition.
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes to plot on
        group_key : tuple
            (heading, outcome) pair for this condition
        """
        heading, outcome = group_key
        trials = self.grouped_trials[group_key]
        outcome_str = 'Success' if outcome == 1 else 'Failure'
        
        # Collect behavioral data for this group
        velocity_x_trials = []
        velocity_y_trials = []
        time_centers = None
        
        for trial_num in trials:
            if trial_num not in self.trial_data:
                continue
                
            trial_data = self.trial_data[trial_num]
            velocity_x = trial_data.get('velocity_x')
            velocity_y = trial_data.get('velocity_y')
            
            if velocity_x is None or velocity_y is None:
                continue
            
            # Create aligned velocity data using same approach as neural data
            align_time = self.align_times[trial_num]
            behavioral_timestamps = trial_data.get('behavioral_timestamps')
            
            if behavioral_timestamps is None:
                continue
            
            # Create time bins relative to alignment
            time_bins = np.arange(ALIGN_WINDOW[0], ALIGN_WINDOW[1] + PSTH_BIN_SIZE, PSTH_BIN_SIZE)
            
            # Align behavioral timestamps to event
            aligned_timestamps = np.array(behavioral_timestamps) - behavioral_timestamps[0] - align_time
            
            # Interpolate velocity to alignment time bins
            from scipy.interpolate import interp1d
            
            # Only use data within the alignment window
            valid_indices = (aligned_timestamps >= ALIGN_WINDOW[0]) & (aligned_timestamps <= ALIGN_WINDOW[1])
            
            if np.sum(valid_indices) < 2:  # Need at least 2 points for interpolation
                continue
            
            valid_times = aligned_timestamps[valid_indices]
            valid_vx = np.array(velocity_x)[valid_indices]
            valid_vy = np.array(velocity_y)[valid_indices]
            
            try:
                # Interpolate to common time grid
                t_centers = time_bins[:-1] + PSTH_BIN_SIZE / 2
                
                interp_vx = interp1d(valid_times, valid_vx, kind='linear', 
                                    bounds_error=False, fill_value=0)(t_centers)
                interp_vy = interp1d(valid_times, valid_vy, kind='linear', 
                                    bounds_error=False, fill_value=0)(t_centers)
                
                velocity_x_trials.append(interp_vx)
                velocity_y_trials.append(interp_vy)
                
                if time_centers is None:
                    time_centers = t_centers
                    
            except:
                continue
        
        if len(velocity_x_trials) == 0:
            # No data available
            ax.text(0.5, 0.5, f'No behavioral data\n{heading}° {outcome_str}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlim(ALIGN_WINDOW)
            return
        
        # Convert to arrays
        velocity_x_trials = np.array(velocity_x_trials)
        velocity_y_trials = np.array(velocity_y_trials)
        
        # Apply DTW alignment to behavioral data (same as neural)
        if len(velocity_x_trials) > 1:
            # Use magnitude as reference for DTW
            magnitude_trials = np.sqrt(velocity_x_trials**2 + velocity_y_trials**2)
            reference_magnitude = np.median(magnitude_trials, axis=0)
            
            aligned_vx_trials = []
            aligned_vy_trials = []
            
            for i in range(len(velocity_x_trials)):
                try:
                    if DTW_AVAILABLE:
                        distance, path = fastdtw(reference_magnitude, magnitude_trials[i], 
                                               dist=lambda x, y: abs(x - y))
                        ref_indices = [p[0] for p in path]
                        trial_indices = [p[1] for p in path]
                    else:
                        ref_indices, trial_indices = self._simple_dtw(reference_magnitude, magnitude_trials[i])
                    
                    # Warp velocity components
                    warped_vx = np.interp(np.arange(len(reference_magnitude)), ref_indices, 
                                        velocity_x_trials[i][trial_indices])
                    warped_vy = np.interp(np.arange(len(reference_magnitude)), ref_indices, 
                                        velocity_y_trials[i][trial_indices])
                    
                    aligned_vx_trials.append(warped_vx)
                    aligned_vy_trials.append(warped_vy)
                    
                except:
                    # Fallback to unaligned
                    aligned_vx_trials.append(velocity_x_trials[i])
                    aligned_vy_trials.append(velocity_y_trials[i])
            
            # Compute averages
            mean_vx = np.mean(aligned_vx_trials, axis=0)
            mean_vy = np.mean(aligned_vy_trials, axis=0)
            sem_vx = np.std(aligned_vx_trials, axis=0) / np.sqrt(len(aligned_vx_trials))
            sem_vy = np.std(aligned_vy_trials, axis=0) / np.sqrt(len(aligned_vy_trials))
        else:
            # Single trial
            mean_vx = velocity_x_trials[0]
            mean_vy = velocity_y_trials[0]
            sem_vx = np.zeros_like(mean_vx)
            sem_vy = np.zeros_like(mean_vy)
        
        # Calculate magnitude
        mean_magnitude = np.sqrt(mean_vx**2 + mean_vy**2)
        sem_magnitude = np.sqrt(sem_vx**2 + sem_vy**2)
        
        # Plot velocity components and magnitude
        ax.plot(time_centers, mean_vx, 'blue', linestyle='-', linewidth=2, 
               label=f'X-velocity', alpha=0.8)
        ax.plot(time_centers, mean_vy, 'green', linestyle='--', linewidth=2, 
               label=f'Y-velocity', alpha=0.8)
        ax.plot(time_centers, mean_magnitude, 'red', linestyle='-', linewidth=3, 
               label=f'Magnitude')
        
        # Add confidence intervals
        ax.fill_between(time_centers, mean_vx - sem_vx, mean_vx + sem_vx,
                       color='blue', alpha=0.2)
        ax.fill_between(time_centers, mean_vy - sem_vy, mean_vy + sem_vy,
                       color='green', alpha=0.2)
        ax.fill_between(time_centers, mean_magnitude - sem_magnitude, mean_magnitude + sem_magnitude,
                       color='red', alpha=0.2)
        
        # Format subplot
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Movement Onset')
        ax.set_ylabel('Velocity (units/s)')
        ax.set_title(f'Time-Warped Behavioral Kinematics - {heading}° {outcome_str}\n'
                    f'{len(trials)} trials averaged after DTW alignment', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ALIGN_WINDOW)
        ax.legend(fontsize=10)
    
    def _align_trials_dtw(self, group_key: Tuple[int, int], channel: int) -> Dict:
        """
        Align trials using DTW for a specific (heading, outcome, channel) group.
        
        Parameters:
        -----------
        group_key : tuple
            (heading, outcome) pair
        channel : int
            Channel to analyze
            
        Returns:
        --------
        dict
            Alignment results and PSTHs
        """
        heading, outcome = group_key
        trials = self.grouped_trials[group_key]
        
        print(f"   🔄 Aligning channel {channel}, heading {heading}°, outcome {'success' if outcome else 'failure'}...")
        
        # Extract spike density functions for all trials
        spike_densities = []
        time_centers = None
        
        for trial_num in trials:
            if trial_num in self.spike_times and channel in self.spike_times[trial_num]:
                spike_times = self.spike_times[trial_num][channel]
                align_time = self.align_times[trial_num]
                
                t_centers, spike_density = self._create_spike_density_function(spike_times, align_time)
                spike_densities.append(spike_density)
                
                if time_centers is None:
                    time_centers = t_centers
        
        if len(spike_densities) == 0:
            return {'aligned_psth': None, 'unaligned_psth': None, 'n_trials': 0}
        
        # Convert to numpy array
        spike_densities = np.array(spike_densities)
        
        # Smooth individual trials
        smoothed_densities = []
        for density in spike_densities:
            if GAUSSIAN_SIGMA > 0:
                sigma_bins = GAUSSIAN_SIGMA / PSTH_BIN_SIZE
                smoothed = ndimage.gaussian_filter1d(density, sigma=sigma_bins)
            else:
                smoothed = density
            smoothed_densities.append(smoothed)
        
        smoothed_densities = np.array(smoothed_densities)
        
        # Calculate unaligned PSTH
        unaligned_psth = np.mean(smoothed_densities, axis=0)
        unaligned_sem = np.std(smoothed_densities, axis=0) / np.sqrt(len(smoothed_densities))
        
        # DTW alignment
        if len(smoothed_densities) > 1:
            # Use median trial as reference
            reference_trial = np.median(smoothed_densities, axis=0)
            
            # Align each trial to reference
            aligned_trials = []
            alignment_paths = []
            
            for i, trial_density in enumerate(smoothed_densities):
                try:
                    if DTW_AVAILABLE:
                        # Use FastDTW
                        distance, path = fastdtw(reference_trial, trial_density, dist=lambda x, y: abs(x - y))
                        ref_indices = [p[0] for p in path]
                        trial_indices = [p[1] for p in path]
                    else:
                        # Use simple DTW
                        ref_indices, trial_indices = self._simple_dtw(reference_trial, trial_density)
                    
                    # Warp trial to match reference
                    warped_trial = np.interp(np.arange(len(reference_trial)), ref_indices, 
                                           trial_density[trial_indices])
                    aligned_trials.append(warped_trial)
                    alignment_paths.append((ref_indices, trial_indices))
                    
                except Exception as e:
                    print(f"     ⚠️  DTW failed for trial {i}: {e}")
                    # Fallback to unaligned
                    aligned_trials.append(trial_density)
                    alignment_paths.append((None, None))
            
            # Calculate aligned PSTH
            aligned_psth = np.mean(aligned_trials, axis=0)
            aligned_sem = np.std(aligned_trials, axis=0) / np.sqrt(len(aligned_trials))
            
        else:
            # Single trial - no alignment needed
            aligned_psth = unaligned_psth
            aligned_sem = unaligned_sem
            alignment_paths = [(None, None)]
        
        # Calculate metrics
        peak_rate_unaligned = np.max(unaligned_psth)
        peak_rate_aligned = np.max(aligned_psth)
        
        # Find peak latency (time of maximum firing)
        peak_idx_unaligned = np.argmax(unaligned_psth)
        peak_idx_aligned = np.argmax(aligned_psth)
        
        peak_latency_unaligned = time_centers[peak_idx_unaligned] if len(time_centers) > peak_idx_unaligned else 0
        peak_latency_aligned = time_centers[peak_idx_aligned] if len(time_centers) > peak_idx_aligned else 0
        
        return {
            'time_centers': time_centers,
            'aligned_psth': aligned_psth,
            'aligned_sem': aligned_sem,
            'unaligned_psth': unaligned_psth,
            'unaligned_sem': unaligned_sem,
            'n_trials': len(trials),
            'peak_rate_aligned': peak_rate_aligned,
            'peak_rate_unaligned': peak_rate_unaligned,
            'peak_latency_aligned': peak_latency_aligned,
            'peak_latency_unaligned': peak_latency_unaligned,
            'alignment_paths': alignment_paths
        }
    
    def compute_aligned_psths(self) -> None:
        """Compute aligned and unaligned PSTHs for all groups and channels."""
        print(f"\n🧮 Computing aligned PSTHs using DTW...")
        
        total_computations = len(self.grouped_trials) * len(self.spike_channels)
        completed = 0
        
        for group_key in self.grouped_trials:
            heading, outcome = group_key
            
            for channel in self.spike_channels:
                # Compute alignment for this (heading, outcome, channel) combination
                result = self._align_trials_dtw(group_key, channel)
                
                # Store results
                result_key = (heading, outcome, channel)
                self.aligned_psths[result_key] = result
                
                completed += 1
                if completed % 10 == 0:
                    print(f"   Progress: {completed}/{total_computations}")
        
        print(f"✅ Computed aligned PSTHs for {len(self.grouped_trials)} groups and {len(self.spike_channels)} channels")
    
    def plot_aligned_psths(self, figsize: Tuple[int, int] = (20, 28)) -> None:
        """
        Create comprehensive plots showing aligned vs unaligned PSTHs with behavioral kinematics.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        print(f"\n📊 Creating aligned PSTH plots with behavioral kinematics...")
        
        if not self.aligned_psths:
            print("   ❌ No aligned PSTH data available.")
            return
        
        # Group by (heading, outcome)
        unique_groups = list(self.grouped_trials.keys())
        n_groups = len(unique_groups)
        n_channels = len(self.spike_channels)
        
        # Create figure with behavioral subplot at top + neural subplots
        # Use gridspec for better control over subplot sizes
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_groups + 1, n_channels, figure=fig, height_ratios=[0.8] + [1.0] * n_groups)
        
        # Create behavioral subplot spanning all columns
        behavioral_ax = fig.add_subplot(gs[0, :])
        
        # Create neural subplot grid
        neural_axes = []
        for group_idx in range(n_groups):
            row_axes = []
            for ch_idx in range(n_channels):
                ax = fig.add_subplot(gs[group_idx + 1, ch_idx])
                row_axes.append(ax)
            neural_axes.append(row_axes)
        
        # Convert to numpy array for easier indexing
        neural_axes = np.array(neural_axes)
        
        # Handle single group/channel cases for neural axes
        if n_groups == 1:
            neural_axes = neural_axes.reshape(1, -1)
        if n_channels == 1:
            neural_axes = neural_axes.reshape(-1, 1)
        if n_groups == 1 and n_channels == 1:
            neural_axes = np.array([[neural_axes]])
        
        # Compute and plot time-warped behavioral kinematics
        print("   🎯 Computing time-warped behavioral averages...")
        self._plot_aligned_behavioral_kinematics(behavioral_ax, unique_groups)
        
        # Plot neural data in lower subplots
        for group_idx, group_key in enumerate(unique_groups):
            heading, outcome = group_key
            outcome_str = 'Success' if outcome == 1 else 'Failure'
            
            for ch_idx, channel in enumerate(self.spike_channels):
                ax = neural_axes[group_idx, ch_idx]
                
                result_key = (heading, outcome, channel)
                if result_key in self.aligned_psths:
                    data = self.aligned_psths[result_key]
                    
                    if data['aligned_psth'] is not None:
                        time_centers = data['time_centers']
                        
                        # Plot unaligned PSTH
                        ax.plot(time_centers, data['unaligned_psth'], 
                               'gray', alpha=0.7, linewidth=1.5, label='Unaligned')
                        ax.fill_between(time_centers, 
                                       data['unaligned_psth'] - data['unaligned_sem'],
                                       data['unaligned_psth'] + data['unaligned_sem'],
                                       color='gray', alpha=0.2)
                        
                        # Plot aligned PSTH
                        ax.plot(time_centers, data['aligned_psth'], 
                               'red', linewidth=2, label='DTW Aligned')
                        ax.fill_between(time_centers, 
                                       data['aligned_psth'] - data['aligned_sem'],
                                       data['aligned_psth'] + data['aligned_sem'],
                                       color='red', alpha=0.3)
                        
                        # Add vertical line at alignment point
                        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                        
                        # Statistics
                        peak_improvement = ((data['peak_rate_aligned'] - data['peak_rate_unaligned']) / 
                                          (data['peak_rate_unaligned'] + 1e-6)) * 100
                        
                        stats_text = (f"Trials: {data['n_trials']}\n"
                                    f"Peak: {data['peak_rate_aligned']:.1f} Hz\n"
                                    f"Improvement: {peak_improvement:.1f}%")
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        # Titles and labels
                        if group_idx == 0:
                            ax.set_title(f'Ch {channel}', fontsize=10, fontweight='bold')
                        
                        if ch_idx == 0:
                            ax.set_ylabel(f'{heading}° {outcome_str}\n(spikes/s)', fontsize=10)
                        
                        if group_idx == n_groups - 1:
                            ax.set_xlabel('Time from movement onset (s)', fontsize=10)
                        
                        # Legend for first subplot
                        if group_idx == 0 and ch_idx == 0:
                            ax.legend(fontsize=8)
                    else:
                        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=10, alpha=0.7)
                
                ax.grid(True, alpha=0.3)
                ax.set_xlim(ALIGN_WINDOW)
        
        plt.suptitle('Aligned vs Unaligned PSTHs using Dynamic Time Warping\n'
                     f'Top: Time-Warped Behavioral Kinematics, Bottom: Neural PSTHs\n'
                     f'Aligned to Movement Onset, σ={GAUSSIAN_SIGMA*1000:.0f}ms smoothing', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save individual plots for each condition
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)
        
        for group_key in unique_groups:
            heading, outcome = group_key
            outcome_str = 'success' if outcome == 1 else 'failure'
            
            for channel in self.spike_channels:
                result_key = (heading, outcome, channel)
                if result_key in self.aligned_psths:
                    data = self.aligned_psths[result_key]
                    
                    if data['aligned_psth'] is not None:
                        # Create individual plot with behavioral subplot on top
                        fig_ind, (ax_behav, ax_neural) = plt.subplots(2, 1, figsize=(12, 10), 
                                                                     height_ratios=[0.8, 1.0])
                        
                        # Plot behavioral kinematics in top subplot
                        self._plot_single_condition_behavioral_kinematics(ax_behav, group_key)
                        
                        # Plot neural data in bottom subplot
                        time_centers = data['time_centers']
                        
                        # Plot both versions
                        ax_neural.plot(time_centers, data['unaligned_psth'], 
                                      'gray', alpha=0.7, linewidth=2, label='Unaligned')
                        ax_neural.fill_between(time_centers, 
                                              data['unaligned_psth'] - data['unaligned_sem'],
                                              data['unaligned_psth'] + data['unaligned_sem'],
                                              color='gray', alpha=0.2)
                        
                        ax_neural.plot(time_centers, data['aligned_psth'], 
                                      'red', linewidth=2, label='DTW Aligned')
                        ax_neural.fill_between(time_centers, 
                                              data['aligned_psth'] - data['aligned_sem'],
                                              data['aligned_psth'] + data['aligned_sem'],
                                              color='red', alpha=0.3)
                        
                        ax_neural.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Movement Onset')
                        
                        ax_neural.set_xlabel('Time from movement onset (s)')
                        ax_neural.set_ylabel('Firing Rate (spikes/s)')
                        ax_neural.set_title(f'Channel {channel} Neural Response')
                        ax_neural.legend()
                        ax_neural.grid(True, alpha=0.3)
                        ax_neural.set_xlim(ALIGN_WINDOW)
                        
                        # Overall figure title
                        fig_ind.suptitle(f'Channel {channel} - Heading {heading}° ({outcome_str.title()})\n'
                                        f'{data["n_trials"]} trials, DTW Aligned Analysis',
                                        fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        
                        # Save
                        filename = f'channel_{channel}_heading_{heading}_outcome_{outcome_str}.png'
                        filepath = figures_dir / filename
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close(fig_ind)
        
        # Save main figure
        main_output_path = 'aligned_psth_dtw_comparison.png'
        plt.savefig(main_output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Main plot saved as: {main_output_path}")
        print(f"   ✅ Individual plots saved in: {figures_dir}")
        
        plt.show()
    
    def export_results(self, output_path: str = 'aligned_psth_results.csv') -> None:
        """
        Export aligned PSTH results to CSV.
        
        Parameters:
        -----------
        output_path : str
            Path to save the CSV file
        """
        print(f"\n💾 Exporting aligned PSTH results to {output_path}...")
        
        if not self.aligned_psths:
            print("   ❌ No results to export.")
            return
        
        export_data = []
        
        for result_key, data in self.aligned_psths.items():
            heading, outcome, channel = result_key
            
            if data['aligned_psth'] is not None:
                # Export time series data
                time_centers = data['time_centers']
                for i, time_center in enumerate(time_centers):
                    if i < len(data['aligned_psth']):
                        export_data.append({
                            'heading': heading,
                            'outcome': 'success' if outcome == 1 else 'failure',
                            'channel': channel,
                            'time_center': time_center,
                            'aligned_firing_rate': data['aligned_psth'][i],
                            'aligned_sem': data['aligned_sem'][i],
                            'unaligned_firing_rate': data['unaligned_psth'][i],
                            'unaligned_sem': data['unaligned_sem'][i],
                            'n_trials': data['n_trials'],
                            'peak_rate_aligned': data['peak_rate_aligned'],
                            'peak_rate_unaligned': data['peak_rate_unaligned'],
                            'peak_latency_aligned': data['peak_latency_aligned'],
                            'peak_latency_unaligned': data['peak_latency_unaligned']
                        })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        print(f"   ✅ Exported {len(df)} rows to {output_path}")
        print(f"   • Headings: {sorted(df['heading'].unique())}")
        print(f"   • Outcomes: {sorted(df['outcome'].unique())}")
        print(f"   • Channels: {len(df['channel'].unique())}")
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        print(f"\n📊 Aligned PSTH Analysis Summary")
        print("=" * 50)
        
        if not self.aligned_psths:
            print("   ❌ No results available.")
            return
        
        print(f"✅ Analysis complete!")
        print(f"   • Total trials processed: {len(self.trial_data)}")
        print(f"   • Valid groups: {len(self.grouped_trials)}")
        print(f"   • Channels analyzed: {len(self.spike_channels)}")
        print(f"   • DTW alignments computed: {len(self.aligned_psths)}")
        
        # Group statistics
        print(f"\n🎯 Group Statistics:")
        for group_key, trials in self.grouped_trials.items():
            heading, outcome = group_key
            outcome_str = 'success' if outcome == 1 else 'failure'
            print(f"   • Heading {heading}° ({outcome_str}): {len(trials)} trials")
        
        # Performance improvements
        print(f"\n📈 DTW Alignment Performance:")
        improvements = []
        for result_key, data in self.aligned_psths.items():
            if data['aligned_psth'] is not None and data['peak_rate_unaligned'] > 0:
                improvement = ((data['peak_rate_aligned'] - data['peak_rate_unaligned']) / 
                              data['peak_rate_unaligned']) * 100
                improvements.append(improvement)
        
        if improvements:
            print(f"   • Mean peak rate improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
            print(f"   • Best improvement: {np.max(improvements):.1f}%")
            print(f"   • Worst improvement: {np.min(improvements):.1f}%")
            print(f"   • Improvements > 10%: {sum(1 for x in improvements if x > 10)}/{len(improvements)}")


def main():
    """Main execution function."""
    print("🧠 Aligned PSTH Analysis with Dynamic Time Warping")
    print("=" * 60)
    
    # Check for DTW availability
    if not DTW_AVAILABLE:
        print("📦 Installing fastdtw for optimal DTW performance...")
        import subprocess
        try:
            subprocess.check_call(['pip', 'install', 'fastdtw'])
            print("✅ FastDTW installed successfully")
        except:
            print("⚠️  FastDTW installation failed, using fallback DTW")
    
    # Initialize analyzer
    analyzer = AlignedPSTHAnalyzer(H5_FILE_PATH, SPIKE_CHANNELS)
    
    # Load and process data
    analyzer.load_trial_data(min_trials_per_group=3)
    
    if not analyzer.grouped_trials:
        print("❌ No valid trial groups found. Exiting.")
        return
    
    # Compute aligned PSTHs
    analyzer.compute_aligned_psths()
    
    # Generate plots
    analyzer.plot_aligned_psths()
    
    # Export results
    analyzer.export_results()
    
    # Print summary
    analyzer.print_summary()
    
    print("\n🎉 Analysis complete!")
    print("   • Check 'figures/' directory for individual plots")
    print("   • Check 'aligned_psth_results.csv' for detailed results")


if __name__ == "__main__":
    main() 