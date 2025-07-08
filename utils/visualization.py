"""
Visualization utilities for neural exploration project.
Handles behavioral task visualization and neural data plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3


class BehavioralVisualizer:
    """
    Class for visualizing behavioral data from Center Out task.
    """
    
    def __init__(self, behavioral_data=None):
        """
        Initialize the behavioral visualizer.
        
        Parameters:
        -----------
        behavioral_data : pandas.DataFrame, optional
            Behavioral data to visualize
        """
        self.behavioral_data = behavioral_data
        self.center_out_targets = self._generate_center_out_targets()
        
    def _generate_center_out_targets(self, n_targets=8, radius=1.0):
        """
        Generate standard Center Out target positions.
        
        Parameters:
        -----------
        n_targets : int
            Number of targets in the ring
        radius : float
            Radius of the target ring
            
        Returns:
        --------
        dict
            Dictionary with target positions
        """
        angles = np.linspace(0, 2*np.pi, n_targets, endpoint=False)
        targets = {}
        
        for i, angle in enumerate(angles):
            targets[f'target_{i}'] = {
                'x': radius * np.cos(angle),
                'y': radius * np.sin(angle),
                'angle': angle,
                'direction': f'{angle*180/np.pi:.0f}°'
            }
        
        return targets
    
    def plot_center_out_layout(self, figsize=(8, 8)):
        """
        Plot the Center Out task layout.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot center
        center = Circle((0, 0), 0.1, color='red', alpha=0.7, label='Center')
        ax.add_patch(center)
        
        # Plot targets
        for i, (target_name, target_info) in enumerate(self.center_out_targets.items()):
            target = Circle(
                (target_info['x'], target_info['y']), 
                0.08, 
                color='blue', 
                alpha=0.7
            )
            ax.add_patch(target)
            
            # Add target labels
            ax.text(
                target_info['x'] * 1.2, 
                target_info['y'] * 1.2, 
                f'T{i}\n{target_info["direction"]}',
                ha='center', 
                va='center',
                fontsize=10
            )
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Center Out Task Layout\n8 Targets in Ring Formation')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_cursor_trajectory(self, trial_data=None, figsize=(10, 8)):
        """
        Plot cursor trajectory for trials.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame, optional
            Trial data with cursor positions
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if trial_data is None:
            trial_data = self.behavioral_data
            
        if trial_data is None:
            raise ValueError("No trial data available for plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Center Out layout first
        self._plot_center_out_background(ax)
        
        # Plot cursor trajectories
        if 'cursor_x' in trial_data.columns and 'cursor_y' in trial_data.columns:
            # Group by trial if trial column exists
            if 'trial' in trial_data.columns:
                for trial_num in trial_data['trial'].unique():
                    trial_subset = trial_data[trial_data['trial'] == trial_num]
                    ax.plot(
                        trial_subset['cursor_x'], 
                        trial_subset['cursor_y'],
                        alpha=0.6,
                        linewidth=1
                    )
            else:
                # Plot all data as single trajectory
                ax.plot(
                    trial_data['cursor_x'], 
                    trial_data['cursor_y'],
                    alpha=0.8,
                    linewidth=1
                )
        
        ax.set_title('Cursor Trajectories - Center Out Task')
        plt.tight_layout()
        return fig
    
    def _plot_center_out_background(self, ax):
        """
        Plot the Center Out task background on given axes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        """
        # Plot center
        center = Circle((0, 0), 0.1, color='red', alpha=0.5, label='Center')
        ax.add_patch(center)
        
        # Plot targets
        for i, (target_name, target_info) in enumerate(self.center_out_targets.items()):
            target = Circle(
                (target_info['x'], target_info['y']), 
                0.08, 
                color='blue', 
                alpha=0.5
            )
            ax.add_patch(target)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    def plot_behavioral_summary(self, figsize=(15, 10)):
        """
        Plot a comprehensive summary of behavioral data.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.behavioral_data is None:
            raise ValueError("No behavioral data available for plotting")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Data overview
        axes[0].text(0.1, 0.9, f"Data Shape: {self.behavioral_data.shape}", 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].text(0.1, 0.8, f"Columns: {len(self.behavioral_data.columns)}", 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].text(0.1, 0.7, f"Time Range: {self.behavioral_data.index[0]} - {self.behavioral_data.index[-1]}", 
                    transform=axes[0].transAxes, fontsize=10)
        axes[0].set_title('Data Overview')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot 2: Center Out layout
        self._plot_center_out_background(axes[1])
        axes[1].set_title('Center Out Layout')
        
        # Plot 3: Column distribution
        if len(self.behavioral_data.columns) > 0:
            numeric_cols = self.behavioral_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sample_col = numeric_cols[0]
                axes[2].hist(self.behavioral_data[sample_col].dropna(), bins=50, alpha=0.7)
                axes[2].set_title(f'Distribution of {sample_col}')
                axes[2].set_xlabel(sample_col)
                axes[2].set_ylabel('Frequency')
        
        # Plot 4: Time series of first numeric column
        if len(self.behavioral_data.columns) > 0:
            numeric_cols = self.behavioral_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sample_col = numeric_cols[0]
                axes[3].plot(self.behavioral_data[sample_col].dropna(), alpha=0.7)
                axes[3].set_title(f'Time Series: {sample_col}')
                axes[3].set_xlabel('Time')
                axes[3].set_ylabel(sample_col)
        
        # Plot 5: Correlation matrix (if multiple numeric columns)
        numeric_cols = self.behavioral_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.behavioral_data[numeric_cols].corr()
            im = axes[4].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Add text annotations
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    text = axes[4].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
            
            # Set ticks and labels
            axes[4].set_xticks(range(len(corr_matrix.columns)))
            axes[4].set_yticks(range(len(corr_matrix)))
            axes[4].set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
            axes[4].set_yticklabels(corr_matrix.index, fontsize=8)
            axes[4].set_title('Correlation Matrix')
        
        # Plot 6: Missing data pattern
        if self.behavioral_data.isnull().sum().sum() > 0:
            missing_data = self.behavioral_data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                axes[5].bar(range(len(missing_data)), missing_data.values)
                axes[5].set_xticks(range(len(missing_data)))
                axes[5].set_xticklabels(missing_data.index, rotation=45)
                axes[5].set_title('Missing Data Count')
                axes[5].set_ylabel('Missing Values')
        
        plt.tight_layout()
        return fig
    
    def plot_trial_behavioral_data(self, trial_num=None, figsize=(15, 10)):
        """
        Plot detailed behavioral data for a specific trial.
        
        Parameters:
        -----------
        trial_num : int, optional
            Trial number to plot. If None, plots the first trial
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.behavioral_data is None:
            raise ValueError("No behavioral data available for plotting")
        
        # Get available trials
        if 'trial' in self.behavioral_data.columns:
            available_trials = sorted(self.behavioral_data['trial'].unique())
            if trial_num is None:
                trial_num = available_trials[0]
            elif trial_num not in available_trials:
                print(f"Trial {trial_num} not found. Available trials: {available_trials[:10]}...")
                trial_num = available_trials[0]
            
            trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial_num].copy()
        else:
            print("No trial column found. Using all data as single trial.")
            trial_data = self.behavioral_data.copy()
            trial_num = "All Data"
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get time axis - prioritize aligned timestamps, then regular timestamps, then sample indices
        time_relative_to_trial = False
        
        if 'timestamp_aligned' in trial_data.columns:
            # Use aligned timestamps (seconds since neural Time Origin)
            time_axis = trial_data['timestamp_aligned']
            time_label = 'Time (seconds since neural start)'
            
            # Option to show time relative to trial start
            if len(trial_data) > 0:
                trial_start_time = trial_data['timestamp_aligned'].iloc[0]
                time_axis_relative = trial_data['timestamp_aligned'] - trial_start_time
                
                # Use relative time if trial is short (< 60 seconds), absolute time otherwise
                if (trial_data['timestamp_aligned'].iloc[-1] - trial_start_time) < 60:
                    time_axis = time_axis_relative
                    time_label = 'Time (seconds from trial start)'
                    time_relative_to_trial = True
                    
        elif 'timestamp' in trial_data.columns:
            # Use regular timestamps
            time_axis = trial_data['timestamp']
            time_label = 'Time (timestamp)'
            
            # Convert to relative time if available
            if len(trial_data) > 0:
                try:
                    # Convert to seconds from trial start for better readability
                    trial_start = trial_data['timestamp'].iloc[0]
                    time_axis = [(t - trial_start).total_seconds() for t in trial_data['timestamp']]
                    time_label = 'Time (seconds from trial start)'
                    time_relative_to_trial = True
                except:
                    pass
        else:
            # Fallback to sample indices
            time_axis = np.arange(len(trial_data))
            time_label = 'Time (samples)'
        
        # Plot 1: Joystick Velocity (raw data)
        ax1 = axes[0, 0]
        if 'velocity_x' in trial_data.columns and 'velocity_y' in trial_data.columns:
            ax1.plot(time_axis, trial_data['velocity_x'], 'b-', label='X Velocity', linewidth=2)
            ax1.plot(time_axis, trial_data['velocity_y'], 'r-', label='Y Velocity', linewidth=2)
            ax1.set_ylabel('Joystick Velocity')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Joystick Velocity - Trial {trial_num}')
        ax1.set_xlabel(time_label)
        
        # Plot 2: Velocity Magnitude and Movement Detection
        ax2 = axes[0, 1]
        if 'velocity_x' in trial_data.columns and 'velocity_y' in trial_data.columns:
            # Calculate velocity magnitude
            vel_mag = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
            
            ax2.plot(time_axis, vel_mag, 'g-', label='Speed', linewidth=2)
            ax2.set_ylabel('Speed (units/s)', color='g')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Detect movement onset
            if len(vel_mag) > 1:
                # Find movement onset threshold (10% of peak velocity)
                peak_vel = np.max(vel_mag)
                movement_threshold = 0.1 * peak_vel
                
                # Find first point above threshold
                movement_onset = np.where(vel_mag > movement_threshold)[0]
                if len(movement_onset) > 0:
                    onset_idx = movement_onset[0]
                    ax2.axvline(time_axis[onset_idx] if time_relative_to_trial else time_axis[onset_idx], 
                               color='red', linestyle='--', alpha=0.7, label='Movement Onset')
                    ax2.axhline(movement_threshold, color='red', linestyle=':', alpha=0.5, label='Threshold')
                    ax2.legend()
                    
                    # Add annotation
                    ax2.annotate(f'Onset: {onset_idx/len(trial_data)*100:.1f}% into trial', 
                                xy=(time_axis[onset_idx] if time_relative_to_trial else time_axis[onset_idx], 
                                    vel_mag[onset_idx]), 
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                arrowprops=dict(arrowstyle='->', color='red'))
        else:
            ax2.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Speed Profile & Movement Onset - Trial {trial_num}')
        ax2.set_xlabel(time_label)
        
        # Plot 3: Target Layout with Trajectory Overlay
        ax3 = axes[1, 0]
        
        # Use the same style as plot_center_out_layout
        # Plot center
        center = Circle((0, 0), 0.1, color='red', alpha=0.7, label='Center')
        ax3.add_patch(center)
        
        # Plot targets using the center_out_targets structure
        current_target_idx = None
        if 'target_index' in trial_data.columns and len(trial_data) > 0:
            current_target_idx = trial_data['target_index'].iloc[0]
        
        for i, (target_name, target_info) in enumerate(self.center_out_targets.items()):
            # Highlight current target
            if current_target_idx is not None and i == current_target_idx:
                target = Circle(
                    (target_info['x'], target_info['y']), 
                    0.12,  # Slightly larger for current target
                    color='red', 
                    alpha=0.8,
                    linewidth=2,
                    edgecolor='darkred'
                )
                ax3.add_patch(target)
            else:
                target = Circle(
                    (target_info['x'], target_info['y']), 
                    0.08, 
                    color='blue', 
                    alpha=0.7
                )
                ax3.add_patch(target)
            
            # Add target labels
            ax3.text(
                target_info['x'] * 1.2, 
                target_info['y'] * 1.2, 
                f'T{i}\n{target_info["direction"]}',
                ha='center', 
                va='center',
                fontsize=9,
                fontweight='bold' if current_target_idx is not None and i == current_target_idx else 'normal'
            )
        
        # Compute and overlay trajectory if velocity data available
        if 'velocity_x' in trial_data.columns and 'velocity_y' in trial_data.columns and len(trial_data) > 1:
            # Compute trajectory from velocity with proper time steps
            if 'timestamp_aligned' in trial_data.columns:
                # Use real time differences from aligned timestamps
                time_diffs = np.diff(trial_data['timestamp_aligned'].values)
                dt_array = np.concatenate([[time_diffs[0]], time_diffs])  # Add first element
            else:
                # Use default constant time step
                dt_array = np.ones(len(trial_data)) * 1.0
            
            # Integrate velocity to get position (starting from center)
            pos_x = np.cumsum(trial_data['velocity_x'].values * dt_array)
            pos_y = np.cumsum(trial_data['velocity_y'].values * dt_array)
            
            # Center the trajectory (start at origin)
            pos_x = pos_x - pos_x[0]
            pos_y = pos_y - pos_y[0]
            
            # Plot trajectory (no scaling - let it show at natural size)
            ax3.plot(pos_x, pos_y, 'g-', linewidth=3, alpha=0.8, label='Trajectory')
            
            # Mark start and end points
            ax3.plot(pos_x[0], pos_y[0], 'go', markersize=8, 
                    label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
            ax3.plot(pos_x[-1], pos_y[-1], 'ro', markersize=8, 
                    label='End', markeredgecolor='darkred', markeredgewidth=2)
            
            # Adjust axis limits dynamically to show both targets and trajectory
            all_x = list(pos_x) + [info['x'] for info in self.center_out_targets.values()]
            all_y = list(pos_y) + [info['y'] for info in self.center_out_targets.values()]
            
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            
            # Set limits with some padding
            padding = 0.3
            x_center = (max(all_x) + min(all_x)) / 2
            y_center = (max(all_y) + min(all_y)) / 2
            
            half_range = max(x_range, y_range) / 2 + padding
            ax3.set_xlim(x_center - half_range, x_center + half_range)
            ax3.set_ylim(y_center - half_range, y_center + half_range)
        else:
            # If no trajectory data, use standard limits
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
        
        # Set equal aspect ratio and other properties
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.legend(loc='upper right', fontsize=8)
        
        # Add trial info with timing
        if current_target_idx is not None:
            outcome = trial_data['trial_outcome'].iloc[0] if 'trial_outcome' in trial_data.columns else 'Unknown'
            target_direction = self.center_out_targets[f'target_{current_target_idx}']['direction']
            
            # Add timing information
            timing_info = ""
            if 'timestamp_aligned' in trial_data.columns:
                trial_start = trial_data['timestamp_aligned'].iloc[0]
                trial_end = trial_data['timestamp_aligned'].iloc[-1]
                trial_duration = trial_end - trial_start
                timing_info = f"\nStart: {trial_start:.3f}s\nDuration: {trial_duration:.3f}s"
            elif time_relative_to_trial:
                trial_duration = time_axis[-1] if hasattr(time_axis, '__len__') else 0
                timing_info = f"\nDuration: {trial_duration:.3f}s"
            
            info_text = f"Target {current_target_idx} ({target_direction})\nOutcome: {outcome}{timing_info}"
            ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightyellow", alpha=0.8))
        
        ax3.set_title(f'Center Out Layout & Trajectory - Trial {trial_num}')
        
        # Plot 4: Velocity Vector Field and Computed Trajectory
        ax4 = axes[1, 1]
        if 'velocity_x' in trial_data.columns and 'velocity_y' in trial_data.columns:
            # Compute trajectory from velocity with proper time steps
            if 'timestamp_aligned' in trial_data.columns:
                # Use real time differences from aligned timestamps
                time_diffs = np.diff(trial_data['timestamp_aligned'].values)
                dt_array = np.concatenate([[time_diffs[0]], time_diffs])  # Add first element
                pos_x = np.cumsum(trial_data['velocity_x'].values * dt_array)
                pos_y = np.cumsum(trial_data['velocity_y'].values * dt_array)
            else:
                # Use default constant time step
                dt = 1.0
                pos_x = np.cumsum(trial_data['velocity_x'] * dt)
                pos_y = np.cumsum(trial_data['velocity_y'] * dt)
            
            # Center the trajectory (start at origin)
            pos_x = pos_x - pos_x[0]
            pos_y = pos_y - pos_y[0]
            
            # Plot computed trajectory
            ax4.plot(pos_x, pos_y, 'b-', linewidth=3, alpha=0.9, label='Computed Trajectory')
            
            # Mark start and end points
            if len(pos_x) > 0:
                ax4.plot(pos_x[0], pos_y[0], 'go', markersize=10, label='Start')
                ax4.plot(pos_x[-1], pos_y[-1], 'ro', markersize=10, label='End')
            
            # Add target position if available
            if 'target_index' in trial_data.columns and len(trial_data) > 0:
                target_idx = trial_data['target_index'].iloc[0]
                n_targets = trial_data['num_targets'].iloc[0] if 'num_targets' in trial_data.columns else 8
                
                try:
                    target_idx_int = int(target_idx)
                    if 0 <= target_idx_int < n_targets:
                        target_angle = target_idx_int * (2 * np.pi / n_targets)  # Convert to radians
                        # Estimate target distance from trajectory end point
                        trajectory_distance = np.sqrt(pos_x[-1]**2 + pos_y[-1]**2)
                        target_distance = max(1.0, trajectory_distance * 1.2)  # A bit beyond trajectory end
                        
                        target_x = target_distance * np.cos(target_angle)
                        target_y = target_distance * np.sin(target_angle)
                        ax4.plot(target_x, target_y, 'rs', markersize=12, 
                                label=f'Target {target_idx_int}')
                except (ValueError, TypeError):
                    pass
            
            # Add center point
            ax4.plot(0, 0, 'ko', markersize=10, label='Center')
            
            ax4.set_xlabel('X Position (integrated)')
            ax4.set_ylabel('Y Position (integrated)')
            ax4.set_aspect('equal')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add text annotation with trajectory stats
            stats_text = f"Trajectory Range:\nX: {pos_x.min():.3f} to {pos_x.max():.3f}\nY: {pos_y.min():.3f} to {pos_y.max():.3f}"
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightblue", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'Computed Trajectory - Trial {trial_num}')
        
        plt.tight_layout()
        return fig
    
    def plot_all_trials_summary(self, max_trials=10, figsize=(15, 8)):
        """
        Plot summary of multiple trials.
        
        Parameters:
        -----------
        max_trials : int
            Maximum number of trials to show
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.behavioral_data is None:
            raise ValueError("No behavioral data available for plotting")
        
        if 'trial' not in self.behavioral_data.columns:
            print("No trial column found. Cannot plot trial summary.")
            return None
        
        available_trials = sorted(self.behavioral_data['trial'].unique())
        trials_to_plot = available_trials[:max_trials]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Trial durations
        ax1 = axes[0, 0]
        trial_durations = []
        duration_unit = 'samples'
        
        for trial in trials_to_plot:
            trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial]
            
            # Calculate actual duration if aligned timestamps available
            if 'timestamp_aligned' in trial_data.columns and len(trial_data) > 1:
                duration = trial_data['timestamp_aligned'].iloc[-1] - trial_data['timestamp_aligned'].iloc[0]
                trial_durations.append(duration)
                duration_unit = 'seconds'
            elif 'timestamp' in trial_data.columns and len(trial_data) > 1:
                # Try to compute duration from timestamps
                try:
                    start_time = trial_data['timestamp'].iloc[0]
                    end_time = trial_data['timestamp'].iloc[-1]
                    if hasattr(start_time, 'total_seconds'):
                        duration = (end_time - start_time).total_seconds()
                    else:
                        duration = len(trial_data)
                    trial_durations.append(duration)
                    duration_unit = 'seconds' if hasattr(start_time, 'total_seconds') else 'samples'
                except:
                    trial_durations.append(len(trial_data))
                    duration_unit = 'samples'
            else:
                trial_durations.append(len(trial_data))
                duration_unit = 'samples'
        
        ax1.bar(range(len(trials_to_plot)), trial_durations, alpha=0.7)
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel(f'Trial Duration ({duration_unit})')
        ax1.set_title('Trial Durations')
        ax1.set_xticks(range(len(trials_to_plot)))
        ax1.set_xticklabels([f'T{t}' for t in trials_to_plot])
        ax1.grid(True, alpha=0.3)
        
        # Add duration statistics
        if trial_durations:
            mean_duration = np.mean(trial_durations)
            std_duration = np.std(trial_durations)
            ax1.axhline(mean_duration, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_duration:.2f}')
            ax1.legend()
        
        # Plot 2: Target distribution
        ax2 = axes[0, 1]
        if 'target_index' in self.behavioral_data.columns:
            target_counts = self.behavioral_data['target_index'].value_counts().sort_index()
            ax2.bar(target_counts.index, target_counts.values, alpha=0.7)
            ax2.set_xlabel('Target Index')
            ax2.set_ylabel('Count')
            ax2.set_title('Target Distribution (All Trials)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Target data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: All computed trajectories
        ax3 = axes[1, 0]
        if 'velocity_x' in self.behavioral_data.columns and 'velocity_y' in self.behavioral_data.columns:
            for i, trial in enumerate(trials_to_plot):
                trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial]
                
                # Compute trajectory from velocity
                if len(trial_data) > 1:
                    # Use proper time steps if aligned timestamps available
                    if 'timestamp_aligned' in trial_data.columns:
                        # Use real time differences from aligned timestamps
                        time_diffs = np.diff(trial_data['timestamp_aligned'].values)
                        dt_array = np.concatenate([[time_diffs[0]], time_diffs])  # Add first element
                        pos_x = np.cumsum(trial_data['velocity_x'].values * dt_array)
                        pos_y = np.cumsum(trial_data['velocity_y'].values * dt_array)
                    else:
                        # Use default constant time step
                        dt = 1.0
                        pos_x = np.cumsum(trial_data['velocity_x'] * dt)
                        pos_y = np.cumsum(trial_data['velocity_y'] * dt)
                    
                    # Center trajectory at origin
                    pos_x = pos_x - pos_x[0]
                    pos_y = pos_y - pos_y[0]
                    
                    ax3.plot(pos_x, pos_y, alpha=0.6, linewidth=1, label=f'Trial {trial}')
            
            # Add center and target positions
            ax3.plot(0, 0, 'ko', markersize=8, label='Center')
            
            # Add target ring (get number of targets from data if available)
            n_targets = 8  # Default
            if 'num_targets' in self.behavioral_data.columns:
                n_targets = self.behavioral_data['num_targets'].iloc[0]
            
            angles = np.linspace(0, 2*np.pi, n_targets, endpoint=False)
            for i, angle in enumerate(angles):
                target_x = 2.0 * np.cos(angle)  # Arbitrary distance
                target_y = 2.0 * np.sin(angle)
                ax3.plot(target_x, target_y, 'rs', markersize=6, alpha=0.7)
            
            ax3.set_xlabel('X Position (integrated)')
            ax3.set_ylabel('Y Position (integrated)')
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            ax3.set_title(f'All Computed Trajectories ({len(trials_to_plot)} trials)')
        else:
            ax3.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Velocity statistics
        ax4 = axes[1, 1]
        if 'velocity_x' in self.behavioral_data.columns and 'velocity_y' in self.behavioral_data.columns:
            max_velocities = []
            for trial in trials_to_plot:
                trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial]
                if len(trial_data) > 0:
                    # Calculate velocity magnitude from velocity components
                    vel_magnitude = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
                    max_vel = np.max(vel_magnitude)
                    max_velocities.append(max_vel)
                else:
                    max_velocities.append(0)
            
            ax4.bar(range(len(trials_to_plot)), max_velocities, alpha=0.7)
            ax4.set_xlabel('Trial Index')
            ax4.set_ylabel('Max Speed (units/s)')
            ax4.set_title('Peak Speeds by Trial')
            ax4.set_xticks(range(len(trials_to_plot)))
            ax4.set_xticklabels([f'T{t}' for t in trials_to_plot])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Velocity data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig


class NeuralVisualizer:
    """
    Class for visualizing neural data.
    """
    
    def __init__(self, neural_data=None, metadata=None):
        """
        Initialize the neural visualizer.
        
        Parameters:
        -----------
        neural_data : dict, optional
            Neural data dictionary
        metadata : dict, optional
            Metadata dictionary
        """
        self.neural_data = neural_data
        self.metadata = metadata or {}
        
    def plot_channel_overview(self, n_channels=16, duration=1.0, figsize=(15, 10)):
        """
        Plot overview of multiple neural channels.
        
        Parameters:
        -----------
        n_channels : int
            Number of channels to plot
        duration : float
            Duration in seconds to plot
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.neural_data is None:
            raise ValueError("No neural data available for plotting")
        
        raw_data = self.neural_data['raw_data']
        times = self.neural_data['times']
        
        # Calculate samples for the desired duration
        sampling_rate = self.metadata.get('sampling_rate', 30000)
        n_samples = int(duration * sampling_rate)
        
        # Select subset of data
        data_subset = raw_data[:n_samples, :n_channels]
        time_subset = times[:n_samples]
        
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
        
        if n_channels == 1:
            axes = [axes]
        
        for i in range(n_channels):
            axes[i].plot(time_subset, data_subset[:, i], 'b-', linewidth=0.5)
            axes[i].set_ylabel(f'Ch {i+1}')
            axes[i].grid(True, alpha=0.3)
            
            # Add some basic stats
            std_val = np.std(data_subset[:, i])
            axes[i].set_ylim([-4*std_val, 4*std_val])
        
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title(f'Neural Data Overview - First {n_channels} Channels ({duration}s)')
        
        plt.tight_layout()
        return fig
    
    def plot_data_structure_summary(self, figsize=(15, 10)):
        """
        Plot a summary of the neural data structure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.neural_data is None:
            raise ValueError("No neural data available for plotting")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        raw_data = self.neural_data['raw_data']
        
        # Plot 1: Data info
        info_text = f"""
        Data Shape: {raw_data.shape}
        Sampling Rate: {self.metadata.get('sampling_rate', 'Unknown')} Hz
        Duration: {self.metadata.get('duration', 'Unknown')} s
        Channels: {self.metadata.get('n_channels', 'Unknown')}
        """
        axes[0].text(0.1, 0.5, info_text, transform=axes[0].transAxes, 
                    fontsize=12, verticalalignment='center')
        axes[0].set_title('Data Structure Info')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot 2: RMS by channel
        rms_values = np.sqrt(np.mean(raw_data**2, axis=0))
        axes[1].plot(rms_values, 'o-')
        axes[1].set_title('RMS by Channel')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('RMS Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Sample of first channel
        sample_duration = min(1.0, self.metadata.get('duration', 1.0))
        sample_size = int(sample_duration * self.metadata.get('sampling_rate', 30000))
        sample_times = self.neural_data['times'][:sample_size]
        sample_data = raw_data[:sample_size, 0]
        
        axes[2].plot(sample_times, sample_data, 'b-', linewidth=0.5)
        axes[2].set_title('Sample Data - Channel 1')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Amplitude histogram
        axes[3].hist(raw_data[:, 0].flatten(), bins=100, alpha=0.7)
        axes[3].set_title('Amplitude Distribution - Channel 1')
        axes[3].set_xlabel('Amplitude')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Power spectrum
        if raw_data.shape[0] > 1000:
            freqs, psd = self._compute_power_spectrum(raw_data[:, 0])
            axes[4].loglog(freqs, psd)
            axes[4].set_title('Power Spectrum - Channel 1')
            axes[4].set_xlabel('Frequency (Hz)')
            axes[4].set_ylabel('Power')
            axes[4].grid(True, alpha=0.3)
        
        # Plot 6: Channel correlation sample
        if raw_data.shape[1] > 1:
            sample_channels = min(10, raw_data.shape[1])
            sample_data = raw_data[:min(10000, raw_data.shape[0]), :sample_channels]
            corr_matrix = np.corrcoef(sample_data.T)
            
            im = axes[5].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[5].set_title(f'Channel Correlation (first {sample_channels})')
            axes[5].set_xlabel('Channel')
            axes[5].set_ylabel('Channel')
            plt.colorbar(im, ax=axes[5])
        
        plt.tight_layout()
        return fig
    
    def _compute_power_spectrum(self, signal):
        """
        Compute power spectrum of a signal.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            Input signal
            
        Returns:
        --------
        tuple
            (frequencies, power spectral density)
        """
        from scipy import signal as scipy_signal
        
        sampling_rate = self.metadata.get('sampling_rate', 30000)
        freqs, psd = scipy_signal.welch(signal, sampling_rate, nperseg=2048)
        
        return freqs, psd


def create_utah_array_layout(figsize=(8, 6)):
    """
    Create a visualization of the standard 96-channel Utah array layout.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Standard 96-channel Utah array is 10x10 with 4 corners missing
    grid_size = 10
    
    # Create grid positions
    channel_positions = {}
    channel_num = 1
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Skip corners
            if (row == 0 and col == 0) or (row == 0 and col == grid_size-1) or \
               (row == grid_size-1 and col == 0) or (row == grid_size-1 and col == grid_size-1):
                continue
            
            channel_positions[channel_num] = (col, grid_size - 1 - row)
            channel_num += 1
    
    # Plot electrode positions
    for channel, (x, y) in channel_positions.items():
        circle = Circle((x, y), 0.35, color='gold', alpha=0.7, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, str(channel), ha='center', va='center', fontsize=6)
    
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('96-Channel Utah Array Layout\n(Standard Configuration)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    return fig 

# =============================================================================
# NEURAL FEATURE VISUALIZATION FUNCTIONS
# =============================================================================

def plot_neural_behavioral_sync(trial_data: dict, features: dict = None, 
                               spike_channels: list = None, trial_number: int = None,
                               figsize: tuple = (15, 10)) -> None:
    """
    Create synchronized visualization of neural and behavioral data.
    
    Args:
        trial_data: Dictionary containing trial data
        features: Dictionary containing extracted features (optional)
        spike_channels: List of spike channel indices
        trial_number: Trial number for title
        figsize: Figure size tuple
    """
    if trial_number is None:
        trial_number = trial_data.get('trial_number', 'Unknown')
    
    print(f"🎨 Creating synchronized visualization for trial {trial_number}...")
    
    # Get data
    neural_data = trial_data['neural_data']
    velocity_x = trial_data.get('velocity_x', None)
    velocity_y = trial_data.get('velocity_y', None)
    behavioral_timestamps = trial_data.get('behavioral_timestamps', None)
    duration = trial_data.get('duration', neural_data.shape[1] / 30000)
    
    # Check behavioral data availability
    has_behavioral = (velocity_x is not None and velocity_y is not None)
    
    if not has_behavioral:
        print("⚠️  No behavioral data found - showing neural data only")
        velocity_x = np.zeros(100)
        velocity_y = np.zeros(100)
        behavioral_timestamps = None
    
    # Create time axes
    neural_time = np.linspace(0, duration, neural_data.shape[1])
    
    if behavioral_timestamps is not None and len(behavioral_timestamps) > 0:
        behavioral_time = behavioral_timestamps - behavioral_timestamps[0]
    else:
        behavioral_time = np.linspace(0, duration, len(velocity_x))
    
    # Select channels for visualization
    if spike_channels is None:
        spike_channels = list(range(min(8, neural_data.shape[0])))
    
    np.random.seed(42)
    n_channels_to_plot = min(6, len(spike_channels))
    if len(spike_channels) > n_channels_to_plot:
        selected_channel_indices = np.random.choice(len(spike_channels), n_channels_to_plot, replace=False)
        selected_channels = [spike_channels[i] for i in selected_channel_indices]
    else:
        selected_channels = spike_channels[:n_channels_to_plot]
    
    # Create figure
    fig, axes = plt.subplots(n_channels_to_plot + 1, 1, figsize=figsize)
    fig.suptitle(f'Trial {trial_number} - Neural & Behavioral Data\n' +
                f'Outcome: {trial_data.get("outcome", "Unknown")} | ' +
                f'Duration: {duration:.2f}s', fontsize=14, fontweight='bold')
    
    # Plot behavioral data (top subplot)
    ax_behavior = axes[0]
    
    if has_behavioral:
        ax_behavior.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_behavior.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        
        # Plot velocity magnitude
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        ax_behavior.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, 
                        label='Magnitude', alpha=0.6)
        
        ax_behavior.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_behavior.legend(loc='upper right')
        
        # Add statistics
        vel_stats = f'Peak: X={np.max(np.abs(velocity_x)):.2f}, Y={np.max(np.abs(velocity_y)):.2f}'
        ax_behavior.text(0.02, 0.95, vel_stats, transform=ax_behavior.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=9, verticalalignment='top')
    else:
        ax_behavior.text(0.5, 0.5, 'No Behavioral Data Available', 
                        transform=ax_behavior.transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.7)
        ax_behavior.set_title('⚠️ No Behavioral Data', fontweight='bold')
    
    ax_behavior.set_ylabel('Velocity')
    ax_behavior.grid(True, alpha=0.3)
    ax_behavior.set_xlim(0, duration)
    
    # Plot neural data (remaining subplots)
    for i, ch_num in enumerate(selected_channels):
        ax = axes[i + 1]
        
        # Get neural signal
        if ch_num < neural_data.shape[0]:
            neural_signal = neural_data[ch_num, :]
        else:
            neural_signal = np.zeros(neural_data.shape[1])
            print(f"⚠️ Channel {ch_num} exceeds available channels")
        
        # Plot neural signal
        ax.plot(neural_time, neural_signal, 'black', linewidth=0.8, alpha=1)
        
        # Add statistics
        signal_std = np.std(neural_signal)
        signal_range = np.ptp(neural_signal)
        
        ax.set_ylabel(f'Ch {ch_num}\n(μV)', fontweight='bold', fontsize=10)
        ax.set_title(f'🧠 Neural Channel {ch_num} | σ={signal_std:.1f}, range={signal_range:.1f}', 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        
        # Only show x-axis label on bottom subplot
        if i == len(selected_channels) - 1:
            ax.set_xlabel('Time (seconds)', fontweight='bold')
        else:
            ax.set_xticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 Trial {trial_number} Summary:")
    print(f"   Neural data: {neural_data.shape[0]} channels, {neural_data.shape[1]} samples")
    
    if has_behavioral:
        print(f"   Behavioral data: {len(velocity_x)} samples")
        print(f"   Velocity peaks: X={np.max(np.abs(velocity_x)):.3f}, Y={np.max(np.abs(velocity_y)):.3f}")
    else:
        print(f"   Behavioral data: Not available")


def plot_feature_overview(features: dict, spike_channels: list, trial_data: dict = None,
                         trial_number: int = None, n_channels: int = 8, figsize: tuple = (15, 12)) -> None:
    """
    Create a quick visualization of the main neural features with behavioral data.
    
    Args:
        features: Dictionary containing extracted features
        spike_channels: List of spike channel indices
        trial_data: Dictionary containing trial data (optional, for behavioral plots)
        trial_number: Trial number for title
        n_channels: Number of channels to display
        figsize: Figure size tuple
    """
    if features is None:
        print("❌ No features to plot")
        return
    
    if trial_number is None:
        trial_number = "Unknown"
    
    # Get neural feature data
    time_axis = features['spike_band']['time_axis']
    spike_rms = features['spike_band']['rms_power']
    lfp_power = features['lfp']['lfp_power']
    gamma_power = features['lfp']['gamma_power']
    crossings = features['threshold']['crossing_counts']
    
    # Get behavioral data if available
    has_behavioral = False
    if trial_data is not None:
        velocity_x = trial_data.get('velocity_x', None)
        velocity_y = trial_data.get('velocity_y', None)
        behavioral_timestamps = trial_data.get('behavioral_timestamps', None)
        duration = trial_data.get('duration', time_axis[-1])
        
        has_behavioral = (velocity_x is not None and velocity_y is not None)
        
        if has_behavioral:
            if behavioral_timestamps is not None and len(behavioral_timestamps) > 0:
                behavioral_time = behavioral_timestamps - behavioral_timestamps[0]
            else:
                behavioral_time = np.linspace(0, duration, len(velocity_x))
    
    # Create figure with behavioral row on top
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Neural Features - Trial {trial_number}', fontsize=16, fontweight='bold')
    
    # Plot behavioral data on top row
    if has_behavioral:
        # Left column behavioral
        ax_beh_left = axes[0, 0]
        ax_beh_left.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_left.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        ax_beh_left.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_left.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_left.set_ylabel('Velocity')
        ax_beh_left.legend(fontsize=8)
        ax_beh_left.grid(True, alpha=0.3)
        ax_beh_left.set_xlim(0, duration)
        
        # Right column behavioral (duplicate for consistency)
        ax_beh_right = axes[0, 1]
        ax_beh_right.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_right.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_right.set_ylabel('Velocity')
        ax_beh_right.legend(fontsize=8)
        ax_beh_right.grid(True, alpha=0.3)
        ax_beh_right.set_xlim(0, duration)
    else:
        # No behavioral data
        for ax_beh in [axes[0, 0], axes[0, 1]]:
            ax_beh.text(0.5, 0.5, 'No Behavioral Data', transform=ax_beh.transAxes, 
                       ha='center', va='center', fontsize=12, alpha=0.7)
            ax_beh.set_title('⚠️ No Behavioral Data', fontweight='bold')
    
    # Plot 1: Spike Band Power
    ax1 = axes[1, 0]
    for i in range(min(n_channels, len(spike_channels))):
        ax1.plot(time_axis, spike_rms[i], label=f'Ch {spike_channels[i]}', alpha=0.7)
    ax1.set_title('Spike Band Power (400-6000 Hz)')
    ax1.set_ylabel('RMS Power')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LFP Power
    ax2 = axes[1, 1]
    for i in range(min(n_channels, len(spike_channels))):
        ax2.plot(time_axis, lfp_power[i], label=f'Ch {spike_channels[i]}', alpha=0.7)
    ax2.set_title('LFP Power (<250 Hz)')
    ax2.set_ylabel('Power')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gamma Power
    ax3 = axes[2, 0]
    for i in range(min(n_channels, len(spike_channels))):
        ax3.plot(time_axis, gamma_power[i], label=f'Ch {spike_channels[i]}', alpha=0.7)
    ax3.set_title('Gamma Power (30-100 Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Power')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Threshold Crossings
    ax4 = axes[2, 1]
    for i in range(min(n_channels, len(spike_channels))):
        ax4.plot(time_axis, crossings[i], label=f'Ch {spike_channels[i]}', alpha=0.7)
    ax4.set_title('Threshold Crossings')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Count per bin')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_channel_comparison(features: dict, spike_channels: list, 
                          channel_list: list, trial_data: dict = None, figsize: tuple = (12, 10)) -> None:
    """
    Compare features across multiple specific channels with behavioral data.
    
    Args:
        features: Dictionary containing extracted features
        spike_channels: List of all spike channel indices
        channel_list: List of specific channels to compare
        trial_data: Dictionary containing trial data (optional, for behavioral plots)
        figsize: Figure size tuple
    """
    if features is None:
        print("❌ No features to compare")
        return
    
    # Find indices of requested channels
    channel_indices = []
    valid_channels = []
    for ch_num in channel_list:
        try:
            idx = spike_channels.index(ch_num)
            channel_indices.append(idx)
            valid_channels.append(ch_num)
        except ValueError:
            print(f"⚠️ Channel {ch_num} not in spike channels list")
    
    if not channel_indices:
        print("❌ No valid channels to compare")
        return
    
    # Get neural feature data
    time_axis = features['spike_band']['time_axis']
    spike_rms = features['spike_band']['rms_power']
    lfp_power = features['lfp']['lfp_power']
    gamma_power = features['lfp']['gamma_power']
    crossings = features['threshold']['crossing_counts']
    
    # Get behavioral data if available
    has_behavioral = False
    if trial_data is not None:
        velocity_x = trial_data.get('velocity_x', None)
        velocity_y = trial_data.get('velocity_y', None)
        behavioral_timestamps = trial_data.get('behavioral_timestamps', None)
        duration = trial_data.get('duration', time_axis[-1])
        
        has_behavioral = (velocity_x is not None and velocity_y is not None)
        
        if has_behavioral:
            if behavioral_timestamps is not None and len(behavioral_timestamps) > 0:
                behavioral_time = behavioral_timestamps - behavioral_timestamps[0]
            else:
                behavioral_time = np.linspace(0, duration, len(velocity_x))
    
    # Create figure with behavioral row on top
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Channel Comparison: {valid_channels}', fontsize=16, fontweight='bold')
    
    # Plot behavioral data on top row
    if has_behavioral:
        # Left column behavioral
        ax_beh_left = axes[0, 0]
        ax_beh_left.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_left.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        ax_beh_left.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_left.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_left.set_ylabel('Velocity')
        ax_beh_left.legend(fontsize=8)
        ax_beh_left.grid(True, alpha=0.3)
        ax_beh_left.set_xlim(0, duration)
        
        # Right column behavioral (duplicate for consistency)
        ax_beh_right = axes[0, 1]
        ax_beh_right.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_right.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_right.set_ylabel('Velocity')
        ax_beh_right.legend(fontsize=8)
        ax_beh_right.grid(True, alpha=0.3)
        ax_beh_right.set_xlim(0, duration)
    else:
        # No behavioral data
        for ax_beh in [axes[0, 0], axes[0, 1]]:
            ax_beh.text(0.5, 0.5, 'No Behavioral Data', transform=ax_beh.transAxes, 
                       ha='center', va='center', fontsize=12, alpha=0.7)
            ax_beh.set_title('⚠️ No Behavioral Data', fontweight='bold')
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_channels)))
    
    # Plot comparisons
    for i, (ch_idx, ch_num) in enumerate(zip(channel_indices, valid_channels)):
        color = colors[i]
        
        # Spike band power
        axes[1, 0].plot(time_axis, spike_rms[ch_idx], 
                       color=color, label=f'Ch {ch_num}', linewidth=2)
        
        # LFP power
        axes[1, 1].plot(time_axis, lfp_power[ch_idx], 
                       color=color, label=f'Ch {ch_num}', linewidth=2)
        
        # Gamma power
        axes[2, 0].plot(time_axis, gamma_power[ch_idx], 
                       color=color, label=f'Ch {ch_num}', linewidth=2)
        
        # Threshold crossings
        axes[2, 1].plot(time_axis, crossings[ch_idx], 
                       color=color, label=f'Ch {ch_num}', linewidth=2)
    
    # Set titles and labels
    axes[1, 0].set_title('Spike Band Power')
    axes[1, 0].set_ylabel('RMS Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('LFP Power')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_title('Gamma Power')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Power')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_title('Threshold Crossings')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Count per bin')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Channel comparison complete!")
    print(f"   • Channels compared: {len(channels_to_compare)}")
    print(f"   • Quality metrics: {len(quality_metrics)} channels")


def plot_channel_detail(features: dict, spike_channels: list, 
                       channel_number: int, trial_data: dict = None, figsize: tuple = (12, 10)) -> None:
    """
    Explore all features for a specific channel in detail with behavioral data.
    
    Args:
        features: Dictionary containing extracted features
        spike_channels: List of spike channel indices
        channel_number: Specific channel number to analyze
        trial_data: Dictionary containing trial data (optional, for behavioral plots)
        figsize: Figure size tuple
    """
    if features is None:
        print("❌ No features to explore")
        return
    
    # Find channel index
    try:
        ch_idx = spike_channels.index(channel_number)
    except ValueError:
        print(f"❌ Channel {channel_number} not in spike channels list")
        return
    
    # Get neural feature data
    time_axis = features['spike_band']['time_axis']
    spike_rms = features['spike_band']['rms_power'][ch_idx]
    lfp_power = features['lfp']['lfp_power'][ch_idx]
    gamma_power = features['lfp']['gamma_power'][ch_idx]
    gamma_amp = features['lfp']['gamma_amplitude'][ch_idx]
    crossings = features['threshold']['crossing_counts'][ch_idx]
    mov_avg = features['voltage']['moving_average'][ch_idx]
    mov_var = features['voltage']['moving_variance'][ch_idx]
    
    # Get behavioral data if available
    has_behavioral = False
    if trial_data is not None:
        velocity_x = trial_data.get('velocity_x', None)
        velocity_y = trial_data.get('velocity_y', None)
        behavioral_timestamps = trial_data.get('behavioral_timestamps', None)
        duration = trial_data.get('duration', time_axis[-1])
        
        has_behavioral = (velocity_x is not None and velocity_y is not None)
        
        if has_behavioral:
            if behavioral_timestamps is not None and len(behavioral_timestamps) > 0:
                behavioral_time = behavioral_timestamps - behavioral_timestamps[0]
            else:
                behavioral_time = np.linspace(0, duration, len(velocity_x))
    
    # Create figure with behavioral row on top
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(f'Channel {channel_number} - All Features', fontsize=16, fontweight='bold')
    
    # Plot behavioral data on top row
    if has_behavioral:
        # Left column behavioral
        ax_beh_left = axes[0, 0]
        ax_beh_left.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_left.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        ax_beh_left.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_left.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_left.set_ylabel('Velocity')
        ax_beh_left.legend(fontsize=8)
        ax_beh_left.grid(True, alpha=0.3)
        ax_beh_left.set_xlim(0, duration)
        
        # Right column behavioral (duplicate for consistency)
        ax_beh_right = axes[0, 1]
        ax_beh_right.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        ax_beh_right.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, label='Magnitude', alpha=0.6)
        ax_beh_right.set_title('🕹️ Behavioral Velocity', fontweight='bold')
        ax_beh_right.set_ylabel('Velocity')
        ax_beh_right.legend(fontsize=8)
        ax_beh_right.grid(True, alpha=0.3)
        ax_beh_right.set_xlim(0, duration)
    else:
        # No behavioral data
        for ax_beh in [axes[0, 0], axes[0, 1]]:
            ax_beh.text(0.5, 0.5, 'No Behavioral Data', transform=ax_beh.transAxes, 
                       ha='center', va='center', fontsize=12, alpha=0.7)
            ax_beh.set_title('⚠️ No Behavioral Data', fontweight='bold')
    
    # Spike band power
    axes[1, 0].plot(time_axis, spike_rms, 'b-', linewidth=2)
    axes[1, 0].set_title('Spike Band Power (400-6000 Hz)')
    axes[1, 0].set_ylabel('RMS Power')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LFP power
    axes[1, 1].plot(time_axis, lfp_power, 'r-', linewidth=2)
    axes[1, 1].set_title('LFP Power (<250 Hz)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Gamma power and amplitude
    ax1 = axes[2, 0]
    ax1.plot(time_axis, gamma_power, 'g-', linewidth=2, label='Gamma Power')
    ax2 = ax1.twinx()
    ax2.plot(time_axis, gamma_amp, 'orange', linewidth=2, label='Gamma Amplitude')
    ax1.set_title('Gamma Features (30-100 Hz)')
    ax1.set_ylabel('Power', color='g')
    ax2.set_ylabel('Amplitude', color='orange')
    ax1.grid(True, alpha=0.3)
    
    # Threshold crossings
    axes[2, 1].bar(time_axis, crossings, width=0.8*(time_axis[1]-time_axis[0]), alpha=0.7, color='purple')
    axes[2, 1].set_title('Threshold Crossings')
    axes[2, 1].set_ylabel('Count per bin')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Moving average
    axes[3, 0].plot(time_axis, mov_avg, 'brown', linewidth=2)
    axes[3, 0].set_title('Moving Average')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Voltage')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Moving variance
    axes[3, 1].plot(time_axis, mov_var, 'pink', linewidth=2)
    axes[3, 1].set_title('Moving Variance')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Variance')
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print channel statistics
    print(f"\n📊 Channel {channel_number} Statistics:")
    print(f"  • Spike RMS Power: {np.mean(spike_rms):.3f} ± {np.std(spike_rms):.3f}")
    print(f"  • LFP Power: {np.mean(lfp_power):.3f} ± {np.std(lfp_power):.3f}")
    print(f"  • Gamma Power: {np.mean(gamma_power):.3f} ± {np.std(gamma_power):.3f}")
    print(f"  • Total Crossings: {np.sum(crossings):.0f}")
    print(f"  • Moving Average: {np.mean(mov_avg):.3f} ± {np.std(mov_avg):.3f}")
    print(f"  • Moving Variance: {np.mean(mov_var):.3f} ± {np.std(mov_var):.3f}")


def plot_behavior_raster_psth(trial_data: dict, spike_channels: list, trial_number: int = None,
                             threshold_multiplier: float = -4.0, psth_bin_size: float = 0.01,
                             psth_sigma: float = 0.02, sampling_rate: int = 30000,
                             raster_plot_type: str = 'vertline', figsize: tuple = (15, 12)) -> None:
    """
    Create a 3x1 plot showing behavioral velocity, spike raster, and PSTH.
    Enhanced with improved raster plot implementation.
    
    Args:
        trial_data: Dictionary containing trial data
        spike_channels: List of spike channel indices
        trial_number: Trial number for title
        threshold_multiplier: Spike detection threshold (-4.0 = -4x RMS)
        psth_bin_size: PSTH bin size in seconds
        psth_sigma: Gaussian smoothing sigma for PSTH in seconds
        sampling_rate: Neural data sampling rate in Hz
        raster_plot_type: Type of raster plot ('horzline', 'vertline', 'scatter')
        figsize: Figure size tuple
    """
    if trial_number is None:
        trial_number = trial_data.get('trial_number', 'Unknown')
    
    print(f"🎨 Creating enhanced behavior-raster-PSTH plot for trial {trial_number}...")
    
    # Get data
    neural_data = trial_data['neural_data']
    velocity_x = trial_data.get('velocity_x', None)
    velocity_y = trial_data.get('velocity_y', None)
    behavioral_timestamps = trial_data.get('behavioral_timestamps', None)
    duration = trial_data.get('duration', neural_data.shape[1] / sampling_rate)
    
    # Check behavioral data availability
    has_behavioral = (velocity_x is not None and velocity_y is not None)
    
    if not has_behavioral:
        print("⚠️  No behavioral data found - showing neural data only")
        velocity_x = np.zeros(100)
        velocity_y = np.zeros(100)
        behavioral_timestamps = None
    
    # Create time axes
    neural_time = np.linspace(0, duration, neural_data.shape[1])
    
    if behavioral_timestamps is not None and len(behavioral_timestamps) > 0:
        behavioral_time = behavioral_timestamps - behavioral_timestamps[0]
    else:
        behavioral_time = np.linspace(0, duration, len(velocity_x))
    
    # Detect spikes using improved method
    print("🔍 Detecting spikes for enhanced raster plot...")
    spike_times_by_channel = []
    all_spike_times = []
    
    for i, ch_num in enumerate(spike_channels):
        if ch_num < neural_data.shape[0]:
            signal = neural_data[ch_num, :]
            
            # Calculate threshold
            rms = np.sqrt(np.mean(signal**2))
            threshold = threshold_multiplier * rms
            
            # Find threshold crossings (negative-going spikes)
            if threshold_multiplier < 0:
                spike_indices = np.where((signal[:-1] > threshold) & (signal[1:] <= threshold))[0]
            else:
                spike_indices = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0]
            
            # Convert to time
            spike_times = spike_indices / sampling_rate
            spike_times_by_channel.append(spike_times)
            all_spike_times.extend(spike_times)
        else:
            spike_times_by_channel.append(np.array([]))
    
    # Create PSTH using advanced kernel smoothing
    print("📊 Computing PSTH with advanced kernel smoothing...")
    psth_bin_edges = np.arange(0, duration + psth_bin_size, psth_bin_size)
    psth_centers = (psth_bin_edges[:-1] + psth_bin_edges[1:]) / 2
    
    # Create spike matrix for advanced PSTH (single trial, multi-channel)
    spike_matrix = np.zeros((1, len(psth_centers)))
    psth_counts, _ = np.histogram(all_spike_times, bins=psth_bin_edges)
    spike_matrix[0, :] = psth_counts
    
    # Use advanced PSTH plotting function
    psth_rate, _, _, _ = plot_spike_psth(
        psth_centers, spike_matrix, psth_bin_size, psth_sigma,
        kernel_type='gauss', plot_error=False, ax=None
    )
    
    # Normalize by number of channels
    psth_rate = psth_rate / len(spike_channels)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2, 1]})
    fig.suptitle(f'Trial {trial_number} - Enhanced Behavior, Raster & PSTH\n' +
                f'Outcome: {trial_data.get("outcome", "Unknown")} | ' +
                f'Duration: {duration:.2f}s | {len(spike_channels)} channels', 
                fontsize=14, fontweight='bold')
    
    # Plot 1: Behavioral Velocity (Top)
    ax_behavior = axes[0]
    
    if has_behavioral:
        ax_behavior.plot(behavioral_time, velocity_x, 'b-', linewidth=2, label='Velocity X', alpha=0.8)
        ax_behavior.plot(behavioral_time, velocity_y, 'r-', linewidth=2, label='Velocity Y', alpha=0.8)
        
        # Plot velocity magnitude
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        ax_behavior.plot(behavioral_time, velocity_magnitude, 'k--', linewidth=2, 
                        label='Magnitude', alpha=0.6)
        
        ax_behavior.set_title('🕹️ Behavioral Velocity', fontweight='bold', fontsize=12)
        ax_behavior.legend(loc='upper right', fontsize=9)
        
        # Add statistics
        vel_stats = f'Peak: X={np.max(np.abs(velocity_x)):.2f}, Y={np.max(np.abs(velocity_y)):.2f}'
        ax_behavior.text(0.02, 0.95, vel_stats, transform=ax_behavior.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=8, verticalalignment='top')
    else:
        ax_behavior.text(0.5, 0.5, 'No Behavioral Data Available', 
                        transform=ax_behavior.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.7)
        ax_behavior.set_title('⚠️ No Behavioral Data', fontweight='bold', fontsize=12)
    
    ax_behavior.set_ylabel('Velocity', fontweight='bold')
    ax_behavior.grid(True, alpha=0.3)
    ax_behavior.set_xlim(0, duration)
    ax_behavior.set_xticks([])  # Remove x-axis labels
    
    # Plot 2: Enhanced Spike Raster (Middle)
    ax_raster = axes[1]
    
    print(f"🎯 Creating enhanced {raster_plot_type} raster plot...")
    
    # Use the enhanced raster plot function
    x_points, y_points = plot_spike_raster(
        spike_times_by_channel,
        trial_numbers=spike_channels,
        plot_type=raster_plot_type,
        spike_duration=0.001,  # 1ms spike duration
        xlim_for_spikes=[0, duration],
        auto_label=False,
        ax=ax_raster
    )
    
    ax_raster.set_title(f'🎯 Enhanced Spike Raster Plot ({raster_plot_type}) - {len(spike_channels)} channels', 
                       fontweight='bold', fontsize=12)
    ax_raster.set_ylabel('Channel Index', fontweight='bold')
    ax_raster.set_xlim(0, duration)
    ax_raster.set_xticks([])  # Remove x-axis labels
    
    # Add channel labels on y-axis
    if len(spike_channels) <= 20:  # Only show labels if not too many channels
        ax_raster.set_yticks(range(1, len(spike_channels) + 1))
        ax_raster.set_yticklabels([f'Ch{ch}' for ch in spike_channels], fontsize=8)
    else:
        # Show fewer ticks for many channels
        tick_indices = np.linspace(1, len(spike_channels), 10, dtype=int)
        ax_raster.set_yticks(tick_indices)
        ax_raster.set_yticklabels([f'Ch{spike_channels[i-1]}' for i in tick_indices], fontsize=8)
    
    # Add spike count statistics
    total_spikes = len(all_spike_times)
    avg_rate = total_spikes / (duration * len(spike_channels))
    spike_stats = f'Total spikes: {total_spikes}, Avg rate: {avg_rate:.1f} Hz/ch'
    ax_raster.text(0.02, 0.95, spike_stats, transform=ax_raster.transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                  fontsize=8, verticalalignment='top')
    
    # Plot 3: PSTH (Bottom)
    ax_psth = axes[2]
    
    ax_psth.plot(psth_centers, psth_rate, 'purple', linewidth=2, alpha=0.8)
    ax_psth.fill_between(psth_centers, psth_rate, alpha=0.3, color='purple')
    
    ax_psth.set_title(f'📊 Population PSTH (smoothed σ={psth_sigma*1000:.0f}ms)', 
                     fontweight='bold', fontsize=12)
    ax_psth.set_xlabel('Time (seconds)', fontweight='bold')
    ax_psth.set_ylabel('Firing Rate\n(spikes/s/ch)', fontweight='bold')
    ax_psth.grid(True, alpha=0.3)
    ax_psth.set_xlim(0, duration)
    
    # Add PSTH statistics
    max_rate = np.max(psth_rate)
    mean_rate = np.mean(psth_rate)
    psth_stats = f'Peak: {max_rate:.1f} Hz/ch, Mean: {mean_rate:.1f} Hz/ch'
    ax_psth.text(0.02, 0.95, psth_stats, transform=ax_psth.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=8, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 Enhanced Trial {trial_number} Summary:")
    print(f"   Neural data: {neural_data.shape[0]} channels, {neural_data.shape[1]} samples")
    print(f"   Spike detection: {total_spikes} spikes detected")
    print(f"   Average firing rate: {avg_rate:.2f} Hz per channel")
    print(f"   Raster plot type: {raster_plot_type}")
    print(f"   PSTH resolution: {psth_bin_size*1000:.1f}ms bins, smoothed with σ={psth_sigma*1000:.0f}ms")
    
    if has_behavioral:
        print(f"   Behavioral data: {len(velocity_x)} samples")
        print(f"   Velocity peaks: X={np.max(np.abs(velocity_x)):.3f}, Y={np.max(np.abs(velocity_y)):.3f}")
    else:
        print(f"   Behavioral data: Not available")
    
    # Find most active channels
    spike_counts_per_channel = [len(times) for times in spike_times_by_channel]
    most_active_indices = np.argsort(spike_counts_per_channel)[-5:][::-1]
    most_active = [(spike_channels[i], spike_counts_per_channel[i]) for i in most_active_indices]
    print(f"   Most active channels: {[f'Ch{ch}({count})' for ch, count in most_active]}")


def plot_multi_trial_raster_comparison(trial_data_list: list, spike_channels: list, 
                                     trial_numbers: list = None, threshold_multiplier: float = -4.0,
                                     sampling_rate: int = 30000, figsize: tuple = (15, 10)) -> None:
    """
    Compare raster plots across multiple trials.
    
    Args:
        trial_data_list: List of trial data dictionaries
        spike_channels: List of spike channel indices
        trial_numbers: List of trial numbers for titles
        threshold_multiplier: Spike detection threshold (-4.0 = -4x RMS)
        sampling_rate: Neural data sampling rate in Hz
        figsize: Figure size tuple
    """
    n_trials = len(trial_data_list)
    if trial_numbers is None:
        trial_numbers = [f"Trial {i+1}" for i in range(n_trials)]
    
    print(f"🎨 Creating multi-trial raster comparison for {n_trials} trials...")
    
    # Create figure
    fig, axes = plt.subplots(n_trials, 1, figsize=figsize, sharex=True)
    if n_trials == 1:
        axes = [axes]  # Make it a list for consistency
    
    fig.suptitle(f'Multi-Trial Raster Comparison ({len(spike_channels)} channels)', 
                fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_channels)))
    
    for trial_idx, (trial_data, trial_num) in enumerate(zip(trial_data_list, trial_numbers)):
        ax = axes[trial_idx]
        
        # Get data
        neural_data = trial_data['neural_data']
        duration = trial_data.get('duration', neural_data.shape[1] / sampling_rate)
        outcome = trial_data.get('outcome', 'Unknown')
        
        # Detect spikes
        spike_times_by_channel = {}
        all_spike_times = []
        
        for i, ch_num in enumerate(spike_channels):
            if ch_num < neural_data.shape[0]:
                signal = neural_data[ch_num, :]
                
                # Calculate threshold
                rms = np.sqrt(np.mean(signal**2))
                threshold = threshold_multiplier * rms
                
                # Find threshold crossings
                if threshold_multiplier < 0:
                    spike_indices = np.where((signal[:-1] > threshold) & (signal[1:] <= threshold))[0]
                else:
                    spike_indices = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0]
                
                # Convert to time
                spike_times = spike_indices / sampling_rate
                spike_times_by_channel[ch_num] = spike_times
                all_spike_times.extend(spike_times)
        
        # Plot raster
        for i, ch_num in enumerate(spike_channels):
            if ch_num in spike_times_by_channel:
                spike_times = spike_times_by_channel[ch_num]
                if len(spike_times) > 0:
                    y_pos = np.full_like(spike_times, i)
                    ax.scatter(spike_times, y_pos, c=[colors[i]], s=1, alpha=0.7, marker='|')
        
        # Format subplot
        total_spikes = len(all_spike_times)
        avg_rate = total_spikes / (duration * len(spike_channels)) if duration > 0 else 0
        
        ax.set_title(f'{trial_num} | {outcome} | {total_spikes} spikes | {avg_rate:.1f} Hz/ch', 
                    fontsize=11)
        ax.set_ylabel('Channel', fontweight='bold', fontsize=10)
        ax.set_ylim(-0.5, len(spike_channels) - 0.5)
        ax.set_xlim(0, max([trial_data.get('duration', trial_data['neural_data'].shape[1] / sampling_rate) 
                           for trial_data in trial_data_list]))
        ax.grid(True, alpha=0.3)
        
        # Only show x-axis labels on bottom plot
        if trial_idx == n_trials - 1:
            ax.set_xlabel('Time (seconds)', fontweight='bold')
        else:
            ax.set_xticks([])
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Multi-trial comparison complete for trials: {trial_numbers}") 


def plot_spike_raster(spikes, trial_numbers=None, plot_type='horzline', spike_duration=0.001, 
                     rel_spike_start_time=0.0, vert_spike_position=0.0, vert_spike_height=1.0,
                     line_format=None, marker_format=None, xlim_for_spikes=None, 
                     sampling_rate=30000, auto_label=True, figsize=(12, 8), ax=None):
    """
    Create efficient spike raster plots supporting multiple input formats and plot types.
    Based on the excellent MATLAB plotSpikeRaster implementation.
    
    Args:
        spikes: Either:
            - 2D numpy array (M trials x N time bins) of binary spike data
            - List of M arrays, each containing spike times for a trial
            - Dict with 'spike_times' key containing list of spike time arrays
        trial_numbers: List of trial numbers for labels (optional)
        plot_type: Type of plot ('horzline', 'vertline', 'scatter', 'horzline2', 'vertline2')
        spike_duration: Duration of spike marks in seconds
        rel_spike_start_time: Relative start time offset
        vert_spike_position: Vertical position offset (0 = centered on trial)
        vert_spike_height: Height of vertical spikes
        line_format: Dict with line formatting (color, linewidth, etc.)
        marker_format: Dict with marker formatting (size, color, etc.)
        xlim_for_spikes: [min, max] time limits for spike time data
        sampling_rate: Sampling rate for binary data conversion
        auto_label: Whether to automatically label axes
        figsize: Figure size tuple
        ax: Matplotlib axis to plot on (optional)
    
    Returns:
        tuple: (x_points, y_points) - coordinates used for plotting
    """
    
    # Set default formats
    if line_format is None:
        line_format = {'color': [0.2, 0.2, 0.2], 'linewidth': 0.5}
    if marker_format is None:
        marker_format = {'s': 1, 'c': [0.2, 0.2, 0.2], 'alpha': 0.7}
    
    # Handle different input formats
    if isinstance(spikes, dict):
        # Extract spike times from dict
        spike_times = spikes.get('spike_times', [])
        is_binary = False
    elif isinstance(spikes, list):
        # List of spike time arrays
        spike_times = spikes
        is_binary = False
    elif isinstance(spikes, np.ndarray):
        if spikes.dtype == bool or np.all(np.isin(spikes, [0, 1])):
            # Binary spike data
            is_binary = True
            spike_data = spikes.astype(bool)
        else:
            # Assume it's spike time data
            spike_times = [spikes[i] for i in range(spikes.shape[0])]
            is_binary = False
    else:
        raise ValueError("Invalid spike data format")
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Process binary vs spike time data
    if is_binary:
        n_trials, n_time_bins = spike_data.shape
        time_per_bin = 1.0 / sampling_rate
        
        # Convert parameters to bin units
        spike_duration_bins = spike_duration / time_per_bin
        rel_start_bins = rel_spike_start_time / time_per_bin
        
        # Set axis limits
        ax.set_xlim(rel_start_bins, n_time_bins + rel_start_bins)
        ax.set_ylim(0, n_trials + 1)
        
        # Generate plot based on type
        if plot_type == 'horzline':
            x_points, y_points = _plot_binary_horzline(
                spike_data, spike_duration_bins, rel_start_bins, vert_spike_position)
        elif plot_type == 'vertline':
            x_points, y_points = _plot_binary_vertline(
                spike_data, rel_start_bins, vert_spike_position, vert_spike_height)
        elif plot_type == 'scatter':
            x_points, y_points = _plot_binary_scatter(
                spike_data, rel_start_bins, vert_spike_position)
        elif plot_type == 'horzline2':
            x_points, y_points = _plot_binary_horzline2(
                spike_data, spike_duration_bins, rel_start_bins, vert_spike_position)
        elif plot_type == 'vertline2':
            x_points, y_points = _plot_binary_vertline2(
                spike_data, rel_start_bins, vert_spike_position, vert_spike_height)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    else:
        # Spike time data
        n_trials = len(spike_times)
        
        # Determine x-axis limits
        if xlim_for_spikes is None:
            all_times = np.concatenate([times for times in spike_times if len(times) > 0])
            if len(all_times) > 0:
                time_range = np.max(all_times) - np.min(all_times)
                padding = 0.0005 * time_range
                xlim_for_spikes = [np.min(all_times) - padding, np.max(all_times) + padding + spike_duration]
            else:
                xlim_for_spikes = [0, 1]
        
        ax.set_xlim(xlim_for_spikes)
        ax.set_ylim(0, n_trials + 1)
        
        # Generate plot based on type
        if plot_type in ['horzline', 'vertline']:
            x_points, y_points = _plot_spiketime_lines(
                spike_times, plot_type, spike_duration, rel_spike_start_time, 
                vert_spike_position, vert_spike_height)
        elif plot_type == 'scatter':
            x_points, y_points = _plot_spiketime_scatter(
                spike_times, rel_spike_start_time, vert_spike_position)
        else:
            raise ValueError(f"Plot type {plot_type} not supported for spike time data")
    
    # Plot the data
    if plot_type in ['horzline', 'vertline', 'horzline2', 'vertline2']:
        ax.plot(x_points, y_points, **line_format)
    elif plot_type == 'scatter':
        ax.scatter(x_points, y_points, **marker_format)
    
    # Formatting
    if not is_binary:
        ax.invert_yaxis()  # Match MATLAB convention for spike times
    
    # Labels and trial numbers
    if trial_numbers is not None:
        ax.set_yticks(range(1, n_trials + 1))
        ax.set_yticklabels([f'Trial {num}' for num in trial_numbers])
    
    if auto_label:
        if is_binary:
            ax.set_xlabel('Time (ms)')
        else:
            ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial')
    
    ax.grid(True, alpha=0.3)
    
    return x_points, y_points


def _plot_binary_horzline(spike_data, spike_duration_bins, rel_start_bins, vert_spike_position):
    """Plot horizontal lines for binary spike data."""
    trials, timebins = np.where(spike_data)
    
    x_points = np.column_stack([
        timebins + rel_start_bins,
        timebins + rel_start_bins + spike_duration_bins,
        np.full(len(timebins), np.nan)
    ]).ravel()
    
    y_points = np.column_stack([
        trials + 1 + vert_spike_position,
        trials + 1 + vert_spike_position,
        np.full(len(trials), np.nan)
    ]).ravel()
    
    return x_points, y_points


def _plot_binary_vertline(spike_data, rel_start_bins, vert_spike_position, vert_spike_height):
    """Plot vertical lines for binary spike data."""
    trials, timebins = np.where(spike_data)
    half_height = vert_spike_height / 2
    
    x_points = np.column_stack([
        timebins + rel_start_bins,
        timebins + rel_start_bins,
        np.full(len(timebins), np.nan)
    ]).ravel()
    
    y_points = np.column_stack([
        trials + 1 - half_height + vert_spike_position,
        trials + 1 + half_height + vert_spike_position,
        np.full(len(trials), np.nan)
    ]).ravel()
    
    return x_points, y_points


def _plot_binary_scatter(spike_data, rel_start_bins, vert_spike_position):
    """Plot scatter points for binary spike data."""
    trials, timebins = np.where(spike_data)
    x_points = timebins + rel_start_bins
    y_points = trials + 1 + vert_spike_position
    
    return x_points, y_points


def _plot_binary_horzline2(spike_data, spike_duration_bins, rel_start_bins, vert_spike_position):
    """Optimized horizontal lines for high-density binary data."""
    n_trials = spike_data.shape[0]
    x_points = []
    y_points = []
    
    for trial in range(n_trials):
        if np.any(spike_data[trial, :]):
            # Find continuous segments of spikes
            spike_diff = np.diff(np.concatenate([[0], spike_data[trial, :].astype(int), [0]]))
            start_x = np.where(spike_diff > 0)[0]
            end_x = np.where(spike_diff < 0)[0]
            
            # Create line segments
            trial_x = np.column_stack([
                start_x + rel_start_bins,
                end_x + rel_start_bins + spike_duration_bins - 1,
                np.full(len(start_x), np.nan)
            ]).ravel()
            
            trial_y = np.full(len(trial_x), trial + 1 + vert_spike_position)
            
            x_points.extend(trial_x)
            y_points.extend(trial_y)
    
    return np.array(x_points), np.array(y_points)


def _plot_binary_vertline2(spike_data, rel_start_bins, vert_spike_position, vert_spike_height):
    """Optimized vertical lines for high-density binary data."""
    n_time_bins = spike_data.shape[1]
    x_points = []
    y_points = []
    
    for time_bin in range(n_time_bins):
        if np.any(spike_data[:, time_bin]):
            # Find continuous segments of trials with spikes
            spike_diff = np.diff(np.concatenate([[0], spike_data[:, time_bin].astype(int), [0]]))
            start_y = np.where(spike_diff > 0)[0]
            end_y = np.where(spike_diff < 0)[0]
            
            # Create line segments
            timebin_y = np.column_stack([
                start_y + vert_spike_position,
                end_y + vert_spike_position,
                np.full(len(start_y), np.nan)
            ]).ravel()
            
            timebin_x = np.full(len(timebin_y), time_bin + rel_start_bins)
            
            x_points.extend(timebin_x)
            y_points.extend(timebin_y)
    
    return np.array(x_points), np.array(y_points)


def _plot_spiketime_lines(spike_times, plot_type, spike_duration, rel_start_time, 
                         vert_spike_position, vert_spike_height):
    """Plot lines for spike time data."""
    total_spikes = sum(len(times) for times in spike_times)
    x_points = np.full(total_spikes * 3, np.nan)
    y_points = np.full(total_spikes * 3, np.nan)
    
    current_idx = 0
    half_height = vert_spike_height / 2
    
    for trial, times in enumerate(spike_times):
        if len(times) == 0:
            continue
            
        n_spikes = len(times)
        
        if plot_type == 'horzline':
            trial_x = np.column_stack([
                times + rel_start_time,
                times + rel_start_time + spike_duration,
                np.full(n_spikes, np.nan)
            ]).ravel()
            
            trial_y = np.column_stack([
                np.full(n_spikes, trial + 1 + vert_spike_position),
                np.full(n_spikes, trial + 1 + vert_spike_position),
                np.full(n_spikes, np.nan)
            ]).ravel()
            
        else:  # vertline
            trial_x = np.column_stack([
                times + rel_start_time,
                times + rel_start_time,
                np.full(n_spikes, np.nan)
            ]).ravel()
            
            trial_y = np.column_stack([
                np.full(n_spikes, trial + 1 - half_height + vert_spike_position),
                np.full(n_spikes, trial + 1 + half_height + vert_spike_position),
                np.full(n_spikes, np.nan)
            ]).ravel()
        
        # Store points
        end_idx = current_idx + n_spikes * 3
        x_points[current_idx:end_idx] = trial_x
        y_points[current_idx:end_idx] = trial_y
        current_idx = end_idx
    
    return x_points[:current_idx], y_points[:current_idx]


def _plot_spiketime_scatter(spike_times, rel_start_time, vert_spike_position):
    """Plot scatter points for spike time data."""
    x_points = []
    y_points = []
    
    for trial, times in enumerate(spike_times):
        if len(times) > 0:
            x_points.extend(times + rel_start_time)
            y_points.extend([trial + 1 + vert_spike_position] * len(times))
    
    return np.array(x_points), np.array(y_points)

# Spike-based visualization functions

def plot_spike_raster_behavioral(analysis_results: Dict, trial_number: int, figsize: Tuple[float, float] = (15, 12)):
    """
    Create a comprehensive spike raster and behavioral visualization.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from SpikeAnalyzer.analyze_trial()
    trial_number : int
        Trial number for title
    figsize : tuple
        Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create visualization - analysis failed")
        return
    
    trial_data = analysis_results['trial_data']
    firing_rates = analysis_results['firing_rates']
    spike_data = analysis_results['spike_data']
    top_channels = analysis_results['top_channels']
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Behavioral data (velocity)
    ax1 = axes[0]
    if trial_data['velocity_x'] is not None and trial_data['velocity_y'] is not None:
        # Create time axis for behavioral data
        if 'behavioral_timestamps' in trial_data:
            time_behavioral = trial_data['behavioral_timestamps']
        else:
            time_behavioral = np.linspace(0, trial_data['metadata']['duration'], len(trial_data['velocity_x']))
        
        ax1.plot(time_behavioral, trial_data['velocity_x'], 'b-', label='Velocity X', linewidth=1.5)
        ax1.plot(time_behavioral, trial_data['velocity_y'], 'r-', label='Velocity Y', linewidth=1.5)
        ax1.set_ylabel('Velocity (mm/s)')
        ax1.set_title(f'Trial {trial_number} - Behavioral Data', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spike raster (top 10 channels)
    ax2 = axes[1]
    y_pos = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_channels[:10])))
    
    duration = trial_data['metadata']['duration']
    sampling_rate = 30000  # Hz
    
    for i, channel in enumerate(top_channels[:10]):
        if channel in spike_data:
            spike_times = spike_data[channel]['spike_times'] / sampling_rate
            ax2.scatter(spike_times, np.ones(len(spike_times)) * y_pos, 
                       c=[colors[i]], s=2, alpha=0.7)
            y_pos += 1
    
    ax2.set_ylabel('Channel')
    ax2.set_title(f'Spike Raster - Top {min(10, len(top_channels))} Active Channels', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(top_channels[:10])))
    ax2.set_yticklabels([f'Ch {ch}' for ch in top_channels[:10]])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Population firing rate
    ax3 = axes[2]
    time_bins = np.linspace(0, duration, len(list(firing_rates.values())[0]))
    
    # Calculate population firing rate
    pop_firing_rate = np.zeros(len(time_bins))
    for channel, rates in firing_rates.items():
        pop_firing_rate += rates
    
    ax3.plot(time_bins, pop_firing_rate, 'k-', linewidth=2, label='Population Rate')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Firing Rate (spikes/s)')
    ax3.set_title('Population Firing Rate', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Spike raster visualization complete!")
    print(f"   • Behavioral data: {len(trial_data['velocity_x'])} samples")
    print(f"   • Spike raster: {len(top_channels[:10])} channels")
    print(f"   • Population rate: {len(time_bins)} time bins")


def plot_spike_firing_rates(analysis_results: Dict, n_channels: int = 6, figsize: Tuple[float, float] = (15, 12)):
    """
    Plot individual channel firing rates.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from SpikeAnalyzer.analyze_trial()
    n_channels : int
        Number of top channels to plot
    figsize : tuple
        Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create visualization - analysis failed")
        return
    
    firing_rates = analysis_results['firing_rates']
    top_channels = analysis_results['top_channels']
    trial_data = analysis_results['trial_data']
    
    # Select channels to plot
    channels_to_plot = top_channels[:n_channels]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Create time axis
    duration = trial_data['metadata']['duration']
    time_bins = np.linspace(0, duration, len(list(firing_rates.values())[0]))
    
    for i, channel in enumerate(channels_to_plot):
        ax = axes[i]
        
        if channel in firing_rates:
            # Plot firing rate
            ax.plot(time_bins, firing_rates[channel], 'b-', linewidth=2, alpha=0.8)
            ax.set_title(f'Channel {channel} - Firing Rate', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (spikes/s)')
            ax.grid(True, alpha=0.3)
            
            # Add channel statistics
            mean_rate = np.mean(firing_rates[channel])
            max_rate = np.max(firing_rates[channel])
            ax.text(0.02, 0.95, f'Mean: {mean_rate:.1f} Hz\nMax: {max_rate:.1f} Hz', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Channel {channel}\nNo Data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Channel {channel} - No Data', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Individual channel analysis complete!")
    print(f"   • Channels analyzed: {len(channels_to_plot)}")


def plot_spike_channel_comparison(analysis_results: Dict, figsize: Tuple[float, float] = (15, 10)):
    """
    Create channel comparison and spike statistics plots.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from SpikeAnalyzer.analyze_trial()
    figsize : tuple
        Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create visualization - analysis failed")
        return
    
    firing_rates = analysis_results['firing_rates']
    quality_metrics = analysis_results['quality_metrics']
    top_channels = analysis_results['top_channels']
    trial_data = analysis_results['trial_data']
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Firing rate comparison (top 4 channels)
    ax1 = axes[0, 0]
    channels_to_compare = top_channels[:4]
    
    for i, channel in enumerate(channels_to_compare):
        if channel in firing_rates:
            time_bins = np.linspace(0, trial_data['metadata']['duration'], len(firing_rates[channel]))
            ax1.plot(time_bins, firing_rates[channel], linewidth=2, 
                    label=f'Ch {channel}', alpha=0.8)
    
    ax1.set_title('Firing Rate Comparison - Top 4 Channels', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Firing Rate (spikes/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spike count per channel
    ax2 = axes[0, 1]
    sorted_metrics = quality_metrics.sort_values('n_spikes', ascending=False)
    top_10_metrics = sorted_metrics.head(10)
    
    bars = ax2.bar(range(len(top_10_metrics)), top_10_metrics['n_spikes'], 
                   color='skyblue', alpha=0.8)
    ax2.set_title('Total Spike Count - Top 10 Channels', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Channel Rank')
    ax2.set_ylabel('Total Spikes')
    ax2.set_xticks(range(len(top_10_metrics)))
    ax2.set_xticklabels([f'Ch {ch}' for ch in top_10_metrics['channel']], rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, top_10_metrics['n_spikes']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: SNR distribution
    ax3 = axes[1, 0]
    ax3.hist(quality_metrics['snr'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('Signal-to-Noise Ratio Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('SNR')
    ax3.set_ylabel('Number of Channels')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Peak-to-peak amplitude vs spike count
    ax4 = axes[1, 1]
    scatter = ax4.scatter(quality_metrics['peak_to_peak'], quality_metrics['n_spikes'], 
                         c=quality_metrics['snr'], cmap='viridis', alpha=0.7)
    ax4.set_title('Peak-to-Peak Amplitude vs Spike Count', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Peak-to-Peak Amplitude')
    ax4.set_ylabel('Total Spikes')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('SNR')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Channel comparison complete!")
    print(f"   • Channels compared: {len(channels_to_compare)}")
    print(f"   • Quality metrics: {len(quality_metrics)} channels")


def plot_spike_channel_detail(analysis_results: Dict, figsize: Tuple[float, float] = (15, 12)):
    """
    Create detailed analysis plot for the most active channel.
    
    Args:
        analysis_results: Results from spike analysis
        figsize: Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create detailed channel analysis - no valid analysis results")
        return
    
    # Import analysis function
    from utils.analysis import analyze_channel_detail
    
    # Get detailed channel analysis
    channel_analysis = analyze_channel_detail(analysis_results)
    
    if not channel_analysis['success']:
        print(f"❌ Cannot create detailed channel analysis: {channel_analysis['error']}")
        return
    
    channel = channel_analysis['channel']
    firing_rate_stats = channel_analysis['firing_rate_stats']
    spike_stats = channel_analysis['spike_stats']
    quality_stats = channel_analysis['quality_stats']
    
    print(f"📊 Creating detailed analysis for channel {channel}...")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Firing rate over time
    ax1 = axes[0, 0]
    ax1.plot(firing_rate_stats['time_bins'], firing_rate_stats['firing_rates'], 
             'b-', linewidth=2)
    ax1.set_title(f'Channel {channel} - Firing Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Firing Rate (spikes/s)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spike waveforms
    ax2 = axes[0, 1]
    if spike_stats and len(spike_stats.get('waveforms', [])) > 0:
        waveforms = spike_stats['waveforms']
        
        # Plot individual waveforms (sample)
        n_sample = min(50, len(waveforms))
        sample_indices = np.random.choice(len(waveforms), n_sample, replace=False)
        
        for i in sample_indices:
            ax2.plot(waveforms[i], 'gray', alpha=0.3, linewidth=0.5)
        
        # Plot mean waveform
        if spike_stats['mean_waveform'] is not None:
            ax2.plot(spike_stats['mean_waveform'], 'r-', linewidth=3, label='Mean')
        
        ax2.set_title(f'Channel {channel} - Spike Waveforms', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No waveform data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Channel {channel} - Spike Waveforms', fontsize=12, fontweight='bold')
    
    # Plot 3: Inter-spike interval histogram
    ax3 = axes[1, 0]
    if spike_stats and len(spike_stats.get('isis', [])) > 0:
        isis = spike_stats['isis']
        ax3.hist(isis, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
        ax3.set_title(f'Channel {channel} - Inter-Spike Intervals', fontsize=12, fontweight='bold')
        ax3.set_xlabel('ISI (ms)')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        ax3.axvline(spike_stats['mean_isi'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {spike_stats["mean_isi"]:.1f}ms')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No spike timing data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'Channel {channel} - Inter-Spike Intervals', fontsize=12, fontweight='bold')
    
    # Plot 4: Quality metrics
    ax4 = axes[1, 1]
    if quality_stats:
        metrics_names = ['Spike\nCount', 'SNR', 'Peak-to-Peak\nAmplitude', 'Consistency']
        metrics_values = [
            quality_stats['n_spikes'], 
            quality_stats['snr'], 
            quality_stats['peak_to_peak'], 
            quality_stats['consistency']
        ]
        
        # Normalize values for comparison (0-1 scale)
        normalized_values = []
        for i, value in enumerate(metrics_values):
            if i == 0:  # Spike count - normalize to reasonable range
                normalized_values.append(min(1.0, value / 500))  # Cap at 500 spikes
            else:
                normalized_values.append(min(1.0, value / 10))  # Cap at 10 for other metrics
        
        bars = ax4.bar(metrics_names, normalized_values, 
                      color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax4.set_title(f'Channel {channel} - Quality Metrics', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Normalized Value')
        ax4.set_ylim(0, 1.1)
        
        # Add actual values as text
        for bar, actual_value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{actual_value:.1f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No quality metrics available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'Channel {channel} - Quality Metrics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"✅ Detailed analysis complete for channel {channel}!")
    if spike_stats:
        print(f"   • Total spikes: {spike_stats.get('n_spikes', 'N/A')}")
    print(f"   • Mean firing rate: {firing_rate_stats['mean_rate']:.1f} Hz")
    print(f"   • Peak firing rate: {firing_rate_stats['max_rate']:.1f} Hz")


def plot_comprehensive_behavior_raster_psth(analysis_results: Dict, trial_number: int, 
                                          psth_bin_size: float = 0.01,
                                          figsize: Tuple[float, float] = (15, 12)):
    """
    Create comprehensive visualization showing behavioral data, spike raster, and PSTH.
    
    Args:
        analysis_results: Results from spike analysis
        trial_number: Trial number being analyzed
        psth_bin_size: Bin size for PSTH in seconds
        figsize: Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create comprehensive visualization - no valid analysis results")
        return
    
    # Extract data
    trial_data = analysis_results['trial_data']
    spike_data = analysis_results['spike_data']
    firing_rates = analysis_results['firing_rates']
    quality_metrics = analysis_results['quality_metrics']
    
    # Get most active channels
    most_active_channels = quality_metrics.nlargest(15, 'n_spikes')['channel'].tolist()
    
    print("📊 Creating comprehensive Behavior-Raster-PSTH visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # === Plot 1: Behavioral Data ===
    ax1 = axes[0]
    if trial_data['velocity_x'] is not None and trial_data['velocity_y'] is not None:
        # Create time axis for behavioral data
        if 'behavioral_timestamps' in trial_data:
            time_behavioral = trial_data['behavioral_timestamps']
        else:
            time_behavioral = np.linspace(0, trial_data['metadata']['duration'], 
                                        len(trial_data['velocity_x']))
        
        # Plot velocity
        ax1.plot(time_behavioral, trial_data['velocity_x'], 'b-', 
                label='Velocity X', linewidth=1.5, alpha=0.8)
        ax1.plot(time_behavioral, trial_data['velocity_y'], 'r-', 
                label='Velocity Y', linewidth=1.5, alpha=0.8)
        
        # Calculate and plot speed
        speed = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
        ax1.plot(time_behavioral, speed, 'k-', label='Speed', linewidth=2, alpha=0.9)
        
        ax1.set_ylabel('Velocity (mm/s)', fontsize=12)
        ax1.set_title(f'Trial {trial_number} - Behavioral Data', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No behavioral data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Trial {trial_number} - Behavioral Data', fontsize=14, fontweight='bold')
    
    # === Plot 2: Spike Raster ===
    ax2 = axes[1]
    y_pos = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(most_active_channels)))
    
    duration = trial_data['metadata']['duration']
    raster_channels = []
    sampling_rate = 30000  # Hz
    
    for i, channel in enumerate(most_active_channels):
        if channel in spike_data:
            spike_times = spike_data[channel]['spike_times'] / sampling_rate
            if len(spike_times) > 0:
                ax2.scatter(spike_times, np.ones(len(spike_times)) * y_pos, 
                           c=[colors[i]], s=1.5, alpha=0.8, rasterized=True)
                raster_channels.append(channel)
                y_pos += 1
    
    ax2.set_ylabel('Channel', fontsize=12)
    ax2.set_title(f'Spike Raster - Top {len(raster_channels)} Active Channels', 
                 fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(raster_channels)))
    ax2.set_yticklabels([f'Ch {ch}' for ch in raster_channels], fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, duration)
    
    # === Plot 3: Population PSTH ===
    ax3 = axes[2]
    
    # Create high-resolution time bins for PSTH
    n_psth_bins = int(duration / psth_bin_size)
    psth_time_bins = np.linspace(0, duration, n_psth_bins)
    
    # Calculate population PSTH from individual spike times
    population_psth = np.zeros(n_psth_bins)
    
    for channel in most_active_channels:
        if channel in spike_data:
            spike_times = spike_data[channel]['spike_times'] / sampling_rate
            if len(spike_times) > 0:
                # Bin spikes for this channel
                hist, _ = np.histogram(spike_times, bins=n_psth_bins, range=(0, duration))
                # Convert to rate (spikes/second)
                channel_psth = hist / psth_bin_size
                population_psth += channel_psth
    
    # Smooth PSTH with Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    sigma_bins = 0.025 / psth_bin_size  # 25ms smoothing in bins
    smoothed_psth = gaussian_filter1d(population_psth, sigma=sigma_bins)
    
    ax3.plot(psth_time_bins, smoothed_psth, 'k-', linewidth=2, 
             label='Population PSTH (smoothed)')
    ax3.fill_between(psth_time_bins, smoothed_psth, alpha=0.3, color='gray')
    
    # Also plot the binned firing rates for comparison
    firing_rate_time = np.linspace(0, duration, len(list(firing_rates.values())[0]))
    population_rate = np.zeros(len(firing_rate_time))
    for channel, rates in firing_rates.items():
        population_rate += rates
    
    ax3.plot(firing_rate_time, population_rate, 'r--', linewidth=1.5, alpha=0.7, 
             label=f'Binned Rate (50ms bins)')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Firing Rate (spikes/s)', fontsize=12)
    ax3.set_title('Population Firing Rate (PSTH)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Comprehensive visualization complete!")
    print(f"   • Behavioral data: {len(trial_data['velocity_x']) if trial_data['velocity_x'] is not None else 0} samples")
    print(f"   • Spike raster: {len(raster_channels)} channels")
    print(f"   • Total spikes: {sum([spike_data[ch]['n_spikes'] for ch in raster_channels if ch in spike_data])}")
    print(f"   • PSTH resolution: {psth_bin_size*1000:.0f}ms bins, 25ms smoothing")
    print(f"   • Duration: {duration:.1f}s")


def plot_neural_behavioral_correlation_analysis(analysis_results: Dict, trial_number: int,
                                              figsize: Tuple[float, float] = (15, 10)):
    """
    Create visualization for neural-behavioral correlation analysis.
    
    Args:
        analysis_results: Results from spike analysis
        trial_number: Trial number being analyzed
        figsize: Figure size (width, height)
    """
    if not analysis_results['success']:
        print("❌ Cannot create correlation analysis - no valid analysis results")
        return
    
    # Get correlation analysis from SpikeAnalyzer
    from utils.analysis import SpikeAnalyzer
    
    # Create temporary analyzer instance to use correlation method
    spike_analyzer = SpikeAnalyzer(spike_channels=[], sampling_rate=30000)
    correlation_results = spike_analyzer.analyze_neural_behavioral_correlation(analysis_results)
    
    if not correlation_results['success']:
        print(f"❌ Cannot create correlation analysis: {correlation_results['error']}")
        return
    
    print("📊 Creating neural-behavioral correlation visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Population firing rate vs velocity
    ax1 = axes[0, 0]
    time_bins = correlation_results['time_bins']
    population_rate = correlation_results['population_rate']
    velocity_aligned = correlation_results['velocity_aligned']
    
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(time_bins, population_rate, 'b-', linewidth=2, 
                    label='Population Firing Rate')
    line2 = ax1_twin.plot(time_bins, velocity_aligned, 'r-', linewidth=2, 
                         label='Velocity Magnitude')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Firing Rate (spikes/s)', color='b')
    ax1_twin.set_ylabel('Velocity (mm/s)', color='r')
    ax1.set_title(f'Population Activity vs Velocity\n(r = {correlation_results["population_correlation"]:.3f})', 
                 fontsize=12, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(velocity_aligned, population_rate, alpha=0.6, s=10)
    ax2.set_xlabel('Velocity Magnitude (mm/s)')
    ax2.set_ylabel('Population Firing Rate (spikes/s)')
    ax2.set_title('Neural-Behavioral Correlation Scatter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(velocity_aligned, population_rate, 1)
    p = np.poly1d(z)
    ax2.plot(velocity_aligned, p(velocity_aligned), "r--", alpha=0.8)
    
    # Plot 3: Individual channel correlations
    ax3 = axes[1, 0]
    channel_correlations = correlation_results['channel_correlations'][:8]  # Top 8
    channels = [str(ch[0]) for ch in channel_correlations]
    correlations = [ch[1] for ch in channel_correlations]
    
    colors = ['red' if corr < 0 else 'blue' for corr in correlations]
    bars = ax3.bar(channels, correlations, color=colors, alpha=0.7)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Individual Channel Correlations', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values as text
    for bar, corr in zip(bars, correlations):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Analysis summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats = correlation_results['stats']
    summary_text = f"""
    CORRELATION ANALYSIS SUMMARY
    
    Population Correlation: {correlation_results['population_correlation']:.3f}
    
    Neural Activity:
    • Peak firing rate: {stats['neural_peak']:.1f} spikes/s
    • Time bins analyzed: {stats['time_bins_analyzed']}
    
    Behavioral Activity:
    • Peak velocity: {stats['velocity_peak']:.1f} mm/s
    
    Top Channel Correlations:
    """
    
    for i, (ch, corr) in enumerate(channel_correlations[:3]):
        summary_text += f"• Channel {ch}: {corr:.3f}\n    "
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Neural-behavioral correlation analysis complete!")
    print(f"   • Population correlation: {correlation_results['population_correlation']:.3f}")
    print(f"   • Top channel correlations: {[f'Ch {ch}: {corr:.3f}' for ch, corr in channel_correlations[:3]]}")


def plot_interactive_exploration_comparison(exploration_results: Dict, original_results: Dict = None,
                                          figsize: Tuple[float, float] = (15, 10)):
    """
    Create visualization comparing different exploration results.
    
    Args:
        exploration_results: Results from interactive exploration
        original_results: Original analysis results for comparison
        figsize: Figure size (width, height)
    """
    if not exploration_results['success']:
        print("❌ Cannot create exploration comparison - no valid exploration results")
        return
    
    trial_number = exploration_results['trial_number']
    summary = exploration_results['summary']
    comparison = exploration_results.get('comparison', {})
    
    print(f"📊 Creating interactive exploration comparison for trial {trial_number}...")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Population firing rate
    ax1 = axes[0, 0]
    explore_results = exploration_results['analysis_results']
    firing_rates = explore_results['firing_rates']
    trial_data = explore_results['trial_data']
    
    duration = trial_data['metadata']['duration']
    firing_rate_time = np.linspace(0, duration, len(list(firing_rates.values())[0]))
    
    pop_rate = np.zeros(len(firing_rate_time))
    for channel, rates in firing_rates.items():
        pop_rate += rates
    
    ax1.plot(firing_rate_time, pop_rate, 'b-', linewidth=2)
    ax1.set_title(f'Trial {trial_number} - Population Firing Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Firing Rate (spikes/s)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top channels spike count
    ax2 = axes[0, 1]
    quality_metrics = explore_results['quality_metrics']
    top_5_metrics = quality_metrics.nlargest(5, 'n_spikes')
    
    ax2.bar(range(len(top_5_metrics)), top_5_metrics['n_spikes'], color='skyblue')
    ax2.set_title(f'Trial {trial_number} - Top 5 Channels (Spike Count)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Channel Rank')
    ax2.set_ylabel('Total Spikes')
    ax2.set_xticks(range(len(top_5_metrics)))
    ax2.set_xticklabels([f'Ch {ch}' for ch in top_5_metrics['channel']])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison metrics (if available)
    ax3 = axes[1, 0]
    if comparison:
        metrics = ['Total Spikes', 'Mean Firing Rate', 'High Quality Channels']
        differences = [comparison['spike_diff'], comparison['rate_diff'], comparison['quality_diff']]
        
        colors = ['red' if diff < 0 else 'green' for diff in differences]
        bars = ax3.bar(metrics, differences, color=colors, alpha=0.7)
        ax3.set_title('Comparison with Original Trial', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Difference')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add values as text
        for bar, diff in zip(bars, differences):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{diff:.1f}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No comparison data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparison with Original Trial', fontsize=12, fontweight='bold')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    EXPLORATION SUMMARY - Trial {trial_number}
    
    Spike Detection Results:
    • Total spikes: {summary['total_spikes']}
    • Mean firing rate: {summary['mean_firing_rate']:.1f} Hz
    • High quality channels: {summary['high_quality_channels']}
    • Active channels: {summary['active_channels']}
    • Total channels: {summary['total_channels_analyzed']}
    
    Top Performing Channels:
    """
    
    for i, ch_info in enumerate(summary['top_channels'][:3]):
        summary_text += f"• Ch {ch_info['channel']}: {ch_info['n_spikes']} spikes, {ch_info['mean_rate']:.1f} Hz\n    "
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Interactive exploration comparison complete!")
    print(f"   • Trial {trial_number}: {summary['total_spikes']} spikes, {summary['mean_firing_rate']:.1f} Hz")
    if comparison:
        print(f"   • Differences: {comparison['spike_diff']} spikes, {comparison['rate_diff']:.1f} Hz")


def plot_spike_psth(time_axis, spike_matrix, bin_size, kernel_width, kernel_type='gauss', 
                   plot_error=True, color='blue', ax=None, linewidth=3, alpha=0.7):
    """
    Plot spike PSTH with various kernel smoothing options.
    
    Python implementation of Nader Nikbakht's MATLAB plotSpikePSTH function.
    
    Parameters:
    -----------
    time_axis : np.ndarray
        Time axis for the data
    spike_matrix : np.ndarray
        Spike matrix (trials x time bins) - can be spike counts or binary
    bin_size : float
        Bin size in seconds
    kernel_width : float
        Width of the smoothing kernel (reasonable values: 0.025-0.250 s)
    kernel_type : str
        Type of kernel ('gauss', 'halfgauss', 'exp', 'epsp')
    plot_error : bool
        Whether to plot error bars (SEM)
    color : str or tuple
        Color for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (if None, creates new figure)
    linewidth : float
        Line width for the plot
    alpha : float
        Alpha (transparency) for error bars
        
    Returns:
    --------
    tuple
        (smoothed_psth, sem, time_axis, ax) - smoothed PSTH, SEM, time axis, and axes
    """
    from scipy import stats
    from scipy import signal
    import matplotlib.pyplot as plt
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure spike_matrix is 2D (trials x time)
    if spike_matrix.ndim == 1:
        spike_matrix = spike_matrix.reshape(1, -1)
    
    # Create smoothing kernel based on type
    if kernel_type == 'gauss':
        # Gaussian kernel
        gauss_width = max(11, int(6 * kernel_width / bin_size) + 1)
        if gauss_width % 2 == 0:  # Make sure it's odd
            gauss_width += 1
        
        kernel_samples = np.arange(-gauss_width//2, gauss_width//2 + 1)
        kernel = stats.norm.pdf(kernel_samples, 0, kernel_width / bin_size)
        
    elif kernel_type == 'halfgauss':
        # Half Gaussian kernel (right half only)
        gauss_width = max(11, int(6 * kernel_width / bin_size) + 1)
        if gauss_width % 2 == 0:
            gauss_width += 1
        
        kernel_samples = np.arange(-gauss_width//2, gauss_width//2 + 1)
        full_kernel = stats.norm.pdf(kernel_samples, 0, kernel_width / bin_size)
        kernel = full_kernel[gauss_width//2:]  # Take right half
        
    elif kernel_type == 'exp':
        # Exponential kernel
        kernel_samples = np.arange(-int(1.0 / bin_size), int(1.0 / bin_size) + 1)
        kernel = stats.expon.pdf(kernel_samples, scale=kernel_width / bin_size) * bin_size
        
    elif kernel_type == 'epsp':
        # EPSP-like kernel with dual exponential
        tau_falling = 0.040  # 40ms falling time constant
        tau_rising = 0.001   # 1ms rising time constant
        
        t = np.arange(0, 0.200, bin_size)  # 200ms kernel
        kernel_falling = stats.expon.pdf(t, scale=tau_falling / bin_size)
        kernel_rising = stats.expon.pdf(t, scale=tau_rising / bin_size)
        
        # EPSP shape: falling * (1 - rising)
        kernel = kernel_falling * (1 - kernel_rising)
        kernel = kernel / np.sum(kernel)  # Normalize
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Calculate mean spike rate across trials
    spike_matrix_mean = np.mean(spike_matrix, axis=0)
    
    # Edge correction factor
    edge_corr_factor = np.convolve(np.ones_like(spike_matrix_mean), kernel, mode='same')
    edge_corr_factor = np.maximum(edge_corr_factor, 1e-10)  # Avoid division by zero
    
    # Convolve with kernel and apply edge correction
    smoothed_mean = np.convolve(spike_matrix_mean, kernel, mode='same')
    
    # Handle shape mismatch by truncating to the minimum length
    min_length = min(len(smoothed_mean), len(edge_corr_factor), len(spike_matrix_mean))
    smoothed_psth = smoothed_mean[:min_length] / (bin_size * edge_corr_factor[:min_length])
    
    # Calculate SEM across trials  
    # Initialize with the same length as smoothed_psth
    spike_density_trials = np.zeros((spike_matrix.shape[0], len(smoothed_psth)))
    
    for i in range(spike_matrix.shape[0]):
        convolved = np.convolve(spike_matrix[i, :], kernel, mode='same')
        
        # Use the same truncation as for smoothed_psth
        spike_density_trials[i, :] = (convolved[:len(smoothed_psth)] / 
                                     (bin_size * edge_corr_factor[:len(smoothed_psth)]))
    
    sem = np.std(spike_density_trials, axis=0) / np.sqrt(spike_matrix.shape[0])
    
    # Handle large SEM values (likely due to edge effects)
    sem = np.minimum(sem, 100)  # Cap SEM at 100 Hz
    
    # Ensure time_axis matches the data length
    if len(time_axis) != len(smoothed_psth):
        time_axis = time_axis[:len(smoothed_psth)]
    
    # Plot
    if plot_error:
        # Plot with error bars using fill_between
        ax.plot(time_axis, smoothed_psth, color=color, linewidth=linewidth, alpha=0.9)
        ax.fill_between(time_axis, smoothed_psth - sem[:len(smoothed_psth)], 
                       smoothed_psth + sem[:len(smoothed_psth)], 
                       color=color, alpha=alpha, label='SEM')
    else:
        ax.plot(time_axis, smoothed_psth, color=color, linewidth=linewidth)
    
    # Format plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate (spikes/s)')
    ax.grid(True, alpha=0.3)
    
    # Add kernel info to title
    kernel_info = f'{kernel_type} kernel (σ={kernel_width*1000:.1f}ms)'
    if ax.get_title():
        ax.set_title(f'{ax.get_title()}\n{kernel_info}')
    else:
        ax.set_title(f'Spike PSTH - {kernel_info}')
    
    return smoothed_psth, sem, time_axis, ax


def plot_multi_condition_psth(spike_data_dict, time_axis, bin_size, kernel_width=0.025, 
                             kernel_type='gauss', figsize=(12, 8), colors=None):
    """
    Plot PSTH for multiple conditions using the advanced kernel smoothing.
    
    Parameters:
    -----------
    spike_data_dict : dict
        Dictionary where keys are condition names and values are spike matrices
    time_axis : np.ndarray
        Time axis for all conditions
    bin_size : float
        Bin size in seconds
    kernel_width : float
        Width of the smoothing kernel
    kernel_type : str
        Type of kernel ('gauss', 'halfgauss', 'exp', 'epsp')
    figsize : tuple
        Figure size
    colors : list, optional
        List of colors for each condition
        
    Returns:
    --------
    dict
        Dictionary containing PSTH results for each condition
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(spike_data_dict)))
    
    results = {}
    
    for i, (condition, spike_matrix) in enumerate(spike_data_dict.items()):
        color = colors[i] if isinstance(colors, (list, tuple)) else colors
        
        # Plot PSTH for this condition
        smoothed_psth, sem, _, _ = plot_spike_psth(
            time_axis, spike_matrix, bin_size, kernel_width, kernel_type,
            plot_error=True, color=color, ax=ax, alpha=0.2
        )
        
        # Store results
        results[condition] = {
            'smoothed_psth': smoothed_psth,
            'sem': sem,
            'time_axis': time_axis,
            'peak_rate': np.max(smoothed_psth),
            'mean_rate': np.mean(smoothed_psth),
            'n_trials': spike_matrix.shape[0]
        }
        
        # Add to legend
        ax.plot([], [], color=color, linewidth=3, alpha=0.9, 
               label=f'{condition} (n={spike_matrix.shape[0]})')
    
    ax.legend()
    ax.set_title(f'Multi-Condition PSTH Comparison\n{kernel_type} kernel (σ={kernel_width*1000:.1f}ms)')
    
    plt.tight_layout()
    plt.show()
    
    return results


def create_psth_from_spike_times(spike_times_list, time_window, bin_size, 
                                align_time=0.0, kernel_width=0.025, kernel_type='gauss'):
    """
    Create PSTH from spike times with advanced kernel smoothing.
    
    Parameters:
    -----------
    spike_times_list : list
        List of spike time arrays (one per trial)
    time_window : tuple
        (start_time, end_time) for the analysis window
    bin_size : float
        Bin size in seconds
    align_time : float
        Time to align to (subtracted from all spike times)
    kernel_width : float
        Width of the smoothing kernel
    kernel_type : str
        Type of kernel ('gauss', 'halfgauss', 'exp', 'epsp')
        
    Returns:
    --------
    tuple
        (time_axis, spike_matrix, smoothed_psth, sem) - ready for plotting
    """
    # Create time axis
    time_bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    time_axis = time_bins[:-1] + bin_size / 2
    
    # Create spike matrix
    spike_matrix = np.zeros((len(spike_times_list), len(time_axis)))
    
    for i, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            # Align spike times
            aligned_spikes = spike_times - align_time
            
            # Only include spikes within the time window
            valid_spikes = aligned_spikes[
                (aligned_spikes >= time_window[0]) & 
                (aligned_spikes <= time_window[1])
            ]
            
            if len(valid_spikes) > 0:
                # Create histogram
                counts, _ = np.histogram(valid_spikes, bins=time_bins)
                spike_matrix[i, :] = counts
    
    # Apply kernel smoothing
    smoothed_psth, sem, _, _ = plot_spike_psth(
        time_axis, spike_matrix, bin_size, kernel_width, kernel_type,
        plot_error=False, ax=None
    )
    
    return time_axis, spike_matrix, smoothed_psth, sem