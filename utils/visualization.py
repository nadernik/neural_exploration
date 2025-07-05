"""
Visualization utilities for neural exploration project.
Handles behavioral task visualization and neural data plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[4])
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
        
        # Get time axis
        if 'timestamp' in trial_data.columns:
            time_axis = trial_data['timestamp']
            time_label = 'Time (timestamp)'
        else:
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
                    ax2.axvline(time_axis.iloc[onset_idx] if hasattr(time_axis, 'iloc') else time_axis[onset_idx], 
                               color='red', linestyle='--', alpha=0.7, label='Movement Onset')
                    ax2.axhline(movement_threshold, color='red', linestyle=':', alpha=0.5, label='Threshold')
                    ax2.legend()
                    
                    # Add annotation
                    ax2.annotate(f'Onset: {onset_idx/len(trial_data)*100:.1f}% into trial', 
                                xy=(time_axis.iloc[onset_idx] if hasattr(time_axis, 'iloc') else time_axis[onset_idx], 
                                    vel_mag.iloc[onset_idx]), 
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
            # Compute trajectory from velocity
            dt = 1.0  # Default time step
            if 'timestamp' in trial_data.columns and len(trial_data) > 1:
                time_diffs = np.diff(pd.to_datetime(trial_data['timestamp']).values)
                if len(time_diffs) > 0:
                    dt = np.median(time_diffs) / np.timedelta64(1, 's')
            
            # Integrate velocity to get position (starting from center)
            pos_x = np.cumsum(trial_data['velocity_x'] * dt)
            pos_y = np.cumsum(trial_data['velocity_y'] * dt)
            
            # Center the trajectory (start at origin)
            pos_x = pos_x - pos_x.iloc[0]
            pos_y = pos_y - pos_y.iloc[0]
            
            # Scale trajectory to fit within the layout (targets are at radius 1.0)
            max_pos = max(np.max(np.abs(pos_x)), np.max(np.abs(pos_y)))
            if max_pos > 0:
                scale_factor = 0.8 / max_pos  # Scale to fit within 80% of target radius
                pos_x_scaled = pos_x * scale_factor
                pos_y_scaled = pos_y * scale_factor
            else:
                pos_x_scaled = pos_x
                pos_y_scaled = pos_y
            
            # Plot trajectory
            ax3.plot(pos_x_scaled, pos_y_scaled, 'g-', linewidth=3, alpha=0.8, label='Trajectory')
            
            # Mark start and end points
            ax3.plot(pos_x_scaled.iloc[0], pos_y_scaled.iloc[0], 'go', markersize=8, 
                    label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
            ax3.plot(pos_x_scaled.iloc[-1], pos_y_scaled.iloc[-1], 'ro', markersize=8, 
                    label='End', markeredgecolor='darkred', markeredgewidth=2)
        
        # Set equal aspect ratio and limits (same as plot_center_out_layout)
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.legend(loc='upper right', fontsize=8)
        
        # Add trial info
        if current_target_idx is not None:
            outcome = trial_data['trial_outcome'].iloc[0] if 'trial_outcome' in trial_data.columns else 'Unknown'
            target_direction = self.center_out_targets[f'target_{current_target_idx}']['direction']
            info_text = f"Target {current_target_idx} ({target_direction})\nOutcome: {outcome}"
            ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightyellow", alpha=0.8))
        
        ax3.set_title(f'Center Out Layout & Trajectory - Trial {trial_num}')
        
        # Plot 4: Velocity Vector Field and Computed Trajectory
        ax4 = axes[1, 1]
        if 'velocity_x' in trial_data.columns and 'velocity_y' in trial_data.columns:
            # Compute approximate trajectory by integrating velocity
            # Assuming constant time steps
            dt = 1.0  # Default time step
            if 'timestamp' in trial_data.columns and len(trial_data) > 1:
                # Try to calculate actual time differences
                time_diffs = np.diff(pd.to_datetime(trial_data['timestamp']).values)
                if len(time_diffs) > 0:
                    dt = np.median(time_diffs) / np.timedelta64(1, 's')
            
            # Integrate velocity to get position (starting from origin)
            pos_x = np.cumsum(trial_data['velocity_x'] * dt)
            pos_y = np.cumsum(trial_data['velocity_y'] * dt)
            
            # Center the trajectory (start at origin)
            pos_x = pos_x - pos_x.iloc[0]
            pos_y = pos_y - pos_y.iloc[0]
            
            # Plot computed trajectory
            ax4.plot(pos_x, pos_y, 'b-', linewidth=2, alpha=0.7, label='Computed Trajectory')
            
            # Mark start and end points
            if len(pos_x) > 0:
                ax4.plot(pos_x.iloc[0], pos_y.iloc[0], 'go', markersize=8, label='Start')
                ax4.plot(pos_x.iloc[-1], pos_y.iloc[-1], 'ro', markersize=8, label='End')
            
            # Add target position if available
            if 'target_index' in trial_data.columns and len(trial_data) > 0:
                target_idx = trial_data['target_index'].iloc[0]
                n_targets = trial_data['num_targets'].iloc[0] if 'num_targets' in trial_data.columns else 8
                
                try:
                    target_idx_int = int(target_idx)
                    if 0 <= target_idx_int < n_targets:
                        target_angle = target_idx_int * (2 * np.pi / n_targets)  # Convert to radians
                        # Estimate target distance from trajectory end point
                        trajectory_distance = np.sqrt(pos_x.iloc[-1]**2 + pos_y.iloc[-1]**2)
                        target_distance = max(1.0, trajectory_distance * 1.2)  # A bit beyond trajectory end
                        
                        target_x = target_distance * np.cos(target_angle)
                        target_y = target_distance * np.sin(target_angle)
                        ax4.plot(target_x, target_y, 'rs', markersize=12, 
                                label=f'Target {target_idx_int}')
                except (ValueError, TypeError):
                    pass
            
            # Add center point
            ax4.plot(0, 0, 'ko', markersize=8, label='Center')
            
            # Add velocity vectors at key points
            n_vectors = min(10, len(trial_data))  # Show up to 10 vectors
            if n_vectors > 1:
                indices = np.linspace(0, len(trial_data)-1, n_vectors, dtype=int)
                for i in indices[::2]:  # Every other vector to avoid clutter
                    ax4.arrow(pos_x.iloc[i], pos_y.iloc[i], 
                             trial_data['velocity_x'].iloc[i] * dt * 10,  # Scale for visibility
                             trial_data['velocity_y'].iloc[i] * dt * 10,
                             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
            
            ax4.set_xlabel('X Position (integrated)')
            ax4.set_ylabel('Y Position (integrated)')
            ax4.set_aspect('equal')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
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
        for trial in trials_to_plot:
            trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial]
            trial_durations.append(len(trial_data))
        
        ax1.bar(range(len(trials_to_plot)), trial_durations, alpha=0.7)
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel('Trial Duration (samples)')
        ax1.set_title('Trial Durations')
        ax1.set_xticks(range(len(trials_to_plot)))
        ax1.set_xticklabels([f'T{t}' for t in trials_to_plot])
        ax1.grid(True, alpha=0.3)
        
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
                    dt = 1.0  # Default time step
                    pos_x = np.cumsum(trial_data['velocity_x'] * dt)
                    pos_y = np.cumsum(trial_data['velocity_y'] * dt)
                    
                    # Center trajectory at origin
                    pos_x = pos_x - pos_x.iloc[0]
                    pos_y = pos_y - pos_y.iloc[0]
                    
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