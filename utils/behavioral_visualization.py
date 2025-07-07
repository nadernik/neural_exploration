"""
Behavioral visualization utilities for center-out task analysis.

This module provides specialized visualization functions for behavioral features including:
- Cursor trajectory plots (overlaid and by target direction)
- Speed profiles and velocity analysis
- Timing metrics visualization (reaction time, movement time)
- Accuracy and error analysis
- Success rate and performance metrics
- Trial-wise trend analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Define colors for different targets
TARGET_COLORS = plt.cm.Set3(np.linspace(0, 1, 8))


class BehavioralPlotter:
    """
    Class for creating behavioral analysis visualizations.
    """
    
    def __init__(self, trial_features: Dict, target_positions: Optional[Dict] = None):
        """
        Initialize the behavioral plotter.
        
        Parameters:
        -----------
        trial_features : dict
            Dictionary of trial features from BehavioralFeatureExtractor
        target_positions : dict, optional
            Dictionary of target positions
        """
        self.trial_features = trial_features
        self.target_positions = target_positions or self._generate_target_positions()
        
    def _generate_target_positions(self, radius=1.0):
        """Generate target positions for center-out task using hardcoded angles."""
        # Hardcoded target angles as specified
        target_angles = {
            0: 90,    # T0: 90°
            1: 45,    # T1: 45°
            2: 0,     # T2: 0°
            3: -45,   # T3: -45°
            4: -90,   # T4: -90°
            5: -135,  # T5: -135°
            6: 180,   # T6: 180°
            7: 135    # T7: 135°
        }
        
        positions = {}
        
        for target_idx, angle_deg in target_angles.items():
            # Convert to radians
            angle_rad = np.deg2rad(angle_deg)
            
            positions[target_idx] = {
                'x': radius * np.cos(angle_rad),
                'y': radius * np.sin(angle_rad),
                'angle': angle_rad,
                'direction': f'{angle_deg}°'
            }
        
        return positions
    
    def plot_cursor_trajectories_overlaid(self, trial_subset: Optional[List[int]] = None, 
                                        color_by_outcome: bool = True, 
                                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot cursor trajectories overlaid on center-out layout.
        
        Parameters:
        -----------
        trial_subset : list, optional
            List of trial numbers to plot. If None, plots all trials.
        color_by_outcome : bool
            If True, colors by trial outcome (win/lose)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if trial_subset is None:
            trial_subset = list(self.trial_features.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot center-out layout
        self._plot_center_out_layout(ax)
        
        # Plot trajectories
        for trial_num in trial_subset:
            if trial_num not in self.trial_features:
                continue
                
            features = self.trial_features[trial_num]
            trajectory = features.get('trajectory_data')
            
            if trajectory is None or len(trajectory) == 0:
                continue
            
            # Determine color
            if color_by_outcome:
                color = 'green' if features['trial_outcome'] == 'win' else 'red'
                alpha = 0.7
            else:
                color = 'blue'
                alpha = 0.5
            
            # Plot trajectory
            ax.plot(trajectory['cursor_x'], trajectory['cursor_y'], 
                   color=color, alpha=alpha, linewidth=1.5)
            
            # Plot start and end points
            ax.scatter(trajectory['cursor_x'].iloc[0], trajectory['cursor_y'].iloc[0], 
                      color='black', s=30, alpha=0.8, zorder=10)
            ax.scatter(trajectory['cursor_x'].iloc[-1], trajectory['cursor_y'].iloc[-1], 
                      color=color, s=20, alpha=0.8, zorder=10)
        
        ax.set_title(f'Cursor Trajectories - Overlaid (n={len(trial_subset)} trials)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(['Start', 'Win', 'Lose'] if color_by_outcome else ['Trajectory'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_trajectories_by_target(self, successful_only: bool = True, 
                                   figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot cursor trajectories grouped by target direction.
        
        Parameters:
        -----------
        successful_only : bool
            If True, only plots successful trials
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Group trials by target
        target_trials = {}
        for trial_num, features in self.trial_features.items():
            if successful_only and features['trial_outcome'] != 'win':
                continue
            
            target_idx = features.get('target_index')
            if target_idx is not None:
                if target_idx not in target_trials:
                    target_trials[target_idx] = []
                target_trials[target_idx].append(trial_num)
        
        # Create subplot grid
        n_targets = len(target_trials)
        cols = 4
        rows = (n_targets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each target
        for i, (target_idx, trials) in enumerate(target_trials.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Plot center-out layout
            self._plot_center_out_layout(ax, highlight_target=target_idx)
            
            # Plot trajectories for this target
            for trial_num in trials:
                features = self.trial_features[trial_num]
                trajectory = features.get('trajectory_data')
                
                if trajectory is None or len(trajectory) == 0:
                    continue
                
                ax.plot(trajectory['cursor_x'], trajectory['cursor_y'], 
                       color=TARGET_COLORS[target_idx % len(TARGET_COLORS)], 
                       alpha=0.7, linewidth=2)
            
            target_info = self.target_positions.get(target_idx, {})
            ax.set_title(f'Target {target_idx} ({target_info.get("direction", "")}) - {len(trials)} trials')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
        
        # Hide unused subplots
        for i in range(n_targets, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_speed_profiles(self, trial_subset: Optional[List[int]] = None, 
                           group_by_target: bool = False, 
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot speed profiles for trials.
        
        Parameters:
        -----------
        trial_subset : list, optional
            List of trial numbers to plot
        group_by_target : bool
            If True, groups by target direction
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if trial_subset is None:
            trial_subset = list(self.trial_features.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if group_by_target:
            # Group by target
            target_trials = {}
            for trial_num in trial_subset:
                if trial_num not in self.trial_features:
                    continue
                target_idx = self.trial_features[trial_num].get('target_index')
                if target_idx is not None:
                    if target_idx not in target_trials:
                        target_trials[target_idx] = []
                    target_trials[target_idx].append(trial_num)
            
            # Plot each target group
            for target_idx, trials in target_trials.items():
                for trial_num in trials:
                    features = self.trial_features[trial_num]
                    speed_profile = features.get('speed_profile')
                    
                    if speed_profile is None:
                        continue
                    
                    ax.plot(speed_profile['time'], speed_profile['speed'], 
                           color=TARGET_COLORS[target_idx % len(TARGET_COLORS)], 
                           alpha=0.6, linewidth=1)
                
                # Add target label
                target_info = self.target_positions.get(target_idx, {})
                ax.plot([], [], color=TARGET_COLORS[target_idx % len(TARGET_COLORS)], 
                       linewidth=2, label=f'Target {target_idx} ({target_info.get("direction", "")})')
        else:
            # Plot all trials
            for trial_num in trial_subset:
                if trial_num not in self.trial_features:
                    continue
                
                features = self.trial_features[trial_num]
                speed_profile = features.get('speed_profile')
                
                if speed_profile is None:
                    continue
                
                color = 'green' if features['trial_outcome'] == 'win' else 'red'
                ax.plot(speed_profile['time'], speed_profile['speed'], 
                       color=color, alpha=0.6, linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (units/s)')
        ax.set_title('Speed Profiles')
        ax.grid(True, alpha=0.3)
        
        if group_by_target:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_timing_histograms(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot histograms of reaction time and movement time.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Collect data
        reaction_times = []
        movement_times = []
        total_times = []
        
        for features in self.trial_features.values():
            if features['reaction_time'] is not None:
                reaction_times.append(features['reaction_time'])
            if features['movement_time'] is not None:
                movement_times.append(features['movement_time'])
            if features['trial_duration'] is not None:
                total_times.append(features['trial_duration'])
        
        # Plot histograms
        if reaction_times:
            axes[0].hist(reaction_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0].set_xlabel('Reaction Time (s)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Reaction Time Distribution\n(n={len(reaction_times)}, mean={np.mean(reaction_times):.3f}s)')
            axes[0].grid(True, alpha=0.3)
        
        if movement_times:
            axes[1].hist(movement_times, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_xlabel('Movement Time (s)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Movement Time Distribution\n(n={len(movement_times)}, mean={np.mean(movement_times):.3f}s)')
            axes[1].grid(True, alpha=0.3)
        
        if total_times:
            axes[2].hist(total_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[2].set_xlabel('Total Trial Duration (s)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'Total Duration Distribution\n(n={len(total_times)}, mean={np.mean(total_times):.3f}s)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_timing_boxplots_by_target(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot boxplots of timing metrics by target direction.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Collect data by target
        target_data = {}
        for features in self.trial_features.values():
            target_idx = features.get('target_index')
            if target_idx is not None:
                if target_idx not in target_data:
                    target_data[target_idx] = {
                        'reaction_times': [],
                        'movement_times': [],
                        'endpoint_errors': [],
                        'max_speeds': []
                    }
                
                if features['reaction_time'] is not None:
                    target_data[target_idx]['reaction_times'].append(features['reaction_time'])
                if features['movement_time'] is not None:
                    target_data[target_idx]['movement_times'].append(features['movement_time'])
                if features['endpoint_error'] is not None:
                    target_data[target_idx]['endpoint_errors'].append(features['endpoint_error'])
                if features['speed_profile'] and 'max_speed' in features['speed_profile']:
                    target_data[target_idx]['max_speeds'].append(features['speed_profile']['max_speed'])
        
        # Create boxplots
        metrics = ['reaction_times', 'movement_times', 'endpoint_errors', 'max_speeds']
        titles = ['Reaction Time by Target', 'Movement Time by Target', 
                 'Endpoint Error by Target', 'Max Speed by Target']
        ylabels = ['Reaction Time (s)', 'Movement Time (s)', 
                  'Endpoint Error (units)', 'Max Speed (units/s)']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = axes[i // 2, i % 2]
            
            # Prepare data for boxplot
            plot_data = []
            target_labels = []
            
            for target_idx in sorted(target_data.keys()):
                data = target_data[target_idx][metric]
                if data:
                    plot_data.append(data)
                    target_info = self.target_positions.get(target_idx, {})
                    target_labels.append(f'T{target_idx}\n{target_info.get("direction", "")}')
            
            if plot_data:
                ax.boxplot(plot_data, labels=target_labels)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_accuracy_heatmap(self, figsize: Tuple[int, int] = (5, 4)) -> plt.Figure:
        """
        Plot accuracy/error heatmap by target direction.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Collect data by target
        target_success = {}
        target_errors = {}
        
        for features in self.trial_features.values():
            target_idx = features.get('target_index')
            if target_idx is not None:
                if target_idx not in target_success:
                    target_success[target_idx] = {'wins': 0, 'total': 0}
                    target_errors[target_idx] = []
                
                target_success[target_idx]['total'] += 1
                if features['trial_outcome'] == 'win':
                    target_success[target_idx]['wins'] += 1
                
                if features['endpoint_error'] is not None:
                    target_errors[target_idx].append(features['endpoint_error'])
        
        # Create success rate heatmap
        success_rates = []
        error_means = []
        target_labels = []
        
        for target_idx in sorted(target_success.keys()):
            success_rate = target_success[target_idx]['wins'] / target_success[target_idx]['total']
            success_rates.append(success_rate)
            
            errors = target_errors[target_idx]
            error_mean = np.mean(errors) if errors else 0
            error_means.append(error_mean)
            
            target_info = self.target_positions.get(target_idx, {})
            target_labels.append(f'T{target_idx} ({target_info.get("direction", "")})')
        
        # Success rate heatmap
        success_matrix = np.array(success_rates).reshape(1, -1)
        im1 = ax1.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(target_labels)))
        ax1.set_xticklabels(target_labels, rotation=45, ha='right')
        ax1.set_yticks([])
        ax1.set_title('Success Rate by Target')
        
        # Add text annotations
        for i, rate in enumerate(success_rates):
            ax1.text(i, 0, f'{rate:.2f}', ha='center', va='center', 
                    color='white' if rate < 0.5 else 'black', fontweight='bold')
        
        # Error heatmap
        error_matrix = np.array(error_means).reshape(1, -1)
        im2 = ax2.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(target_labels)))
        ax2.set_xticklabels(target_labels, rotation=45, ha='right')
        ax2.set_yticks([])
        ax2.set_title('Mean Endpoint Error by Target')
        
        # Add text annotations
        for i, error in enumerate(error_means):
            ax2.text(i, 0, f'{error:.2f}', ha='center', va='center', 
                    color='white' if error > np.max(error_means)/2 else 'black', fontweight='bold')
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Success Rate')
        plt.colorbar(im2, ax=ax2, label='Mean Error')
        
        plt.tight_layout()
        return fig
    
    def plot_success_rate_bars(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot bar chart of success rate by target.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collect data by target
        target_success = {}
        for features in self.trial_features.values():
            target_idx = features.get('target_index')
            if target_idx is not None:
                if target_idx not in target_success:
                    target_success[target_idx] = {'wins': 0, 'total': 0}
                
                target_success[target_idx]['total'] += 1
                if features['trial_outcome'] == 'win':
                    target_success[target_idx]['wins'] += 1
        
        # Create bar chart
        target_indices = sorted(target_success.keys())
        success_rates = []
        trial_counts = []
        
        for target_idx in target_indices:
            success_rate = target_success[target_idx]['wins'] / target_success[target_idx]['total']
            success_rates.append(success_rate)
            trial_counts.append(target_success[target_idx]['total'])
        
        # Create bars
        bars = ax.bar(range(len(target_indices)), success_rates, 
                     color=[TARGET_COLORS[i % len(TARGET_COLORS)] for i in target_indices],
                     alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, rate, count) in enumerate(zip(bars, success_rates, trial_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        target_labels = []
        for target_idx in target_indices:
            target_info = self.target_positions.get(target_idx, {})
            target_labels.append(f'T{target_idx}\n{target_info.get("direction", "")}')
        
        ax.set_xticks(range(len(target_indices)))
        ax.set_xticklabels(target_labels)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Target Direction')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_trial_trends(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot trial-wise trends over the session.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Collect data ordered by trial number
        trial_nums = sorted(self.trial_features.keys())
        
        reaction_times = []
        movement_times = []
        endpoint_errors = []
        success_outcomes = []
        
        for trial_num in trial_nums:
            features = self.trial_features[trial_num]
            
            reaction_times.append(features['reaction_time'])
            movement_times.append(features['movement_time'])
            endpoint_errors.append(features['endpoint_error'])
            success_outcomes.append(1 if features['trial_outcome'] == 'win' else 0)
        
        # Plot trends
        # Reaction time trend
        valid_rt = [(t, rt) for t, rt in zip(trial_nums, reaction_times) if rt is not None]
        if valid_rt:
            t_rt, rt_vals = zip(*valid_rt)
            axes[0, 0].plot(t_rt, rt_vals, 'o-', alpha=0.7, markersize=3)
            axes[0, 0].set_xlabel('Trial Number')
            axes[0, 0].set_ylabel('Reaction Time (s)')
            axes[0, 0].set_title('Reaction Time Trend')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Movement time trend
        valid_mt = [(t, mt) for t, mt in zip(trial_nums, movement_times) if mt is not None]
        if valid_mt:
            t_mt, mt_vals = zip(*valid_mt)
            axes[0, 1].plot(t_mt, mt_vals, 'o-', alpha=0.7, markersize=3, color='green')
            axes[0, 1].set_xlabel('Trial Number')
            axes[0, 1].set_ylabel('Movement Time (s)')
            axes[0, 1].set_title('Movement Time Trend')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Endpoint error trend
        valid_ee = [(t, ee) for t, ee in zip(trial_nums, endpoint_errors) if ee is not None]
        if valid_ee:
            t_ee, ee_vals = zip(*valid_ee)
            axes[1, 0].plot(t_ee, ee_vals, 'o-', alpha=0.7, markersize=3, color='red')
            axes[1, 0].set_xlabel('Trial Number')
            axes[1, 0].set_ylabel('Endpoint Error')
            axes[1, 0].set_title('Endpoint Error Trend')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate trend (moving average)
        window_size = 10
        if len(success_outcomes) >= window_size:
            success_ma = pd.Series(success_outcomes).rolling(window=window_size, center=True).mean()
            axes[1, 1].plot(trial_nums, success_ma, 'o-', alpha=0.7, markersize=3, color='purple')
            axes[1, 1].set_xlabel('Trial Number')
            axes[1, 1].set_ylabel('Success Rate (Moving Average)')
            axes[1, 1].set_title(f'Success Rate Trend (window={window_size})')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_center_out_layout(self, ax, highlight_target: Optional[int] = None):
        """
        Plot the center-out task layout on given axes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        highlight_target : int, optional
            Target index to highlight
        """
        # Get dynamic sizing
        x_min, x_max, y_min, y_max = self._get_plot_limits()
        target_size = self._get_target_circle_size()
        center_size = target_size * 1.2  # Center slightly larger
        
        # Plot center
        center = Circle((0, 0), center_size, color='red', alpha=0.5)
        ax.add_patch(center)
        
        # Plot targets
        for target_idx, target_info in self.target_positions.items():
            color = 'orange' if target_idx == highlight_target else 'blue'
            alpha = 0.8 if target_idx == highlight_target else 0.3
            
            target = Circle((target_info['x'], target_info['y']), target_size, 
                          color=color, alpha=alpha)
            ax.add_patch(target)
            
            # Add target label - position relative to target distance from center
            target_distance = np.sqrt(target_info['x']**2 + target_info['y']**2)
            label_scale = 1.2 if target_distance > 0.1 else 2.0  # Scale label distance
            
            ax.text(target_info['x'] * label_scale, target_info['y'] * label_scale, 
                   f'T{target_idx}', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Use dynamic limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def plot_correct_trials_overlay(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot overlay of all correct trials showing center-out design.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot center-out layout (this now uses dynamic sizing)
        self._plot_center_out_layout(ax)
        
        # Plot all successful trials in same color
        successful_trials = [trial_num for trial_num, features in self.trial_features.items() 
                           if features['trial_outcome'] == 'win']
        
        for trial_num in successful_trials:
            features = self.trial_features[trial_num]
            trajectory = features.get('trajectory_data')
            
            if trajectory is None or len(trajectory) == 0:
                continue
            
            # Plot trajectory in same color to show center-out pattern
            ax.plot(trajectory['cursor_x'], trajectory['cursor_y'], 
                   color='green', alpha=0.6, linewidth=2)
        
        ax.set_title(f'Correct Trials Overlay - Center-Out Design\n({len(successful_trials)} successful trials)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    

    
    def set_plot_scaling(self, padding_factor: Optional[float] = None, 
                        target_circle_size: Optional[float] = None):
        """
        Manually adjust plot scaling parameters.
        
        Parameters:
        -----------
        padding_factor : float, optional
            Fraction of range to add as padding (default: auto-calculated)
        target_circle_size : float, optional
            Size of target circles (default: auto-calculated)
        """
        if padding_factor is not None:
            self._custom_padding = padding_factor
        else:
            self._custom_padding = None
            
        if target_circle_size is not None:
            self._custom_circle_size = target_circle_size
        else:
            self._custom_circle_size = None
            
        print(f"📏 Plot scaling updated:")
        if padding_factor is not None:
            print(f"  - Custom padding factor: {padding_factor}")
        if target_circle_size is not None:
            print(f"  - Custom circle size: {target_circle_size}")
        if padding_factor is None and target_circle_size is None:
            print(f"  - Reset to automatic scaling")
    
    def _get_plot_limits(self, padding_factor: float = 0.3) -> Tuple[float, float, float, float]:
        """
        Calculate appropriate plot limits based on actual target positions.
        
        Parameters:
        -----------
        padding_factor : float
            Fraction of range to add as padding (default: 0.3 = 30% padding)
            
        Returns:
        --------
        tuple
            (x_min, x_max, y_min, y_max) for plot limits
        """
        # Use custom padding if set
        if hasattr(self, '_custom_padding') and self._custom_padding is not None:
            padding_factor = self._custom_padding
            
        if not self.target_positions:
            # Fallback to default if no target positions
            return -1.5, 1.5, -1.5, 1.5
        
        # Get all target coordinates
        x_coords = [pos['x'] for pos in self.target_positions.values()]
        y_coords = [pos['y'] for pos in self.target_positions.values()]
        
        # Add center point
        x_coords.append(0.0)
        y_coords.append(0.0)
        
        # Calculate ranges
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure minimum range for visibility
        min_range = 0.5
        if x_range < min_range:
            x_center = (x_max + x_min) / 2
            x_min = x_center - min_range / 2
            x_max = x_center + min_range / 2
            x_range = min_range
        if y_range < min_range:
            y_center = (y_max + y_min) / 2
            y_min = y_center - min_range / 2
            y_max = y_center + min_range / 2
            y_range = min_range
        
        # Apply padding
        x_padding = x_range * padding_factor
        y_padding = y_range * padding_factor
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        return x_min, x_max, y_min, y_max
    
    def _get_target_circle_size(self) -> float:
        """
        Calculate appropriate target circle size based on target spacing.
        
        Returns:
        --------
        float
            Radius for target circles
        """
        # Use custom size if set
        if hasattr(self, '_custom_circle_size') and self._custom_circle_size is not None:
            return self._custom_circle_size
            
        if not self.target_positions:
            return 0.08  # Default size
        
        # Calculate average distance between adjacent targets
        positions = list(self.target_positions.values())
        if len(positions) < 2:
            return 0.08
        
        # Get distances from center
        distances = [np.sqrt(pos['x']**2 + pos['y']**2) for pos in positions]
        avg_distance = np.mean(distances)
        
        # Scale circle size to ~8% of average distance
        circle_size = avg_distance * 0.08
        
        # Ensure reasonable bounds
        return max(0.02, min(0.15, circle_size)) 