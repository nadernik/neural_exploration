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