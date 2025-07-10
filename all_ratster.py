"""
All-Channel Raster Plot with Behavioral Data
============================================

This script creates a comprehensive visualization showing:
1. Behavioral measures (velocity x, y, and magnitude) on top
2. Raster plot of neural spikes across all 96 channels below

Usage:
    # Single trial
    python all_ratster.py --trial 1 --h5_file "path/to/data.h5" --save "output.png"
    
    # Grand figure with multiple trials
    python all_ratster.py --grand_figure --trials "1-10" --save "grand_figure.png"

Features:
- Handles all available channels (up to 96)
- Spike detection across all channels
- Combined behavioral and neural visualization
- Grand figure mode for multiple trials (6x wider)
- Configurable parameters
- High-quality output with proper scaling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import sys
import h5py
from typing import Dict, List, Optional, Tuple
from scipy import signal

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from spike_detection import SpikeDetector
from h5_data_loader import H5DataLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default data file path - update this to your H5 file location
DEFAULT_H5_FILE = r"D:\Data\ScienceCorp\trials_aligned.h5"

# Default trial to visualize
DEFAULT_TRIAL = 10
# Spike detection parameters
SAMPLING_RATE = 30000  # Hz
THRESHOLD_FACTOR = 5.0  # Spike detection threshold
SPIKE_WINDOW = (-10, 32)  # samples around spike

# Display parameters
FIGURE_SIZE = (3, 15)  # Large figure for all channels
BEHAVIORAL_HEIGHT_RATIO = 3  # Height ratio for behavioral plot
RASTER_HEIGHT_RATIO = 20  # Height ratio for raster plot
SPIKE_MARKER_SIZE = 1  # Size of spike markers
SPIKE_ALPHA = 1.0  # Transparency of spike markers
CHANNEL_SPACING = 1.0  # Vertical spacing between channels

# Color schemes
BEHAVIORAL_COLORS = {
    'velocity_x': '#2E86AB',  # Blue
    'velocity_y': '#A23B72',  # Red/Pink
    'velocity_magnitude': '#F18F01'  # Orange
}

# Generate colors for channels (using a colormap)
def generate_channel_colors(n_channels: int) -> List[str]:
    """Generate distinct colors for each channel."""
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / n_channels) for i in range(n_channels)]
    return colors

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_trial_data(h5_file_path: str, trial_number: int) -> Optional[Dict]:
    """
    Load trial data from H5 file.
    
    Args:
        h5_file_path: Path to H5 file
        trial_number: Trial number to load
        
    Returns:
        Dictionary containing trial data or None if failed
    """
    print(f"📂 Loading trial {trial_number} from {h5_file_path}")
    
    if not Path(h5_file_path).exists():
        print(f"❌ H5 file not found: {h5_file_path}")
        return None
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            trial_key = f'trial_{trial_number}'
            
            if trial_key not in f:
                print(f"❌ Trial {trial_number} not found in H5 file")
                available_trials = [k for k in f.keys() if k.startswith('trial_')]
                print(f"Available trials: {available_trials}")
                return None
            
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
                print(f"❌ No neural data found. Available keys: {available_keys}")
                return None
            
            # Load behavioral data
            velocity_x = trial_group.get('velocity_x', None)
            velocity_y = trial_group.get('velocity_y', None)
            behavioral_timestamps = trial_group.get('behavioral_timestamps', None)
            
            if velocity_x is not None:
                velocity_x = velocity_x[:]
            if velocity_y is not None:
                velocity_y = velocity_y[:]
            if behavioral_timestamps is not None:
                behavioral_timestamps = behavioral_timestamps[:]
            
            # Load metadata
            metadata = dict(trial_group.attrs)
            duration = metadata.get('duration', neural_data.shape[1] / SAMPLING_RATE)
            
            trial_data = {
                'neural_data': neural_data,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'behavioral_timestamps': behavioral_timestamps,
                'duration': duration,
                'metadata': metadata,
                'trial_number': trial_number
            }
            
            print(f"✅ Trial {trial_number} loaded successfully:")
            print(f"   • Neural data shape: {neural_data.shape}")
            print(f"   • Duration: {duration:.2f} seconds")
            print(f"   • Behavioral data: {'Available' if velocity_x is not None else 'Not available'}")
            
            return trial_data
            
    except Exception as e:
        print(f"❌ Error loading trial data: {e}")
        return None

def detect_spikes_all_channels(neural_data: np.ndarray, 
                             threshold_factor: float = THRESHOLD_FACTOR,
                             use_all_channels: bool = True) -> Dict[int, Dict]:
    """
    Detect spikes across all channels.
    
    Args:
        neural_data: Neural data array (channels x samples)
        threshold_factor: Threshold factor for spike detection
        use_all_channels: If True, use all channels; if False, use only good channels
        
    Returns:
        Dictionary with channel-wise spike data
    """
    n_channels = neural_data.shape[0]
    
    if use_all_channels:
        # Use all available channels
        channels_to_use = list(range(min(n_channels, 96)))  # Limit to 96 channels
    else:
        # Use predefined good channels
        good_channels = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
        channels_to_use = [ch for ch in good_channels if ch < n_channels]
    
    print(f"🔍 Detecting spikes across {len(channels_to_use)} channels...")
    
    # Initialize spike detector
    spike_detector = SpikeDetector(
        sampling_rate=SAMPLING_RATE,
        threshold_factor=threshold_factor,
        spike_window=SPIKE_WINDOW,
        good_channels=channels_to_use
    )
    
    # Detect spikes
    spike_data = spike_detector.detect_spikes_all_channels(neural_data)
    
    # Add empty entries for channels without spikes
    for ch in channels_to_use:
        if ch not in spike_data:
            spike_data[ch] = {
                'spike_times': np.array([]),
                'spike_waveforms': np.array([]).reshape(0, spike_detector.spike_length),
                'n_spikes': 0
            }
    
    total_spikes = sum(data['n_spikes'] for data in spike_data.values())
    print(f"✅ Spike detection complete: {total_spikes} spikes across {len(spike_data)} channels")
    
    return spike_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_all_channel_raster_plot(trial_data: Dict, spike_data: Dict[int, Dict],
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[float, float] = FIGURE_SIZE,
                                  ax_behavior=None, ax_raster=None) -> None:
    """
    Create a comprehensive raster plot with behavioral data for all channels.
    
    Args:
        trial_data: Dictionary containing trial data
        spike_data: Dictionary with channel-wise spike data
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    trial_number = trial_data['trial_number']
    duration = trial_data['duration']
    
    print(f"🎨 Creating all-channel raster plot for trial {trial_number}...")
    
    # Get channel list (sorted)
    channels = sorted(spike_data.keys())
    n_channels = len(channels)
    
    # Create figure with custom grid (only if axes not provided)
    if ax_behavior is None or ax_raster is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[BEHAVIORAL_HEIGHT_RATIO, RASTER_HEIGHT_RATIO],
                              hspace=0.1)
        ax_behavior = fig.add_subplot(gs[0])
        ax_raster = fig.add_subplot(gs[1])
        standalone_plot = True
    else:
        standalone_plot = False
    
    # =========================================================================
    # TOP PANEL: BEHAVIORAL DATA
    # =========================================================================
    
    # Plot behavioral data if available
    if (trial_data['velocity_x'] is not None and 
        trial_data['velocity_y'] is not None):
        
        # Get behavioral data
        vel_x = np.array(trial_data['velocity_x'])
        vel_y = np.array(trial_data['velocity_y'])
        
        # Create time axis for behavioral data
        if trial_data['behavioral_timestamps'] is not None:
            behavior_time = np.array(trial_data['behavioral_timestamps'])
            # Convert to relative time if needed
            if behavior_time[0] > 100:  # Likely absolute timestamps
                behavior_time = behavior_time - behavior_time[0]
        else:
            behavior_time = np.linspace(0, duration, len(vel_x))
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        
        # Plot velocity traces
        ax_behavior.plot(behavior_time, vel_x, 
                        color=BEHAVIORAL_COLORS['velocity_x'], 
                        linewidth=2, label='Velocity X', alpha=0.9)
        ax_behavior.plot(behavior_time, vel_y, 
                        color=BEHAVIORAL_COLORS['velocity_y'], 
                        linewidth=2, label='Velocity Y', alpha=0.9)
        ax_behavior.plot(behavior_time, velocity_magnitude, 
                        color=BEHAVIORAL_COLORS['velocity_magnitude'], 
                        linewidth=2, label='Speed', alpha=0.9)
        
        # Format behavioral plot
        ax_behavior.set_ylabel('Velocity (units/s)', fontsize=12, fontweight='bold')
        ax_behavior.legend(loc='upper right', fontsize=10)
        ax_behavior.grid(True, alpha=0.3)
        
        # Print behavioral data statistics
        print(f"📊 Behavioral data statistics:")
        print(f"   • Velocity X: [{vel_x.min():.2f}, {vel_x.max():.2f}]")
        print(f"   • Velocity Y: [{vel_y.min():.2f}, {vel_y.max():.2f}]")
        print(f"   • Speed: [{velocity_magnitude.min():.2f}, {velocity_magnitude.max():.2f}]")
        
    else:
        # No behavioral data available
        ax_behavior.text(0.5, 0.5, 'No behavioral data available', 
                        transform=ax_behavior.transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.6)
        ax_behavior.set_ylabel('Behavior', fontsize=9, fontweight='bold')
    
    # Set x-axis limits and remove x-axis labels (shared with raster plot)
    ax_behavior.set_xlim(0, duration)
    ax_behavior.set_xticklabels([])
    
    # Add title
    outcome = trial_data['metadata'].get('outcome', 'Unknown')
    ax_behavior.set_title(f'Trial {trial_number} - All Channel Raster Plot (Outcome: {outcome})', 
                         fontsize=9, fontweight='bold', pad=20)
    
    # =========================================================================
    # BOTTOM PANEL: RASTER PLOT
    # =========================================================================
    
    # Plot spikes for each channel
    print(f"🎯 Plotting spikes for {n_channels} channels...")
    
    channel_labels = []
    channel_positions = []
    
    for i, channel in enumerate(channels):
        y_pos = i * CHANNEL_SPACING
        channel_positions.append(y_pos)
        
        # Get spike times in seconds
        spike_times = spike_data[channel]['spike_times'] / SAMPLING_RATE
        n_spikes = len(spike_times)
        
        # Plot vertical lines for each spike
        if n_spikes > 0:
            ax_raster.vlines(spike_times, y_pos - 0.4, y_pos + 0.4, 
                           colors='black', linewidth=SPIKE_MARKER_SIZE, alpha=SPIKE_ALPHA)
        
        # Create channel label
        channel_labels.append(f'Ch{channel:02d} ({n_spikes})')
    
    # Format raster plot
    ax_raster.set_xlim(0, duration)
    ax_raster.set_ylim(-0.5, n_channels * CHANNEL_SPACING - 0.5)
    ax_raster.set_xlabel('Time (seconds)', fontsize=8, fontweight='bold')
    ax_raster.set_ylabel('Channel', fontsize=8, fontweight='bold')
    
    # Set y-axis ticks and labels
    ax_raster.set_yticks(channel_positions)
    ax_raster.set_yticklabels(channel_labels, fontsize=8)
    
    # Add grid for time reference
    ax_raster.grid(True, axis='x', alpha=0.3)
    
    # Add summary statistics
    total_spikes = sum(data['n_spikes'] for data in spike_data.values())
    active_channels = sum(1 for data in spike_data.values() if data['n_spikes'] > 0)
    avg_firing_rate = total_spikes / (duration * n_channels) if duration > 0 else 0
    
    # Add statistics text
    stats_text = (f"Channels: {n_channels} | Active: {active_channels} | "
                 f"Total spikes: {total_spikes} | Avg rate: {avg_firing_rate:.1f} Hz")
    
    ax_raster.text(0.02, 0.98, stats_text, transform=ax_raster.transAxes, 
                  fontsize=10, verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Tight layout and save/show only for standalone plots
    if standalone_plot:
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        else:
            plt.show()
        
        print(f"✅ All-channel raster plot complete!")

def create_grand_multi_trial_figure(h5_file_path: str, trial_numbers: List[int],
                                  threshold_factor: float = THRESHOLD_FACTOR,
                                  use_all_channels: bool = True,
                                  save_path: Optional[str] = None) -> None:
    """
    Create a grand figure showing multiple trials in a grid layout.
    
    Args:
        h5_file_path: Path to H5 file
        trial_numbers: List of trial numbers to plot
        threshold_factor: Threshold factor for spike detection
        use_all_channels: Whether to use all channels or just good ones
        save_path: Optional path to save the plot
    """
    n_trials = len(trial_numbers)
    
    # Calculate grid dimensions (prefer more columns than rows)
    if n_trials <= 5:
        n_cols = n_trials
        n_rows = 1
    elif n_trials <= 10:
        n_cols = 5
        n_rows = 2
    else:
        n_cols = int(np.ceil(np.sqrt(n_trials)))
        n_rows = int(np.ceil(n_trials / n_cols))
    
    # Create grand figure (6 times wider)
    grand_figsize = (FIGURE_SIZE[0] * 6, FIGURE_SIZE[1])
    fig = plt.figure(figsize=grand_figsize)
    
    print(f"🎨 Creating grand figure with {n_trials} trials ({n_rows}x{n_cols} grid)")
    print(f"📐 Figure size: {grand_figsize}")
    
    # Create main gridspec for overall layout
    gs_main = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Create subplots for each trial
    for idx, trial_num in enumerate(trial_numbers):
        print(f"\n📊 Processing trial {trial_num} ({idx+1}/{n_trials})...")
        
        # Load trial data
        trial_data = load_trial_data(h5_file_path, trial_num)
        if trial_data is None:
            print(f"⚠️  Skipping trial {trial_num} - failed to load")
            continue
        
        # Detect spikes
        spike_data = detect_spikes_all_channels(
            trial_data['neural_data'], 
            threshold_factor=threshold_factor,
            use_all_channels=use_all_channels
        )
        
        # Create subplot grid for this trial (behavioral + raster)
        gs_trial = gridspec.GridSpecFromSubplotSpec(
            2, 1, 
            subplot_spec=gs_main[idx],
            height_ratios=[BEHAVIORAL_HEIGHT_RATIO, RASTER_HEIGHT_RATIO],
            hspace=0.1
        )
        
        ax_behavior = fig.add_subplot(gs_trial[0])
        ax_raster = fig.add_subplot(gs_trial[1])
        
        # Create the plot for this trial
        create_all_channel_raster_plot(
            trial_data, 
            spike_data,
            ax_behavior=ax_behavior,
            ax_raster=ax_raster
        )
        
        # Adjust font sizes for smaller subplots
        ax_behavior.tick_params(labelsize=6)
        ax_raster.tick_params(labelsize=6)
        
        # Adjust title font size
        ax_behavior.set_title(f'Trial {trial_num}', fontsize=8, fontweight='bold')
        
        # Remove some labels to reduce clutter
        if idx % n_cols != 0:  # Not leftmost column
            ax_behavior.set_ylabel('')
            ax_raster.set_ylabel('')
        
        if idx < n_trials - n_cols:  # Not bottom row
            ax_raster.set_xlabel('')
    
    # Add overall title
    fig.suptitle('Multi-Trial Neural Activity Raster Plots', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Grand figure saved to: {save_path}")
    else:
        plt.show()
    
    print(f"✅ Grand multi-trial figure complete!")

def print_spike_summary(spike_data: Dict[int, Dict]) -> None:
    """Print a summary of spike detection results."""
    print(f"\n📈 SPIKE DETECTION SUMMARY")
    print("=" * 50)
    
    channels = sorted(spike_data.keys())
    total_spikes = sum(data['n_spikes'] for data in spike_data.values())
    active_channels = [ch for ch in channels if spike_data[ch]['n_spikes'] > 0]
    
    print(f"• Total channels analyzed: {len(channels)}")
    print(f"• Active channels (with spikes): {len(active_channels)}")
    print(f"• Total spikes detected: {total_spikes}")
    
    if len(active_channels) > 0:
        avg_spikes = total_spikes / len(active_channels)
        print(f"• Average spikes per active channel: {avg_spikes:.1f}")
        
        # Show top 10 channels
        channel_spike_counts = [(ch, spike_data[ch]['n_spikes']) for ch in channels]
        top_channels = sorted(channel_spike_counts, key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 CHANNELS:")
        for i, (ch, count) in enumerate(top_channels, 1):
            if count > 0:
                print(f"   {i:2d}. Channel {ch:2d}: {count:3d} spikes")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to create all-channel raster plot."""
    parser = argparse.ArgumentParser(
        description='Create all-channel raster plot with behavioral data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--trial', type=int, default=DEFAULT_TRIAL,
                       help=f'Trial number to visualize (default: {DEFAULT_TRIAL})')
    parser.add_argument('--h5_file', type=str, default=DEFAULT_H5_FILE,
                       help='Path to H5 data file')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the plot (if not specified, shows plot)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_FACTOR,
                       help=f'Spike detection threshold factor (default: {THRESHOLD_FACTOR})')
    parser.add_argument('--good_channels_only', action='store_true',
                       help='Use only the predefined good channels (21 channels) for spike detection')
    parser.add_argument('--max_channels', type=int, default=96,
                       help='Maximum number of channels to plot (default: 96)')
    parser.add_argument('--grand_figure', action='store_true',
                       help='Create grand figure with multiple trials')
    parser.add_argument('--trials', type=str, default='1-10',
                       help='Trial range for grand figure (e.g., "1-10" or "1,3,5,7") (default: "1-10")')
    
    args = parser.parse_args()
    
    print(f"🧠 ALL-CHANNEL RASTER PLOT GENERATOR")
    print("=" * 50)
    print(f"H5 File: {args.h5_file}")
    print(f"Threshold: {args.threshold}x")
    print(f"Use all channels: {not args.good_channels_only}")
    print(f"Max channels: {args.max_channels}")
    print(f"Grand figure: {args.grand_figure}")
    
    if args.grand_figure:
        # Parse trial range
        if '-' in args.trials:
            # Range format: "1-10"
            start, end = map(int, args.trials.split('-'))
            trial_numbers = list(range(start, end + 1))
        else:
            # List format: "1,3,5,7"
            trial_numbers = [int(x.strip()) for x in args.trials.split(',')]
        
        print(f"Trials: {trial_numbers}")
        
        # Create grand figure
        create_grand_multi_trial_figure(
            args.h5_file,
            trial_numbers,
            threshold_factor=args.threshold,
            use_all_channels=not args.good_channels_only,
            save_path=args.save
        )
    else:
        # Single trial mode
        print(f"Trial: {args.trial}")
        
        # Load trial data
        trial_data = load_trial_data(args.h5_file, args.trial)
        if trial_data is None:
            return
        
        # Detect spikes
        spike_data = detect_spikes_all_channels(
            trial_data['neural_data'], 
            threshold_factor=args.threshold,
            use_all_channels=not args.good_channels_only
        )
        
        # Limit channels if requested
        if len(spike_data) > args.max_channels:
            # Keep top channels by spike count
            channels_by_spikes = sorted(spike_data.keys(), 
                                      key=lambda ch: spike_data[ch]['n_spikes'], 
                                      reverse=True)
            selected_channels = channels_by_spikes[:args.max_channels]
            spike_data = {ch: spike_data[ch] for ch in selected_channels}
            print(f"⚠️  Limited to top {args.max_channels} channels by spike count")
        
        # Print summary
        print_spike_summary(spike_data)
        
        # Create plot
        create_all_channel_raster_plot(
            trial_data, 
            spike_data, 
            save_path=args.save,
            figsize=FIGURE_SIZE
        )

if __name__ == "__main__":
    main() 