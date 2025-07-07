"""
Neural Spike Raster Plot Viewer
==============================

This script creates a raster plot showing detected spikes for all neurons in a given trial.
Uses vertical lines to display spike times across all channels.

Usage:
    python view_trial_raster.py

Configuration:
    - Modify TRIAL_NUMBER to view different trials
    - Modify H5_FILE_PATH to use different data files
    - Modify display parameters for customization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from spike_detection import SpikeDetector
from h5_data_loader import H5DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data file path - update this to your H5 file location
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"

# Trial to visualize
TRIAL_NUMBER = 1

# Spike detection parameters
SAMPLING_RATE = 30000  # Hz
THRESHOLD_FACTOR = 5.0  # Spike detection threshold
SPIKE_WINDOW = (-10, 32)  # samples around spike
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]

# Display parameters
FIGURE_SIZE = (15, 10)
COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
SPIKE_LINE_WIDTH = 0.5
CHANNEL_SPACING = 1.0  # Vertical spacing between channels
MAX_CHANNELS_TO_DISPLAY = 20  # Maximum channels to display (for readability)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_trial_data(h5_file_path: str, trial_number: int) -> Dict:
    """
    Load trial data from H5 file.
    
    Args:
        h5_file_path: Path to H5 file
        trial_number: Trial number to load
        
    Returns:
        Dictionary containing trial data
    """
    print(f"📂 Loading trial {trial_number} from {h5_file_path}")
    
    try:
        loader = H5DataLoader(h5_file_path)
        trial_data = loader.load_trial_data(trial_number)
        
        print(f"✅ Trial {trial_number} loaded successfully:")
        print(f"   • Neural data shape: {trial_data['neural_data'].shape}")
        print(f"   • Duration: {trial_data.get('duration', 'Unknown')} seconds")
        print(f"   • Outcome: {trial_data.get('outcome', 'Unknown')}")
        
        return trial_data
        
    except Exception as e:
        print(f"❌ Error loading trial data: {e}")
        return None


def detect_spikes(neural_data: np.ndarray, channels: List[int]) -> Dict[int, Dict]:
    """
    Detect spikes in neural data for specified channels.
    
    Args:
        neural_data: Neural data array (channels x samples)
        channels: List of channel indices to process
        
    Returns:
        Dictionary with spike data for each channel
    """
    print(f"🔍 Detecting spikes on {len(channels)} channels...")
    
    detector = SpikeDetector(
        sampling_rate=SAMPLING_RATE,
        threshold_factor=THRESHOLD_FACTOR,
        spike_window=SPIKE_WINDOW,
        good_channels=channels
    )
    
    spike_data = detector.detect_spikes_all_channels(neural_data)
    
    total_spikes = sum(data['n_spikes'] for data in spike_data.values())
    print(f"✅ Spike detection complete: {total_spikes} total spikes detected")
    
    return spike_data


def create_raster_plot(spike_data: Dict[int, Dict], trial_duration: float, 
                      trial_number: int, trial_data: Optional[Dict] = None, 
                      save_path: Optional[str] = None) -> None:
    """
    Create a raster plot showing spike times for all channels with behavioral data on top.
    
    Args:
        spike_data: Dictionary with spike data for each channel
        trial_duration: Duration of trial in seconds
        trial_number: Trial number for title
        trial_data: Optional trial data containing behavioral information
        save_path: Optional path to save the plot
    """
    print(f"📊 Creating raster plot with behavior for trial {trial_number}...")
    
    # Get channels with spikes
    channels_with_spikes = [ch for ch, data in spike_data.items() if data['n_spikes'] > 0]
    
    if not channels_with_spikes:
        print("⚠️  No spikes detected in any channel!")
        return
    
    # Limit channels for display readability
    display_channels = channels_with_spikes[:MAX_CHANNELS_TO_DISPLAY]
    if len(channels_with_spikes) > MAX_CHANNELS_TO_DISPLAY:
        print(f"⚠️  Showing only first {MAX_CHANNELS_TO_DISPLAY} channels with spikes")
    
    # Create figure with two subplots (behavior:raster = 1:20 ratio)
    fig, (ax_behavior, ax_raster) = plt.subplots(2, 1, figsize=FIGURE_SIZE, 
                                                 height_ratios=[1, 20], 
                                                 sharex=True)
    
    # Plot behavioral data in top subplot
    if trial_data is not None and 'velocity_x' in trial_data and 'velocity_y' in trial_data:
        # Get behavioral data
        vel_x = np.array(trial_data['velocity_x'])
        vel_y = np.array(trial_data['velocity_y'])
        
        print(f"📊 Behavioral data found:")
        print(f"   • Velocity X: {len(vel_x)} samples, range: [{vel_x.min():.3f}, {vel_x.max():.3f}]")
        print(f"   • Velocity Y: {len(vel_y)} samples, range: [{vel_y.min():.3f}, {vel_y.max():.3f}]")
        
        # Create time axis for behavioral data
        if 'behavioral_timestamps' in trial_data:
            behavior_time = np.array(trial_data['behavioral_timestamps'])
            # Convert to relative time if needed
            if behavior_time[0] > 100:  # Likely absolute timestamps
                behavior_time = behavior_time - behavior_time[0]
            print(f"   • Using behavioral_timestamps: {len(behavior_time)} samples")
        else:
            behavior_time = np.linspace(0, trial_duration, len(vel_x))
            print(f"   • Generated time axis: {len(behavior_time)} samples")
        
        print(f"   • Time range: [{behavior_time.min():.3f}, {behavior_time.max():.3f}] seconds")
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        print(f"   • Speed range: [{velocity_magnitude.min():.3f}, {velocity_magnitude.max():.3f}]")
        
        # Check if data has meaningful values
        if np.all(np.abs(vel_x) < 1e-6) and np.all(np.abs(vel_y) < 1e-6):
            print("⚠️  Warning: Behavioral data appears to be all zeros or very small values")
        
        # Plot velocity traces with better visibility
        line1 = ax_behavior.plot(behavior_time, vel_x, 'b-', linewidth=2, label='Vel X', alpha=0.8)
        line2 = ax_behavior.plot(behavior_time, vel_y, 'r-', linewidth=2, label='Vel Y', alpha=0.8)
        line3 = ax_behavior.plot(behavior_time, velocity_magnitude, 'k-', linewidth=2.5, label='Speed', alpha=0.9)
        
        # Format behavior plot with better scaling
        ax_behavior.set_ylabel('Velocity', fontsize=10)
        ax_behavior.set_xlim(0, trial_duration)
        
        # Set y-limits to show the actual data range
        all_data = np.concatenate([vel_x, vel_y, velocity_magnitude])
        if not np.all(all_data == 0):
            y_min, y_max = np.min(all_data), np.max(all_data)
            y_range = y_max - y_min
            if y_range > 0:
                ax_behavior.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ax_behavior.set_ylim(-0.1, 0.1)
        else:
            ax_behavior.set_ylim(-0.1, 0.1)
            ax_behavior.text(0.5, 0.5, 'Behavioral data is all zeros', 
                           transform=ax_behavior.transAxes, ha='center', va='center',
                           fontsize=10, alpha=0.7)
        
        ax_behavior.grid(True, alpha=0.3)
        ax_behavior.legend(loc='upper right', fontsize=8)
        ax_behavior.set_title(f'Trial {trial_number} - Behavior & Neural Activity', fontsize=14, fontweight='bold')
        
        # Remove x-axis labels (shared with raster plot)
        ax_behavior.set_xticklabels([])
        
        print(f"✅ Behavioral plot created with y-range: {ax_behavior.get_ylim()}")
        
    else:
        # Check what data is available
        available_keys = list(trial_data.keys()) if trial_data else []
        print(f"⚠️  Behavioral data not found. Available keys: {available_keys}")
        
        # No behavioral data available - just show a placeholder
        ax_behavior.text(0.5, 0.5, 'No behavioral data available', 
                        transform=ax_behavior.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
        ax_behavior.set_xlim(0, trial_duration)
        ax_behavior.set_ylim(0, 1)
        ax_behavior.set_ylabel('Behavior', fontsize=10)
        ax_behavior.set_title(f'Trial {trial_number} - Neural Activity', fontsize=14, fontweight='bold')
        ax_behavior.set_xticklabels([])
    
    # Plot spike raster in bottom subplot
    y_positions = {}
    for i, channel in enumerate(display_channels):
        y_pos = i * CHANNEL_SPACING
        y_positions[channel] = y_pos
        
        # Get spike times in seconds
        spike_times = spike_data[channel]['spike_times'] / SAMPLING_RATE
        n_spikes = len(spike_times)
        
        # Plot vertical lines for each spike
        color = COLORS[i % len(COLORS)]
        ax_raster.vlines(spike_times, y_pos - 0.4, y_pos + 0.4, 
                        colors=color, linewidth=SPIKE_LINE_WIDTH, alpha=0.7)
        
        # Add channel label
        ax_raster.text(-0.05 * trial_duration, y_pos, f'Ch {channel}\n({n_spikes})', 
                      ha='right', va='center', fontsize=8)
    
    # Format raster plot
    ax_raster.set_xlim(0, trial_duration)
    ax_raster.set_ylim(-0.5, len(display_channels) * CHANNEL_SPACING - 0.5)
    ax_raster.set_xlabel('Time (seconds)', fontsize=12)
    ax_raster.set_ylabel('Channel', fontsize=12)
    
    # Remove y-axis ticks (we have custom labels)
    ax_raster.set_yticks([])
    
    # Add grid for time reference
    ax_raster.grid(True, axis='x', alpha=0.3)
    
    # Add summary statistics
    total_spikes = sum(spike_data[ch]['n_spikes'] for ch in display_channels)
    mean_rate = total_spikes / trial_duration / len(display_channels)
    
    ax_raster.text(0.02, 0.98, f'Avg Rate: {mean_rate:.1f} Hz\nThreshold: {THRESHOLD_FACTOR}x', 
                  transform=ax_raster.transAxes, va='top', ha='left', 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    else:
        plt.show()
    
    print(f"✅ Raster plot created successfully!")


def print_spike_summary(spike_data: Dict[int, Dict]) -> None:
    """
    Print summary statistics of detected spikes.
    
    Args:
        spike_data: Dictionary with spike data for each channel
    """
    print(f"\n📈 SPIKE DETECTION SUMMARY:")
    print(f"=" * 50)
    
    total_spikes = 0
    active_channels = 0
    
    for channel, data in spike_data.items():
        n_spikes = data['n_spikes']
        if n_spikes > 0:
            active_channels += 1
            total_spikes += n_spikes
    
    print(f"• Total channels analyzed: {len(spike_data)}")
    print(f"• Active channels (with spikes): {active_channels}")
    print(f"• Total spikes detected: {total_spikes}")
    
    if active_channels > 0:
        avg_spikes_per_channel = total_spikes / active_channels
        print(f"• Average spikes per active channel: {avg_spikes_per_channel:.1f}")
    
    # Show top channels
    top_channels = sorted(spike_data.items(), key=lambda x: x[1]['n_spikes'], reverse=True)[:10]
    print(f"\n🏆 TOP 10 CHANNELS:")
    for i, (channel, data) in enumerate(top_channels):
        if data['n_spikes'] > 0:
            print(f"  {i+1:2d}. Channel {channel:2d}: {data['n_spikes']:3d} spikes")


def main():
    """Main function to create spike raster plot."""
    # Declare global variables at the top
    global TRIAL_NUMBER, H5_FILE_PATH, THRESHOLD_FACTOR
    
    parser = argparse.ArgumentParser(description='View neural spike raster plot')
    parser.add_argument('--trial', type=int, default=TRIAL_NUMBER,
                       help=f'Trial number to visualize (default: {TRIAL_NUMBER})')
    parser.add_argument('--h5_file', type=str, default=H5_FILE_PATH,
                       help='Path to H5 data file')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the plot (if not specified, shows plot)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_FACTOR,
                       help=f'Spike detection threshold factor (default: {THRESHOLD_FACTOR})')
    
    args = parser.parse_args()
    
    # Update global parameters
    TRIAL_NUMBER = args.trial
    H5_FILE_PATH = args.h5_file
    THRESHOLD_FACTOR = args.threshold
    
    print(f"🧠 NEURAL SPIKE RASTER VIEWER")
    print(f"=" * 50)
    print(f"Trial: {TRIAL_NUMBER}")
    print(f"H5 File: {H5_FILE_PATH}")
    print(f"Threshold: {THRESHOLD_FACTOR}x")
    print(f"Channels: {len(SPIKE_CHANNELS)}")
    
    # Check if file exists
    if not Path(H5_FILE_PATH).exists():
        print(f"❌ H5 file not found: {H5_FILE_PATH}")
        print(f"Please update the H5_FILE_PATH variable or use --h5_file argument")
        return
    
    # Load trial data
    trial_data = load_trial_data(H5_FILE_PATH, TRIAL_NUMBER)
    if trial_data is None:
        return
    
    # Detect spikes
    spike_data = detect_spikes(trial_data['neural_data'], SPIKE_CHANNELS)
    
    # Print summary
    print_spike_summary(spike_data)
    
    # Create raster plot
    trial_duration = trial_data.get('duration', 
                                   trial_data['neural_data'].shape[1] / SAMPLING_RATE)
    
    create_raster_plot(spike_data, trial_duration, TRIAL_NUMBER, trial_data, args.save)


if __name__ == "__main__":
    main() 