"""
Example Usage of Neural Spike Raster Plot Functions
==================================================

This script demonstrates how to use the functions from view_trial_raster.py
in other workflows or for batch processing multiple trials.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from view_trial_raster import load_trial_data, detect_spikes, create_raster_plot, print_spike_summary

# Configuration
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]


def analyze_single_trial(trial_number: int, save_plot: bool = False):
    """
    Analyze a single trial and create raster plot.
    
    Args:
        trial_number: Trial number to analyze
        save_plot: Whether to save the plot to file
    """
    print(f"\n🔍 Analyzing Trial {trial_number}")
    print("=" * 40)
    
    # Load trial data
    trial_data = load_trial_data(H5_FILE_PATH, trial_number)
    if trial_data is None:
        print(f"❌ Failed to load trial {trial_number}")
        return None
    
    # Detect spikes
    spike_data = detect_spikes(trial_data['neural_data'], SPIKE_CHANNELS)
    
    # Print summary
    print_spike_summary(spike_data)
    
    # Create raster plot
    trial_duration = trial_data.get('duration', 
                                   trial_data['neural_data'].shape[1] / 30000)
    
    save_path = f"trial_{trial_number}_raster.png" if save_plot else None
    create_raster_plot(spike_data, trial_duration, trial_number, trial_data, save_path)
    
    return spike_data


def compare_multiple_trials(trial_numbers: list, save_plots: bool = False):
    """
    Compare spike activity across multiple trials.
    
    Args:
        trial_numbers: List of trial numbers to compare
        save_plots: Whether to save individual plots
    """
    print(f"\n🔍 COMPARING MULTIPLE TRIALS: {trial_numbers}")
    print("=" * 60)
    
    trial_summaries = []
    
    for trial_num in trial_numbers:
        print(f"\n--- Processing Trial {trial_num} ---")
        
        # Load and analyze trial
        trial_data = load_trial_data(H5_FILE_PATH, trial_num)
        if trial_data is None:
            continue
            
        spike_data = detect_spikes(trial_data['neural_data'], SPIKE_CHANNELS)
        
        # Calculate summary statistics
        total_spikes = sum(data['n_spikes'] for data in spike_data.values())
        active_channels = sum(1 for data in spike_data.values() if data['n_spikes'] > 0)
        duration = trial_data.get('duration', trial_data['neural_data'].shape[1] / 30000)
        avg_rate = total_spikes / duration / len(SPIKE_CHANNELS)
        
        trial_summaries.append({
            'trial': trial_num,
            'total_spikes': total_spikes,
            'active_channels': active_channels,
            'duration': duration,
            'avg_rate': avg_rate,
            'outcome': trial_data.get('outcome', 'Unknown')
        })
        
        # Create individual raster plot
        if save_plots:
            save_path = f"trial_{trial_num}_comparison_raster.png"
            create_raster_plot(spike_data, duration, trial_num, trial_data, save_path)
    
    # Print comparison summary
    print(f"\n📊 TRIAL COMPARISON SUMMARY:")
    print("=" * 60)
    print(f"{'Trial':<6} {'Spikes':<7} {'Active Ch':<9} {'Duration':<9} {'Avg Rate':<9} {'Outcome':<8}")
    print("-" * 60)
    
    for summary in trial_summaries:
        print(f"{summary['trial']:<6} {summary['total_spikes']:<7} "
              f"{summary['active_channels']:<9} {summary['duration']:<9.1f} "
              f"{summary['avg_rate']:<9.1f} {summary['outcome']:<8}")
    
    return trial_summaries


def find_trials_with_high_activity(threshold_spikes: int = 200, max_trials: int = 20):
    """
    Find trials with spike activity above a threshold.
    
    Args:
        threshold_spikes: Minimum number of spikes to consider "high activity"
        max_trials: Maximum number of trials to check
        
    Returns:
        List of trial numbers with high spike activity
    """
    print(f"\n🔍 FINDING TRIALS WITH HIGH ACTIVITY (>{threshold_spikes} spikes)")
    print("=" * 60)
    
    high_activity_trials = []
    
    for trial_num in range(1, max_trials + 1):
        print(f"Checking trial {trial_num}...", end=" ")
        
        try:
            # Load trial data
            trial_data = load_trial_data(H5_FILE_PATH, trial_num)
            if trial_data is None:
                print("❌ Failed to load")
                continue
                
            # Detect spikes
            spike_data = detect_spikes(trial_data['neural_data'], SPIKE_CHANNELS)
            
            # Count total spikes
            total_spikes = sum(data['n_spikes'] for data in spike_data.values())
            
            if total_spikes >= threshold_spikes:
                high_activity_trials.append(trial_num)
                print(f"✅ {total_spikes} spikes")
            else:
                print(f"⚪ {total_spikes} spikes")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print(f"\n🏆 FOUND {len(high_activity_trials)} HIGH-ACTIVITY TRIALS:")
    print(f"Trials: {high_activity_trials}")
    
    return high_activity_trials


def batch_generate_raster_plots(trial_numbers: list, output_dir: str = "raster_plots"):
    """
    Generate raster plots for multiple trials and save to directory.
    
    Args:
        trial_numbers: List of trial numbers to process
        output_dir: Directory to save plots
    """
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📊 BATCH GENERATING RASTER PLOTS")
    print(f"Output directory: {output_path}")
    print("=" * 60)
    
    successful_plots = 0
    
    for trial_num in trial_numbers:
        print(f"Processing trial {trial_num}...")
        
        try:
            # Load and analyze trial
            trial_data = load_trial_data(H5_FILE_PATH, trial_num)
            if trial_data is None:
                continue
                
            spike_data = detect_spikes(trial_data['neural_data'], SPIKE_CHANNELS)
            
            # Generate plot
            trial_duration = trial_data.get('duration', 
                                          trial_data['neural_data'].shape[1] / 30000)
            save_path = output_path / f"trial_{trial_num:02d}_raster.png"
            
            create_raster_plot(spike_data, trial_duration, trial_num, trial_data, str(save_path))
            successful_plots += 1
            
        except Exception as e:
            print(f"❌ Error processing trial {trial_num}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {successful_plots} raster plots!")
    print(f"📁 Saved to: {output_path}")


def main():
    """Main function demonstrating various usage patterns."""
    print("🧠 NEURAL SPIKE RASTER PLOT - USAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Analyze a single trial
    print("\n1️⃣ SINGLE TRIAL ANALYSIS:")
    analyze_single_trial(1, save_plot=True)
    
    # Example 2: Compare multiple trials
    print("\n2️⃣ MULTIPLE TRIAL COMPARISON:")
    compare_multiple_trials([1, 2, 3, 4, 5], save_plots=False)
    
    # Example 3: Find high-activity trials
    print("\n3️⃣ HIGH-ACTIVITY TRIAL SEARCH:")
    high_activity_trials = find_trials_with_high_activity(threshold_spikes=150, max_trials=10)
    
    # Example 4: Batch generate plots
    if high_activity_trials:
        print("\n4️⃣ BATCH PLOT GENERATION:")
        batch_generate_raster_plots(high_activity_trials[:3], "example_raster_plots")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main() 