# Data Alignment Example for Neural Exploration Project
# This script demonstrates how to align behavioral and neural data using the Time Origin

import sys
import os
sys.path.append('utils')

from data_loader import DataLoader
from visualization import BehavioralVisualizer, NeuralVisualizer

def main():
    """
    Example of aligning behavioral and neural data.
    """
    
    # File paths (update these to your actual file paths)
    ns6_file = "path/to/your/neural_data.ns6"
    csv_file = "path/to/your/behavioral_data.csv"
    
    print("=== Neural Data Alignment Example ===")
    print()
    
    # 1. Initialize DataLoader
    loader = DataLoader()
    
    # 2. Load Neural Data (this extracts Time Origin)
    # Note: If you call this multiple times with the same file, it will skip reloading
    print("Step 1: Loading neural data...")
    neural_data = loader.load_neural_data(ns6_file)
    
    if neural_data is None:
        print("Failed to load neural data. Please check file path and format.")
        return
    
    print(f"Time Origin extracted: {loader.time_origin}")
    print()
    
    # 2b. Demonstrate that calling load again skips the reload
    print("Step 1b: Calling load_neural_data again (should skip reload)...")
    neural_data_again = loader.load_neural_data(ns6_file)  # This will skip reloading
    print()
    
    # 3. Load Behavioral Data
    print("Step 2: Loading behavioral data...")
    behavioral_data = loader.load_behavioral_data(csv_file)
    
    if behavioral_data is None:
        print("Failed to load behavioral data. Please check file path and format.")
        return
    
    # 3b. Demonstrate that calling load again skips the reload
    print("Step 2b: Calling load_behavioral_data again (should skip reload)...")
    behavioral_data_again = loader.load_behavioral_data(csv_file)  # This will skip reloading
    print()
    
    # 4. Align Timestamps
    print("Step 3: Aligning timestamps...")
    alignment_info = loader.align_timestamps()
    
    if alignment_info is None:
        print("Failed to align timestamps.")
        return
    
    print()
    print("=== Alignment Results ===")
    print(f"Neural started {-alignment_info['time_offset_seconds']:.1f} seconds before behavioral")
    print(f"Overlap duration: {alignment_info['overlap_duration']:.1f} seconds")
    print(f"Alignment quality: {alignment_info['alignment_quality']}")
    print()
    
    # 5. Extract Overlapping Data
    print("Step 4: Extracting overlapping data...")
    overlapping_data = loader.get_overlapping_data()
    
    if overlapping_data is None:
        print("Failed to extract overlapping data.")
        return
    
    print(f"Overlapping period: {overlapping_data['duration']:.1f} seconds")
    print()
    
    # 6. Segment Aligned Trials
    print("Step 5: Segmenting aligned trials...")
    aligned_trials = loader.segment_aligned_trials()
    
    if aligned_trials is None:
        print("Failed to segment aligned trials.")
        return
    
    print(f"Segmented {aligned_trials['n_trials']} aligned trials")
    print()
    
    # 7. Convert to Blackrock Ticks (for precise timing)
    print("Step 6: Converting to Blackrock time ticks...")
    
    # Example: Convert first 10 behavioral timestamps to ticks
    behavioral_times = loader.behavioral_data['timestamp_aligned'][:10]
    ticks = loader.convert_to_blackrock_ticks(behavioral_times)
    
    print("First 10 behavioral timestamps in Blackrock ticks:")
    for i, (time_sec, tick) in enumerate(zip(behavioral_times, ticks)):
        print(f"  Sample {i}: {time_sec:.6f} sec = {tick} ticks")
    
    print()
    
    # 8. Visualize Aligned Data
    print("Step 7: Creating visualizations...")
    
    # Create visualizer with aligned data
    viz = BehavioralVisualizer(loader.behavioral_data)
    
    # Plot trial with aligned timestamps
    fig = viz.plot_trial_behavioral_data(trial_num=0)
    
    print("Behavioral visualization created (using aligned timestamps)")
    print()
    
    # 9. Summary
    print("=== Summary ===")
    print(f"✓ Neural Time Origin: {loader.time_origin}")
    print(f"✓ Time offset: {alignment_info['time_offset_seconds']:.3f} seconds")
    print(f"✓ Overlap duration: {alignment_info['overlap_duration']:.1f} seconds")
    print(f"✓ Aligned trials: {aligned_trials['n_trials']}")
    print(f"✓ Alignment quality: {alignment_info['alignment_quality']}")
    print(f"✓ Optimization: Files are cached and won't reload unnecessarily")
    
    # Show trial timing information
    print("\nFirst 5 aligned trials:")
    trial_info = aligned_trials['trial_info']
    for i in range(min(5, len(trial_info))):
        trial = trial_info.iloc[i]
        print(f"  Trial {i}: {trial['start_time_aligned']:.3f} to {trial['end_time_aligned']:.3f} sec, "
              f"Duration: {trial['duration']:.3f} sec, Outcome: {trial['outcome']}")

def analyze_alignment_quality(alignment_info):
    """
    Analyze and report on alignment quality.
    
    Parameters:
    -----------
    alignment_info : dict
        Alignment information from align_timestamps()
    """
    print("\n=== Alignment Quality Analysis ===")
    
    overlap_duration = alignment_info['overlap_duration']
    time_offset = alignment_info['time_offset_seconds']
    
    # Quality assessment
    if overlap_duration > 300:  # 5 minutes
        quality = "Excellent"
    elif overlap_duration > 60:   # 1 minute
        quality = "Good"
    elif overlap_duration > 10:   # 10 seconds
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"Overlap Duration: {overlap_duration:.1f} seconds")
    print(f"Time Offset: {time_offset:.3f} seconds")
    print(f"Quality Assessment: {quality}")
    
    # Recommendations
    print("\nRecommendations:")
    if overlap_duration < 60:
        print("- Consider checking timestamp formats")
        print("- Verify that both recordings are from the same session")
    
    if abs(time_offset) > 3600:  # 1 hour
        print("- Large time offset detected - verify time zones")
    
    print(f"- Neural recording started {abs(time_offset):.1f} seconds {'before' if time_offset < 0 else 'after'} behavioral")

def demonstrate_force_reload():
    """
    Demonstrate how to force reload files when needed.
    """
    print("\n=== Force Reload Demonstration ===")
    
    loader = DataLoader()
    
    # File paths
    ns6_file = "path/to/your/neural_data.ns6"
    csv_file = "path/to/your/behavioral_data.csv"
    
    # Load data normally
    print("1. Loading data normally...")
    neural_data = loader.load_neural_data(ns6_file)
    behavioral_data = loader.load_behavioral_data(csv_file)
    
    # Try to load again (will skip)
    print("\n2. Loading same files again (will skip)...")
    neural_data = loader.load_neural_data(ns6_file)
    behavioral_data = loader.load_behavioral_data(csv_file)
    
    # Force reload
    print("\n3. Force reloading the same files...")
    neural_data = loader.load_neural_data(ns6_file, force_reload=True)
    behavioral_data = loader.load_behavioral_data(csv_file, force_reload=True)
    
    print("Force reload completed!")

if __name__ == "__main__":
    main()
    
    # Uncomment to see force reload demonstration
    # demonstrate_force_reload() 