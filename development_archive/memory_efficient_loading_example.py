"""
Memory-Efficient Neural Data Loading Example

This script demonstrates how to load large .ns6 files without running into memory errors.
The original error "unable to allocate 16.1 GiB for an array" can be resolved by using
the memory-efficient loading options.
"""

import sys
import os
sys.path.append('utils')

from data_loader import DataLoader

def main():
    # Initialize the data loader
    loader = DataLoader()
    
    # Replace with your actual file path
    ns6_file_path = "your_file.ns6"  # Update this path
    
    print("=== Memory-Efficient Neural Data Loading Options ===\n")
    
    # Option 1: Load only a subset of channels (e.g., first 32 channels)
    print("1. Loading first 32 channels only:")
    print("   loader.load_neural_data(ns6_file_path, max_channels=32)")
    
    # Option 2: Load only a short duration (e.g., first 60 seconds)
    print("\n2. Loading first 60 seconds only:")
    print("   loader.load_neural_data(ns6_file_path, max_duration=60)")
    
    # Option 3: Downsample the data (e.g., reduce sampling rate by 10x)
    print("\n3. Downsampling by factor of 10:")
    print("   loader.load_neural_data(ns6_file_path, downsample_factor=10)")
    
    # Option 4: Combine all options for maximum memory efficiency
    print("\n4. Combining all options (recommended for large files):")
    print("   loader.load_neural_data(ns6_file_path, max_channels=16, max_duration=30, downsample_factor=5)")
    
    # Example usage - uncomment and modify as needed
    if os.path.exists(ns6_file_path):
        print(f"\n=== Loading {ns6_file_path} ===")
        
        # Start with very conservative settings
        neural_data = loader.load_neural_data(
            ns6_file_path,
            max_channels=16,      # Only first 16 channels
            max_duration=30,      # Only first 30 seconds  
            downsample_factor=10  # Reduce sampling rate by 10x
        )
        
        if neural_data is not None:
            print("Success! Neural data loaded.")
            print(f"Shape: {neural_data['raw_data'].shape}")
            print(f"Duration: {neural_data['raw_data'].shape[0] / neural_data['sampling_rate']:.2f} seconds")
            print(f"Channels: {neural_data['raw_data'].shape[1]}")
            print(f"Sampling rate: {neural_data['sampling_rate']} Hz")
        else:
            print("Failed to load neural data.")
    else:
        print(f"\nFile not found: {ns6_file_path}")
        print("Please update the file path in this script.")

def progressive_loading_example():
    """
    Example of progressive loading - start small and increase as needed
    """
    loader = DataLoader()
    ns6_file_path = "your_file.ns6"  # Update this path
    
    print("\n=== Progressive Loading Strategy ===")
    
    # Step 1: Start with minimal data to check if file loads
    print("Step 1: Loading minimal data (1 channel, 10 seconds, 20x downsample)")
    try:
        neural_data = loader.load_neural_data(
            ns6_file_path,
            max_channels=1,
            max_duration=10,
            downsample_factor=20
        )
        if neural_data is not None:
            print("✓ Minimal loading successful")
        else:
            print("✗ Minimal loading failed")
            return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 2: Increase to more channels
    print("\nStep 2: Loading more channels (16 channels, 30 seconds, 10x downsample)")
    try:
        neural_data = loader.load_neural_data(
            ns6_file_path,
            max_channels=16,
            max_duration=30,
            downsample_factor=10,
            force_reload=True
        )
        if neural_data is not None:
            print("✓ Medium loading successful")
        else:
            print("✗ Medium loading failed")
            return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 3: Increase duration and reduce downsampling
    print("\nStep 3: Loading longer duration (32 channels, 60 seconds, 5x downsample)")
    try:
        neural_data = loader.load_neural_data(
            ns6_file_path,
            max_channels=32,
            max_duration=60,
            downsample_factor=5,
            force_reload=True
        )
        if neural_data is not None:
            print("✓ Extended loading successful")
            print(f"Final data shape: {neural_data['raw_data'].shape}")
            print(f"Memory usage: ~{neural_data['raw_data'].nbytes / (1024**3):.2f} GB")
        else:
            print("✗ Extended loading failed")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Consider reducing parameters further")

def memory_estimation():
    """
    Helper function to estimate memory usage for different loading options
    """
    print("\n=== Memory Usage Estimation ===")
    
    # Typical .ns6 file parameters
    total_channels = 96
    sampling_rate = 30000  # Hz
    duration_minutes = 60  # minutes
    
    total_samples = sampling_rate * duration_minutes * 60
    
    print(f"Full file specs:")
    print(f"  - Channels: {total_channels}")
    print(f"  - Sampling rate: {sampling_rate} Hz")
    print(f"  - Duration: {duration_minutes} minutes")
    print(f"  - Total samples: {total_samples:,}")
    
    # Memory calculations (4 bytes per float32)
    scenarios = [
        ("Full file", total_channels, duration_minutes * 60, 1),
        ("32 channels, 60s", 32, 60, 1),
        ("16 channels, 30s", 16, 30, 1),
        ("16 channels, 60s, 10x downsample", 16, 60, 10),
        ("8 channels, 30s, 5x downsample", 8, 30, 5),
    ]
    
    print("\nMemory usage estimates:")
    for name, channels, duration, downsample in scenarios:
        samples = (sampling_rate * duration) // downsample
        memory_gb = (samples * channels * 4) / (1024**3)
        print(f"  {name}: {memory_gb:.2f} GB")

if __name__ == "__main__":
    main()
    # progressive_loading_example()  # Uncomment to run progressive loading
    memory_estimation() 