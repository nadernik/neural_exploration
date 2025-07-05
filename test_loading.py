# Test script for efficient neural data loading
import sys
sys.path.append('utils')

from data_loader import DataLoader
import numpy as np

# File paths
NS6_FILE_PATH = r"D:\Data\ScienceCorp\neural.ns6"
CSV_FILE_PATH = r"D:\Data\ScienceCorp\actions.csv"

def test_efficient_loading():
    """Test the corrected efficient neural data loading approach."""
    
    print("🧪 Testing corrected efficient neural data loading")
    print("="*60)
    
    # Initialize loader
    loader = DataLoader(ns6_file_path=NS6_FILE_PATH, csv_file_path=CSV_FILE_PATH)
    
    # Load behavioral data
    print("Loading behavioral data...")
    behavioral_data = loader.load_behavioral_data()
    
    if behavioral_data is None:
        print("❌ Failed to load behavioral data")
        return False
    
    print(f"✅ Behavioral data loaded: {behavioral_data.shape}")
    
    # Get first 5 trials for quick test
    first_5_trials = sorted(behavioral_data['trial'].unique())[:5]
    print(f"Testing with trials: {first_5_trials}")
    
    # Test the corrected loading method
    print("\nTesting load_neural_data_for_trials_simple...")
    
    neural_data = loader.load_neural_data_for_trials_simple(
        NS6_FILE_PATH,
        trial_numbers=first_5_trials,
        buffer_seconds=2.0,     # Small buffer for quick test
        max_channels=16,        # Few channels for quick test
        downsample_factor=4,    # Aggressive downsampling for quick test
        force_reload=True
    )
    
    if neural_data is not None:
        print("\n✅ SUCCESS! Neural data loaded successfully")
        print(f"Shape: {neural_data['raw_data'].shape}")
        print(f"Memory: {neural_data['raw_data'].nbytes / 1024**2:.1f} MB")
        print(f"Duration: {neural_data['metadata']['duration']:.2f} seconds")
        return True
    else:
        print("❌ FAILED! Neural data loading failed")
        return False

if __name__ == "__main__":
    test_efficient_loading() 