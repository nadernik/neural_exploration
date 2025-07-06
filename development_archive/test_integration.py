#!/usr/bin/env python3
"""
Test script for the neural-behavioral integration functionality.

This script tests the basic functionality without requiring actual data files.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

def test_import():
    """Test that the integration script can be imported."""
    try:
        from neural_behavioral_integration import NeuralBehavioralIntegrator
        print("✅ Successfully imported NeuralBehavioralIntegrator")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

def create_test_behavioral_data():
    """Create a sample behavioral CSV file for testing."""
    
    # Create test data
    base_time = datetime(2025, 3, 25, 9, 22, 28)  # Behavioral session start time
    
    data = []
    for i in range(1000):  # 1000 samples
        timestamp = base_time + timedelta(seconds=i * 0.1)  # 10 Hz sampling
        
        # Create some trials
        trial_start = (i % 100) == 0  # Trial every 10 seconds
        trial_win = (i % 100) == 50 and i > 0  # Win 5 seconds after start
        trial_lose = (i % 100) == 60 and i > 0  # Lose 6 seconds after start
        
        data.append({
            'timestamp': timestamp,
            'velocity_x': np.random.normal(0, 1),
            'velocity_y': np.random.normal(0, 1),
            'trial_start': trial_start,
            'trial_win': trial_win,
            'trial_lose': trial_lose,
            'target_index': (i // 100) % 8  # Target 0-7
        })
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name, df

def test_behavioral_loading():
    """Test loading and parsing of behavioral data."""
    try:
        from neural_behavioral_integration import NeuralBehavioralIntegrator
        
        # Create test data
        csv_file, original_df = create_test_behavioral_data()
        
        # Create integrator
        integrator = NeuralBehavioralIntegrator(
            neural_file="dummy.ns6",  # Won't be used in this test
            behavioral_file=csv_file,
            output_file="dummy.h5"
        )
        
        # Test loading
        loaded_df = integrator.load_behavioral_data()
        
        # Verify data
        assert len(loaded_df) == len(original_df), "Data length mismatch"
        assert 'timestamp' in loaded_df.columns, "Missing timestamp column"
        assert loaded_df['timestamp'].dtype == 'datetime64[ns]', "Timestamp not parsed as datetime"
        
        # Clean up
        os.unlink(csv_file)
        
        print("✅ Behavioral data loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Behavioral data loading test failed: {e}")
        return False

def test_trial_segmentation():
    """Test trial segmentation without neural data."""
    try:
        from neural_behavioral_integration import NeuralBehavioralIntegrator
        
        # Create test data
        csv_file, original_df = create_test_behavioral_data()
        
        # Create integrator
        integrator = NeuralBehavioralIntegrator(
            neural_file="dummy.ns6",
            behavioral_file=csv_file,
            output_file="dummy.h5"
        )
        
        # Load behavioral data
        integrator.load_behavioral_data()
        
        # Segment trials
        trials = integrator.segment_trials()
        
        # Verify trials
        assert len(trials) > 0, "No trials found"
        
        # Check trial structure
        for trial in trials:
            assert 'trial_number' in trial, "Missing trial_number"
            assert 'start_time' in trial, "Missing start_time"
            assert 'end_time' in trial, "Missing end_time"
            assert 'outcome' in trial, "Missing outcome"
            assert 'duration' in trial, "Missing duration"
            assert trial['outcome'] in ['win', 'lose'], f"Invalid outcome: {trial['outcome']}"
        
        # Clean up
        os.unlink(csv_file)
        
        print(f"✅ Trial segmentation test passed ({len(trials)} trials found)")
        return True
        
    except Exception as e:
        print(f"❌ Trial segmentation test failed: {e}")
        return False

def test_time_conversion():
    """Test time conversion functionality."""
    try:
        from neural_behavioral_integration import NeuralBehavioralIntegrator
        
        # Create integrator
        integrator = NeuralBehavioralIntegrator(
            neural_file="dummy.ns6",
            behavioral_file="dummy.csv",
            output_file="dummy.h5"
        )
        
        # Set neural start time
        integrator.neural_start_time = datetime(2025, 3, 25, 9, 22, 53)
        
        # Test time conversion
        test_time = datetime(2025, 3, 25, 9, 23, 53)  # 60 seconds later
        seconds = integrator.time_to_neural_seconds(test_time)
        
        assert seconds == 60.0, f"Expected 60.0 seconds, got {seconds}"
        
        print("✅ Time conversion test passed")
        return True
        
    except Exception as e:
        print(f"❌ Time conversion test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("Neural-Behavioral Integration Tests")
    print("=" * 40)
    
    tests = [
        test_import,
        test_behavioral_loading,
        test_trial_segmentation,
        test_time_conversion
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 