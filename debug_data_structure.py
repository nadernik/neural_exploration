#!/usr/bin/env python3
"""
Debug script to examine trial data structure and behavioral data availability.
"""
import numpy as np
import h5py
from neural_feature_extraction import NeuralFeatureExtractor, find_h5_file

def examine_trial_structure(trial_number=12):
    """Examine the structure of trial data and check for behavioral data."""
    print("🔍 TRIAL DATA STRUCTURE DIAGNOSTIC")
    print("=" * 50)
    
    # Find H5 file
    h5_file = find_h5_file()
    if h5_file is None:
        print("❌ No H5 files found!")
        return
    
    print(f"📁 Using H5 file: {h5_file}")
    
    # Initialize extractor
    extractor = NeuralFeatureExtractor()
    
    # Load trial data
    print(f"\n🔍 Loading trial {trial_number}...")
    trial_data = extractor.load_trial_data(h5_file, trial_number)
    
    print(f"\n📊 TRIAL DATA STRUCTURE:")
    print(f"   Top-level keys: {list(trial_data.keys())}")
    
    for key, value in trial_data.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: {value.shape} array")
        elif isinstance(value, dict):
            print(f"   {key}: dictionary with keys: {list(value.keys())}")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    print(f"      {subkey}: {subvalue.shape} array")
                else:
                    print(f"      {subkey}: {type(subvalue)} - {subvalue}")
        else:
            print(f"   {key}: {type(value)} - {value}")
    
    print(f"\n🔍 EXAMINING H5 FILE DIRECTLY:")
    with h5py.File(h5_file, 'r') as f:
        trial_key = f'trial_{trial_number}'
        if trial_key in f:
            trial_group = f[trial_key]
            print(f"   HDF5 datasets: {list(trial_group.keys())}")
            print(f"   HDF5 attributes: {list(trial_group.attrs.keys())}")
            
            # Check each dataset
            for dataset_name in trial_group.keys():
                dataset = trial_group[dataset_name]
                print(f"   {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
                
                # Sample some data
                if dataset_name in ['velocity_x', 'velocity_y']:
                    sample_data = dataset[:10]  # First 10 samples
                    print(f"      Sample data: {sample_data}")
                    print(f"      Non-zero values: {np.count_nonzero(sample_data)}/10")
        else:
            print(f"   ❌ Trial {trial_number} not found in HDF5 file")
    
    return trial_data

def check_behavioral_data_availability(h5_file, max_trials=10):
    """Check which trials have behavioral data available."""
    print(f"\n🔍 CHECKING BEHAVIORAL DATA AVAILABILITY (first {max_trials} trials):")
    print("-" * 50)
    
    with h5py.File(h5_file, 'r') as f:
        trial_keys = [k for k in f.keys() if k.startswith('trial_')]
        trial_keys = sorted(trial_keys)[:max_trials]
        
        behavioral_summary = []
        
        for trial_key in trial_keys:
            trial_group = f[trial_key]
            trial_num = int(trial_key.split('_')[1])
            
            summary = {
                'trial': trial_num,
                'has_neural': 'neural' in trial_group,
                'has_velocity_x': 'velocity_x' in trial_group,
                'has_velocity_y': 'velocity_y' in trial_group,
                'has_behavioral_timestamps': 'behavioral_timestamps' in trial_group,
            }
            
            if summary['has_velocity_x']:
                vel_x = trial_group['velocity_x'][:]
                vel_y = trial_group['velocity_y'][:]
                summary['velocity_x_shape'] = vel_x.shape
                summary['velocity_y_shape'] = vel_y.shape
                summary['velocity_x_nonzero'] = np.count_nonzero(vel_x)
                summary['velocity_y_nonzero'] = np.count_nonzero(vel_y)
                summary['velocity_x_range'] = f"{np.min(vel_x):.3f} to {np.max(vel_x):.3f}"
                summary['velocity_y_range'] = f"{np.min(vel_y):.3f} to {np.max(vel_y):.3f}"
            
            behavioral_summary.append(summary)
        
        # Print summary
        print(f"{'Trial':<6} {'Neural':<7} {'Vel_X':<7} {'Vel_Y':<7} {'Timestamps':<11} {'VelX Shape':<12} {'VelX Range':<20}")
        print("-" * 90)
        
        for summary in behavioral_summary:
            neural = "✅" if summary['has_neural'] else "❌"
            vel_x = "✅" if summary['has_velocity_x'] else "❌"
            vel_y = "✅" if summary['has_velocity_y'] else "❌"
            timestamps = "✅" if summary['has_behavioral_timestamps'] else "❌"
            
            vel_x_shape = str(summary.get('velocity_x_shape', 'N/A'))
            vel_x_range = summary.get('velocity_x_range', 'N/A')
            
            print(f"{summary['trial']:<6} {neural:<7} {vel_x:<7} {vel_y:<7} {timestamps:<11} {vel_x_shape:<12} {vel_x_range:<20}")
    
    return behavioral_summary

if __name__ == "__main__":
    # Run diagnostic
    trial_data = examine_trial_structure(trial_number=12)
    
    if trial_data:
        h5_file = find_h5_file()
        if h5_file:
            behavioral_summary = check_behavioral_data_availability(h5_file, max_trials=10)
            
            # Check if any trials have behavioral data
            has_behavioral = any(s['has_velocity_x'] for s in behavioral_summary)
            has_active_behavioral = any(s.get('velocity_x_nonzero', 0) > 0 for s in behavioral_summary)
            
            print(f"\n📊 SUMMARY:")
            print(f"   Trials with velocity_x data: {sum(s['has_velocity_x'] for s in behavioral_summary)}/{len(behavioral_summary)}")
            print(f"   Trials with non-zero velocity: {sum(s.get('velocity_x_nonzero', 0) > 0 for s in behavioral_summary)}/{len(behavioral_summary)}")
            
            if has_behavioral:
                print(f"   ✅ Behavioral data is available!")
                if not has_active_behavioral:
                    print(f"   ⚠️  Warning: Behavioral data exists but appears to be all zeros")
            else:
                print(f"   ❌ No behavioral data found in examined trials") 