"""
Diagnostic utilities for neural feature exploration.

This module provides functions to examine trial data structure, 
check behavioral data availability, and validate data quality.
"""

import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple
from neural_feature_extraction import NeuralFeatureExtractor


def diagnose_trial_data(trial_number: int, sampling_rate: int = 30000) -> Optional[Dict]:
    """
    Diagnose trial data structure and behavioral data availability.
    
    Args:
        trial_number: Trial number to examine
        sampling_rate: Neural data sampling rate in Hz
        
    Returns:
        Dictionary containing trial data if successful, None if failed
    """
    print(f"🔍 DIAGNOSING TRIAL {trial_number} DATA STRUCTURE")
    print("=" * 50)
    
    # Use hardcoded H5 file path
    h5_file = r"D:\Data\ScienceCorp\trials_aligned.h5"
    
    print(f"📁 Using H5 file: {h5_file}")
    
    # Initialize extractor
    extractor = NeuralFeatureExtractor(sampling_rate=sampling_rate)
    
    # Load trial data
    print(f"\n🔍 Loading trial {trial_number}...")
    try:
        trial_data = extractor.load_trial_data(h5_file, trial_number)
    except Exception as e:
        print(f"❌ Failed to load trial {trial_number}: {e}")
        return None
    
    print(f"\n📊 TRIAL DATA STRUCTURE:")
    print(f"   Available keys: {list(trial_data.keys())}")
    
    # Examine each key
    for key, value in trial_data.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: {value.shape} array")
            if key in ['velocity_x', 'velocity_y']:
                non_zero = np.count_nonzero(value)
                print(f"      Non-zero values: {non_zero}/{len(value)}")
                if non_zero > 0:
                    print(f"      Range: {np.min(value):.3f} to {np.max(value):.3f}")
                    print(f"      Mean: {np.mean(value):.3f}, Std: {np.std(value):.3f}")
        elif value is None:
            print(f"   {key}: None (missing)")
        else:
            print(f"   {key}: {type(value).__name__} - {value}")
    
    return trial_data


def check_behavioral_data_availability(max_trials: int = 10) -> List[Dict]:
    """
    Check which trials have behavioral data available.
    
    Args:
        max_trials: Maximum number of trials to check
        
    Returns:
        List of dictionaries with trial information
    """
    print(f"\n🔍 CHECKING BEHAVIORAL DATA AVAILABILITY (first {max_trials} trials):")
    print("-" * 50)
    
    h5_file = r"D:\Data\ScienceCorp\trials_aligned.h5"
    
    behavioral_summary = []
    
    with h5py.File(h5_file, 'r') as f:
        trial_keys = [k for k in f.keys() if k.startswith('trial_')]
        trial_keys = sorted(trial_keys, key=lambda x: int(x.split('_')[1]))[:max_trials]
        
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
            
            # Get additional info about velocity data
            if summary['has_velocity_x']:
                vel_x = trial_group['velocity_x'][:]
                vel_y = trial_group['velocity_y'][:]
                summary['velocity_x_shape'] = vel_x.shape
                summary['velocity_y_shape'] = vel_y.shape
                summary['velocity_x_nonzero'] = np.count_nonzero(vel_x)
                summary['velocity_y_nonzero'] = np.count_nonzero(vel_y)
                summary['velocity_x_range'] = f"{np.min(vel_x):.3f} to {np.max(vel_x):.3f}"
                summary['velocity_y_range'] = f"{np.min(vel_y):.3f} to {np.max(vel_y):.3f}"
                
                # Get trial outcome if available
                if 'outcome' in trial_group.attrs:
                    outcome = trial_group.attrs['outcome']
                    if isinstance(outcome, bytes):
                        outcome = outcome.decode()
                    summary['outcome'] = outcome
                else:
                    summary['outcome'] = 'unknown'
            
            behavioral_summary.append(summary)
        
        # Print summary table
        print(f"{'Trial':<6} {'Neural':<7} {'Vel_X':<7} {'Vel_Y':<7} {'Timestamps':<11} {'Outcome':<8} {'VelX Shape':<12} {'VelX Range':<20}")
        print("-" * 100)
        
        for summary in behavioral_summary:
            neural = "✅" if summary['has_neural'] else "❌"
            vel_x = "✅" if summary['has_velocity_x'] else "❌"
            vel_y = "✅" if summary['has_velocity_y'] else "❌"
            timestamps = "✅" if summary['has_behavioral_timestamps'] else "❌"
            outcome = summary.get('outcome', 'N/A')
            
            vel_x_shape = str(summary.get('velocity_x_shape', 'N/A'))
            vel_x_range = summary.get('velocity_x_range', 'N/A')
            
            print(f"{summary['trial']:<6} {neural:<7} {vel_x:<7} {vel_y:<7} {timestamps:<11} {outcome:<8} {vel_x_shape:<12} {vel_x_range:<20}")
    
    return behavioral_summary


def find_trials_with_movement(min_movement_threshold: float = 0.01, max_trials: int = 20) -> List[int]:
    """
    Find trials that have significant movement (non-zero velocity).
    
    Args:
        min_movement_threshold: Minimum movement threshold
        max_trials: Maximum number of trials to check
        
    Returns:
        List of trial numbers with significant movement
    """
    print(f"\n🔍 FINDING TRIALS WITH MOVEMENT (threshold: {min_movement_threshold}):")
    print("-" * 50)
    
    h5_file = r"D:\Data\ScienceCorp\trials_aligned.h5"
    
    trials_with_movement = []
    
    with h5py.File(h5_file, 'r') as f:
        trial_keys = [k for k in f.keys() if k.startswith('trial_')]
        trial_keys = sorted(trial_keys, key=lambda x: int(x.split('_')[1]))[:max_trials]
        
        for trial_key in trial_keys:
            trial_group = f[trial_key]
            trial_num = int(trial_key.split('_')[1])
            
            if 'velocity_x' in trial_group and 'velocity_y' in trial_group:
                vel_x = trial_group['velocity_x'][:]
                vel_y = trial_group['velocity_y'][:]
                
                # Calculate movement metrics
                max_vel_x = np.max(np.abs(vel_x))
                max_vel_y = np.max(np.abs(vel_y))
                rms_vel = np.sqrt(np.mean(vel_x**2 + vel_y**2))
                
                if max_vel_x > min_movement_threshold or max_vel_y > min_movement_threshold:
                    outcome = trial_group.attrs.get('outcome', b'unknown')
                    if isinstance(outcome, bytes):
                        outcome = outcome.decode()
                    
                    trials_with_movement.append(trial_num)
                    print(f"   Trial {trial_num:3d}: Max vel = ({max_vel_x:.3f}, {max_vel_y:.3f}), RMS = {rms_vel:.3f}, Outcome = {outcome}")
    
    print(f"\n✅ Found {len(trials_with_movement)} trials with movement: {trials_with_movement}")
    return trials_with_movement


def validate_trial_data(trial_data: Dict) -> Dict[str, bool]:
    """
    Validate trial data quality and completeness.
    
    Args:
        trial_data: Dictionary containing trial data
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'has_neural_data': False,
        'has_behavioral_data': False,
        'neural_data_valid': False,
        'behavioral_data_active': False,
        'timestamps_aligned': False,
        'duration_consistent': False
    }
    
    # Check neural data
    if 'neural_data' in trial_data and trial_data['neural_data'] is not None:
        validation['has_neural_data'] = True
        neural_data = trial_data['neural_data']
        
        # Check if neural data is valid (not all zeros, reasonable range)
        if neural_data.shape[1] > 0 and np.std(neural_data) > 0.1:
            validation['neural_data_valid'] = True
    
    # Check behavioral data
    velocity_x = trial_data.get('velocity_x')
    velocity_y = trial_data.get('velocity_y')
    
    if velocity_x is not None and velocity_y is not None:
        validation['has_behavioral_data'] = True
        
        # Check if behavioral data shows movement
        if (np.count_nonzero(velocity_x) > 0 or np.count_nonzero(velocity_y) > 0):
            validation['behavioral_data_active'] = True
    
    # Check timestamp alignment
    behavioral_timestamps = trial_data.get('behavioral_timestamps')
    if behavioral_timestamps is not None and len(behavioral_timestamps) > 1:
        validation['timestamps_aligned'] = True
    
    # Check duration consistency
    duration = trial_data.get('duration')
    if duration is not None and duration > 0:
        if validation['has_neural_data']:
            neural_duration = trial_data['neural_data'].shape[1] / 30000  # Assuming 30kHz
            if abs(duration - neural_duration) < 1.0:  # Within 1 second
                validation['duration_consistent'] = True
    
    return validation


def print_validation_summary(validation: Dict[str, bool], trial_number: int):
    """
    Print a summary of trial data validation results.
    
    Args:
        validation: Dictionary with validation results
        trial_number: Trial number being validated
    """
    print(f"\n📋 TRIAL {trial_number} VALIDATION SUMMARY:")
    print("-" * 40)
    
    status_map = {True: "✅", False: "❌"}
    
    print(f"   Neural data present: {status_map[validation['has_neural_data']]}")
    print(f"   Neural data valid: {status_map[validation['neural_data_valid']]}")
    print(f"   Behavioral data present: {status_map[validation['has_behavioral_data']]}")
    print(f"   Behavioral data active: {status_map[validation['behavioral_data_active']]}")
    print(f"   Timestamps aligned: {status_map[validation['timestamps_aligned']]}")
    print(f"   Duration consistent: {status_map[validation['duration_consistent']]}")
    
    # Overall assessment
    critical_checks = ['has_neural_data', 'neural_data_valid', 'has_behavioral_data']
    critical_passed = all(validation[check] for check in critical_checks)
    
    print(f"\n📊 Overall Assessment: {'✅ GOOD' if critical_passed else '⚠️ NEEDS ATTENTION'}")
    
    if not critical_passed:
        print("   Recommendations:")
        if not validation['has_neural_data']:
            print("   - Check neural data loading")
        if not validation['neural_data_valid']:
            print("   - Verify neural data quality")
        if not validation['has_behavioral_data']:
            print("   - Check behavioral data availability")
        if not validation['behavioral_data_active']:
            print("   - Try a different trial with movement")


def run_comprehensive_diagnostic(trial_number: int, sampling_rate: int = 30000) -> Dict:
    """
    Run a comprehensive diagnostic on a trial.
    
    Args:
        trial_number: Trial number to examine
        sampling_rate: Neural data sampling rate in Hz
        
    Returns:
        Dictionary with diagnostic results
    """
    print(f"🔍 COMPREHENSIVE DIAGNOSTIC - TRIAL {trial_number}")
    print("=" * 60)
    
    # Load and examine trial data
    trial_data = diagnose_trial_data(trial_number, sampling_rate)
    
    if trial_data is None:
        return {'success': False, 'trial_data': None, 'validation': None}
    
    # Validate trial data
    validation = validate_trial_data(trial_data)
    print_validation_summary(validation, trial_number)
    
    # Check behavioral data availability across multiple trials
    behavioral_summary = check_behavioral_data_availability(max_trials=10)
    
    # Find trials with movement
    movement_trials = find_trials_with_movement(min_movement_threshold=0.01, max_trials=20)
    
    results = {
        'success': True,
        'trial_data': trial_data,
        'validation': validation,
        'behavioral_summary': behavioral_summary,
        'movement_trials': movement_trials
    }
    
    return results 