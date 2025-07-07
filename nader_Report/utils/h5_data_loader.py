"""
H5 Data loading utilities for neural exploration project.
Handles integrated H5 files containing both neural and behavioral data.
"""

import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class H5DataLoader:
    """
    Class for loading and preprocessing data from integrated H5 files.
    
    The H5 file is expected to have the following structure:
    - Global attributes: creation_date, neural_file, behavioral_file, etc.
    - Trial groups: trial_1, trial_2, ..., trial_N
    - Each trial contains:
        - neural: Neural data (channels x time)
        - velocity_x, velocity_y: Behavioral velocity data
        - behavioral_timestamps: Timestamps for behavioral data
        - Attributes: outcome, target_index, duration, start_time, end_time, etc.
    """
    
    def __init__(self, h5_file_path: str):
        """
        Initialize the H5 data loader.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to the integrated H5 file
        """
        self.h5_file_path = h5_file_path
        self.behavioral_data = None
        self.metadata = {}
        self.trial_info = {}
        
        # Validate file exists
        if not Path(h5_file_path).exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file_path}")
    
    def load_file_metadata(self) -> Dict:
        """
        Load global metadata from H5 file.
        
        Returns:
        --------
        dict
            Dictionary containing file metadata
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            metadata = {}
            for key, value in f.attrs.items():
                if isinstance(value, bytes):
                    metadata[key] = value.decode('utf-8')
                else:
                    metadata[key] = value
            
            # Get trial information
            trial_keys = [k for k in f.keys() if k.startswith('trial_')]
            metadata['total_trials'] = len(trial_keys)
            metadata['available_trials'] = sorted([int(k.split('_')[1]) for k in trial_keys])
            
        self.metadata = metadata
        return metadata
    
    def get_trial_info(self, trial_numbers: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get information about trials in the H5 file.
        
        Parameters:
        -----------
        trial_numbers : list, optional
            List of trial numbers to get info for. If None, gets info for all trials.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trial information
        """
        trial_info = []
        
        with h5py.File(self.h5_file_path, 'r') as f:
            trial_keys = [k for k in f.keys() if k.startswith('trial_')]
            
            if trial_numbers is None:
                trial_numbers = sorted([int(k.split('_')[1]) for k in trial_keys])
            
            for trial_num in trial_numbers:
                trial_key = f'trial_{trial_num}'
                
                if trial_key not in f:
                    continue
                
                trial_group = f[trial_key]
                
                # Extract trial metadata
                info = {'trial_number': trial_num}
                
                # Get attributes
                for attr_name in trial_group.attrs.keys():
                    value = trial_group.attrs[attr_name]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    info[attr_name] = value
                
                # Get data shapes
                if 'neural' in trial_group:
                    info['neural_shape'] = trial_group['neural'].shape
                    info['neural_channels'] = trial_group['neural'].shape[0]
                    info['neural_samples'] = trial_group['neural'].shape[1]
                
                if 'velocity_x' in trial_group:
                    info['behavioral_samples'] = trial_group['velocity_x'].shape[0]
                
                if 'behavioral_timestamps' in trial_group:
                    info['behavioral_duration'] = (
                        trial_group['behavioral_timestamps'][-1] - 
                        trial_group['behavioral_timestamps'][0]
                    )
                
                trial_info.append(info)
        
        return pd.DataFrame(trial_info)
    
    def load_trial_data(self, trial_number: int) -> Dict:
        """
        Load all data for a specific trial.
        
        Parameters:
        -----------
        trial_number : int
            Trial number to load
            
        Returns:
        --------
        dict
            Dictionary containing trial data and metadata
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            trial_key = f'trial_{trial_number}'
            
            if trial_key not in f:
                raise KeyError(f"Trial {trial_number} not found in H5 file")
            
            trial_group = f[trial_key]
            
            # Load data
            trial_data = {
                'trial_number': trial_number,
                'neural_data': None,
                'velocity_x': None,
                'velocity_y': None,
                'behavioral_timestamps': None,
            }
            
            # Load neural data
            if 'neural' in trial_group:
                trial_data['neural_data'] = trial_group['neural'][:]
            elif 'neural_data' in trial_group:
                trial_data['neural_data'] = trial_group['neural_data'][:]
            
            # Load behavioral data
            if 'velocity_x' in trial_group:
                trial_data['velocity_x'] = trial_group['velocity_x'][:]
            
            if 'velocity_y' in trial_group:
                trial_data['velocity_y'] = trial_group['velocity_y'][:]
            
            if 'behavioral_timestamps' in trial_group:
                trial_data['behavioral_timestamps'] = trial_group['behavioral_timestamps'][:]
            
            # Load metadata
            for attr_name in trial_group.attrs.keys():
                value = trial_group.attrs[attr_name]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                trial_data[attr_name] = value
        
        return trial_data
    
    def load_behavioral_data(self, trial_numbers: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load behavioral data from all trials and create a consolidated DataFrame.
        
        Parameters:
        -----------
        trial_numbers : list, optional
            List of trial numbers to load. If None, loads all trials.
            
        Returns:
        --------
        pandas.DataFrame
            Consolidated behavioral data with trial segmentation
        """
        if trial_numbers is None:
            # Get all available trials
            with h5py.File(self.h5_file_path, 'r') as f:
                trial_keys = [k for k in f.keys() if k.startswith('trial_')]
                trial_numbers = sorted([int(k.split('_')[1]) for k in trial_keys])
        
        all_behavioral_data = []
        
        for trial_num in trial_numbers:
            try:
                trial_data = self.load_trial_data(trial_num)
                
                # Skip if no behavioral data
                if (trial_data['velocity_x'] is None or 
                    trial_data['velocity_y'] is None or 
                    trial_data['behavioral_timestamps'] is None):
                    continue
                
                # Create DataFrame for this trial
                n_samples = len(trial_data['velocity_x'])
                
                # Convert timestamps to datetime
                if isinstance(trial_data['behavioral_timestamps'][0], (int, float)):
                    # Assume Unix timestamps
                    timestamps = pd.to_datetime(trial_data['behavioral_timestamps'], unit='s')
                else:
                    timestamps = pd.to_datetime(trial_data['behavioral_timestamps'])
                
                trial_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'velocity_x': trial_data['velocity_x'],
                    'velocity_y': trial_data['velocity_y'],
                    'trial': trial_num,
                    'trial_outcome': trial_data.get('outcome', 'unknown'),
                    'target_index': trial_data.get('target_index', -1),
                })
                
                # Add trial markers
                trial_df['trial_start'] = False
                trial_df['trial_win'] = False
                trial_df['trial_lose'] = False
                
                # Mark trial start (first row)
                trial_df.iloc[0, trial_df.columns.get_loc('trial_start')] = True
                
                # Mark trial end based on outcome
                if trial_data.get('outcome') == 'win':
                    trial_df.iloc[-1, trial_df.columns.get_loc('trial_win')] = True
                elif trial_data.get('outcome') == 'lose':
                    trial_df.iloc[-1, trial_df.columns.get_loc('trial_lose')] = True
                
                # Add additional metadata as constant columns
                if 'duration' in trial_data:
                    trial_df['trial_duration'] = trial_data['duration']
                
                all_behavioral_data.append(trial_df)
                
            except Exception as e:
                print(f"Warning: Failed to load trial {trial_num}: {e}")
                continue
        
        if not all_behavioral_data:
            raise ValueError("No behavioral data could be loaded from any trials")
        
        # Concatenate all trials
        behavioral_data = pd.concat(all_behavioral_data, ignore_index=True)
        
        # Sort by timestamp to ensure proper ordering
        behavioral_data = behavioral_data.sort_values('timestamp').reset_index(drop=True)
        
        # Store for later use
        self.behavioral_data = behavioral_data
        
        print(f"✅ Loaded behavioral data from {len(trial_numbers)} trials")
        print(f"   - Total samples: {len(behavioral_data)}")
        print(f"   - Time span: {behavioral_data['timestamp'].max() - behavioral_data['timestamp'].min()}")
        print(f"   - Trials: {sorted(behavioral_data['trial'].unique())}")
        
        # Show trial outcomes
        if 'trial_outcome' in behavioral_data.columns:
            outcomes = behavioral_data.groupby('trial')['trial_outcome'].first().value_counts()
            print(f"   - Trial outcomes: {dict(outcomes)}")
        
        return behavioral_data
    
    def load_multiple_trials_data(self, trial_numbers: List[int]) -> Dict[int, Dict]:
        """
        Load data for multiple trials.
        
        Parameters:
        -----------
        trial_numbers : list
            List of trial numbers to load
            
        Returns:
        --------
        dict
            Dictionary mapping trial numbers to trial data
        """
        trials_data = {}
        
        for trial_num in trial_numbers:
            try:
                trials_data[trial_num] = self.load_trial_data(trial_num)
            except Exception as e:
                print(f"Warning: Failed to load trial {trial_num}: {e}")
                continue
        
        return trials_data
    
    def get_summary_info(self) -> Dict:
        """
        Get summary information about the H5 file.
        
        Returns:
        --------
        dict
            Summary information
        """
        if not self.metadata:
            self.load_file_metadata()
        
        trial_info_df = self.get_trial_info()
        
        summary = {
            'h5_file_path': self.h5_file_path,
            'file_size_mb': Path(self.h5_file_path).stat().st_size / (1024 * 1024),
            'total_trials': self.metadata.get('total_trials', 0),
            'available_trials': self.metadata.get('available_trials', []),
            'global_metadata': self.metadata,
        }
        
        if len(trial_info_df) > 0:
            summary.update({
                'trial_outcomes': trial_info_df['outcome'].value_counts().to_dict() if 'outcome' in trial_info_df.columns else {},
                'trial_durations': {
                    'mean': trial_info_df['duration'].mean() if 'duration' in trial_info_df.columns else None,
                    'std': trial_info_df['duration'].std() if 'duration' in trial_info_df.columns else None,
                    'min': trial_info_df['duration'].min() if 'duration' in trial_info_df.columns else None,
                    'max': trial_info_df['duration'].max() if 'duration' in trial_info_df.columns else None,
                },
                'targets_used': sorted(trial_info_df['target_index'].unique()) if 'target_index' in trial_info_df.columns else [],
            })
        
        return summary


 