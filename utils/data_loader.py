"""
Data loading utilities for neural exploration project.
Handles .ns6 files using Neo library and behavioral CSV data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

try:
    import neo
    from neo.io import BlackrockIO
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False
    print("Neo library not available. Please install with: pip install neo")


class DataLoader:
    """
    Class for loading and preprocessing neural and behavioral data.
    """
    
    def __init__(self, ns6_file_path=None, csv_file_path=None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        ns6_file_path : str, optional
            Path to the .ns6 neural data file
        csv_file_path : str, optional
            Path to the behavioral CSV file
        """
        self.ns6_file_path = ns6_file_path
        self.csv_file_path = csv_file_path
        self.neural_data = None
        self.behavioral_data = None
        self.metadata = {}
        
    def load_behavioral_data(self, csv_file_path=None):
        """
        Load behavioral data from CSV file and segment trials.
        
        Parameters:
        -----------
        csv_file_path : str, optional
            Path to CSV file. If None, uses self.csv_file_path
            
        Returns:
        --------
        pandas.DataFrame
            Behavioral data with proper timestamps and trial segmentation
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path
            
        if csv_file_path is None:
            raise ValueError("No CSV file path provided")
            
        try:
            # Load CSV data
            self.behavioral_data = pd.read_csv(csv_file_path)
            
            # Process timestamps if available
            if 'timestamp' in self.behavioral_data.columns:
                self.behavioral_data['timestamp'] = pd.to_datetime(
                    self.behavioral_data['timestamp'], 
                    errors='coerce'
                )
            
            # Segment trials based on trial_start and trial_win/trial_lose flags
            self.behavioral_data = self._segment_trials(self.behavioral_data)
            
            print(f"Behavioral data loaded successfully!")
            print(f"Shape: {self.behavioral_data.shape}")
            print(f"Columns: {list(self.behavioral_data.columns)}")
            
            # Report trial segmentation results
            if 'trial' in self.behavioral_data.columns:
                n_trials = self.behavioral_data['trial'].nunique()
                print(f"Trials segmented: {n_trials} trials identified")
                
                # Show trial outcomes
                if 'trial_outcome' in self.behavioral_data.columns:
                    outcomes = self.behavioral_data.groupby('trial')['trial_outcome'].first().value_counts()
                    print(f"Trial outcomes: {dict(outcomes)}")
            
            return self.behavioral_data
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None
    
    def _segment_trials(self, data):
        """
        Segment trials based on trial_start and trial_win/trial_lose flags.
        
        Trial Definition Heuristic:
        - Start: First row where trial_start == True
        - End: First subsequent row where trial_win == True OR trial_lose == True
        
        This assumes:
        - trial_start flags the beginning of a trial (center hold or cue presentation)
        - trial_win or trial_lose marks behavioral resolution
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw behavioral data with columns:
            - trial_start: Boolean flag for trial beginnings
            - trial_win: Boolean flag for successful trial endings
            - trial_lose: Boolean flag for failed trial endings
            
        Returns:
        --------
        pandas.DataFrame
            Data with added columns:
            - trial: Trial numbers (1 to N)
            - trial_outcome: 'win', 'lose', or 'incomplete'
        """
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Initialize trial-related columns
        data['trial'] = -1
        data['trial_outcome'] = 'unknown'
        
        # Check for required columns
        required_cols = ['trial_start']
        outcome_cols = ['trial_win', 'trial_lose']
        
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: Missing required columns for trial segmentation: {required_cols}")
            return data
            
        if not any(col in data.columns for col in outcome_cols):
            print(f"Warning: Missing outcome columns for trial segmentation: {outcome_cols}")
            return data
        
        # Find trial start and end points
        trial_starts = data.index[data['trial_start'] == True].tolist()
        
        if len(trial_starts) == 0:
            print("Warning: No trial start markers found")
            return data
        
        print(f"Found {len(trial_starts)} trial start markers")
        
        # Segment each trial
        current_trial = 0
        
        for i, start_idx in enumerate(trial_starts):
            current_trial += 1
            
            # Find the end of this trial
            if i + 1 < len(trial_starts):
                # Look for trial end before next trial start
                search_end = trial_starts[i + 1]
            else:
                # Last trial - search to end of data
                search_end = len(data)
            
            # Look for trial_win or trial_lose in this range
            trial_segment = data.iloc[start_idx:search_end]
            
            # Find trial outcome
            win_indices = trial_segment.index[trial_segment.get('trial_win', False) == True]
            lose_indices = trial_segment.index[trial_segment.get('trial_lose', False) == True]
            
            # Determine trial end and outcome
            if len(win_indices) > 0 and len(lose_indices) > 0:
                # Both win and lose flags - take the first one
                first_win = win_indices[0] if len(win_indices) > 0 else float('inf')
                first_lose = lose_indices[0] if len(lose_indices) > 0 else float('inf')
                
                if first_win < first_lose:
                    end_idx = first_win
                    outcome = 'win'
                else:
                    end_idx = first_lose
                    outcome = 'lose'
            elif len(win_indices) > 0:
                end_idx = win_indices[0]
                outcome = 'win'
            elif len(lose_indices) > 0:
                end_idx = lose_indices[0]
                outcome = 'lose'
            else:
                # No clear end - extend to next trial start or end of data
                end_idx = search_end - 1
                outcome = 'incomplete'
            
            # Assign trial number and outcome to this segment
            data.loc[start_idx:end_idx, 'trial'] = current_trial
            data.loc[start_idx:end_idx, 'trial_outcome'] = outcome
        
        # Remove rows that aren't part of any trial
        data = data[data['trial'] != -1].copy()
        
        print(f"Trial segmentation complete: {current_trial} trials identified")
        
        return data
    
    def load_neural_data_neo(self, ns6_file_path=None):
        """
        Load neural data using Neo library.
        
        Parameters:
        -----------
        ns6_file_path : str, optional
            Path to .ns6 file. If None, uses self.ns6_file_path
            
        Returns:
        --------
        dict
            Dictionary containing neural data and metadata
        """
        if not NEO_AVAILABLE:
            raise ImportError("Neo library not available")
            
        if ns6_file_path is None:
            ns6_file_path = self.ns6_file_path
            
        if ns6_file_path is None:
            raise ValueError("No .ns6 file path provided")
            
        try:
            # Load using Neo
            reader = BlackrockIO(filename=ns6_file_path)
            
            # Read the data
            block = reader.read_block()
            
            # Extract information
            self.neural_data = {}
            self.metadata = {}
            
            # Get sampling rate and other metadata
            for segment in block.segments:
                for signal in segment.analogsignals:
                    self.metadata['sampling_rate'] = float(signal.sampling_rate)
                    self.metadata['n_channels'] = signal.shape[1]
                    self.metadata['duration'] = float(signal.duration)
                    self.metadata['t_start'] = float(signal.t_start)
                    
                    # Store the actual data
                    self.neural_data['raw_data'] = signal.magnitude
                    self.neural_data['times'] = signal.times
                    self.neural_data['channel_names'] = [f'ch_{i:02d}' for i in range(signal.shape[1])]
                    break
                break
            
            print(f"Neural data loaded successfully using Neo!")
            print(f"Sampling rate: {self.metadata['sampling_rate']} Hz")
            print(f"Number of channels: {self.metadata['n_channels']}")
            print(f"Duration: {self.metadata['duration']:.2f} seconds")
            print(f"Data shape: {self.neural_data['raw_data'].shape}")
            
            return self.neural_data
            
        except Exception as e:
            print(f"Error loading neural data with Neo: {e}")
            return None
    

    
    def load_neural_data(self, ns6_file_path=None):
        """
        Load neural data using Neo library.
        
        Parameters:
        -----------
        ns6_file_path : str, optional
            Path to .ns6 file. If None, uses self.ns6_file_path
            
        Returns:
        --------
        dict
            Dictionary containing neural data and metadata
        """
        return self.load_neural_data_neo(ns6_file_path)
    
    def get_data_info(self):
        """
        Get summary information about loaded data.
        
        Returns:
        --------
        dict
            Summary of loaded data
        """
        info = {
            'behavioral_data_loaded': self.behavioral_data is not None,
            'neural_data_loaded': self.neural_data is not None,
            'metadata': self.metadata.copy() if self.metadata else {}
        }
        
        if self.behavioral_data is not None:
            info['behavioral_shape'] = self.behavioral_data.shape
            info['behavioral_columns'] = list(self.behavioral_data.columns)
            
        if self.neural_data is not None:
            info['neural_shape'] = self.neural_data['raw_data'].shape
            info['neural_channels'] = len(self.neural_data['channel_names'])
            
        return info 