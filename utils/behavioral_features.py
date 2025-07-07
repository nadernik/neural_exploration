"""
Behavioral feature extraction utilities for center-out task analysis.

This module provides functions for extracting behavioral features from trial data including:
- Reaction time and movement time analysis
- Cursor trajectory reconstruction and analysis  
- Velocity and speed profiles
- Endpoint accuracy and path metrics
- Trial outcome analysis

Works with both CSV-based behavioral data and integrated H5 files.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BehavioralFeatureExtractor:
    """
    Class for extracting behavioral features from center-out task data.
    """
    
    def __init__(self, behavioral_data: pd.DataFrame = None):
        """
        Initialize the behavioral feature extractor.
        
        Parameters:
        -----------
        behavioral_data : pandas.DataFrame, optional
            Behavioral data with trial segmentation. Can be None if loading from H5.
        """
        self.behavioral_data = behavioral_data.copy() if behavioral_data is not None else None
        self.target_positions = self._generate_target_positions()
        self.h5_loader = None
        
    def _generate_target_positions(self, radius=1.0):
        """
        Generate target positions for center-out task using hardcoded angles.
        
        Parameters:
        -----------
        radius : float
            Target radius from center (default: 1.0)
            
        Returns:
        --------
        dict
            Dictionary mapping target_index to (x, y) positions
        """
        # Hardcoded target angles as specified
        target_angles = {
            0: 90,    # T0: 90°
            1: 45,    # T1: 45°
            2: 0,     # T2: 0°
            3: -45,   # T3: -45°
            4: -90,   # T4: -90°
            5: -135,  # T5: -135°
            6: 180,   # T6: 180°
            7: 135    # T7: 135°
        }
        
        positions = {}
        
        for target_idx, angle_deg in target_angles.items():
            # Convert to radians
            angle_rad = np.deg2rad(angle_deg)
            
            positions[target_idx] = {
                'x': radius * np.cos(angle_rad),
                'y': radius * np.sin(angle_rad),
                'angle': angle_rad,
                'direction': f'{angle_deg}°'
            }
        
        return positions
    
    def load_from_h5(self, h5_file_path: str, trial_numbers: Optional[List[int]] = None):
        """
        Load behavioral data from an H5 file.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to the H5 file
        trial_numbers : list, optional
            List of trial numbers to load. If None, loads all trials.
        """
        from .h5_data_loader import H5DataLoader
        
        self.h5_loader = H5DataLoader(h5_file_path)
        self.behavioral_data = self.h5_loader.load_behavioral_data(trial_numbers)
        
        print(f"✅ Loaded behavioral data from H5 file: {h5_file_path}")
        return self.behavioral_data
    
    def extract_trial_features_from_h5_trial(self, trial_data: Dict) -> Dict:
        """
        Extract behavioral features from a single H5 trial data dictionary.
        
        Parameters:
        -----------
        trial_data : dict
            Trial data dictionary from H5DataLoader.load_trial_data()
            
        Returns:
        --------
        dict
            Dictionary with all extracted features
        """
        trial_number = trial_data['trial_number']
        
        # Convert H5 trial data to DataFrame format
        if (trial_data['velocity_x'] is None or 
            trial_data['velocity_y'] is None or 
            trial_data['behavioral_timestamps'] is None):
            return {}
        
        # Convert timestamps
        if isinstance(trial_data['behavioral_timestamps'][0], (int, float)):
            timestamps = pd.to_datetime(trial_data['behavioral_timestamps'], unit='s')
        else:
            timestamps = pd.to_datetime(trial_data['behavioral_timestamps'])
        
        # Create trial DataFrame
        trial_df = pd.DataFrame({
            'timestamp': timestamps,
            'velocity_x': trial_data['velocity_x'],
            'velocity_y': trial_data['velocity_y'],
            'trial': trial_number,
            'trial_outcome': trial_data.get('outcome', 'unknown'),
            'target_index': trial_data.get('target_index', -1),
        })
        
        # Add trial markers
        trial_df['trial_start'] = False
        trial_df['trial_win'] = False
        trial_df['trial_lose'] = False
        
        # Mark trial start and end
        trial_df.iloc[0, trial_df.columns.get_loc('trial_start')] = True
        if trial_data.get('outcome') == 'win':
            trial_df.iloc[-1, trial_df.columns.get_loc('trial_win')] = True
        elif trial_data.get('outcome') == 'lose':
            trial_df.iloc[-1, trial_df.columns.get_loc('trial_lose')] = True
        
        # Extract features using existing methods
        trial_df = self.reconstruct_cursor_trajectory(trial_df)
        
        features = {
            'trial_number': trial_number,
            'trial_outcome': trial_data.get('outcome', 'unknown'),
            'target_index': trial_data.get('target_index', -1),
            'reaction_time': self.extract_reaction_time(trial_df),
            'movement_time': self.extract_movement_time(trial_df),
            'endpoint_error': self.extract_endpoint_error(trial_df),
            'speed_profile': self.extract_speed_profile(trial_df),
            'path_metrics': self.extract_path_metrics(trial_df),
            'trial_duration': trial_data.get('duration', (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()),
            'trajectory_data': trial_df[['timestamp', 'cursor_x', 'cursor_y', 'velocity_x', 'velocity_y']].copy()
        }
        
        return features
    
    def reconstruct_cursor_trajectory(self, trial_data: pd.DataFrame, 
                                    start_position: Tuple[float, float] = (0.0, 0.0)) -> pd.DataFrame:
        """
        Reconstruct cursor trajectory from velocity data.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data with velocity_x and velocity_y columns
        start_position : tuple
            Starting position (x, y) for trajectory reconstruction
            
        Returns:
        --------
        pandas.DataFrame
            Trial data with added cursor_x and cursor_y columns
        """
        trial_data = trial_data.copy()
        
        # Calculate time intervals
        if 'timestamp' in trial_data.columns:
            time_diffs = trial_data['timestamp'].diff().dt.total_seconds()
            time_diffs.iloc[0] = 0.0  # First point has no time diff
        else:
            # Assume uniform sampling if no timestamp
            time_diffs = pd.Series([0.02] * len(trial_data))  # 50Hz default
        
        # Integrate velocity to get position
        dx = trial_data['velocity_x'] * time_diffs
        dy = trial_data['velocity_y'] * time_diffs
        
        # Cumulative sum to get trajectory
        trial_data['cursor_x'] = start_position[0] + dx.cumsum()
        trial_data['cursor_y'] = start_position[1] + dy.cumsum()
        
        return trial_data
    
    def extract_reaction_time(self, trial_data: pd.DataFrame, 
                            movement_threshold: float = 0.05) -> Optional[float]:
        """
        Extract reaction time from trial start to movement onset.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data
        movement_threshold : float
            Speed threshold for movement detection (units/s)
            
        Returns:
        --------
        float or None
            Reaction time in seconds, or None if not detectable
        """
        # Find trial start
        trial_start_idx = trial_data.index[trial_data['trial_start'] == True]
        if len(trial_start_idx) == 0:
            return None
        
        trial_start_time = trial_data.loc[trial_start_idx[0], 'timestamp']
        
        # Calculate speed
        speed = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
        
        # Find movement onset (first point above threshold after trial start)
        post_start_data = trial_data[trial_data['timestamp'] > trial_start_time]
        post_start_speed = np.sqrt(post_start_data['velocity_x']**2 + post_start_data['velocity_y']**2)
        
        movement_onset_idx = post_start_speed[post_start_speed > movement_threshold].index
        if len(movement_onset_idx) == 0:
            return None
        
        movement_onset_time = post_start_data.loc[movement_onset_idx[0], 'timestamp']
        reaction_time = (movement_onset_time - trial_start_time).total_seconds()
        
        return reaction_time
    
    def extract_movement_time(self, trial_data: pd.DataFrame, 
                            movement_threshold: float = 0.05) -> Optional[float]:
        """
        Extract movement time from movement onset to target reach.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data
        movement_threshold : float
            Speed threshold for movement detection (units/s)
            
        Returns:
        --------
        float or None
            Movement time in seconds, or None if not detectable
        """
        # Find movement onset
        speed = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
        movement_onset_idx = speed[speed > movement_threshold].index
        if len(movement_onset_idx) == 0:
            return None
        
        movement_onset_time = trial_data.loc[movement_onset_idx[0], 'timestamp']
        
        # Find trial end (win or lose)
        trial_end_idx = trial_data.index[
            (trial_data['trial_win'] == True) | (trial_data['trial_lose'] == True)
        ]
        if len(trial_end_idx) == 0:
            return None
        
        trial_end_time = trial_data.loc[trial_end_idx[0], 'timestamp']
        movement_time = (trial_end_time - movement_onset_time).total_seconds()
        
        return movement_time
    
    def extract_speed_profile(self, trial_data: pd.DataFrame) -> Dict:
        """
        Extract speed profile features from trial data.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data
            
        Returns:
        --------
        dict
            Dictionary with speed profile features
        """
        # Calculate speed
        speed = np.sqrt(trial_data['velocity_x']**2 + trial_data['velocity_y']**2)
        
        # Calculate time array
        if 'timestamp' in trial_data.columns:
            time_from_start = (trial_data['timestamp'] - trial_data['timestamp'].iloc[0]).dt.total_seconds()
        else:
            time_from_start = np.arange(len(trial_data)) * 0.02  # 50Hz default
        
        features = {
            'speed': speed.values,
            'time': time_from_start.values,
            'max_speed': np.max(speed),
            'mean_speed': np.mean(speed),
            'peak_speed_time': time_from_start.iloc[np.argmax(speed)],
            'speed_variability': np.std(speed)
        }
        
        return features
    
    def extract_endpoint_error(self, trial_data: pd.DataFrame) -> Optional[float]:
        """
        Extract endpoint error (distance from target at movement end).
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data with reconstructed trajectory
            
        Returns:
        --------
        float or None
            Endpoint error in distance units, or None if not calculable
        """
        # Check if we have cursor position data
        if 'cursor_x' not in trial_data.columns or 'cursor_y' not in trial_data.columns:
            # Reconstruct trajectory if needed
            trial_data = self.reconstruct_cursor_trajectory(trial_data)
        
        # Get target position
        if 'target_index' not in trial_data.columns:
            return None
        
        target_idx = trial_data['target_index'].iloc[0]
        if target_idx not in self.target_positions:
            return None
        
        target_pos = self.target_positions[target_idx]
        
        # Get endpoint position (last position in trial)
        endpoint_x = trial_data['cursor_x'].iloc[-1]
        endpoint_y = trial_data['cursor_y'].iloc[-1]
        
        # Calculate distance to target
        error = np.sqrt((endpoint_x - target_pos['x'])**2 + (endpoint_y - target_pos['y'])**2)
        
        return error
    
    def extract_path_metrics(self, trial_data: pd.DataFrame) -> Dict:
        """
        Extract path length and curvature metrics.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            Single trial data with reconstructed trajectory
            
        Returns:
        --------
        dict
            Dictionary with path metrics
        """
        # Check if we have cursor position data
        if 'cursor_x' not in trial_data.columns or 'cursor_y' not in trial_data.columns:
            # Reconstruct trajectory if needed
            trial_data = self.reconstruct_cursor_trajectory(trial_data)
        
        x = trial_data['cursor_x'].values
        y = trial_data['cursor_y'].values
        
        # Calculate path length
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        path_length = np.sum(segment_lengths)
        
        # Calculate straight-line distance to target
        if 'target_index' in trial_data.columns:
            target_idx = trial_data['target_index'].iloc[0]
            if target_idx in self.target_positions:
                target_pos = self.target_positions[target_idx]
                straight_distance = np.sqrt((x[0] - target_pos['x'])**2 + (y[0] - target_pos['y'])**2)
            else:
                straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        else:
            straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        
        # Calculate path efficiency
        path_efficiency = straight_distance / path_length if path_length > 0 else 0
        
        # Calculate curvature (simplified as total angular change)
        if len(x) > 2:
            # Calculate angles between consecutive segments
            angles = np.arctan2(dy, dx)
            angle_changes = np.diff(angles)
            # Normalize angles to [-pi, pi]
            angle_changes = ((angle_changes + np.pi) % (2*np.pi)) - np.pi
            total_curvature = np.sum(np.abs(angle_changes))
        else:
            total_curvature = 0.0
        
        return {
            'path_length': path_length,
            'straight_distance': straight_distance,
            'path_efficiency': path_efficiency,
            'total_curvature': total_curvature
        }
    
    def extract_trial_features(self, trial_number: int) -> Dict:
        """
        Extract all behavioral features for a single trial.
        
        Parameters:
        -----------
        trial_number : int
            Trial number to analyze
            
        Returns:
        --------
        dict
            Dictionary with all extracted features
        """
        # Get trial data
        trial_data = self.behavioral_data[self.behavioral_data['trial'] == trial_number].copy()
        
        if len(trial_data) == 0:
            return {}
        
        # Reconstruct cursor trajectory
        trial_data = self.reconstruct_cursor_trajectory(trial_data)
        
        # Extract features
        features = {
            'trial_number': trial_number,
            'trial_outcome': trial_data['trial_outcome'].iloc[0],
            'target_index': trial_data['target_index'].iloc[0] if 'target_index' in trial_data.columns else None,
            'reaction_time': self.extract_reaction_time(trial_data),
            'movement_time': self.extract_movement_time(trial_data),
            'endpoint_error': self.extract_endpoint_error(trial_data),
            'speed_profile': self.extract_speed_profile(trial_data),
            'path_metrics': self.extract_path_metrics(trial_data),
            'trial_duration': (trial_data['timestamp'].iloc[-1] - trial_data['timestamp'].iloc[0]).total_seconds(),
            'trajectory_data': trial_data[['timestamp', 'cursor_x', 'cursor_y', 'velocity_x', 'velocity_y']].copy()
        }
        
        return features
    
    def extract_all_trials_features(self, trial_numbers: Optional[List[int]] = None) -> Dict:
        """
        Extract features for all trials or specified trials.
        
        Parameters:
        -----------
        trial_numbers : list, optional
            List of trial numbers to analyze. If None, analyzes all trials.
            
        Returns:
        --------
        dict
            Dictionary with features for all trials
        """
        if self.behavioral_data is not None:
            # Use DataFrame-based extraction
            if trial_numbers is None:
                trial_numbers = sorted(self.behavioral_data['trial'].unique())
            
            all_features = {}
            
            for trial_num in trial_numbers:
                try:
                    features = self.extract_trial_features(trial_num)
                    if features:  # Only add if features were successfully extracted
                        all_features[trial_num] = features
                except Exception as e:
                    print(f"Warning: Failed to extract features for trial {trial_num}: {e}")
                    continue
            
            return all_features
        
        elif self.h5_loader is not None:
            # Use H5-based extraction
            if trial_numbers is None:
                metadata = self.h5_loader.load_file_metadata()
                trial_numbers = metadata.get('available_trials', [])
            
            all_features = {}
            
            for trial_num in trial_numbers:
                try:
                    trial_data = self.h5_loader.load_trial_data(trial_num)
                    features = self.extract_trial_features_from_h5_trial(trial_data)
                    if features:  # Only add if features were successfully extracted
                        all_features[trial_num] = features
                except Exception as e:
                    print(f"Warning: Failed to extract features for trial {trial_num}: {e}")
                    continue
            
            return all_features
        
        else:
            raise ValueError("No behavioral data loaded. Use load_from_h5() or provide behavioral_data during initialization.")
    
    def extract_all_trials_features_from_h5(self, h5_file_path: str, trial_numbers: Optional[List[int]] = None) -> Dict:
        """
        Extract features for all trials directly from H5 file without loading into memory.
        
        Parameters:
        -----------
        h5_file_path : str
            Path to the H5 file
        trial_numbers : list, optional
            List of trial numbers to analyze. If None, analyzes all trials.
            
        Returns:
        --------
        dict
            Dictionary with features for all trials
        """
        from .h5_data_loader import H5DataLoader
        
        h5_loader = H5DataLoader(h5_file_path)
        
        if trial_numbers is None:
            metadata = h5_loader.load_file_metadata()
            trial_numbers = metadata.get('available_trials', [])
        
        all_features = {}
        
        print(f"🚀 Extracting features from H5 file for {len(trial_numbers)} trials...")
        
        for i, trial_num in enumerate(trial_numbers):
            try:
                trial_data = h5_loader.load_trial_data(trial_num)
                features = self.extract_trial_features_from_h5_trial(trial_data)
                if features:  # Only add if features were successfully extracted
                    all_features[trial_num] = features
                
                # Progress update
                if (i + 1) % 10 == 0 or (i + 1) == len(trial_numbers):
                    print(f"   Processed {i + 1}/{len(trial_numbers)} trials")
                    
            except Exception as e:
                print(f"Warning: Failed to extract features for trial {trial_num}: {e}")
                continue
        
        print(f"✅ Feature extraction complete! Processed {len(all_features)} trials successfully.")
        
        return all_features
    
    def compute_summary_statistics(self, trial_features: Dict) -> Dict:
        """
        Compute summary statistics across all trials.
        
        Parameters:
        -----------
        trial_features : dict
            Dictionary of trial features from extract_all_trials_features
            
        Returns:
        --------
        dict
            Summary statistics across trials
        """
        if not trial_features:
            return {}
        
        # Collect metrics across trials
        reaction_times = []
        movement_times = []
        endpoint_errors = []
        max_speeds = []
        path_lengths = []
        path_efficiencies = []
        
        successful_trials = []
        failed_trials = []
        
        for trial_num, features in trial_features.items():
            if features['reaction_time'] is not None:
                reaction_times.append(features['reaction_time'])
            if features['movement_time'] is not None:
                movement_times.append(features['movement_time'])
            if features['endpoint_error'] is not None:
                endpoint_errors.append(features['endpoint_error'])
            if 'max_speed' in features['speed_profile']:
                max_speeds.append(features['speed_profile']['max_speed'])
            if 'path_length' in features['path_metrics']:
                path_lengths.append(features['path_metrics']['path_length'])
            if 'path_efficiency' in features['path_metrics']:
                path_efficiencies.append(features['path_metrics']['path_efficiency'])
            
            # Trial outcomes
            if features['trial_outcome'] == 'win':
                successful_trials.append(trial_num)
            else:
                failed_trials.append(trial_num)
        
        # Calculate summary statistics
        summary = {
            'n_trials': len(trial_features),
            'success_rate': len(successful_trials) / len(trial_features) if trial_features else 0,
            'successful_trials': successful_trials,
            'failed_trials': failed_trials,
            'reaction_time': {
                'mean': np.mean(reaction_times) if reaction_times else None,
                'std': np.std(reaction_times) if reaction_times else None,
                'median': np.median(reaction_times) if reaction_times else None,
                'values': reaction_times
            },
            'movement_time': {
                'mean': np.mean(movement_times) if movement_times else None,
                'std': np.std(movement_times) if movement_times else None,
                'median': np.median(movement_times) if movement_times else None,
                'values': movement_times
            },
            'endpoint_error': {
                'mean': np.mean(endpoint_errors) if endpoint_errors else None,
                'std': np.std(endpoint_errors) if endpoint_errors else None,
                'median': np.median(endpoint_errors) if endpoint_errors else None,
                'values': endpoint_errors
            },
            'max_speed': {
                'mean': np.mean(max_speeds) if max_speeds else None,
                'std': np.std(max_speeds) if max_speeds else None,
                'median': np.median(max_speeds) if max_speeds else None,
                'values': max_speeds
            },
            'path_length': {
                'mean': np.mean(path_lengths) if path_lengths else None,
                'std': np.std(path_lengths) if path_lengths else None,
                'median': np.median(path_lengths) if path_lengths else None,
                'values': path_lengths
            },
            'path_efficiency': {
                'mean': np.mean(path_efficiencies) if path_efficiencies else None,
                'std': np.std(path_efficiencies) if path_efficiencies else None,
                'median': np.median(path_efficiencies) if path_efficiencies else None,
                'values': path_efficiencies
            }
        }
        
        return summary
    
 