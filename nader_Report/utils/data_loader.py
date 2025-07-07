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
        self.neural_metadata = {}
        self.behavioral_metadata = {}
        self.time_origin = None  # Global time reference from .ns6 file
        
    def load_behavioral_data(self, csv_file_path=None, force_reload=False):
        """
        Load behavioral data from CSV file and segment trials.
        
        Parameters:
        -----------
        csv_file_path : str, optional
            Path to CSV file. If None, uses self.csv_file_path
        force_reload : bool, optional
            If True, reload even if same file is already loaded
            
        Returns:
        --------
        pandas.DataFrame
            Behavioral data with proper timestamps and trial segmentation
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path
            
        if csv_file_path is None:
            raise ValueError("No CSV file path provided")
        
        # Check if same file is already loaded
        if (not force_reload and 
            self.behavioral_data is not None and 
            self.behavioral_metadata.get('file_path') == csv_file_path):
            print(f"Behavioral data from {csv_file_path} is already loaded.")
            print(f"Skipping reload (use force_reload=True to reload anyway)")
            print(f"Loaded data info:")
            print(f"  - Shape: {self.behavioral_data.shape}")
            print(f"  - Columns: {list(self.behavioral_data.columns)}")
            if 'trial' in self.behavioral_data.columns:
                n_trials = self.behavioral_data['trial'].nunique()
                print(f"  - Trials: {n_trials}")
            return self.behavioral_data
            
        try:
            # Load CSV data
            self.behavioral_data = pd.read_csv(csv_file_path)
            
            # Process timestamps if available
            if 'timestamp' in self.behavioral_data.columns:
                self.behavioral_data['timestamp'] = pd.to_datetime(
                    self.behavioral_data['timestamp'], 
                    errors='coerce'
                )
                
                # Store metadata including file path
                first_timestamp = self.behavioral_data['timestamp'].iloc[0]
                last_timestamp = self.behavioral_data['timestamp'].iloc[-1]
                duration = (last_timestamp - first_timestamp).total_seconds()
                
                self.behavioral_metadata = {
                    'first_timestamp': first_timestamp,
                    'last_timestamp': last_timestamp,
                    'duration': duration,
                    'n_samples': len(self.behavioral_data),
                    'file_path': csv_file_path
                }
            else:
                self.behavioral_metadata = {
                    'file_path': csv_file_path,
                    'n_samples': len(self.behavioral_data)
                }
            
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
        
        # Convert boolean columns to proper boolean type (handle 1/0, 'True'/'False', etc.)
        for col in ['trial_start', 'trial_win', 'trial_lose']:
            if col in data.columns:
                # Handle different boolean representations
                if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numeric: 1 = True, 0 = False
                    data[col] = data[col] > 0
                elif data[col].dtype == 'bool':
                    # Already boolean, keep as is
                    pass
                else:
                    # String representations: convert various formats to boolean
                    data[col] = data[col].astype(str).str.lower().str.strip().isin(['true', '1', '1.0', 'yes'])
        
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
            win_indices = trial_segment.index[trial_segment.get('trial_win', pd.Series([False]*len(trial_segment))) == True]
            lose_indices = trial_segment.index[trial_segment.get('trial_lose', pd.Series([False]*len(trial_segment))) == True]
            
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
    
    def load_neural_data(self, file_path, force_reload=False, 
                        max_channels=None, max_duration=None, 
                        start_time=0, downsample_factor=1):
        """
        Load neural data from .ns6 file using Neo with memory-efficient options.
        
        Parameters:
        -----------
        file_path : str
            Path to the .ns6 file
        force_reload : bool, optional
            If True, reload even if same file is already loaded
        max_channels : int, optional
            Maximum number of channels to load (loads first N channels)
        max_duration : float, optional
            Maximum duration to load in seconds (loads from start_time)
        start_time : float, optional
            Start time in seconds (default: 0)
        downsample_factor : int, optional
            Downsample factor (1 = no downsampling, 2 = half sample rate, etc.)
            
        Returns:
        --------
        dict
            Dictionary containing neural data and metadata
        """
        # Check if same file is already loaded
        if (not force_reload and 
            self.neural_data is not None and 
            self.neural_metadata.get('file_path') == file_path):
            print(f"Neural data from {file_path} is already loaded.")
            print(f"Skipping reload (use force_reload=True to reload anyway)")
            print(f"Loaded data info:")
            print(f"  - Shape: {self.neural_data['raw_data'].shape}")
            print(f"  - Duration: {self.neural_metadata['duration']:.1f} seconds")
            print(f"  - Channels: {self.neural_metadata['n_channels']}")
            print(f"  - Sampling rate: {self.neural_metadata['sampling_rate']} Hz")
            print(f"  - Time Origin: {self.time_origin}")
            return self.neural_data
        
        print(f"Loading neural data from: {file_path}")
        
        # Print memory-efficient loading options
        if max_channels is not None:
            print(f"  - Loading only first {max_channels} channels")
        if max_duration is not None:
            print(f"  - Loading only {max_duration} seconds starting from {start_time}s")
        if downsample_factor > 1:
            print(f"  - Downsampling by factor of {downsample_factor}")
        
        try:
            # Create Neo reader for Blackrock files
            reader = neo.BlackrockIO(filename=file_path)
            
            # First, get basic file info without loading data
            try:
                # Get Time Origin from the reader metadata
                if hasattr(reader, 'datetime'):
                    self.time_origin = reader.datetime
                elif hasattr(reader, 'rec_datetime'):
                    self.time_origin = reader.rec_datetime
                else:
                    # Try to get from header
                    try:
                        if hasattr(reader, 'header'):
                            self.time_origin = reader.header.get('datetime', None)
                        elif hasattr(reader, '_read_header'):
                            header = reader._read_header()
                            self.time_origin = header.get('datetime', None)
                    except:
                        pass
                
                # If no time origin found, use a default
                if self.time_origin is None:
                    print("Warning: Could not extract Time Origin from .ns6 file. Using provided timestamp.")
                    self.time_origin = datetime(2025, 3, 25, 9, 22, 53, tzinfo=timezone.utc)
                
                print(f"Neural data Time Origin: {self.time_origin}")
                
                # Get basic file info
                header = reader.header
                if 'signal_channels' in header:
                    total_channels = len(header['signal_channels'])
                    original_sampling_rate = float(header['signal_channels'][0][2])  # sampling rate
                    
                    print(f"File contains {total_channels} channels at {original_sampling_rate} Hz")
                    
                    # Estimate memory usage
                    if max_duration is None:
                        # Try to get duration from header
                        if hasattr(reader, 'segment_duration'):
                            total_duration = reader.segment_duration(0)
                        else:
                            total_duration = None
                    else:
                        total_duration = max_duration
                    
                    if total_duration is not None:
                        channels_to_load = min(total_channels, max_channels or total_channels)
                        effective_sampling_rate = original_sampling_rate / downsample_factor
                        estimated_samples = int(total_duration * effective_sampling_rate)
                        estimated_memory_gb = (estimated_samples * channels_to_load * 4) / (1024**3)  # 4 bytes per float32
                        
                        print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB")
                        
                        if estimated_memory_gb > 4.0:
                            print("WARNING: Estimated memory usage is high. Consider reducing:")
                            print(f"  - max_channels (currently: {channels_to_load})")
                            print(f"  - max_duration (currently: {total_duration})")
                            print(f"  - downsample_factor (currently: {downsample_factor})")
                
            except Exception as e:
                print(f"Could not get file info: {e}")
                print("Proceeding with data loading...")
            
            # Read the data with memory-efficient options
            if max_duration is not None or max_channels is not None:
                # Use lazy loading approach
                block = reader.read_block(lazy=True)
                segment = block.segments[0]
                analog_signals = segment.analogsignals
                
                if len(analog_signals) > 0:
                    raw_signal = analog_signals[0]
                    
                    # Calculate time indices
                    sampling_rate = float(raw_signal.sampling_rate.magnitude)
                    start_idx = int(start_time * sampling_rate)
                    
                    if max_duration is not None:
                        end_idx = int((start_time + max_duration) * sampling_rate)
                    else:
                        end_idx = None
                    
                    # Load data with slicing
                    if end_idx is not None:
                        raw_data = raw_signal.magnitude[start_idx:end_idx]
                        times = raw_signal.times.magnitude[start_idx:end_idx]
                    else:
                        raw_data = raw_signal.magnitude[start_idx:]
                        times = raw_signal.times.magnitude[start_idx:]
                    
                    # Adjust times to start from 0
                    times = times - times[0]
                    
                    # Channel selection
                    if max_channels is not None and raw_data.shape[1] > max_channels:
                        raw_data = raw_data[:, :max_channels]
                        print(f"Selected first {max_channels} channels out of {raw_signal.shape[1]} available")
                    
                    # Downsampling
                    if downsample_factor > 1:
                        raw_data = raw_data[::downsample_factor]
                        times = times[::downsample_factor]
                        sampling_rate = sampling_rate / downsample_factor
                        print(f"Downsampled by factor of {downsample_factor}, new sampling rate: {sampling_rate} Hz")
                    
                else:
                    print("No analog signals found in the neural data file.")
                    return None
                    
            else:
                # Standard loading (for small files)
                block = reader.read_block()
                segment = block.segments[0]
                analog_signals = segment.analogsignals
                
                if len(analog_signals) > 0:
                    raw_signal = analog_signals[0]
                    raw_data = raw_signal.magnitude
                    times = raw_signal.times.magnitude
                    sampling_rate = float(raw_signal.sampling_rate.magnitude)
                    
                    # Downsampling if requested
                    if downsample_factor > 1:
                        raw_data = raw_data[::downsample_factor]
                        times = times[::downsample_factor]
                        sampling_rate = sampling_rate / downsample_factor
                        print(f"Downsampled by factor of {downsample_factor}, new sampling rate: {sampling_rate} Hz")
                        
                else:
                    print("No analog signals found in the neural data file.")
                    return None
            
            print(f"Loaded neural data shape: {raw_data.shape}")
            print(f"Effective sampling rate: {sampling_rate} Hz")
            print(f"Loaded duration: {len(times)/sampling_rate:.2f} seconds")
            print(f"Number of channels: {raw_data.shape[1] if len(raw_data.shape) > 1 else 1}")
            
            # Store metadata
            self.neural_metadata = {
                'sampling_rate': sampling_rate,
                'n_channels': raw_data.shape[1] if len(raw_data.shape) > 1 else 1,
                'duration': len(times) / sampling_rate,
                'time_origin': self.time_origin,
                'file_path': file_path,
                'max_channels': max_channels,
                'max_duration': max_duration,
                'start_time': start_time,
                'downsample_factor': downsample_factor
            }
            
            # Store data
            self.neural_data = {
                'raw_data': raw_data,
                'times': times,
                'sampling_rate': sampling_rate,
                'time_origin': self.time_origin
            }
            
            print("Neural data loaded successfully!")
            return self.neural_data
            
        except Exception as e:
            print(f"Error loading neural data: {e}")
            print("Make sure the Neo library is installed and the file format is supported.")
            
            # Suggest memory-efficient options
            if "unable to allocate" in str(e).lower() or "memory" in str(e).lower():
                print("\nMemory Error Solutions:")
                print("1. Load fewer channels: loader.load_neural_data(file_path, max_channels=32)")
                print("2. Load shorter duration: loader.load_neural_data(file_path, max_duration=60)")
                print("3. Downsample the data: loader.load_neural_data(file_path, downsample_factor=10)")
                print("4. Combine options: loader.load_neural_data(file_path, max_channels=16, max_duration=30, downsample_factor=5)")
            
            return None
    
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
            'neural_metadata': self.neural_metadata.copy() if self.neural_metadata else {},
            'behavioral_metadata': self.behavioral_metadata.copy() if self.behavioral_metadata else {}
        }
        
        if self.behavioral_data is not None:
            info['behavioral_shape'] = self.behavioral_data.shape
            info['behavioral_columns'] = list(self.behavioral_data.columns)
            
        if self.neural_data is not None:
            info['neural_shape'] = self.neural_data['raw_data'].shape
            info['neural_channels'] = len(self.neural_data['channel_names'])
            
        return info 

    def align_timestamps(self):
        """
        Align behavioral and neural data timestamps using the neural Time Origin as reference.
        
        Returns:
        --------
        dict
            Dictionary containing alignment information and aligned timestamps
        """
        if self.neural_data is None or self.behavioral_data is None:
            print("Error: Both neural and behavioral data must be loaded first.")
            return None
        
        if self.time_origin is None:
            print("Error: No time origin available from neural data.")
            return None
        
        print("Aligning behavioral and neural timestamps...")
        
        # Convert neural times to absolute timestamps
        neural_start_time = self.time_origin
        neural_times_absolute = [neural_start_time + pd.Timedelta(seconds=t) for t in self.neural_data['times']]
        
        # Get behavioral timestamps
        if 'timestamp' in self.behavioral_data.columns:
            behavioral_times = self.behavioral_data['timestamp']
        else:
            print("Error: No timestamp column found in behavioral data.")
            return None
        
        # Calculate alignment offset
        behavioral_start = behavioral_times.iloc[0]
        neural_start = neural_times_absolute[0]
        
        # Time offset between the two data streams
        time_offset = (behavioral_start - neural_start).total_seconds()
        
        print(f"Neural recording started at: {neural_start}")
        print(f"Behavioral recording started at: {behavioral_start}")
        print(f"Time offset: {time_offset:.3f} seconds")
        print(f"Neural started {-time_offset:.1f} seconds {'before' if time_offset < 0 else 'after'} behavioral")
        
        # Convert all timestamps to seconds since neural time origin
        behavioral_times_aligned = [(t - neural_start_time).total_seconds() for t in behavioral_times]
        neural_times_aligned = list(self.neural_data['times'])
        
        # Add aligned timestamps to behavioral data
        self.behavioral_data['timestamp_aligned'] = behavioral_times_aligned
        
        # Add aligned timestamps to neural data
        self.neural_data['times_aligned'] = neural_times_aligned
        
        # Find overlapping time period
        behavioral_start_sec = min(behavioral_times_aligned)
        behavioral_end_sec = max(behavioral_times_aligned)
        neural_start_sec = min(neural_times_aligned)
        neural_end_sec = max(neural_times_aligned)
        
        overlap_start = max(behavioral_start_sec, neural_start_sec)
        overlap_end = min(behavioral_end_sec, neural_end_sec)
        overlap_duration = overlap_end - overlap_start
        
        print(f"Behavioral data spans: {behavioral_start_sec:.3f} to {behavioral_end_sec:.3f} seconds")
        print(f"Neural data spans: {neural_start_sec:.3f} to {neural_end_sec:.3f} seconds")
        print(f"Overlapping period: {overlap_start:.3f} to {overlap_end:.3f} seconds")
        print(f"Overlap duration: {overlap_duration:.3f} seconds")
        
        alignment_info = {
            'time_origin': self.time_origin,
            'time_offset_seconds': time_offset,
            'behavioral_start_aligned': behavioral_start_sec,
            'behavioral_end_aligned': behavioral_end_sec,
            'neural_start_aligned': neural_start_sec,
            'neural_end_aligned': neural_end_sec,
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'overlap_duration': overlap_duration,
            'alignment_quality': 'good' if overlap_duration > 0 else 'poor'
        }
        
        print(f"Alignment completed! Quality: {alignment_info['alignment_quality']}")
        return alignment_info
    
    def get_overlapping_data(self, start_time=None, end_time=None):
        """
        Extract overlapping portions of behavioral and neural data.
        
        Parameters:
        -----------
        start_time : float, optional
            Start time in seconds since neural time origin
        end_time : float, optional
            End time in seconds since neural time origin
            
        Returns:
        --------
        dict
            Dictionary containing overlapping data portions
        """
        if 'timestamp_aligned' not in self.behavioral_data.columns:
            print("Error: Data not aligned yet. Run align_timestamps() first.")
            return None
        
        # Determine time range
        if start_time is None or end_time is None:
            # Find natural overlap
            behavioral_times = self.behavioral_data['timestamp_aligned']
            neural_times = self.neural_data['times_aligned']
            
            start_time = max(min(behavioral_times), min(neural_times))
            end_time = min(max(behavioral_times), max(neural_times))
        
        print(f"Extracting overlapping data from {start_time:.3f} to {end_time:.3f} seconds")
        
        # Filter behavioral data
        behavioral_mask = (
            (self.behavioral_data['timestamp_aligned'] >= start_time) &
            (self.behavioral_data['timestamp_aligned'] <= end_time)
        )
        behavioral_overlap = self.behavioral_data[behavioral_mask].copy()
        
        # Filter neural data
        neural_mask = (
            (self.neural_data['times_aligned'] >= start_time) &
            (self.neural_data['times_aligned'] <= end_time)
        )
        neural_overlap = {
            'raw_data': self.neural_data['raw_data'][neural_mask],
            'times': self.neural_data['times'][neural_mask],
            'times_aligned': np.array(self.neural_data['times_aligned'])[neural_mask],
            'sampling_rate': self.neural_data['sampling_rate']
        }
        
        print(f"Extracted {len(behavioral_overlap)} behavioral samples")
        print(f"Extracted {len(neural_overlap['times'])} neural samples")
        
        return {
            'behavioral_data': behavioral_overlap,
            'neural_data': neural_overlap,
            'time_range': (start_time, end_time),
            'duration': end_time - start_time
        }
    
    def convert_to_blackrock_ticks(self, timestamps):
        """
        Convert timestamps to Blackrock time ticks for precise alignment.
        
        Parameters:
        -----------
        timestamps : array-like
            Timestamps in seconds since neural time origin
            
        Returns:
        --------
        numpy.ndarray
            Timestamps in Blackrock time ticks (30 kHz clock)
        """
        # Blackrock systems typically use 30 kHz clock for timing
        blackrock_clock_rate = 30000  # Hz
        
        # Convert seconds to ticks
        ticks = np.array(timestamps) * blackrock_clock_rate
        
        return ticks.astype(np.int64)
    
    def segment_aligned_trials(self):
        """
        Segment trials using aligned timestamps.
        
        Returns:
        --------
        dict
            Dictionary containing segmented trial data
        """
        if 'timestamp_aligned' not in self.behavioral_data.columns:
            print("Error: Data not aligned yet. Run align_timestamps() first.")
            return None
        
        print("Segmenting aligned trials...")
        
        # Use existing trial segmentation logic but with aligned timestamps
        data = self.behavioral_data.copy()
        
        # Find trial boundaries
        trial_starts = data[data['trial_start'] == True].index
        trial_ends_win = data[data['trial_win'] == True].index
        trial_ends_lose = data[data['trial_lose'] == True].index
        
        # Combine and sort all end points
        all_ends = np.concatenate([trial_ends_win, trial_ends_lose])
        all_ends = np.sort(all_ends)
        
        trials = []
        trial_id = 0
        
        for start_idx in trial_starts:
            # Find the next end point after this start
            end_candidates = all_ends[all_ends > start_idx]
            
            if len(end_candidates) > 0:
                end_idx = end_candidates[0]
                
                # Extract trial data
                trial_data = data.iloc[start_idx:end_idx+1].copy()
                trial_data['trial'] = trial_id
                
                # Determine outcome
                if end_idx in trial_ends_win:
                    trial_data['trial_outcome'] = 'win'
                elif end_idx in trial_ends_lose:
                    trial_data['trial_outcome'] = 'lose'
                else:
                    trial_data['trial_outcome'] = 'incomplete'
                
                trials.append(trial_data)
                trial_id += 1
        
        if trials:
            all_trials = pd.concat(trials, ignore_index=True)
            print(f"Successfully segmented {len(trials)} aligned trials")
            
            # Add trial timing information
            trial_info = []
            for trial_num in range(len(trials)):
                trial_data = trials[trial_num]
                trial_info.append({
                    'trial_id': trial_num,
                    'start_time_aligned': trial_data['timestamp_aligned'].iloc[0],
                    'end_time_aligned': trial_data['timestamp_aligned'].iloc[-1],
                    'duration': trial_data['timestamp_aligned'].iloc[-1] - trial_data['timestamp_aligned'].iloc[0],
                    'outcome': trial_data['trial_outcome'].iloc[0],
                    'target_index': trial_data['target_index'].iloc[0] if 'target_index' in trial_data.columns else None
                })
            
            trial_info_df = pd.DataFrame(trial_info)
            
            return {
                'trial_data': all_trials,
                'trial_info': trial_info_df,
                'n_trials': len(trials)
            }
        else:
            print("No valid trials found in aligned data")
            return None
    
    def load_neural_data_for_trials(self, file_path, trial_numbers=None, 
                                    buffer_seconds=5.0, force_reload=False,
                                    max_channels=None, downsample_factor=1):
        """
        Load neural data for specific behavioral trials using lazy loading.
        
        This method uses behavioral trial timing to determine the exact time window
        needed and loads only that portion of the neural data using Neo's lazy loading.
        
        Parameters:
        -----------
        file_path : str
            Path to the .ns6 file
        trial_numbers : list, optional
            List of trial numbers to load. If None, uses first 10 trials
        buffer_seconds : float, optional
            Buffer time in seconds to add before/after trial window
        force_reload : bool, optional
            If True, reload even if same file is already loaded
        max_channels : int, optional
            Maximum number of channels to load
        downsample_factor : int, optional
            Downsample factor (1 = no downsampling)
            
        Returns:
        --------
        dict
            Dictionary containing neural data and metadata
        """
        # Check if behavioral data is available
        if self.behavioral_data is None:
            print("❌ No behavioral data available. Load behavioral data first.")
            return None
            
        if 'trial' not in self.behavioral_data.columns:
            print("❌ No trial segmentation found in behavioral data.")
            return None
        
        # Determine which trials to load
        available_trials = sorted(self.behavioral_data['trial'].unique())
        if trial_numbers is None:
            trial_numbers = available_trials[:10]  # First 10 trials by default
        
        # Validate trial numbers
        valid_trials = [t for t in trial_numbers if t in available_trials]
        if len(valid_trials) == 0:
            print(f"❌ No valid trials found. Available: {available_trials[:10]}...")
            return None
        
        # Calculate time window needed
        trial_data = self.behavioral_data[self.behavioral_data['trial'].isin(valid_trials)]
        
        if 'timestamp' not in trial_data.columns:
            print("❌ No timestamp column found in behavioral data.")
            return None
        
        # Get time range with buffer
        start_time_behavioral = trial_data['timestamp'].iloc[0]
        end_time_behavioral = trial_data['timestamp'].iloc[-1]
        duration_behavioral = (end_time_behavioral - start_time_behavioral).total_seconds()
        
        # Account for time offset between neural and behavioral data
        # Neural data starts at 2025-03-25T9:22:53Z
        # Behavioral data starts at 2025-03-25T9:22:28Z (25 second offset)
        time_offset = 25.0  # seconds
        
        # Calculate neural data time window
        neural_start_time = max(0, time_offset - buffer_seconds)
        neural_duration = duration_behavioral + 2 * buffer_seconds
        
        print(f"Loading neural data for trials: {valid_trials}")
        print(f"Behavioral time range: {duration_behavioral:.2f} seconds")
        print(f"Neural time window: {neural_start_time:.2f} to {neural_start_time + neural_duration:.2f} seconds")
        print(f"Total duration to load: {neural_duration:.2f} seconds")
        
        # Check if same window is already loaded
        cache_key = f"{file_path}_{neural_start_time}_{neural_duration}_{max_channels}_{downsample_factor}"
        if (not force_reload and 
            hasattr(self, '_neural_cache_key') and 
            self._neural_cache_key == cache_key):
            print("Neural data for this trial window is already loaded.")
            return self.neural_data
        
        print(f"Loading neural data from: {file_path}")
        print(f"Using lazy loading for time window: {neural_start_time:.1f} to {neural_start_time + neural_duration:.1f} seconds")
        
        try:
            # Create Neo reader 
            reader = neo.BlackrockIO(filename=file_path)
            
            # Get Time Origin
            try:
                if hasattr(reader, 'datetime'):
                    self.time_origin = reader.datetime
                elif hasattr(reader, 'rec_datetime'):
                    self.time_origin = reader.rec_datetime
                else:
                    self.time_origin = datetime(2025, 3, 25, 9, 22, 53, tzinfo=timezone.utc)
            except:
                self.time_origin = datetime(2025, 3, 25, 9, 22, 53, tzinfo=timezone.utc)
            
            print(f"Neural data Time Origin: {self.time_origin}")
            
            # Read block in lazy mode to get proxies
            block = reader.read_block(lazy=True)
            segment = block.segments[0]
            
            if len(segment.analogsignals) == 0:
                print("❌ No analog signals found in the neural data file.")
                return None
                
            # Get the analog signal proxy (lazy-loaded)
            analog_signal = segment.analogsignals[0]
            
            # Get sampling rate and calculate indices
            sampling_rate = float(analog_signal.sampling_rate.magnitude)
            start_idx = int(neural_start_time * sampling_rate)
            end_idx = int((neural_start_time + neural_duration) * sampling_rate)
            
            print(f"Original sampling rate: {sampling_rate} Hz")
            print(f"Loading samples {start_idx} to {end_idx}")
            
            # Estimate memory usage before loading
            n_samples = end_idx - start_idx
            n_channels = analog_signal.shape[1]
            if max_channels is not None:
                n_channels = min(n_channels, max_channels)
            
            estimated_memory_mb = (n_samples * n_channels * 4) / (1024**2)  # 4 bytes per float32
            print(f"Estimated memory usage: {estimated_memory_mb:.1f} MB")
            
            # Load the specific time window using lazy loading
            # The analog_signal is a proxy object - we need to load it properly
            print("Loading data from proxy object...")
            
            # For AnalogSignalProxy, we need to load the data using the proxy's load method
            # or access it through the segment
            try:
                if hasattr(analog_signal, 'load'):
                    # Method 1: Use the proxy's load method
                    loaded_signal = analog_signal.load()
                    raw_data_full = loaded_signal.magnitude
                else:
                    # Method 2: Load the full signal from the proxy
                    raw_data_full = analog_signal[:]
                    if hasattr(raw_data_full, 'magnitude'):
                        raw_data_full = raw_data_full.magnitude
                
                # Now slice the loaded data to get our time window
                raw_data = raw_data_full[start_idx:end_idx]
                
                # Apply channel selection if requested
                if max_channels is not None and raw_data.shape[1] > max_channels:
                    raw_data = raw_data[:, :max_channels]
                    print(f"Selected first {max_channels} channels out of {raw_data_full.shape[1]} available")
                
            except Exception as e:
                print(f"Error with proxy loading method: {e}")
                # Fallback: Load the entire segment and slice
                print("Trying alternative loading method...")
                
                # Re-read the block without lazy loading for the specific time window
                reader.close()
                reader = neo.BlackrockIO(filename=file_path)
                
                # Calculate time window in seconds
                t_start = neural_start_time
                t_stop = neural_start_time + neural_duration
                
                # Load only the specific time window
                block = reader.read_block(
                    time_slice=(t_start, t_stop),
                    lazy=False
                )
                
                if len(block.segments) > 0 and len(block.segments[0].analogsignals) > 0:
                    loaded_signal = block.segments[0].analogsignals[0]
                    raw_data = loaded_signal.magnitude
                    
                    # Apply channel selection if requested
                    if max_channels is not None and raw_data.shape[1] > max_channels:
                        raw_data = raw_data[:, :max_channels]
                        print(f"Selected first {max_channels} channels out of {loaded_signal.shape[1]} available")
                else:
                    raise Exception("No data found in the specified time window")
            
            # Create time axis relative to the loaded segment
            times = np.arange(raw_data.shape[0]) / sampling_rate
            
            # Apply downsampling if requested
            if downsample_factor > 1:
                raw_data = raw_data[::downsample_factor]
                times = times[::downsample_factor]
                sampling_rate = sampling_rate / downsample_factor
                print(f"Downsampled by factor of {downsample_factor}, new sampling rate: {sampling_rate} Hz")
            
            print(f"Loaded neural data shape: {raw_data.shape}")
            print(f"Effective sampling rate: {sampling_rate} Hz")
            print(f"Loaded duration: {len(raw_data)/sampling_rate:.2f} seconds")
            print(f"Actual memory usage: {raw_data.nbytes / 1024**2:.1f} MB")
            
            # Store metadata
            self.neural_metadata = {
                'sampling_rate': sampling_rate,
                'n_channels': raw_data.shape[1],
                'duration': len(times) / sampling_rate,
                'start_time': neural_start_time,
                'end_time': neural_start_time + neural_duration,
                'time_origin': self.time_origin,
                'file_path': file_path,
                'loaded_trials': valid_trials,
                'buffer_seconds': buffer_seconds,
                'downsample_factor': downsample_factor
            }
            
            # Store neural data
            self.neural_data = {
                'raw_data': raw_data,
                'times': times,
                'timestamps': times + neural_start_time,  # Absolute timestamps
                'sampling_rate': sampling_rate,
                'channels': list(range(raw_data.shape[1])),
                'metadata': self.neural_metadata.copy()
            }
            
            # Cache the loading parameters
            self._neural_cache_key = cache_key
            
            print("✅ Neural data loaded successfully using lazy loading!")
            
            return self.neural_data
            
        except Exception as e:
            print(f"❌ Error loading neural data: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Ensure reader is closed
            try:
                reader.close()
            except:
                pass 

    def load_neural_data_for_trials_simple(self, file_path, trial_numbers=None, 
                                          buffer_seconds=5.0, force_reload=False,
                                          max_channels=None, downsample_factor=1):
        """
        Load neural data for specific behavioral trials using direct time slicing.
        
        This is a simpler, more reliable approach that uses Neo's time_slice parameter
        directly in read_block instead of handling lazy loading proxies.
        
        Parameters:
        -----------
        file_path : str
            Path to the .ns6 file
        trial_numbers : list, optional
            List of trial numbers to load. If None, uses first 10 trials
        buffer_seconds : float, optional
            Buffer time in seconds to add before/after trial window
        force_reload : bool, optional
            If True, reload even if same file is already loaded
        max_channels : int, optional
            Maximum number of channels to load
        downsample_factor : int, optional
            Downsample factor (1 = no downsampling)
            
        Returns:
        --------
        dict
            Dictionary containing neural data and metadata
        """
        # Check if behavioral data is available
        if self.behavioral_data is None:
            print("❌ No behavioral data available. Load behavioral data first.")
            return None
            
        if 'trial' not in self.behavioral_data.columns:
            print("❌ No trial segmentation found in behavioral data.")
            return None
        
        # Determine which trials to load
        available_trials = sorted(self.behavioral_data['trial'].unique())
        if trial_numbers is None:
            trial_numbers = available_trials[:10]  # First 10 trials by default
        
        # Validate trial numbers
        valid_trials = [t for t in trial_numbers if t in available_trials]
        if len(valid_trials) == 0:
            print(f"❌ No valid trials found. Available: {available_trials[:10]}...")
            return None
        
        # Calculate time window needed
        trial_data = self.behavioral_data[self.behavioral_data['trial'].isin(valid_trials)]
        
        if 'timestamp' not in trial_data.columns:
            print("❌ No timestamp column found in behavioral data.")
            return None
        
        # Get time range with buffer
        start_time_behavioral = trial_data['timestamp'].iloc[0]
        end_time_behavioral = trial_data['timestamp'].iloc[-1]
        duration_behavioral = (end_time_behavioral - start_time_behavioral).total_seconds()
        
        # Account for time offset between neural and behavioral data
        time_offset = 25.0  # seconds
        
        # Calculate neural data time window
        neural_start_time = max(0, time_offset - buffer_seconds)
        neural_end_time = neural_start_time + duration_behavioral + 2 * buffer_seconds
        
        print(f"Loading neural data for trials: {valid_trials}")
        print(f"Behavioral time range: {duration_behavioral:.2f} seconds")
        print(f"Neural time window: {neural_start_time:.2f} to {neural_end_time:.2f} seconds")
        
        # Check if same window is already loaded
        cache_key = f"{file_path}_{neural_start_time}_{neural_end_time}_{max_channels}_{downsample_factor}"
        if (not force_reload and 
            hasattr(self, '_neural_cache_key') and 
            self._neural_cache_key == cache_key):
            print("Neural data for this trial window is already loaded.")
            return self.neural_data
        
        print(f"Loading neural data from: {file_path}")
        print(f"Using direct time slicing: {neural_start_time:.1f} to {neural_end_time:.1f} seconds")
        
        reader = None
        try:
            # Create Neo reader
            reader = neo.BlackrockIO(filename=file_path)
            
            # Get Time Origin
            try:
                if hasattr(reader, 'datetime'):
                    self.time_origin = reader.datetime
                elif hasattr(reader, 'rec_datetime'):
                    self.time_origin = reader.rec_datetime
                else:
                    self.time_origin = datetime(2025, 3, 25, 9, 22, 53, tzinfo=timezone.utc)
            except:
                self.time_origin = datetime(2025, 3, 25, 9, 22, 53, tzinfo=timezone.utc)
            
            print(f"Neural data Time Origin: {self.time_origin}")
            
            # Load data using manual time slicing - BlackrockIO doesn't support time_slice parameter
            print("Loading data with manual time slicing...")
            
            # First, read the block to get the signal information
            block = reader.read_block(lazy=True)  # Start with lazy to get structure
            
            if len(block.segments) == 0:
                print("❌ No segments found in the specified time window.")
                return None
                
            segment = block.segments[0]
            
            if len(segment.analogsignals) == 0:
                print("❌ No analog signals found in the neural data file.")
                return None
                
            # Get the analog signal proxy
            analog_signal_proxy = segment.analogsignals[0]
            
            # Get sampling rate from the proxy
            sampling_rate = float(analog_signal_proxy.sampling_rate.magnitude)
            
            print(f"Original sampling rate: {sampling_rate} Hz")
            print(f"Total signal shape: {analog_signal_proxy.shape}")
            
            # Calculate sample indices for the time window we want
            start_sample = int(neural_start_time * sampling_rate)
            end_sample = int(neural_end_time * sampling_rate)
            
            print(f"Loading samples {start_sample} to {end_sample}")
            
            # Load data from the proxy using proper Neo methods
            # AnalogSignalProxy is not subscriptable, so we need to use other approaches
            print("Loading data from proxy using Neo methods...")
            
            try:
                # Method 1: Try to load the proxy and then slice
                if hasattr(analog_signal_proxy, 'load'):
                    print("Using proxy.load() method...")
                    loaded_signal = analog_signal_proxy.load()
                    raw_data_full = loaded_signal.magnitude
                    
                    # Now slice the loaded data
                    raw_data = raw_data_full[start_sample:end_sample]
                    
                    # Apply channel selection
                    if max_channels is not None and raw_data.shape[1] > max_channels:
                        raw_data = raw_data[:, :max_channels]
                        print(f"Selected first {max_channels} channels out of {raw_data_full.shape[1]} available")
                        
                else:
                    # Method 2: Try time_slice method if available
                    if hasattr(analog_signal_proxy, 'time_slice'):
                        print("Using proxy.time_slice() method...")
                        t_start = start_sample / sampling_rate
                        t_stop = end_sample / sampling_rate
                        sliced_signal = analog_signal_proxy.time_slice(t_start, t_stop)
                        raw_data = sliced_signal.magnitude
                        
                        # Apply channel selection
                        if max_channels is not None and raw_data.shape[1] > max_channels:
                            raw_data = raw_data[:, :max_channels]
                            print(f"Selected first {max_channels} channels out of {sliced_signal.shape[1]} available")
                    else:
                        # Method 3: Simple fallback - load manageable chunk
                        print("Using simple chunk-based loading fallback...")
                        
                        # Close the lazy reader and use non-lazy loading
                        reader.close()
                        reader = neo.BlackrockIO(filename=file_path)
                        
                        # Load just a reasonable chunk that includes our time window
                        # Add some buffer but keep it manageable
                        chunk_duration = min(60.0, neural_end_time - neural_start_time + 20)  # Max 60 seconds
                        chunk_start_time = max(0, neural_start_time - 5)  # 5 second buffer before
                        
                        print(f"Loading {chunk_duration:.1f} second chunk starting at {chunk_start_time:.1f}s")
                        
                        # Calculate sample indices for the chunk
                        chunk_start_sample = int(chunk_start_time * sampling_rate)
                        chunk_samples = int(chunk_duration * sampling_rate)
                        
                        # Load the block without lazy loading
                        block = reader.read_block(lazy=False)
                        segment = block.segments[0]
                        analog_signal_full = segment.analogsignals[0]
                        
                        print(f"Full signal shape: {analog_signal_full.shape}")
                        
                        # Extract just the chunk we need
                        chunk_end_sample = min(chunk_start_sample + chunk_samples, analog_signal_full.shape[0])
                        chunk_data = analog_signal_full.magnitude[chunk_start_sample:chunk_end_sample]
                        
                        print(f"Chunk data shape: {chunk_data.shape}")
                        
                        # Now extract our specific time window from the chunk
                        window_start_in_chunk = start_sample - chunk_start_sample
                        window_end_in_chunk = end_sample - chunk_start_sample
                        
                        # Make sure indices are valid
                        window_start_in_chunk = max(0, window_start_in_chunk)
                        window_end_in_chunk = min(chunk_data.shape[0], window_end_in_chunk)
                        
                        raw_data = chunk_data[window_start_in_chunk:window_end_in_chunk]
                        
                        # Apply channel selection
                        if max_channels is not None and raw_data.shape[1] > max_channels:
                            raw_data = raw_data[:, :max_channels]
                            print(f"Selected first {max_channels} channels out of {chunk_data.shape[1]} available")
                            
            except Exception as e:
                print(f"Error with proxy methods: {e}")
                raise e
            
            print(f"Loaded data shape: {raw_data.shape}")
            
            # Create time axis
            times = np.arange(raw_data.shape[0]) / sampling_rate
            
            # Apply downsampling if requested
            if downsample_factor > 1:
                raw_data = raw_data[::downsample_factor]
                times = times[::downsample_factor]
                sampling_rate = sampling_rate / downsample_factor
                print(f"Downsampled by factor of {downsample_factor}, new sampling rate: {sampling_rate} Hz")
            
            print(f"Final neural data shape: {raw_data.shape}")
            print(f"Effective sampling rate: {sampling_rate} Hz")
            print(f"Loaded duration: {len(raw_data)/sampling_rate:.2f} seconds")
            print(f"Actual memory usage: {raw_data.nbytes / 1024**2:.1f} MB")
            
            # Store metadata
            self.neural_metadata = {
                'sampling_rate': sampling_rate,
                'n_channels': raw_data.shape[1],
                'duration': len(raw_data) / sampling_rate,
                'start_time': neural_start_time,
                'end_time': neural_end_time,
                'time_origin': self.time_origin,
                'file_path': file_path,
                'loaded_trials': valid_trials,
                'buffer_seconds': buffer_seconds,
                'downsample_factor': downsample_factor
            }
            
            # Store neural data
            self.neural_data = {
                'raw_data': raw_data,
                'times': times,
                'timestamps': times + neural_start_time,  # Absolute timestamps
                'sampling_rate': sampling_rate,
                'channels': list(range(raw_data.shape[1])),
                'metadata': self.neural_metadata.copy()
            }
            
            # Cache the loading parameters
            self._neural_cache_key = cache_key
            
            print("✅ Neural data loaded successfully using direct time slicing!")
            
            return self.neural_data
            
        except Exception as e:
            print(f"❌ Error loading neural data: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Ensure reader is closed
            if reader is not None:
                try:
                    reader.close()
                except:
                    pass 