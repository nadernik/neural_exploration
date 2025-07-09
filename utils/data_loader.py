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

# Import the new NSX header parser
from .nsx_header_parser import parse_nsx_header, get_time_origin, NSXHeaderParseError

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
        
        # Cache for loading
        self._behavioral_cache_key = None
        self._neural_cache_key = None
        
    def load_behavioral_data(self, csv_file_path=None, force_reload=False):
        """
        Load behavioral data from CSV file and perform trial segmentation.
        
        Parameters:
        -----------
        csv_file_path : str, optional
            Path to the CSV file. If None, uses self.csv_file_path
        force_reload : bool, optional
            If True, reload even if same file is already loaded
            
        Returns:
        --------
        pandas.DataFrame
            Behavioral data with trial segmentation
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path
        
        if csv_file_path is None:
            print("❌ No CSV file path provided")
            return None
            
        # Check if we need to reload
        cache_key = (csv_file_path, force_reload)
        if (not force_reload and 
            self._behavioral_cache_key == cache_key and 
            self.behavioral_data is not None):
            print("✅ Using cached behavioral data")
            return self.behavioral_data
            
        print(f"📂 Loading behavioral data from: {csv_file_path}")
        
        try:
            # Load CSV data
            data = pd.read_csv(csv_file_path)
            print(f"✅ Loaded {len(data)} behavioral data points")
            
            # Parse timestamps
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                print(f"✅ Parsed timestamps from {data['timestamp'].min()} to {data['timestamp'].max()}")
            else:
                print("❌ No timestamp column found in behavioral data")
                return None
            
            # Perform trial segmentation
            print("🔄 Performing trial segmentation...")
            data = self._segment_trials(data)
            
            # Store data and metadata
            self.behavioral_data = data
            self.behavioral_metadata = {
                'file_path': csv_file_path,
                'n_trials': len(data['trial'].unique()) if 'trial' in data.columns else 0,
                'duration': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() if 'timestamp' in data.columns else 0,
                'total_samples': len(data),
                'columns': list(data.columns)
            }
            
            # Cache the loading parameters
            self._behavioral_cache_key = cache_key
            
            print(f"✅ Behavioral data loaded successfully! {self.behavioral_metadata['n_trials']} trials found")
            return self.behavioral_data
            
        except Exception as e:
            print(f"❌ Error loading behavioral data: {e}")
            return None
    
    def _segment_trials(self, data):
        """
        Segment behavioral data into trials based on trial_start column.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw behavioral data
            
        Returns:
        --------
        pandas.DataFrame
            Data with trial segmentation
        """
        if 'trial_start' not in data.columns:
            print("❌ No trial_start column found. Cannot segment trials.")
            return data
            
        # Initialize trial column
        data['trial'] = -1
        data['trial_outcome'] = 'unknown'
        
        # Find trial starts
        trial_starts = data[data['trial_start'] == True].index
        print(f"Found {len(trial_starts)} trial starts")
        
        current_trial = 0
        
        for i, start_idx in enumerate(trial_starts):
            # Determine end of current trial
            if i + 1 < len(trial_starts):
                end_idx = trial_starts[i + 1] - 1
            else:
                end_idx = len(data) - 1
            
            # Assign trial number
            data.loc[start_idx:end_idx, 'trial'] = current_trial
            
            # Determine trial outcome
            trial_data = data.loc[start_idx:end_idx]
            if trial_data['success'].any():
                outcome = 'success'
            elif trial_data['failed'].any():
                outcome = 'failure'
            else:
                outcome = 'unknown'
            
            data.loc[start_idx:end_idx, 'trial_outcome'] = outcome
            current_trial += 1
        
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
        # Check if we need to reload
        cache_key = (file_path, max_channels, max_duration, start_time, downsample_factor, force_reload)
        if (not force_reload and 
            self._neural_cache_key == cache_key and 
            self.neural_data is not None):
            print("✅ Using cached neural data")
            return self.neural_data
        
        print(f"📂 Loading neural data from: {file_path}")
        
        if not NEO_AVAILABLE:
            print("❌ Neo library not available. Cannot load neural data.")
            return None
        
        try:
            # First, extract time origin from NSX header
            try:
                print("📅 Extracting time origin from NSX header...")
                self.time_origin = get_time_origin(file_path)
                print(f"✅ Time origin extracted: {self.time_origin}")
            except NSXHeaderParseError as e:
                print(f"❌ Failed to extract time origin from NSX header: {e}")
                print("⚠️  Cannot proceed without time origin - this is required for alignment")
                return None
            
            # Create Neo reader for Blackrock files
            reader = neo.BlackrockIO(filename=file_path)
            
            # Get basic file info
            header = reader.header
            if header:
                print("📊 File header information:")
                for key, value in header.items():
                    print(f"  • {key}: {value}")
            
            # Lazy loading approach for memory efficiency
            print("🔄 Using lazy loading for memory efficiency...")
            
            # Load block with lazy loading
            block = reader.read_block(lazy=True)
            segment = block.segments[0]
            analog_signals = segment.analogsignals
            
            if len(analog_signals) > 0:
                # Get the first analog signal (proxy)
                analog_signal_proxy = analog_signals[0]
                
                # Get signal properties
                total_samples = analog_signal_proxy.shape[0]
                n_channels = analog_signal_proxy.shape[1]
                sampling_rate = float(analog_signal_proxy.sampling_rate.magnitude)
                
                print(f"📊 Signal properties:")
                print(f"  • Total samples: {total_samples:,}")
                print(f"  • Channels: {n_channels}")
                print(f"  • Sampling rate: {sampling_rate} Hz")
                print(f"  • Total duration: {total_samples/sampling_rate:.2f} seconds")
                
                # Calculate memory requirements
                bytes_per_sample = 2  # Assuming 16-bit integers
                max_samples = int(max_duration * sampling_rate) if max_duration else total_samples
                max_channels_to_load = min(max_channels or n_channels, n_channels)
                
                estimated_memory = max_samples * max_channels_to_load * bytes_per_sample
                print(f"💾 Estimated memory usage: {estimated_memory / 1024**2:.1f} MB")
                
                # Check if we need to use time slicing
                if max_duration or start_time > 0:
                    # Calculate sample indices
                    start_sample = int(start_time * sampling_rate)
                    if max_duration:
                        end_sample = start_sample + int(max_duration * sampling_rate)
                    else:
                        end_sample = total_samples
                    
                    # Ensure we don't exceed bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(end_sample, total_samples)
                    
                    print(f"📊 Loading samples {start_sample} to {end_sample}")
                    
                    # Load the proxy to get actual data
                    loaded_signal = analog_signal_proxy.load()
                    raw_data = loaded_signal.magnitude[start_sample:end_sample]
                    
                    # Create time array
                    times = np.arange(end_sample - start_sample) / sampling_rate + start_time
                    
                else:
                    # Load all data
                    print("📊 Loading all data...")
                    loaded_signal = analog_signal_proxy.load()
                    raw_data = loaded_signal.magnitude
                    times = np.arange(len(raw_data)) / sampling_rate
                
                # Apply channel selection
                if max_channels is not None and raw_data.shape[1] > max_channels:
                    raw_data = raw_data[:, :max_channels]
                    print(f"Selected first {max_channels} channels out of {n_channels} available")
                
                # Downsampling
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
            print(f"Actual memory usage: {raw_data.nbytes / 1024**2:.1f} MB")
            
            # Store metadata
            self.neural_metadata = {
                'sampling_rate': sampling_rate,
                'n_channels': raw_data.shape[1],
                'duration': len(times) / sampling_rate,
                'start_time': start_time,
                'time_origin': self.time_origin,
                'file_path': file_path,
                'downsample_factor': downsample_factor
            }
            
            # Store neural data
            self.neural_data = {
                'raw_data': raw_data,
                'times': times,
                'timestamps': times,  # For compatibility
                'sampling_rate': sampling_rate,
                'channels': list(range(raw_data.shape[1])),
                'metadata': self.neural_metadata.copy()
            }
            
            # Cache the loading parameters
            self._neural_cache_key = cache_key
            
            print("✅ Neural data loaded successfully!")
            
            return self.neural_data
            
        except Exception as e:
            print(f"❌ Error loading neural data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_time_origin_from_nsx_header(self, file_path):
        """
        Extract time origin from NSX file header using proper binary parsing.
        
        Parameters:
        -----------
        file_path : str
            Path to the NSX file
            
        Returns:
        --------
        datetime or None
            Time origin if successfully extracted, None otherwise
        """
        try:
            return get_time_origin(file_path)
        except NSXHeaderParseError as e:
            print(f"❌ Failed to extract time origin from NSX header: {e}")
            return None 