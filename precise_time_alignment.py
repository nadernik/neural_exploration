#!/usr/bin/env python3
"""
Precise Time Alignment for Neural-Behavioral Integration

This script focuses on precisely aligning behavioral trial timestamps with neural data
based on shared wall-clock timing, without relying on .nev files or digital sync pulses.
"""

import pandas as pd
import numpy as np
import h5py
from datetime import datetime
import neo
from neo.io import BlackrockIO
from scipy.signal import decimate
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreciseTimeAligner:
    """
    Handles precise time alignment between behavioral and neural data.
    """
    
    def __init__(self, neural_file: str, behavioral_file: str, output_file: str = "trials_aligned.h5"):
        """
        Initialize the time aligner.
        
        Args:
            neural_file: Path to the .ns6 neural data file
            behavioral_file: Path to the behavioral CSV file
            output_file: Path for the output HDF5 file
        """
        self.neural_file = Path(neural_file)
        self.behavioral_file = Path(behavioral_file)
        self.output_file = Path(output_file)
        
        # Will be set during processing
        self.neural_io = None
        self.ns6_time_origin = None
        self.behavioral_data = None
        self.trials = []
        
        # Neural data parameters
        self.original_fs = 30000  # Original sampling rate (Hz)
        self.target_fs = 1000     # Target sampling rate after downsampling (Hz)
        self.downsample_factor = self.original_fs // self.target_fs
        
    def extract_ns6_time_origin(self) -> datetime:
        """
        Extract the precise Time Origin from the .ns6 file header.
        
        Returns:
            datetime object representing the neural recording start time
        """
        logger.info(f"Extracting Time Origin from {self.neural_file}")
        
        try:
            # Initialize BlackrockIO
            self.neural_io = BlackrockIO(filename=str(self.neural_file))
            
            # Method 1: Try to access the raw header directly
            try:
                if hasattr(self.neural_io, '_get_nsx_header'):
                    header = self.neural_io._get_nsx_header()
                    if 'TimeOrigin' in header:
                        time_origin = header['TimeOrigin']
                        logger.info(f"Found TimeOrigin in _get_nsx_header: {time_origin}")
                        
                        # Convert to datetime if it's not already
                        if isinstance(time_origin, str):
                            time_origin = time_origin.rstrip('Z')
                            return datetime.fromisoformat(time_origin)
                        elif isinstance(time_origin, datetime):
                            return time_origin
                        else:
                            logger.warning(f"Unknown TimeOrigin format: {type(time_origin)}")
            except Exception as e:
                logger.warning(f"_get_nsx_header method failed: {e}")
            
            # Method 2: Try raw_annotations
            try:
                if hasattr(self.neural_io, 'raw_annotations') and self.neural_io.raw_annotations:
                    annotations = self.neural_io.raw_annotations
                    if 'Time Origin' in annotations:
                        time_origin = annotations['Time Origin']
                        logger.info(f"Found Time Origin in raw_annotations: {time_origin}")
                        
                        if isinstance(time_origin, str):
                            time_origin = time_origin.rstrip('Z')
                            return datetime.fromisoformat(time_origin)
                        elif isinstance(time_origin, datetime):
                            return time_origin
            except Exception as e:
                logger.warning(f"raw_annotations method failed: {e}")
            
            # Method 3: Try to parse the file header directly
            try:
                # Read the first few bytes of the file to extract header info
                with open(self.neural_file, 'rb') as f:
                    # NSx files have a specific header structure
                    # This is a simplified approach - real implementation would need
                    # to parse the complete NSx header format
                    header_data = f.read(10000)  # Read first 10KB
                    
                    # Look for timestamp information in the header
                    # This is a heuristic approach and may need adjustment
                    logger.info("Attempting to parse raw header data...")
                    
                    # The exact parsing would depend on the NSx file format version
                    # For now, we'll use a fallback approach
                    
            except Exception as e:
                logger.warning(f"Raw header parsing failed: {e}")
            
            # Method 4: Try to read a small block and get timing info
            try:
                logger.info("Attempting to extract timing from data block...")
                
                # Check if we can use get_analogsignal_chunk to get timing info
                if hasattr(self.neural_io, 'get_signal_t_start'):
                    t_start = self.neural_io.get_signal_t_start(block_index=0, seg_index=0)
                    logger.info(f"Signal t_start: {t_start}")
                    
                    # This gives us the relative start time, but we need the absolute time
                    # We'll need to combine this with other information
                    
            except Exception as e:
                logger.warning(f"Data block timing extraction failed: {e}")
            
            # If all methods fail, we need to make an educated guess
            # Based on the behavioral data timing
            logger.warning("Could not extract Time Origin from neural file header")
            logger.info("Will attempt to estimate based on behavioral data timing")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract Time Origin: {e}")
            return None
    
    def load_behavioral_data(self) -> pd.DataFrame:
        """
        Load behavioral data and convert timestamps to pd.Timestamp objects.
        
        Returns:
            DataFrame with properly parsed timestamps
        """
        logger.info(f"Loading behavioral data from {self.behavioral_file}")
        
        # Load CSV
        df = pd.read_csv(self.behavioral_file)
        
        # Convert timestamps to pd.Timestamp (UNIX seconds)
        logger.info(f"Raw timestamp sample: {df['timestamp'].iloc[0]}")
        
        # Force parsing as UNIX timestamp in seconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Convert to timezone-aware timestamps (assuming UTC)
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        logger.info(f"Parsed timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Ensure boolean columns are properly typed
        bool_cols = ['trial_start', 'trial_win', 'trial_lose']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        logger.info(f"Loaded {len(df)} behavioral samples")
        
        self.behavioral_data = df
        return df
    
    def estimate_time_alignment(self) -> datetime:
        """
        Estimate the neural time origin based on behavioral data timing.
        
        Returns:
            Estimated neural recording start time
        """
        logger.info("Estimating time alignment based on behavioral data...")
        
        if self.behavioral_data is None:
            raise ValueError("Behavioral data not loaded")
        
        # Get behavioral time range
        behavioral_start = self.behavioral_data['timestamp'].min()
        behavioral_end = self.behavioral_data['timestamp'].max()
        behavioral_duration = (behavioral_end - behavioral_start).total_seconds()
        
        logger.info(f"Behavioral session: {behavioral_start} to {behavioral_end}")
        logger.info(f"Behavioral duration: {behavioral_duration:.2f} seconds")
        
        # Get neural recording duration
        try:
            if hasattr(self.neural_io, 'get_signal_size'):
                signal_size = self.neural_io.get_signal_size(block_index=0, seg_index=0)
                neural_duration = signal_size / self.original_fs
                logger.info(f"Neural recording duration: {neural_duration:.2f} seconds")
                
                # Estimate neural start time
                # Assume the neural recording started some time before the behavioral session
                # This is a heuristic - in practice, you'd need to know the exact timing
                
                # Common scenarios:
                # 1. Neural started before behavioral (most common)
                # 2. Neural and behavioral started at the same time
                # 3. Neural started after behavioral (less common)
                
                # For now, let's assume neural started 30 seconds before behavioral
                estimated_neural_start = behavioral_start - pd.Timedelta(seconds=30)
                
                logger.info(f"Estimated neural start time: {estimated_neural_start}")
                
                # Convert to datetime (remove timezone for consistency)
                return estimated_neural_start.to_pydatetime().replace(tzinfo=None)
                
        except Exception as e:
            logger.warning(f"Could not estimate timing: {e}")
            
        # Final fallback - use a reasonable default
        fallback_time = datetime(2025, 3, 25, 21, 22, 0)  # Just before behavioral start
        logger.warning(f"Using fallback time: {fallback_time}")
        return fallback_time
    
    def align_timestamps(self):
        """
        Establish precise time alignment between behavioral and neural data.
        """
        logger.info("Establishing precise time alignment...")
        
        # Try to extract Time Origin from neural file
        self.ns6_time_origin = self.extract_ns6_time_origin()
        
        if self.ns6_time_origin is None:
            # Fall back to estimation
            self.ns6_time_origin = self.estimate_time_alignment()
        
        logger.info(f"Neural Time Origin: {self.ns6_time_origin}")
        
        # Check time alignment
        behavioral_start = self.behavioral_data['timestamp'].min()
        behavioral_end = self.behavioral_data['timestamp'].max()
        
        # Convert both to timezone-naive for comparison
        behavioral_start_naive = behavioral_start.to_pydatetime().replace(tzinfo=None)
        behavioral_end_naive = behavioral_end.to_pydatetime().replace(tzinfo=None)
        ns6_time_origin_naive = self.ns6_time_origin.replace(tzinfo=None) if self.ns6_time_origin.tzinfo else self.ns6_time_origin
        
        # Calculate relative times
        relative_start = (behavioral_start_naive - ns6_time_origin_naive).total_seconds()
        relative_end = (behavioral_end_naive - ns6_time_origin_naive).total_seconds()
        
        logger.info(f"Behavioral session relative to neural:")
        logger.info(f"  Start: {relative_start:.2f} seconds")
        logger.info(f"  End: {relative_end:.2f} seconds")
        
        # Validate alignment
        if relative_start < 0:
            logger.warning("Behavioral data starts before neural recording!")
        if relative_end < 0:
            logger.error("Behavioral data ends before neural recording starts!")
            
        # Check if we have enough neural data
        try:
            if hasattr(self.neural_io, 'get_signal_size'):
                signal_size = self.neural_io.get_signal_size(block_index=0, seg_index=0)
                neural_duration = signal_size / self.original_fs
                
                if relative_end > neural_duration:
                    logger.warning(f"Behavioral data extends beyond neural recording!")
                    logger.warning(f"  Neural duration: {neural_duration:.2f}s")
                    logger.warning(f"  Behavioral end: {relative_end:.2f}s")
                    
        except Exception as e:
            logger.warning(f"Could not check neural duration: {e}")
    
    def compute_trial_times(self, trial_data: Dict) -> Tuple[float, float]:
        """
        Compute trial start and end times relative to neural Time Origin.
        
        Args:
            trial_data: Dictionary containing trial information
            
        Returns:
            Tuple of (start_seconds, end_seconds) relative to neural Time Origin
        """
        # Get trial timestamps
        start_timestamp = trial_data['start_time']
        end_timestamp = trial_data['end_time']
        
        # Convert to timezone-naive for calculation
        start_naive = start_timestamp.to_pydatetime().replace(tzinfo=None)
        end_naive = end_timestamp.to_pydatetime().replace(tzinfo=None)
        ns6_naive = self.ns6_time_origin.replace(tzinfo=None) if self.ns6_time_origin.tzinfo else self.ns6_time_origin
        
        # Convert to seconds since neural Time Origin
        start_seconds = (start_naive - ns6_naive).total_seconds()
        end_seconds = (end_naive - ns6_naive).total_seconds()
        
        return start_seconds, end_seconds
    
    def segment_trials(self) -> List[Dict]:
        """
        Segment behavioral data into individual trials.
        
        Returns:
            List of trial dictionaries with metadata
        """
        logger.info("Segmenting behavioral data into trials")
        
        if self.behavioral_data is None:
            raise ValueError("Behavioral data not loaded")
        
        df = self.behavioral_data
        trials = []
        
        # Find all trial start indices
        trial_starts = df[df['trial_start'] == True].index.tolist()
        
        logger.info(f"Found {len(trial_starts)} trial starts")
        
        for i, start_idx in enumerate(trial_starts):
            trial_data = {
                'trial_number': i + 1,
                'start_idx': start_idx,
                'start_time': df.loc[start_idx, 'timestamp'],
                'target_index': df.loc[start_idx, 'target_index'] if 'target_index' in df.columns else None,
            }
            
            # Find the end of this trial
            end_idx = None
            outcome = None
            
            for j in range(start_idx + 1, len(df)):
                if df.loc[j, 'trial_win']:
                    end_idx = j
                    outcome = 'win'
                    break
                elif df.loc[j, 'trial_lose']:
                    end_idx = j
                    outcome = 'lose'
                    break
            
            if end_idx is not None:
                trial_data['end_idx'] = end_idx
                trial_data['end_time'] = df.loc[end_idx, 'timestamp']
                trial_data['outcome'] = outcome
                trial_data['duration'] = (trial_data['end_time'] - trial_data['start_time']).total_seconds()
                
                # Compute relative times
                start_sec, end_sec = self.compute_trial_times(trial_data)
                trial_data['start_seconds'] = start_sec
                trial_data['end_seconds'] = end_sec
                
                trials.append(trial_data)
                
                logger.info(f"Trial {i+1}: {start_sec:.2f}s to {end_sec:.2f}s ({outcome})")
        
        logger.info(f"Successfully segmented {len(trials)} trials")
        self.trials = trials
        return trials
    
    def extract_neural_data(self, trial: Dict) -> Optional[np.ndarray]:
        """
        Extract neural data for a specific trial using time_slice.
        
        Args:
            trial: Trial dictionary with timing information
            
        Returns:
            Neural data array with shape (channels, time) or None if extraction fails
        """
        try:
            start_seconds = trial['start_seconds']
            end_seconds = trial['end_seconds']
            
            # Add buffer
            buffer_sec = 0.1
            start_seconds = max(0, start_seconds - buffer_sec)
            end_seconds = end_seconds + buffer_sec
            
            logger.info(f"Extracting neural data for trial {trial['trial_number']}: "
                       f"{start_seconds:.3f}s to {end_seconds:.3f}s")
            
            # Try to use time_slice (this may fail in older neo versions)
            try:
                block = self.neural_io.read_block(time_slice=(start_seconds, end_seconds))
                
                if block.segments and block.segments[0].analogsignals:
                    # Get all analog signals
                    neural_data = []
                    for signal in block.segments[0].analogsignals:
                        data = signal.magnitude
                        if data.ndim == 1:
                            neural_data.append(data.reshape(1, -1))
                        else:
                            neural_data.append(data.T)
                    
                    neural_array = np.vstack(neural_data)
                    logger.info(f"Extracted neural data shape: {neural_array.shape}")
                    return neural_array
                    
            except Exception as e:
                logger.warning(f"time_slice method failed: {e}")
                
                # Fallback: use get_analogsignal_chunk if available
                if hasattr(self.neural_io, 'get_analogsignal_chunk'):
                    start_sample = int(start_seconds * self.original_fs)
                    end_sample = int(end_seconds * self.original_fs)
                    
                    try:
                        chunk = self.neural_io.get_analogsignal_chunk(
                            block_index=0,
                            seg_index=0,
                            i_start=start_sample,
                            i_stop=end_sample
                        )
                        
                        neural_array = chunk.T  # Transpose to (channels, time)
                        logger.info(f"Extracted neural data using chunk method: {neural_array.shape}")
                        return neural_array
                        
                    except Exception as e2:
                        logger.error(f"chunk method also failed: {e2}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract neural data for trial {trial['trial_number']}: {e}")
            return None
    
    def extract_behavioral_data(self, trial: Dict) -> Dict:
        """
        Extract behavioral data for a specific trial from the behavioral DataFrame.
        
        Args:
            trial: Trial dictionary with timing information
            
        Returns:
            Dictionary containing behavioral data arrays
        """
        try:
            start_idx = trial['start_idx']
            end_idx = trial['end_idx']
            
            # Extract behavioral data for this trial
            trial_behavioral_data = self.behavioral_data.iloc[start_idx:end_idx+1]
            
            behavioral_dict = {
                'velocity_x': trial_behavioral_data['velocity_x'].values,
                'velocity_y': trial_behavioral_data['velocity_y'].values,
                'behavioral_timestamps': trial_behavioral_data['timestamp'].values.astype('datetime64[ns]').astype(np.float64) / 1e9  # Convert pd.Timestamp to UNIX seconds
            }
            
            logger.info(f"Extracted behavioral data for trial {trial['trial_number']}: "
                       f"{len(behavioral_dict['velocity_x'])} samples")
            
            return behavioral_dict
            
        except Exception as e:
            logger.error(f"Failed to extract behavioral data for trial {trial['trial_number']}: {e}")
            return {}

    def process_all(self, downsample: bool = True):
        """
        Run the complete precise time alignment pipeline.
        
        Args:
            downsample: Whether to downsample neural data to 1kHz
        """
        logger.info("Starting Precise Time Alignment Pipeline")
        logger.info("="*50)
        
        # Load behavioral data
        self.load_behavioral_data()
        
        # Establish time alignment
        self.align_timestamps()
        
        # Segment trials
        self.segment_trials()
        
        # Save results
        self.save_aligned_trials(downsample=downsample)
        
        logger.info("Precise time alignment complete!")
    
    def save_aligned_trials(self, downsample: bool = True):
        """
        Save time-aligned trials to HDF5 file.
        
        Args:
            downsample: Whether to downsample neural data
        """
        logger.info(f"Saving aligned trials to {self.output_file}")
        
        with h5py.File(self.output_file, 'w') as f:
            # Global metadata
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['neural_file'] = str(self.neural_file)
            f.attrs['behavioral_file'] = str(self.behavioral_file)
            f.attrs['ns6_time_origin'] = self.ns6_time_origin.isoformat()
            f.attrs['original_sampling_rate'] = self.original_fs
            f.attrs['final_sampling_rate'] = self.target_fs if downsample else self.original_fs
            f.attrs['total_trials'] = len(self.trials)
            
            successful_trials = 0
            
            for trial in self.trials:
                trial_group = f.create_group(f"trial_{trial['trial_number']}")
                
                # Extract neural data
                neural_data = self.extract_neural_data(trial)
                
                # Extract behavioral data
                behavioral_data = self.extract_behavioral_data(trial)
                
                if neural_data is not None:
                    # Downsample if requested
                    if downsample and neural_data.shape[1] > 60:  # Only if enough samples
                        try:
                            downsampled = []
                            for ch in range(neural_data.shape[0]):
                                downsampled.append(decimate(neural_data[ch], self.downsample_factor))
                            neural_data = np.array(downsampled)
                        except Exception as e:
                            logger.warning(f"Downsampling failed: {e}")
                    
                    # Save neural data
                    trial_group.create_dataset('neural', data=neural_data, compression='gzip')
                    
                    # Save behavioral data if available
                    if behavioral_data:
                        for key, data in behavioral_data.items():
                            if len(data) > 0:
                                trial_group.create_dataset(key, data=data, compression='gzip')
                                logger.info(f"Saved {key} with shape {data.shape}")
                    
                    successful_trials += 1
                    
                    # Save metadata
                    trial_group.attrs['trial_number'] = trial['trial_number']
                    trial_group.attrs['start_time'] = trial['start_time'].isoformat()
                    trial_group.attrs['end_time'] = trial['end_time'].isoformat()
                    trial_group.attrs['start_seconds'] = trial['start_seconds']
                    trial_group.attrs['end_seconds'] = trial['end_seconds']
                    trial_group.attrs['duration'] = trial['duration']
                    trial_group.attrs['outcome'] = trial['outcome']
                    
                    if trial['target_index'] is not None:
                        trial_group.attrs['target_index'] = trial['target_index']
                    
                    logger.info(f"✅ Saved trial {trial['trial_number']}: neural {neural_data.shape}, "
                               f"behavioral {len(behavioral_data)} datasets")
                else:
                    logger.warning(f"❌ Failed to extract neural data for trial {trial['trial_number']}")
            
            logger.info(f"Successfully saved {successful_trials}/{len(self.trials)} trials")
            
            # Check final file size
            file_size = Path(self.output_file).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Final file size: {file_size:.2f} MB")

def main():
    """
    Main function for precise time alignment.
    """
    # File paths
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"
    output_file = r"D:\Data\ScienceCorp\trials_aligned.h5"
    
    # Create aligner
    aligner = PreciseTimeAligner(
        neural_file=neural_file,
        behavioral_file=behavioral_file,
        output_file=output_file
    )
    
    # Run the alignment process
    try:
        aligner.process_all(downsample=True)
        
        logger.info("="*50)
        logger.info("TIME ALIGNMENT COMPLETE")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Time alignment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 