#!/usr/bin/env python3
"""
Fixed Neural-Behavioral Data Integration Script

This version works with older neo versions that don't support time_slice.
Uses get_analogsignal_chunk instead of read_block with time_slice.
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
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuralBehavioralIntegratorFixed:
    """
    Fixed version that works with older neo versions without time_slice support.
    """
    
    def __init__(self, neural_file: str, behavioral_file: str, output_file: str = "trials.h5"):
        """
        Initialize the integrator.
        
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
        self.neural_start_time = None
        self.behavioral_data = None
        self.trials = []
        
        # Neural data parameters
        self.original_fs = 30000  # Original sampling rate (Hz)
        self.target_fs = 1000     # Target sampling rate after downsampling (Hz)
        self.downsample_factor = self.original_fs // self.target_fs
        
        # Neural file metadata
        self.signal_size = None
        self.channel_count = None
        self.sampling_rate = None
        
    def load_behavioral_data(self) -> pd.DataFrame:
        """
        Load and parse behavioral data from CSV with improved timestamp handling.
        
        Returns:
            DataFrame with parsed behavioral data
        """
        logger.info(f"Loading behavioral data from {self.behavioral_file}")
        
        # Load CSV
        df = pd.read_csv(self.behavioral_file)
        
        # Parse timestamps with better handling
        if 'timestamp' in df.columns:
            # The timestamps appear to be in a high-precision format
            # Let's examine the raw values first
            logger.info(f"Raw timestamp sample: {df['timestamp'].iloc[0]}")
            logger.info(f"Timestamp data type: {df['timestamp'].dtype}")
            
            # Force parsing as UNIX timestamp in seconds - this is the correct format
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                logger.info("Used UNIX timestamp (seconds) parsing")
            except Exception as e:
                logger.error(f"Failed to parse timestamps as UNIX seconds: {e}")
                # Fallback to other methods
                try:
                    # Try direct datetime parsing
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info("Used direct datetime parsing as fallback")
                except:
                    try:
                        # Try parsing as UNIX timestamp in milliseconds
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        logger.info("Used UNIX timestamp (milliseconds) parsing")
                    except:
                        # Try parsing as UNIX timestamp in microseconds
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
                        logger.info("Used UNIX timestamp (microseconds) parsing")
            
            # Check if the parsed timestamps make sense
            timestamp_range = df['timestamp'].max() - df['timestamp'].min()
            logger.info(f"Parsed timestamp range: {timestamp_range.total_seconds():.2f} seconds")
            
            # If the range is too small (< 1 minute), the timestamps are probably wrong
            if timestamp_range.total_seconds() < 60:
                logger.warning("Timestamp range is suspiciously small - check timestamp format!")
                
        else:
            raise ValueError("No 'timestamp' column found in behavioral data")
        
        # Ensure boolean columns are properly typed
        bool_cols = ['trial_start', 'trial_win', 'trial_lose']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        logger.info(f"Loaded {len(df)} behavioral samples")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.behavioral_data = df
        return df
    
    def setup_neural_io(self) -> BlackrockIO:
        """
        Setup neural data I/O and get metadata.
        
        Returns:
            BlackrockIO object for neural data access
        """
        logger.info(f"Setting up neural I/O for {self.neural_file}")
        
        # Initialize BlackrockIO
        try:
            self.neural_io = BlackrockIO(filename=str(self.neural_file), lazy=True)
            logger.info("Using lazy loading")
        except TypeError:
            self.neural_io = BlackrockIO(filename=str(self.neural_file))
            logger.info("Lazy loading not supported, using regular loading")
        
        # Get neural file metadata
        try:
            # Get sampling rate
            self.sampling_rate = self.neural_io.get_signal_sampling_rate()
            logger.info(f"Sampling rate: {self.sampling_rate} Hz")
            
            # Get signal size (need to provide block_index and seg_index)
            try:
                # Most files have block_index=0, seg_index=0
                self.signal_size = self.neural_io.get_signal_size(block_index=0, seg_index=0)
                logger.info(f"Signal size: {self.signal_size} samples")
            except Exception as e:
                logger.warning(f"Could not get signal size: {e}")
                self.signal_size = None
            
            # Get channel count
            if hasattr(self.neural_io, 'signal_channels_count'):
                self.channel_count = self.neural_io.signal_channels_count(block_index=0, seg_index=0)
                logger.info(f"Channel count: {self.channel_count}")
            else:
                logger.warning("Could not get channel count")
                self.channel_count = 96  # Default for Utah array
            
        except Exception as e:
            logger.warning(f"Error getting neural metadata: {e}")
            self.sampling_rate = 30000  # Default
            self.channel_count = 96
        
        # Get the time origin from header
        try:
            # Try different ways to access the header information
            time_origin = None
            
            # Method 1: Try to get from raw_annotations
            if hasattr(self.neural_io, 'raw_annotations'):
                time_origin = self.neural_io.raw_annotations.get('Time Origin', None)
            
            # Method 2: Try to get from header
            if time_origin is None and hasattr(self.neural_io, 'header'):
                time_origin = self.neural_io.header.get('Time Origin', None)
            
            if time_origin:
                # Parse the time origin (e.g., "2025-03-25T09:22:53Z")
                if isinstance(time_origin, str):
                    # Remove 'Z' and parse
                    time_origin = time_origin.rstrip('Z')
                    self.neural_start_time = datetime.fromisoformat(time_origin)
                else:
                    self.neural_start_time = time_origin
                logger.info(f"Found Time Origin in header: {self.neural_start_time}")
            else:
                # Fallback: use the provided time
                self.neural_start_time = datetime.fromisoformat("2025-03-25T09:22:53")
                logger.warning("Time Origin not found in header, using default: 2025-03-25T09:22:53Z")
                
        except Exception as e:
            logger.warning(f"Could not parse Time Origin from header: {e}")
            # Use the provided default time
            self.neural_start_time = datetime.fromisoformat("2025-03-25T09:22:53")
        
        logger.info(f"Neural recording start time: {self.neural_start_time}")
        
        return self.neural_io
    
    def segment_trials(self) -> List[Dict]:
        """
        Segment behavioral data into individual trials.
        
        Returns:
            List of trial dictionaries with metadata
        """
        logger.info("Segmenting behavioral data into trials")
        
        if self.behavioral_data is None:
            raise ValueError("Behavioral data not loaded. Call load_behavioral_data() first.")
        
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
            
            # Find the end of this trial (next trial_win or trial_lose)
            end_idx = None
            outcome = None
            
            # Look for trial end from current position onwards
            for j in range(start_idx, len(df)):
                if df.loc[j, 'trial_win']:
                    end_idx = j
                    outcome = 'win'
                    break
                elif df.loc[j, 'trial_lose']:
                    end_idx = j
                    outcome = 'lose'
                    break
            
            # If we found an end, complete the trial info
            if end_idx is not None:
                trial_data['end_idx'] = end_idx
                trial_data['end_time'] = df.loc[end_idx, 'timestamp']
                trial_data['outcome'] = outcome
                trial_data['duration'] = (trial_data['end_time'] - trial_data['start_time']).total_seconds()
                
                # Extract behavioral data for this trial
                trial_behavioral = df.loc[start_idx:end_idx].copy()
                trial_data['behavioral_data'] = trial_behavioral
                
                trials.append(trial_data)
            else:
                logger.warning(f"Trial {i+1} has no clear end (win/lose). Skipping.")
        
        logger.info(f"Successfully segmented {len(trials)} complete trials")
        self.trials = trials
        return trials
    
    def time_to_neural_seconds(self, timestamp: datetime) -> float:
        """
        Convert behavioral timestamp to seconds relative to neural recording start.
        
        Args:
            timestamp: Behavioral timestamp
            
        Returns:
            Time in seconds relative to neural recording start
        """
        if self.neural_start_time is None:
            raise ValueError("Neural start time not set. Call setup_neural_io() first.")
        
        # Convert to seconds relative to neural start
        delta = timestamp - self.neural_start_time
        return delta.total_seconds()
    
    def extract_neural_data_chunk(self, trial: Dict) -> Optional[np.ndarray]:
        """
        Extract neural data using get_analogsignal_chunk method.
        
        Args:
            trial: Trial dictionary with timing information
            
        Returns:
            Neural data array with shape (channels, time) or None if extraction fails
        """
        try:
            # Convert behavioral times to neural recording time
            start_seconds = self.time_to_neural_seconds(trial['start_time'])
            end_seconds = self.time_to_neural_seconds(trial['end_time'])
            
            # Add small buffer to ensure we capture the full trial
            buffer_sec = 0.1  # 100ms buffer
            start_seconds = max(0, start_seconds - buffer_sec)
            end_seconds = end_seconds + buffer_sec
            
            # Convert to sample indices
            start_sample = int(start_seconds * self.sampling_rate)
            end_sample = int(end_seconds * self.sampling_rate)
            
            # Check bounds and handle negative times
            if start_sample < 0:
                logger.warning(f"Trial start time is before neural recording start. Adjusting from {start_sample} to 0")
                start_sample = 0
            
            if end_sample < 0:
                logger.error(f"Trial end time is before neural recording start. Skipping trial.")
                return None
            
            if self.signal_size is not None:
                if end_sample > self.signal_size:
                    logger.warning(f"Trial end time is after neural recording end. Adjusting from {end_sample} to {self.signal_size}")
                    end_sample = self.signal_size
            
            sample_count = end_sample - start_sample
            
            # Ensure we have a valid sample range
            if sample_count <= 0:
                logger.error(f"Invalid sample range: {start_sample} to {end_sample} ({sample_count} samples)")
                return None
            
            logger.info(f"Extracting neural data for trial {trial['trial_number']}: "
                       f"samples {start_sample} to {end_sample} ({sample_count} samples)")
            
            # Extract neural data using get_analogsignal_chunk
            try:
                # Try to get all channels at once
                neural_chunk = self.neural_io.get_analogsignal_chunk(
                    block_index=0,
                    seg_index=0,
                    i_start=start_sample,
                    i_stop=end_sample,
                    channel_indexes=None  # Get all channels
                )
                
                # neural_chunk should be (time, channels) - transpose to (channels, time)
                neural_data = neural_chunk.T
                
                logger.info(f"Successfully extracted neural data shape: {neural_data.shape}")
                
                return neural_data
                
            except Exception as e:
                logger.error(f"get_analogsignal_chunk failed: {e}")
                
                # Fallback: try to get channels individually
                logger.info("Trying to read channels individually...")
                
                neural_data_list = []
                for ch in range(min(96, self.channel_count)):  # Limit to 96 channels
                    try:
                        chunk = self.neural_io.get_analogsignal_chunk(
                            block_index=0,
                            seg_index=0,
                            i_start=start_sample,
                            i_stop=end_sample,
                            channel_indexes=[ch]
                        )
                        neural_data_list.append(chunk.flatten())
                    except Exception as e2:
                        logger.warning(f"Failed to read channel {ch}: {e2}")
                        # Add zeros for missing channels
                        neural_data_list.append(np.zeros(sample_count))
                
                if neural_data_list:
                    neural_data = np.array(neural_data_list)
                    logger.info(f"Individual channel reading successful: {neural_data.shape}")
                    return neural_data
                else:
                    logger.error("All channel reading methods failed")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to extract neural data for trial {trial['trial_number']}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def downsample_neural_data(self, neural_data: np.ndarray) -> np.ndarray:
        """
        Downsample neural data from 30kHz to 1kHz.
        
        Args:
            neural_data: Neural data array with shape (channels, time)
            
        Returns:
            Downsampled neural data array
        """
        logger.info(f"Downsampling neural data from {self.original_fs}Hz to {self.target_fs}Hz")
        
        # Check if data is long enough to downsample
        min_samples_needed = self.downsample_factor * 2  # Need at least 2 output samples
        if neural_data.shape[1] < min_samples_needed:
            logger.warning(f"Neural data too short for downsampling ({neural_data.shape[1]} samples < {min_samples_needed}). Returning original data.")
            return neural_data
        
        # Downsample each channel
        downsampled_channels = []
        for i in range(neural_data.shape[0]):
            try:
                # Use scipy.signal.decimate for anti-aliasing
                downsampled = decimate(neural_data[i], self.downsample_factor, axis=0)
                downsampled_channels.append(downsampled)
            except Exception as e:
                logger.warning(f"Failed to downsample channel {i}: {e}. Using original data.")
                # Use simple downsampling as fallback
                downsampled = neural_data[i][::self.downsample_factor]
                downsampled_channels.append(downsampled)
        
        downsampled_array = np.array(downsampled_channels)
        logger.info(f"Downsampled neural data shape: {downsampled_array.shape}")
        
        return downsampled_array
    
    def save_trials_to_hdf5(self, downsample: bool = True):
        """
        Save all trials to HDF5 file.
        
        Args:
            downsample: Whether to downsample neural data to 1kHz
        """
        logger.info(f"Saving trials to {self.output_file}")
        
        if not self.trials:
            raise ValueError("No trials to save. Call segment_trials() first.")
        
        with h5py.File(self.output_file, 'w') as f:
            # Add global metadata
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['neural_file'] = str(self.neural_file)
            f.attrs['behavioral_file'] = str(self.behavioral_file)
            f.attrs['neural_start_time'] = self.neural_start_time.isoformat()
            f.attrs['original_sampling_rate'] = self.original_fs
            f.attrs['final_sampling_rate'] = self.target_fs if downsample else self.original_fs
            f.attrs['total_trials'] = len(self.trials)
            
            # Process each trial
            successful_trials = 0
            failed_trials = 0
            
            for trial in self.trials:
                trial_group = f.create_group(f"trial_{trial['trial_number']}")
                
                # Extract neural data using chunk method
                neural_data = self.extract_neural_data_chunk(trial)
                
                if neural_data is not None:
                    # Downsample if requested
                    if downsample:
                        neural_data = self.downsample_neural_data(neural_data)
                    
                    # Save neural data
                    trial_group.create_dataset('neural', data=neural_data, compression='gzip')
                    successful_trials += 1
                    
                    # Add trial metadata as attributes
                    trial_group.attrs['trial_number'] = trial['trial_number']
                    trial_group.attrs['start_time'] = trial['start_time'].isoformat()
                    trial_group.attrs['end_time'] = trial['end_time'].isoformat()
                    trial_group.attrs['duration'] = trial['duration']
                    trial_group.attrs['outcome'] = trial['outcome']
                    
                    if trial['target_index'] is not None:
                        trial_group.attrs['target_index'] = trial['target_index']
                    
                    # Save behavioral data for this trial
                    behavioral_data = trial['behavioral_data']
                    
                    # Convert behavioral data to arrays for HDF5 storage
                    timestamps = np.array([t.timestamp() for t in behavioral_data['timestamp']])
                    trial_group.create_dataset('behavioral_timestamps', data=timestamps)
                    
                    if 'velocity_x' in behavioral_data.columns:
                        trial_group.create_dataset('velocity_x', data=behavioral_data['velocity_x'].values)
                    if 'velocity_y' in behavioral_data.columns:
                        trial_group.create_dataset('velocity_y', data=behavioral_data['velocity_y'].values)
                    
                    logger.info(f"✅ Saved trial {trial['trial_number']} with neural shape: {neural_data.shape}")
                else:
                    failed_trials += 1
                    logger.warning(f"❌ Skipped trial {trial['trial_number']} due to neural data extraction failure")
        
        logger.info(f"Successfully saved {successful_trials} trials, {failed_trials} failed")
        
        # Check final file size
        if self.output_file.exists():
            file_size = self.output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Final H5 file size: {file_size:.2f} MB")
    
    def process_all(self, downsample: bool = True):
        """
        Run the complete processing pipeline.
        
        Args:
            downsample: Whether to downsample neural data to 1kHz
        """
        logger.info("Starting complete neural-behavioral integration pipeline")
        
        # Load behavioral data
        self.load_behavioral_data()
        
        # Setup neural I/O
        self.setup_neural_io()
        
        # Segment trials
        self.segment_trials()
        
        # Save to HDF5
        self.save_trials_to_hdf5(downsample=downsample)
        
        logger.info("Processing complete!")

def main():
    """
    Main function to run the fixed neural-behavioral integration.
    """
    logger.info("Starting Fixed Neural-Behavioral Integration")
    logger.info("="*50)
    
    # File paths
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"
    output_file = r"D:\Data\ScienceCorp\trials_fixed.h5"
    
    # Check if files exist
    if not Path(neural_file).exists():
        logger.error(f"Neural file not found: {neural_file}")
        return
    
    if not Path(behavioral_file).exists():
        logger.error(f"Behavioral file not found: {behavioral_file}")
        return
    
    # Create integrator
    integrator = NeuralBehavioralIntegratorFixed(
        neural_file=neural_file,
        behavioral_file=behavioral_file,
        output_file=output_file
    )
    
    # Run the complete pipeline
    try:
        integrator.process_all(downsample=True)
        
        # Print summary
        logger.info("="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Neural file: {neural_file}")
        logger.info(f"Behavioral file: {behavioral_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Total trials processed: {len(integrator.trials)}")
        logger.info(f"Neural sampling rate: {integrator.original_fs}Hz -> {integrator.target_fs}Hz")
        
        # Check output file
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Output file size: {file_size:.2f} MB")
            logger.info("✅ Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 