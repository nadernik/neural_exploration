#!/usr/bin/env python3
"""
Neural-Behavioral Data Integration Script

This script processes large neural recordings (.ns6) and behavioral data (CSV) to create
trial-segmented HDF5 files for efficient analysis.

Author: Neural Exploration Team
Date: 2025
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

class NeuralBehavioralIntegrator:
    """
    A class to integrate neural recordings with behavioral data and save trials to HDF5.
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
        
    def load_behavioral_data(self) -> pd.DataFrame:
        """
        Load and parse behavioral data from CSV.
        
        Returns:
            DataFrame with parsed behavioral data
        """
        logger.info(f"Loading behavioral data from {self.behavioral_file}")
        
        # Load CSV
        df = pd.read_csv(self.behavioral_file)
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            # The timestamps appear to be in a high-precision format
            # Let's examine the raw values first
            logger.info(f"Raw timestamp sample: {df['timestamp'].iloc[0]}")
            logger.info(f"Timestamp data type: {df['timestamp'].dtype}")
            
            # Try different parsing approaches
            try:
                # First try direct datetime parsing
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info("Used direct datetime parsing")
            except:
                try:
                    # Try parsing as UNIX timestamp in seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    logger.info("Used UNIX timestamp (seconds) parsing")
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
        Setup neural data I/O.
        
        Returns:
            BlackrockIO object for neural data access
        """
        logger.info(f"Setting up neural I/O for {self.neural_file}")
        
        # Initialize BlackrockIO - try with and without lazy parameter
        try:
            self.neural_io = BlackrockIO(filename=str(self.neural_file), lazy=True)
            logger.info("Using lazy loading")
        except TypeError:
            # Fallback for older versions of neo that don't support lazy parameter
            logger.info("Lazy loading not supported, using regular loading")
            self.neural_io = BlackrockIO(filename=str(self.neural_file))
        
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
            
            # Method 3: Try to read a minimal block and get datetime from there
            if time_origin is None:
                try:
                    # Read just the first second to get timing info
                    block = self.neural_io.read_block(time_slice=(0, 1))
                    if block.segments and block.segments[0].analogsignals:
                        signal = block.segments[0].analogsignals[0]
                        # Get the t_start time
                        t_start = signal.t_start
                        if hasattr(t_start, 'rescale'):
                            t_start = float(t_start.rescale('s'))
                        # This gives us relative time, but we need absolute time
                        logger.info(f"Signal t_start: {t_start}")
                except Exception as e:
                    logger.warning(f"Could not read test block: {e}")
            
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
    
    def extract_neural_data(self, trial: Dict) -> Optional[np.ndarray]:
        """
        Extract neural data for a specific trial.
        
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
            
            logger.info(f"Extracting neural data for trial {trial['trial_number']}: "
                       f"{start_seconds:.3f}s to {end_seconds:.3f}s")
            
            # Read the specific time slice - handle different neo versions
            try:
                # Try with lazy parameter first
                block = self.neural_io.read_block(
                    lazy=False, 
                    time_slice=(start_seconds, end_seconds)
                )
            except TypeError:
                # Fallback for older versions that don't support lazy parameter
                try:
                    block = self.neural_io.read_block(
                        time_slice=(start_seconds, end_seconds)
                    )
                except TypeError:
                    # If time_slice is not supported, we need a different approach
                    logger.warning("Time slice not supported in this neo version")
                    logger.warning("This version of neo requires loading the entire file, which may fail due to memory constraints")
                    logger.warning(f"Skipping neural data extraction for trial {trial['trial_number']}")
                    return None
            
            if not block.segments:
                logger.warning(f"No segments found for trial {trial['trial_number']}")
                return None
            
            # Get the first segment
            segment = block.segments[0]
            
            if not segment.analogsignals:
                logger.warning(f"No analog signals found for trial {trial['trial_number']}")
                return None
            
            # Combine all analog signals (channels)
            neural_data = []
            for signal in segment.analogsignals:
                # Convert to numpy array and transpose to get (channels, time)
                data = signal.magnitude
                
                # Handle different data shapes
                if data.ndim == 1:
                    # Single channel
                    neural_data.append(data.reshape(1, -1))
                elif data.ndim == 2:
                    # Multiple channels - transpose to get (channels, time)
                    neural_data.append(data.T)
                else:
                    logger.warning(f"Unexpected data shape: {data.shape}")
                    continue
            
            if not neural_data:
                logger.warning(f"No valid neural data found for trial {trial['trial_number']}")
                return None
            
            # Concatenate all channels
            neural_array = np.vstack(neural_data)
            
            logger.info(f"Extracted neural data shape: {neural_array.shape}")
            
            return neural_array
            
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
        
        # Downsample each channel
        downsampled_channels = []
        for i in range(neural_data.shape[0]):
            # Use scipy.signal.decimate for anti-aliasing
            downsampled = decimate(neural_data[i], self.downsample_factor, axis=0)
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
            for trial in self.trials:
                trial_group = f.create_group(f"trial_{trial['trial_number']}")
                
                # Extract neural data
                neural_data = self.extract_neural_data(trial)
                
                if neural_data is not None:
                    # Downsample if requested
                    if downsample:
                        neural_data = self.downsample_neural_data(neural_data)
                    
                    # Save neural data
                    trial_group.create_dataset('neural', data=neural_data, compression='gzip')
                    
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
                    
                    logger.info(f"Saved trial {trial['trial_number']} with neural shape: {neural_data.shape}")
                else:
                    logger.warning(f"Skipping trial {trial['trial_number']} due to neural data extraction failure")
        
        logger.info(f"Successfully saved {len(self.trials)} trials to {self.output_file}")
    
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

def check_dependencies():
    """
    Check if all required dependencies are available and their versions.
    """
    try:
        import neo
        logger.info(f"Neo version: {neo.__version__}")
        
        # Check if BlackrockIO is available
        try:
            from neo.io import BlackrockIO
            logger.info("BlackrockIO available")
        except ImportError:
            logger.error("BlackrockIO not available in this neo version")
            return False
            
        import pandas as pd
        logger.info(f"Pandas version: {pd.__version__}")
        
        import h5py
        logger.info(f"h5py version: {h5py.__version__}")
        
        import scipy
        logger.info(f"Scipy version: {scipy.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    """
    Main function to run the neural-behavioral integration.
    """
    logger.info("Starting Neural-Behavioral Integration")
    logger.info("="*50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install missing dependencies")
        return
    
    # File paths (modify these as needed)
    neural_file = "neural.ns6"
    behavioral_file = "actions.csv"
    output_file = "trials.h5"
    
    # Check if files exist
    if not Path(neural_file).exists():
        logger.error(f"Neural file not found: {neural_file}")
        logger.info("Please update the neural_file path in the script")
        return
    
    if not Path(behavioral_file).exists():
        logger.error(f"Behavioral file not found: {behavioral_file}")
        logger.info("Please update the behavioral_file path in the script")
        return
    
    # Create integrator
    integrator = NeuralBehavioralIntegrator(
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
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please check that your input files exist and paths are correct")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.info("Common issues:")
        logger.info("1. Check that your .ns6 file is valid and readable")
        logger.info("2. Verify that your CSV has the required columns")
        logger.info("3. Ensure you have write permissions for the output directory")
        logger.info("4. Check that you have enough disk space")
        raise

if __name__ == "__main__":
    main() 