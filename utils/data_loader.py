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
        Load behavioral data from CSV file.
        
        Parameters:
        -----------
        csv_file_path : str, optional
            Path to CSV file. If None, uses self.csv_file_path
            
        Returns:
        --------
        pandas.DataFrame
            Behavioral data with proper timestamps
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
            
            print(f"Behavioral data loaded successfully!")
            print(f"Shape: {self.behavioral_data.shape}")
            print(f"Columns: {list(self.behavioral_data.columns)}")
            
            return self.behavioral_data
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None
    
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