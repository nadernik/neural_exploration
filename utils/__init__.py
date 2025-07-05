"""
Neural Data Exploration Utilities

This package provides utilities for loading and visualizing neural data
from Center Out tasks with Utah array recordings.
"""

from .data_loader import DataLoader
from .visualization import BehavioralVisualizer, NeuralVisualizer, create_utah_array_layout

__all__ = [
    'DataLoader',
    'BehavioralVisualizer', 
    'NeuralVisualizer',
    'create_utah_array_layout'
]

__version__ = '1.0.0'
__author__ = 'Neural Data Exploration Team' 