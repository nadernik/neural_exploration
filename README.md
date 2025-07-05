# Neural Data Exploration Project

A modular Python toolkit for exploring neural data from a Center Out task with 96-channel Utah array recordings.

## Overview

This project provides tools for analyzing neural data from a cynomolgus macaque performing a Center Out task. The data consists of:
- **Neural Data**: 96-channel Utah array recordings (.ns6 format) from M1 hand knob area
- **Behavioral Data**: CSV file with task timing and cursor positions
- **Task**: Center Out with 8 targets, 127 successful trials

## Project Structure

```
neural_exploration/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── neural_exploration.ipynb     # Main exploration notebook
└── utils/
    ├── data_loader.py          # Data loading utilities
    └── visualization.py        # Visualization tools
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure data paths**:
   - Open `neural_exploration.ipynb`
   - Update `NS6_FILE_PATH` and `CSV_FILE_PATH` with your data file locations

3. **Run the notebook**:
   - Start Jupyter: `jupyter notebook neural_exploration.ipynb`
   - Or use JupyterLab: `jupyter lab neural_exploration.ipynb`

## Features

### Data Loading (`utils/data_loader.py`)
- **Neo library support**: Robust .ns6 file loading with Neo
- **Metadata extraction**: Sampling rate, duration, channel info
- **Error handling**: Graceful handling of missing files or libraries

### Visualization (`utils/visualization.py`)
- **Behavioral Task**: Center Out task layout with 8 targets
- **Utah Array**: 96-channel electrode layout visualization
- **Neural Data**: Multi-channel signal overview and structure analysis
- **Data Quality**: RMS, correlation, and power spectrum analysis
- **Trial Analysis**: Individual trial visualization with joystick velocity
- **Multi-Trial Summary**: Comparative analysis across multiple trials

### Main Notebook (`neural_exploration.ipynb`)
- **Step-by-step exploration**: Guided analysis workflow
- **Interactive visualizations**: Comprehensive plots and summaries
- **Data characterization**: Signal quality and structure assessment
- **Individual trial analysis**: Detailed joystick velocity and target visualization
- **Multi-trial comparisons**: Statistical analysis across trials
- **Helper functions**: Easy trial selection and analysis tools
- **Next steps guidance**: Clear path for detailed analysis

## Dependencies

The project requires these Python packages:
- `numpy`, `pandas`, `matplotlib`, `seaborn` (core data analysis)
- `scipy`, `scikit-learn` (signal processing and machine learning)
- `jupyter`, `ipywidgets` (interactive notebooks)
- `neo`, `elephant` (neural data analysis)

## Data Format Requirements

### Neural Data (.ns6)
- Blackrock Microsystems format
- 96-channel Utah array
- Broadband recordings (typically 30 kHz sampling)
- Contains LFP and spike information

### Behavioral Data (.csv)
- Task timing information
- Cursor position data
- Trial segmentation
- Target information

## Key Analysis Capabilities

1. **Signal Preprocessing**
   - Bandpass filtering for LFP and spike bands
   - Artifact detection and removal
   - Noise characterization

2. **Feature Extraction**
   - LFP power in frequency bands
   - Spike detection and firing rates
   - Spectral and time-frequency features

3. **Neural-Behavioral Analysis**
   - Trial alignment and segmentation
   - Movement prediction from neural activity
   - Target classification
   - Temporal dynamics analysis

4. **Modeling and Decoding**
   - Kalman filters for continuous decoding
   - Machine learning models
   - Performance evaluation

## Usage Examples

### Basic Data Loading
```python
from utils.data_loader import DataLoader

loader = DataLoader(ns6_file_path="data.ns6", csv_file_path="behavior.csv")
neural_data = loader.load_neural_data()
behavioral_data = loader.load_behavioral_data()
```

### Visualization
```python
from utils.visualization import BehavioralVisualizer, NeuralVisualizer

# Behavioral task layout
behav_viz = BehavioralVisualizer(behavioral_data)
fig = behav_viz.plot_center_out_layout()

# Individual trial analysis
fig = behav_viz.plot_trial_behavioral_data(trial_num=5)

# Multi-trial summary
fig = behav_viz.plot_all_trials_summary(max_trials=10)

# Neural data overview
neural_viz = NeuralVisualizer(neural_data, metadata)
fig = neural_viz.plot_channel_overview()
```

## Google Colab Compatibility

This code is designed to work in Google Colab:
1. Upload the project files to your Google Drive
2. Mount your Drive in Colab
3. Update file paths to point to your Drive locations
4. Install dependencies with `!pip install -r requirements.txt`

## Troubleshooting

### Common Issues

1. **Import errors**: Install missing packages with `pip install package_name`
2. **File not found**: Check and update file paths in the configuration section
3. **Memory issues**: Large neural datasets may require memory management
4. **Neo library issues**: Ensure Neo is properly installed with `pip install neo`

### Performance Tips

- **Memory usage**: Monitor RAM usage with large datasets
- **Processing time**: Some analyses are computationally intensive
- **Data quality**: Check for artifacts and noisy channels before analysis

## Contributing

This is a research tool designed for flexibility and modularity. Feel free to:
- Add new analysis modules
- Extend visualization capabilities
- Improve data loading robustness
- Add new decoding algorithms

## License

This project is for research and educational purposes. Please cite appropriately if used in publications.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the notebook documentation
3. Examine the utility module docstrings
4. Test with minimal data samples first 
