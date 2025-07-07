# Neural Feature Extraction Guide

This guide explains how to use the `neural_feature_extraction.py` script to extract advanced neural signal features from your spike channels.

## Overview

The script extracts 4 types of neural features from manually identified spike channels:

1. **Signal Features (Spike-free)**: Band power (400-6000 Hz) in time bins
2. **Local Field Potential (LFP)**: Low-pass filtered (<250 Hz) with gamma band analysis
3. **Simple Voltage Features**: Moving average and variance
4. **Thresholded Spike Counts**: Threshold crossings per time bin

## Prerequisites

- HDF5 file with neural data (created using `precise_time_alignment.py`)
- Python environment with required packages (see `requirements.txt`)

## Configuration

### Key Parameters (Top of Script)

```python
# File and trial selection
H5_FILE = 'aligned_neural_behavioral_data.h5'  # Auto-detected if not found
TRIAL_NUMBER = 1  # Trial to analyze

# Manually identified spike channels
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]

# Signal processing parameters
SAMPLING_RATE = 30000  # Hz
TIME_BIN_SIZE = 0.05   # seconds (50 ms bins)

# Filter parameters
SPIKE_BAND_LOW = 400    # Hz - spike band lower bound
SPIKE_BAND_HIGH = 6000  # Hz - spike band upper bound
LFP_CUTOFF = 250       # Hz - LFP low-pass cutoff
GAMMA_LOW = 30         # Hz - gamma band lower bound
GAMMA_HIGH = 100       # Hz - gamma band upper bound

# Threshold parameters
THRESHOLD_MULTIPLIER = -4  # Negative threshold multiplier (-4x RMS)
```

### Customization Options

1. **Change trial**: Modify `TRIAL_NUMBER` to analyze different trials
2. **Adjust time bins**: Change `TIME_BIN_SIZE` (e.g., 0.1 for 100ms bins)
3. **Modify frequency bands**: 
   - Spike band: Adjust `SPIKE_BAND_LOW` and `SPIKE_BAND_HIGH`
   - LFP: Modify `LFP_CUTOFF`
   - Gamma: Change `GAMMA_LOW` and `GAMMA_HIGH`
4. **Threshold sensitivity**: Adjust `THRESHOLD_MULTIPLIER` (more negative = higher threshold)

## Usage

### Basic Usage

```bash
python neural_feature_extraction.py
```

The script will:
1. Auto-detect your HDF5 file
2. Load the specified trial
3. Extract all 4 feature types
4. Create a comprehensive visualization
5. Display summary statistics
6. Optionally save features to NumPy file

### Expected Output

```
🧠 Neural Feature Extraction Tool
==================================================
📁 Using H5 file: aligned_neural_behavioral_data.h5
📂 Loading trial 1 from aligned_neural_behavioral_data.h5...
🔍 Analyzing 21 spike channels...
📊 Neural data shape: (96, 1080000)
🚀 Extracting features for 21 channels...
📊 Data shape: (21, 1080000)
⏱️  Time bin size: 50 ms
🔍 Extracting spike band power (400-6000 Hz)...
✅ Spike band power extraction completed
🧠 Extracting LFP features (low-pass <250 Hz, gamma 30-100 Hz)...
✅ LFP feature extraction completed
⚡ Extracting voltage features (moving average and variance)...
✅ Voltage feature extraction completed
🎯 Extracting threshold crossings (-4x RMS)...
✅ Threshold crossing extraction completed
🔍 Feature extraction completed!
📊 Creating visualization...
📊 Plot saved as neural_features_trial_001.png
```

## Output Files

### 1. Visualization (`neural_features_trial_XXX.png`)

The script generates a comprehensive 8-panel figure showing:

- **Spike Band Power (RMS)**: High-frequency activity (400-6000 Hz)
- **LFP Power**: Low-frequency local field potential
- **Gamma Power**: Gamma band activity (30-100 Hz)
- **Threshold Crossings**: Spike-like events per time bin
- **Moving Variance**: Signal variability over time
- **Feature Heatmap**: Spike band power across all channels
- **Summary Statistics**: Feature comparison across channels

### 2. Feature Data (`neural_features_trial_XXX.npz`)

Optional NumPy compressed file containing:

```python
# Load saved features
import numpy as np
features = np.load('neural_features_trial_001.npz', allow_pickle=True)

# Access different feature types
spike_features = features['spike_band'].item()
lfp_features = features['lfp'].item()
voltage_features = features['voltage'].item()
threshold_features = features['threshold'].item()

# Example: RMS power data
rms_power = spike_features['rms_power']  # Shape: (n_channels, n_time_bins)
time_axis = spike_features['time_axis']  # Time points for each bin
```

## Feature Descriptions

### 1. Spike Band Power (400-6000 Hz)
- **Purpose**: Capture high-frequency activity associated with action potentials
- **Method**: Bandpass filter + RMS power in time bins
- **Output**: 
  - `rms_power`: RMS power values (channels × time_bins)
  - `log_power`: Log-transformed power values

### 2. LFP Features (<250 Hz)
- **Purpose**: Analyze local field potentials and gamma activity
- **Method**: Low-pass filter + gamma bandpass + Hilbert transform
- **Output**:
  - `lfp_power`: LFP power (channels × time_bins)
  - `gamma_power`: Gamma band power (channels × time_bins)
  - `gamma_amplitude`: Gamma amplitude envelope (channels × time_bins)

### 3. Voltage Features
- **Purpose**: Simple statistical measures of raw signal
- **Method**: Moving average and variance in time bins
- **Output**:
  - `moving_average`: Mean voltage per bin (channels × time_bins)
  - `moving_variance`: Voltage variance per bin (channels × time_bins)

### 4. Threshold Crossings
- **Purpose**: Detect spike-like events without full spike sorting
- **Method**: Count negative threshold crossings (-4× RMS)
- **Output**:
  - `crossing_counts`: Number of crossings per bin (channels × time_bins)
  - `thresholds`: Threshold values per channel

## Troubleshooting

### Common Issues

1. **File not found**: Script auto-detects H5 files in current directory and `development_archive/`
2. **Trial not found**: Script tries multiple trial naming conventions (`trial_001`, `trial_1`, etc.)
3. **Memory issues**: Reduce `TIME_BIN_SIZE` or process fewer channels
4. **Slow processing**: Large files may take several minutes for feature extraction

### Error Messages

- `"No H5 files found!"`: Run `precise_time_alignment.py` first
- `"Trial X not found"`: Check available trials in your H5 file
- `"No neural data found"`: Check H5 file structure

### Performance Tips

1. **Faster processing**: Increase `TIME_BIN_SIZE` (e.g., 0.1 for 100ms bins)
2. **Memory optimization**: Process trials individually rather than all at once
3. **Selective analysis**: Modify `SPIKE_CHANNELS` list to focus on specific channels

## Interpreting Results

### Feature Statistics Example

```
📈 Feature Statistics:
  • Spike RMS Power: 0.613 ± 0.004    # High-frequency activity strength
  • LFP Power: 0.017 ± 0.001          # Low-frequency field potential
  • Gamma Power: 0.005 ± 0.001        # Gamma oscillation strength
  • Total Crossings: 22 across all channels  # Spike-like events
```

### Active Channel Identification

```
🔥 Most active channels: [74, 41, 46, 2, 42]
```

These channels show the highest combined activity across all feature types.

## Advanced Usage

### Batch Processing Multiple Trials

```python
# Process all trials in a loop
for trial_num in range(1, 141):  # Adjust range as needed
    TRIAL_NUMBER = trial_num
    # Run feature extraction
    # Save results with trial-specific names
```

### Custom Frequency Bands

```python
# For beta band analysis
BETA_LOW = 13
BETA_HIGH = 30

# Add custom filter in extract_lfp_features()
sos_beta = signal.butter(4, [BETA_LOW/self.nyquist, BETA_HIGH/self.nyquist], 
                        btype='band', output='sos')
```

### Integration with Behavioral Data

The script can be extended to correlate neural features with behavioral events:

```python
# Example: Correlate spike power with movement velocity
if 'behavioral_data' in trial_data and trial_data['behavioral_data']:
    velocity = trial_data['behavioral_data']['velocity_x']
    # Align behavioral and neural timescales
    # Compute correlations
```

## Next Steps

1. **Decode Movement**: Use features to predict behavioral variables
2. **Trial Comparison**: Compare features across different trial outcomes
3. **Channel Analysis**: Identify channels with task-related activity
4. **Time Analysis**: Examine feature evolution during trial phases

This feature extraction provides a solid foundation for advanced neural decoding and analysis of your spike channel data. 