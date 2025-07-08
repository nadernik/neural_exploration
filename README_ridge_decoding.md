# Ridge Regression Decoding from Scratch

A comprehensive Python implementation for decoding 2D cursor velocity from neural data recorded with a 96-channel Utah array in the primary motor cortex (M1) during a center-out joystick task.

## Overview

This script implements a complete pipeline for neural decoding using ridge regression, including:

1. **Neural Data Loading**: Extracts broadband neural signals from .ns5 Blackrock files
2. **Spike Detection**: Uses PyWaveClus-style spike detection algorithms
3. **Time Binning**: Converts spike trains to firing rates in fixed-width time bins
4. **Behavioral Alignment**: Aligns neural data with behavioral velocity data using movement onset as time zero
5. **Ridge Regression**: Trains a regularized linear model to predict 2D velocity
6. **Cross-Validation**: Evaluates model performance using cross-validation
7. **Hyperparameter Optimization**: Finds optimal regularization parameters
8. **Visualization**: Generates comprehensive plots of results

## Key Features

### Movement Onset Detection
- Defines time zero as the first `trial_start = TRUE` event with non-zero velocity
- Filters out trials without clear movement onset
- Handles multiple trials for robust dataset creation

### Neural Processing
- Loads .ns5 files using the Neo library
- Applies PyWaveClus-style spike detection with configurable thresholds
- Bins spikes into 100ms time windows (configurable)
- Uses only validated good channels from existing analysis

### Ridge Regression Implementation
- Separate models for velocity_x and velocity_y components
- Optional feature normalization using StandardScaler
- Comprehensive evaluation metrics (R², correlation, MSE)
- Cross-validation with configurable folds

### Performance Evaluation
- Train/test split evaluation
- Cross-validation for robust performance estimation
- Multiple metrics: R², Pearson correlation, speed correlation
- Direction accuracy analysis

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn neo h5py
```

## Usage

### 1. Configuration

Edit the configuration section in `ridge_decoding_from_scratch.py`:

```python
# File paths - UPDATE THESE FOR YOUR DATA
NS5_FILE_PATH = "path/to/your/neural_data.ns5"  # Your .ns5 file
CSV_FILE_PATH = "path/to/your/behavioral_data.csv"  # Your behavioral CSV

# Neural parameters
GOOD_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
BIN_SIZE = 0.1  # seconds (100ms bins)
THRESHOLD_FACTOR = 5.0  # Spike detection threshold

# Decoding parameters
TIME_WINDOW = (-1.0, 2.0)  # seconds relative to movement onset
ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Ridge regularization
```

### 2. Data Format Requirements

#### Neural Data (.ns5 file)
- Blackrock .ns5 file with 96 channels
- 30 kHz sampling rate
- Continuous broadband neural recordings

#### Behavioral Data (CSV file)
Required columns:
- `timestamp`: DateTime stamps for each sample
- `trial_start`: Boolean flag indicating trial start events
- `velocity_x`: X-component of cursor velocity
- `velocity_y`: Y-component of cursor velocity

Optional columns:
- `trial_win`, `trial_lose`: Trial outcome flags
- `target_x`, `target_y`: Target positions

### 3. Running the Script

```bash
python ridge_decoding_from_scratch.py
```

The script will:
1. Load and validate your data files
2. Detect spikes in neural data
3. Find movement onsets in behavioral data
4. Create aligned neural-behavioral dataset
5. Optimize hyperparameters using cross-validation
6. Train the final ridge regression model
7. Generate comprehensive visualizations

### 4. Output Files

The script generates several output files:

- `ridge_decoding_results.png`: Main results visualization
- `hyperparameter_optimization.png`: Hyperparameter search results
- Console output with detailed metrics and progress

## Algorithm Details

### Spike Detection
Uses PyWaveClus-style detection with:
- Bandpass filtering (300-3000 Hz)
- Median-based noise estimation
- Threshold-based spike detection
- Refractory period enforcement

### Time Alignment
Movement onset detection:
1. Find all `trial_start = TRUE` events
2. For each trial start, look for first non-zero velocity
3. Extract time window around movement onset
4. Align neural and behavioral data to movement time

### Ridge Regression
- Separate models for velocity_x and velocity_y
- L2 regularization with cross-validated α selection
- Feature normalization using StandardScaler
- Comprehensive evaluation metrics

## Performance Metrics

The script evaluates decoding performance using:

1. **R² Score**: Coefficient of determination for each velocity component
2. **Pearson Correlation**: Linear correlation between true and predicted velocities
3. **Speed Correlation**: Correlation between true and predicted speed magnitudes
4. **Cross-Validation**: Robust performance estimation using k-fold CV

## Visualization

Generated plots include:

1. **True vs Predicted Scatter**: Shows decoding accuracy for each velocity component
2. **Speed Comparison**: Compares predicted vs true movement speeds
3. **Time Series Example**: Shows actual decoding performance over time
4. **Hyperparameter Search**: Cross-validation results for different α values

## Troubleshooting

### Common Issues

1. **File Not Found**: Verify file paths are correct
2. **Memory Issues**: Reduce time window or number of channels for large datasets
3. **No Movement Onsets**: Check behavioral data format and movement threshold
4. **Poor Performance**: Try different time windows or regularization parameters

### Data Quality Checks

The script includes automatic validation:
- Checks for required columns in behavioral data
- Validates neural data format and sampling rate
- Ensures sufficient movement trials for training
- Reports data quality metrics

## Customization

### Adjusting Parameters

- **Time Window**: Modify `TIME_WINDOW` to change the analysis period around movement onset
- **Bin Size**: Change `BIN_SIZE` to adjust temporal resolution
- **Channels**: Update `GOOD_CHANNELS` based on your channel validation
- **Spike Detection**: Adjust `THRESHOLD_FACTOR` for spike detection sensitivity

### Adding Features

The modular design allows easy extension:
- Add new neural features (LFP, spectral power, etc.)
- Implement different alignment strategies
- Try alternative decoding algorithms
- Add additional behavioral variables

## Expected Performance

Typical performance ranges for M1 velocity decoding:
- **R²**: 0.1-0.4 for instantaneous velocity
- **Correlation**: 0.3-0.7 for velocity components
- **Speed Correlation**: 0.4-0.8 for movement speed

Performance depends on:
- Data quality and recording duration
- Number of active neurons
- Task complexity and movement variability
- Temporal resolution of analysis

## References

1. Georgopoulos, A. P., Schwartz, A. B., & Kettner, R. E. (1986). Neuronal population coding of movement direction. *Science*, 233(4771), 1416-1419.

2. Wu, W., Gao, Y., Bienenstock, E., Donoghue, J. P., & Black, M. J. (2006). Bayesian population decoding of motor cortical activity using a Kalman filter. *Neural Computation*, 18(1), 80-118.

3. Quian Quiroga, R., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661-1687.

## License

This code is provided for research and educational purposes. Please cite appropriately if used in publications. 