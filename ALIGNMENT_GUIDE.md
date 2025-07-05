# Neural-Behavioral Data Alignment Guide

## Overview

This guide explains how to align behavioral and neural data using the Time Origin from the .ns6 file as a global reference. The alignment accounts for the ~25 second offset between neural and behavioral recording start times.

## Key Features

✅ **Time Origin Extraction**: Automatically extracts Time Origin from .ns6 file header  
✅ **Precise Alignment**: Uses neural Time Origin as global reference  
✅ **Overlap Detection**: Identifies overlapping time periods  
✅ **Blackrock Time Ticks**: Converts to 30 kHz clock for sub-millisecond precision  
✅ **Trial Segmentation**: Segments aligned trials for analysis  
✅ **Smart Caching**: Avoids reloading files if already loaded (saves time!)  

## Data Timeline

```
Behavioral (CSV):  |-------- 2025-03-25T09:22:28Z --------|
Neural (.ns6):              |-------- 2025-03-25T09:22:53Z --------|
Offset:                     |<-- ~25 seconds -->|
```

## Quick Start

```python
from utils.data_loader import DataLoader
from utils.visualization import BehavioralVisualizer

# Initialize loader
loader = DataLoader()

# Load data (Time Origin extracted automatically)
neural_data = loader.load_neural_data("your_data.ns6")
behavioral_data = loader.load_behavioral_data("your_data.csv")

# Align timestamps
alignment_info = loader.align_timestamps()

# Extract overlapping data
overlapping_data = loader.get_overlapping_data()

# Segment aligned trials
aligned_trials = loader.segment_aligned_trials()
```

## Expected Output

```
Neural recording started at: 2025-03-25 09:22:53+00:00
Behavioral recording started at: 2025-03-25 09:22:28+00:00
Time offset: -25.000 seconds
Neural started 25.0 seconds before behavioral
Overlap duration: XXX.X seconds
Alignment quality: good
```

## Available Methods

### 1. `align_timestamps()`
- Aligns behavioral and neural timestamps using neural Time Origin
- Adds `timestamp_aligned` columns to both datasets
- Returns alignment information dictionary

### 2. `get_overlapping_data()`
- Extracts overlapping portions of both datasets
- Returns synchronized data for analysis

### 3. `convert_to_blackrock_ticks()`
- Converts timestamps to Blackrock time ticks (30 kHz)
- Provides sub-millisecond precision for precise alignment

### 4. `segment_aligned_trials()`
- Segments trials using aligned timestamps
- Returns trial data with aligned timing information

## Smart Caching Optimization

**Problem**: Large .ns6 files can take a long time to load (minutes for large recordings).

**Solution**: The data loader now checks if the same file is already loaded before attempting to reload it.

### How It Works

```python
# First load - reads from disk (slow)
neural_data = loader.load_neural_data("data.ns6")

# Subsequent loads of same file - uses cached data (fast!)
neural_data = loader.load_neural_data("data.ns6")  # Skips reload
```

### Expected Output

```
# First call
Loading neural data from: data.ns6
Neural data loaded successfully!

# Second call
Neural data from data.ns6 is already loaded.
Skipping reload (use force_reload=True to reload anyway)
Loaded data info:
  - Shape: (18000000, 96)
  - Duration: 600.0 seconds
  - Channels: 96
  - Sampling rate: 30000.0 Hz
```

### Force Reload When Needed

Sometimes you may want to reload the same file (e.g., if file was updated):

```python
# Force reload even if already loaded
neural_data = loader.load_neural_data("data.ns6", force_reload=True)
behavioral_data = loader.load_behavioral_data("data.csv", force_reload=True)
```

### Benefits

- **Faster iteration**: No waiting for reload during analysis
- **Better workflow**: Run cells multiple times without penalty
- **Memory efficient**: Same data instance reused
- **Automatic**: Works transparently, no code changes needed

## Alignment Quality Assessment

| Overlap Duration | Quality | Description |
|------------------|---------|-------------|
| > 5 minutes      | Excellent | Plenty of data for analysis |
| 1-5 minutes      | Good | Sufficient for basic analysis |
| 10-60 seconds    | Fair | Limited but usable |
| < 10 seconds     | Poor | Check timestamp formats |

## Troubleshooting

### Common Issues

1. **"No time origin available"**
   - Check if .ns6 file is corrupted
   - Verify Neo library installation
   - File might be missing header information

2. **"Poor alignment quality"**
   - Check timestamp formats in CSV
   - Verify both files are from same session
   - Check for timezone issues

3. **Large time offset (> 1 hour)**
   - Verify timezone settings
   - Check if timestamps are in different formats

### Debugging

Enable debugging in the visualization:
```python
# This will show detailed trajectory computation info
viz = BehavioralVisualizer(behavioral_data)
fig = viz.plot_trial_behavioral_data(trial_num=0)
```

## Advanced Usage

### Precise Neural-Behavioral Analysis

```python
# Get trial start times in Blackrock ticks
trial_starts_sec = aligned_trials['trial_info']['start_time_aligned']
trial_starts_ticks = loader.convert_to_blackrock_ticks(trial_starts_sec)

# Extract neural data around behavioral events
for i, start_tick in enumerate(trial_starts_ticks):
    # Find neural samples around this behavioral event
    # with sub-millisecond precision
    pass
```

### Custom Time Windows

```python
# Extract specific time window
custom_data = loader.get_overlapping_data(
    start_time=10.0,  # 10 seconds after neural start
    end_time=60.0     # 60 seconds after neural start
)
```

## File Structure

```
neural_exploration/
├── utils/
│   ├── data_loader.py          # Enhanced with alignment methods
│   └── visualization.py        # Debug info for trajectory plots
├── data_alignment_example.py   # Complete alignment example
├── neural_exploration_with_alignment.ipynb  # Jupyter notebook
└── ALIGNMENT_GUIDE.md          # This guide
```

## Next Steps

After successful alignment, you can:

1. **Analyze trial-locked neural activity**
   - Compute peri-event time histograms (PETHs)
   - Examine neural firing around behavioral events

2. **Movement analysis**
   - Correlate neural activity with joystick movements
   - Analyze preparatory activity before movement onset

3. **Population dynamics**
   - Study neural population responses to different targets
   - Analyze trial-to-trial variability

4. **Spike sorting**
   - Identify single units from multi-unit recordings
   - Analyze single-unit responses to behavioral events

## Support

If you encounter issues:
1. Check that both data files are from the same recording session
2. Verify file paths and formats
3. Ensure Neo library is properly installed
4. Check alignment quality and overlap duration

The alignment system provides sub-millisecond precision for neural-behavioral analysis, enabling precise investigation of neural correlates of behavior in the Center Out task. 