# Aligned PSTH Analysis with Dynamic Time Warping

This script implements dynamic time warping (DTW) to temporally align neural responses across trials with the same target direction (heading) and outcome, using spike time data rather than raw voltages or binned RMS.

## Features

- **Trial Grouping**: Groups trials by unique (heading, outcome) pairs (8 directions × 2 outcomes)
- **Spike Detection**: Uses existing spike detection utilities to extract spike times
- **Movement Onset Detection**: Detects first non-zero velocity after trial_start flag
- **DTW Alignment**: Applies dynamic time warping to align trials to a reference template
- **PSTH Computation**: Computes both aligned and unaligned PSTHs for comparison
- **Behavioral Alignment**: Applies same DTW warping to velocity data for direct comparison
- **Dual Visualization**: 
  - Main comprehensive plot with all conditions overlaid
  - Individual plots for each channel/heading/outcome combination
- **Individual Condition Analysis**: Each saved plot includes:
  - Top subplot: Time-warped behavioral kinematics (X, Y, magnitude)
  - Bottom subplot: Neural PSTH comparison (aligned vs unaligned)
- **Export**: Saves detailed results to CSV for further analysis

## Usage

### Basic Usage

```bash
python aligned_psth.py
```

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The script will automatically install `fastdtw` if not available and fall back to a custom DTW implementation if installation fails.

## Configuration

Key parameters can be adjusted in the script:

- `SPIKE_CHANNELS`: List of neural channels to analyze (currently set to [0, 1, 2] for testing)
- `PSTH_BIN_SIZE`: Time bin size for PSTH computation (default: 10ms)
- `ALIGN_WINDOW`: Time window around alignment event (default: -0.5 to 1.0 seconds)
- `GAUSSIAN_SIGMA`: Smoothing parameter for PSTH (default: 25ms)
- `H5_FILE_PATH`: Path to the H5 data file
- `DTW_RADIUS`: DTW constraint radius (default: 10% of sequence length)

### Quick Testing Configuration

For faster testing and development, the script is currently configured with:
```python
SPIKE_CHANNELS = [0, 1, 2]  # Reduced from full 21-channel list
```

To analyze all available channels, uncomment the full list:
```python
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
```

## Input Data Format

The script expects:
- `spike_times`: Nested structure where `spike_times[trial][channel]` contains spike times in seconds
- `align_times`: Array of event times per trial for alignment (movement onset)
- `trial_headings`: Array of target directions (0-7 for 8 targets)
- `trial_outcomes`: Array of trial outcomes (1=success, 0=failure)

## Output

### Files Generated

1. **`aligned_psth_dtw_comparison.png`**: Main comparison plot showing aligned vs unaligned PSTHs
2. **`figures/channel_{c}_heading_{h}_outcome_{o}.png`**: Individual plots for each condition (with behavioral kinematics on top)
3. **`aligned_psth_results.csv`**: Detailed results including:
   - Time series data for aligned and unaligned PSTHs
   - Peak firing rates and latencies
   - Trial counts per condition
   - Alignment performance metrics

### Plot Interpretations

**Main Comprehensive Plot:**
- **Top Behavioral Panel**: All conditions overlaid with different colors
- **Bottom Neural Panels**: Grid of aligned vs unaligned PSTHs for each channel/condition

**Individual Condition Plots** (`figures/channel_X_heading_Y_outcome_Z.png`):
- **Top Subplot - Behavioral Kinematics**:
  - **Blue solid line**: X-velocity component (time-warped average)
  - **Green dashed line**: Y-velocity component (time-warped average)  
  - **Red bold line**: Velocity magnitude (time-warped average)
  - **Shaded areas**: Confidence intervals (SEM)
  - **Vertical dotted line**: Movement onset alignment point

- **Bottom Subplot - Neural PSTH**:
  - **Gray trace**: Unaligned PSTH (traditional averaging)
  - **Red trace**: DTW-aligned PSTH (temporally warped)
  - **Shaded areas**: Standard error of the mean
  - **Vertical dashed line**: Movement onset alignment point

## Algorithm Details

### DTW Alignment Process

**Neural Data:**
1. **Spike Density Functions**: Convert spike trains to binned firing rates
2. **Reference Template**: Use median trial as alignment reference
3. **Warping**: Apply DTW to align each trial to the reference
4. **Averaging**: Compute mean across aligned trials

**Behavioral Data:**
1. **Velocity Interpolation**: Align velocity data to same time bins as neural data
2. **DTW Alignment**: Apply same temporal warping used for neural data
3. **Component Alignment**: Warp X-velocity, Y-velocity, and magnitude separately
4. **Average Kinematics**: Compute time-warped average velocity profiles

### Alignment Event Detection

Movement onset is detected as:
- **Primary**: First non-zero velocity after trial start (since velocity=0 indicates no movement)
- **Fallback**: Trial midpoint if no movement detected or if velocity data unavailable

This approach correctly interprets the behavioral data structure where:
- `trial_start` flag marks the beginning of each trial
- Velocity values of exactly 0 indicate no movement (not noise)
- Movement onset is the first sample with non-zero velocity after trial start

## Performance Metrics

The script reports:
- Peak firing rate improvements from alignment
- Latency changes
- Number of trials per condition
- Overall alignment effectiveness

## Troubleshooting

### Common Issues

1. **FastDTW Installation**: If `fastdtw` installation fails, the script uses a custom DTW implementation
2. **Memory Usage**: Large datasets may require adjusting bin sizes or time windows
3. **Insufficient Trials**: Groups with fewer than 3 trials are excluded by default

### Dependencies

- Python 3.7+
- NumPy, SciPy, Matplotlib, Pandas
- H5py for data loading
- FastDTW (optional, for faster DTW computation)

## Performance and Scaling

### Current Test Configuration
- **Channels**: 3 (reduced from 21 for faster testing)
- **Expected Groups**: ~14 (heading, outcome) combinations
- **Computation Time**: ~2-3 minutes for 3 channels
- **Output Files**: ~42 individual plots + 1 comprehensive plot

### Full Analysis Scaling
- **All Channels**: 21 channels would generate ~294 individual plots
- **Estimated Time**: ~15-20 minutes for full channel set
- **Memory Usage**: Moderate (handles ~127 trials efficiently)
- **Disk Space**: ~50-100MB for all plots and CSV export

### Optimization Tips
- Use test configuration (3 channels) for initial validation
- FastDTW installation recommended for faster alignment
- Adjust `ALIGN_WINDOW` to reduce computation if needed
- Set `min_trials_per_group` higher to exclude sparse conditions

## Example Output

```
🧠 Aligned PSTH Analysis with Dynamic Time Warping
============================================================
📊 Loading trial data and grouping by (heading, outcome)...
✅ Loaded 127 trials
   • Valid groups: 14
   • Heading 0°, success: 18 trials
   • Heading 45°, success: 15 trials
   • Heading 90°, failure: 4 trials
   ...
🧮 Computing aligned PSTHs using DTW...
   🔄 Aligning channel 0, heading 0°, outcome success...
   🔄 Aligning channel 1, heading 45°, outcome success...
   Progress: 20/42
📊 Creating aligned PSTH plots with behavioral kinematics...
   🎯 Computing time-warped behavioral averages...
   ✅ Main plot saved as: aligned_psth_dtw_comparison.png
   ✅ Individual plots saved in: figures/
💾 Exporting aligned PSTH results to aligned_psth_results.csv...
   ✅ Exported 12,450 rows to aligned_psth_results.csv
   • Headings: [0, 45, 90, 135, 180, 225, 270, 315]
   • Outcomes: ['failure', 'success']
   • Channels: 3
📊 Aligned PSTH Analysis Summary
   • Total trials processed: 127
   • Valid groups: 14
   • Channels analyzed: 3
   • DTW alignments computed: 42
   • Mean peak rate improvement: 23.5% ± 15.2%
   • Best improvement: 67.3%
   • Improvements > 10%: 32/42
🎉 Analysis complete!
   • Check 'figures/' directory for individual plots
   • Check 'aligned_psth_results.csv' for detailed results
```

## Citation

If you use this script in your research, please cite the relevant DTW and PSTH alignment literature.

### Related Methods
- **Dynamic Time Warping**: Sakoe & Chiba (1978) for temporal sequence alignment
- **Neural Spike Analysis**: Rieke et al. (1999) "Spikes: Exploring the Neural Code"
- **Motor Cortex PSTH**: Churchland et al. (2012) for motor cortex temporal dynamics
- **Trial Alignment**: Russo et al. (2018) for movement onset alignment methods

## Script Information

**Version**: 1.0  
**Author**: Neural Exploration Analysis Suite  
**Last Updated**: 2024  
**Purpose**: DTW-based temporal alignment of motor cortex responses to behavioral kinematics 