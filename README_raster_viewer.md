# Neural Spike Raster Plot Viewer

A standalone script to visualize neural spike activity across all channels for a given trial using raster plots with vertical lines.

## Features

- **Spike Detection**: Uses PyWaveClus-style spike detection with configurable threshold
- **Behavioral + Neural Plot**: Shows behavioral velocity data above spike raster (1:20 ratio)
- **Raster Plot**: Shows spike times as vertical lines across all channels
- **Multi-channel Support**: Handles up to 20 channels simultaneously for readability
- **Customizable Parameters**: Configurable via command line or script modification
- **Summary Statistics**: Displays spike counts and firing rates per channel

## Quick Start

### Basic Usage

```bash
python view_trial_raster.py
```

This will create a raster plot for Trial 1 using default parameters.

### Advanced Usage

```bash
# View different trial
python view_trial_raster.py --trial 5

# Use different H5 file
python view_trial_raster.py --h5_file "path/to/your/data.h5"

# Adjust spike detection threshold
python view_trial_raster.py --threshold 3.0

# Save plot instead of displaying
python view_trial_raster.py --save "trial_1_raster.png"

# Combine multiple options
python view_trial_raster.py --trial 10 --threshold 4.0 --save "trial_10_raster.png"
```

### Command Line Arguments

- `--trial`: Trial number to visualize (default: 1)
- `--h5_file`: Path to H5 data file 
- `--threshold`: Spike detection threshold factor (default: 5.0)
- `--save`: Path to save the plot (if not specified, shows plot)

## Configuration

### File Path Setup

Update the `H5_FILE_PATH` variable in the script to point to your H5 data file:

```python
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"
```

### Spike Detection Parameters

```python
SAMPLING_RATE = 30000  # Hz
THRESHOLD_FACTOR = 5.0  # Spike detection threshold
SPIKE_WINDOW = (-10, 32)  # samples around spike
SPIKE_CHANNELS = [0, 1, 2, 3, 6, 32, 39, 40, 41, 42, 46, 49, 53, 67, 68, 73, 74, 75, 76, 77, 84]
```

### Display Parameters

```python
FIGURE_SIZE = (15, 10)
SPIKE_LINE_WIDTH = 0.5
CHANNEL_SPACING = 1.0  # Vertical spacing between channels
MAX_CHANNELS_TO_DISPLAY = 20  # Maximum channels to display
```

## Output

The script generates a combined behavioral and neural plot showing:

**Top Panel (Behavior - 1/21 of plot height):**
- **Velocity Traces**: X-velocity (blue), Y-velocity (red), and speed (black)
- **Shared Time Axis**: Aligned with neural data below

**Bottom Panel (Neural Raster - 20/21 of plot height):**
- **Vertical Lines**: Each spike appears as a vertical line
- **Color-coded Channels**: Different colors for each channel
- **Channel Labels**: Channel numbers and spike counts on the left
- **Time Grid**: Horizontal grid lines for time reference
- **Summary Statistics**: Average firing rate and threshold information

### Console Output

```
🧠 NEURAL SPIKE RASTER VIEWER
==================================================
Trial: 1
H5 File: D:\Data\ScienceCorp\trials_aligned.h5
Threshold: 5.0x
Channels: 21

📂 Loading trial 1 from D:\Data\ScienceCorp\trials_aligned.h5
✅ Trial 1 loaded successfully:
   • Neural data shape: (128, 150000)
   • Duration: 5.0 seconds
   • Outcome: win

🔍 Detecting spikes on 21 channels...
✅ Spike detection complete: 347 total spikes detected

📈 SPIKE DETECTION SUMMARY:
==================================================
• Total channels analyzed: 21
• Active channels (with spikes): 15
• Total spikes detected: 347
• Average spikes per active channel: 23.1

🏆 TOP 10 CHANNELS:
   1. Channel  0:  45 spikes
   2. Channel  1:  38 spikes
   3. Channel  2:  32 spikes
   ...
```

## Plot Features

### Plot Elements

**Behavior Panel:**
- **Y-axis**: Velocity values 
- **X-axis**: Time in seconds (shared with raster)
- **Traces**: Velocity X (blue), Y (red), Speed (black)
- **Legend**: Color-coded velocity components

**Raster Panel:**
- **Y-axis**: Each channel occupies a horizontal band
- **X-axis**: Time in seconds (shared with behavior)
- **Vertical Lines**: Each spike appears as a colored vertical line
- **Labels**: Channel numbers and spike counts displayed on the left
- **Statistics Box**: Shows average firing rate and threshold setting

### Visual Enhancements

- **Color Coding**: Each channel has a unique color
- **Grid Lines**: Horizontal grid for time reference
- **Transparency**: Semi-transparent spike lines for better visibility
- **Spacing**: Optimal vertical spacing between channels

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- h5py
- Custom modules: `spike_detection.py`, `h5_data_loader.py`

## Examples

### View Trial 1 with Default Settings
```bash
python view_trial_raster.py
```

### View Trial 5 with Lower Threshold
```bash
python view_trial_raster.py --trial 5 --threshold 3.0
```

### Generate High-Resolution Plot
```bash
python view_trial_raster.py --trial 10 --save "high_res_raster.png"
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Update `H5_FILE_PATH` to your actual data file location
2. **No Spikes Detected**: Try lowering the threshold factor (e.g., `--threshold 3.0`)
3. **Too Many Channels**: The script automatically limits to 20 channels for readability

### Error Messages

- `❌ H5 file not found`: Check the file path
- `⚠️ No spikes detected`: Try different threshold or trial
- `❌ Error loading trial data`: Check trial number exists

## Customization

### Adding New Features

The script is modular and can be extended:

- Modify `create_raster_plot()` to change visualization style
- Update `SPIKE_CHANNELS` to analyze different channel sets
- Adjust `COLORS` array to change color scheme
- Modify `MAX_CHANNELS_TO_DISPLAY` for different channel limits

### Integration

The script can be integrated into larger workflows:

```python
from view_trial_raster import load_trial_data, detect_spikes, create_raster_plot

# Use functions in other scripts
trial_data = load_trial_data(h5_file_path, trial_number)
spike_data = detect_spikes(trial_data['neural_data'], channels)
create_raster_plot(spike_data, duration, trial_num, trial_data, save_path)
``` 