# Advanced PSTH Plotting

This document describes the advanced PSTH (Peri-Stimulus Time Histogram) plotting functionality implemented in the neural exploration codebase, based on Nader Nikbakht's MATLAB `plotSpikePSTH` function.

## Overview

The advanced PSTH plotting system provides sophisticated kernel-based smoothing with multiple kernel types, proper edge correction, and SEM calculation across trials. It's designed to replace simple Gaussian smoothing with more physiologically relevant kernel shapes.

## Key Features

### 🎯 Multiple Kernel Types
- **Gaussian**: Standard Gaussian kernel for general smoothing
- **Half-Gaussian**: Right-half of Gaussian (causal smoothing)
- **Exponential**: Exponential decay kernel
- **EPSP**: Dual-exponential EPSP-like kernel (physiologically realistic)

### 🎯 Advanced Processing
- **Edge Correction**: Proper normalization at signal boundaries
- **SEM Calculation**: Standard error of the mean across trials
- **Flexible Input**: Works with spike matrices or spike time lists
- **Multi-Condition Support**: Compare multiple conditions simultaneously

### 🎯 Integration
- **Seamless Integration**: Works with existing codebase
- **Consistent API**: Same interface as original functions
- **Backward Compatibility**: Existing code continues to work

## Functions

### `plot_spike_psth()`
Main function for plotting spike PSTH with advanced kernel smoothing.

```python
from utils.visualization import plot_spike_psth

# Basic usage
smoothed_psth, sem, time_axis, ax = plot_spike_psth(
    time_axis, spike_matrix, bin_size=0.01, kernel_width=0.025,
    kernel_type='gauss', plot_error=True, color='blue'
)
```

**Parameters:**
- `time_axis`: Time axis for the data
- `spike_matrix`: Spike matrix (trials × time bins)
- `bin_size`: Bin size in seconds
- `kernel_width`: Kernel width in seconds (typically 0.025-0.250s)
- `kernel_type`: 'gauss', 'halfgauss', 'exp', or 'epsp'
- `plot_error`: Whether to plot error bars (SEM)
- `color`: Plot color
- `ax`: Matplotlib axes (optional)

### `plot_multi_condition_psth()`
Plot PSTH for multiple conditions with overlaid curves.

```python
from utils.visualization import plot_multi_condition_psth

# Compare multiple conditions
results = plot_multi_condition_psth(
    spike_data_dict, time_axis, bin_size=0.01, 
    kernel_width=0.025, kernel_type='gauss'
)
```

### `create_psth_from_spike_times()`
Create PSTH directly from spike times.

```python
from utils.visualization import create_psth_from_spike_times

# From spike times
time_axis, spike_matrix, smoothed_psth, sem = create_psth_from_spike_times(
    spike_times_list, time_window=(0, 2.0), bin_size=0.01,
    kernel_width=0.025, kernel_type='gauss'
)
```

## Kernel Types

### Gaussian Kernel
```python
kernel_type='gauss'
```
- **Use case**: General purpose smoothing
- **Properties**: Symmetric, smooth
- **Physiological relevance**: Moderate

### Half-Gaussian Kernel
```python
kernel_type='halfgauss'
```
- **Use case**: Causal smoothing (no future information)
- **Properties**: Asymmetric, right-sided
- **Physiological relevance**: Good for real-time applications

### Exponential Kernel
```python
kernel_type='exp'
```
- **Use case**: Exponential decay patterns
- **Properties**: Asymmetric, exponential decay
- **Physiological relevance**: Models synaptic decay

### EPSP Kernel
```python
kernel_type='epsp'
```
- **Use case**: Physiologically realistic smoothing
- **Properties**: Dual-exponential (fast rise, slow decay)
- **Physiological relevance**: Models post-synaptic potentials
- **Parameters**: 
  - τ_rising = 1ms
  - τ_falling = 40ms

## Updated Files

The following files have been updated to use the new PSTH plotting:

### `utils/visualization.py`
- ✅ Added `plot_spike_psth()` function
- ✅ Added `plot_multi_condition_psth()` function  
- ✅ Added `create_psth_from_spike_times()` function
- ✅ Updated `plot_behavior_raster_psth()` to use advanced smoothing

### `individual_channel_psth_by_target.py`
- ✅ Updated to use `plot_spike_psth()` for smoothing
- ✅ Replaced `gaussian_filter1d()` with advanced kernels
- ✅ Better spike matrix handling

### `aligned_psth.py`
- ✅ Updated `_create_spike_density_function()` to use advanced smoothing
- ✅ Better alignment and smoothing integration

## Usage Examples

### Basic PSTH with Different Kernels

```python
import numpy as np
from utils.visualization import plot_spike_psth

# Generate spike matrix (20 trials × 200 time bins)
spike_matrix = np.random.poisson(0.1, (20, 200))
time_axis = np.linspace(0, 2.0, 200)

# Try different kernels
for kernel_type in ['gauss', 'halfgauss', 'exp', 'epsp']:
    smoothed_psth, sem, _, ax = plot_spike_psth(
        time_axis, spike_matrix, bin_size=0.01, 
        kernel_width=0.025, kernel_type=kernel_type
    )
```

### Multi-Condition Comparison

```python
from utils.visualization import plot_multi_condition_psth

# Compare different conditions
conditions = {
    'Low Rate': spike_matrix_low,
    'High Rate': spike_matrix_high
}

results = plot_multi_condition_psth(
    conditions, time_axis, bin_size=0.01,
    kernel_type='epsp'  # Use physiologically realistic kernel
)
```

### Integration with Existing Code

```python
# Existing code continues to work
from utils.visualization import plot_behavior_raster_psth

# This now uses advanced PSTH smoothing internally
plot_behavior_raster_psth(
    trial_data, spike_channels, 
    psth_sigma=0.025  # Now uses advanced kernel smoothing
)
```

## Testing

Run the test suite to verify functionality:

```bash
python test_psth_kernels.py
```

The test suite includes:
- ✅ Kernel type testing with synthetic data
- ✅ Multi-condition PSTH comparison
- ✅ Real data testing (if available)
- ✅ Error handling and edge cases

## Performance Considerations

### Kernel Width Selection
- **Fast applications**: 0.010-0.025s (10-25ms)
- **General use**: 0.025-0.050s (25-50ms)  
- **Smooth visualization**: 0.050-0.100s (50-100ms)
- **Very smooth**: 0.100-0.250s (100-250ms)

### Kernel Type Selection
- **Speed**: Gaussian > Half-Gaussian > Exponential > EPSP
- **Realism**: EPSP > Exponential > Half-Gaussian > Gaussian
- **Smoothness**: Gaussian > EPSP > Half-Gaussian > Exponential

## Best Practices

### 1. Kernel Selection
```python
# For general analysis
kernel_type='gauss', kernel_width=0.025

# For physiologically realistic analysis
kernel_type='epsp', kernel_width=0.025

# For real-time/causal analysis
kernel_type='halfgauss', kernel_width=0.025
```

### 2. Error Handling
```python
# Always check for edge effects
if len(smoothed_psth) != len(time_axis):
    print("⚠️ Edge effects detected - consider larger time window")
```

### 3. Multi-Trial Analysis
```python
# Use appropriate SEM for error bars
plot_spike_psth(time_axis, spike_matrix, plot_error=True)
```

## Migration Guide

### From Simple Gaussian
```python
# Old way
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(psth, sigma=sigma_bins)

# New way
from utils.visualization import plot_spike_psth
smoothed_psth, sem, _, _ = plot_spike_psth(
    time_axis, spike_matrix, bin_size, kernel_width,
    kernel_type='gauss', plot_error=False
)
```

### From Custom Smoothing
```python
# Old way
kernel = np.exp(-0.5 * (t/sigma)**2)
smoothed = np.convolve(psth, kernel, mode='same')

# New way
smoothed_psth, sem, _, _ = plot_spike_psth(
    time_axis, spike_matrix, bin_size, kernel_width,
    kernel_type='epsp', plot_error=True
)
```

## Troubleshooting

### Common Issues

1. **Shape mismatch errors**
   - Solution: Check that spike_matrix dimensions match time_axis length
   - The function automatically handles minor mismatches

2. **Edge effects**
   - Solution: Use larger time windows or smaller kernels
   - The function includes automatic edge correction

3. **Poor smoothing**
   - Solution: Adjust `kernel_width` parameter
   - Try different `kernel_type` for better fit

### Debug Mode
```python
# Enable debug information
smoothed_psth, sem, time_axis, ax = plot_spike_psth(
    time_axis, spike_matrix, bin_size, kernel_width,
    kernel_type='gauss', plot_error=True
)

print(f"Input shape: {spike_matrix.shape}")
print(f"Output shape: {smoothed_psth.shape}")
print(f"Time axis length: {len(time_axis)}")
```

## References

1. Nikbakht, N. (2014). MATLAB `plotSpikePSTH` function. SISSA - Trieste.
2. Kernel smoothing techniques in neurophysiology
3. Edge correction methods for convolution-based smoothing

## Contributing

To extend the kernel types:

1. Add new kernel type in `plot_spike_psth()` function
2. Implement kernel generation logic
3. Add proper normalization
4. Update tests and documentation
5. Test with real neural data

---

**Note**: This implementation maintains backward compatibility while providing enhanced functionality. Existing code will continue to work, but new code should use the advanced features for better results. 