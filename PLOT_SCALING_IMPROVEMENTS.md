# Plot Scaling Improvements Guide

## Overview

The behavioral visualization system now automatically scales plots to fit your actual target positions perfectly, eliminating oversized plots and ensuring optimal use of screen space.

## What Changed

### Before 🔴
- Fixed plot limits (-1.5 to 1.5 for both X and Y axes)
- Fixed target circle size (0.08 radius)
- Fixed center circle size (0.1 radius)
- Many plots showed mostly empty space around small target areas

### After ✅
- **Dynamic plot limits** based on actual target positions
- **Adaptive target circle sizes** scaled to target spacing
- **Smart label positioning** that adjusts to target distances
- **Proportional center circles** that scale with targets
- **Automatic padding** for perfect visualization

## Technical Implementation

### 1. Dynamic Plot Limits (`_get_plot_limits()`)

```python
def _get_plot_limits(self, padding_factor=0.3):
    # Analyzes actual target positions
    # Calculates appropriate X/Y ranges
    # Adds configurable padding
    # Ensures minimum visibility range
    return x_min, x_max, y_min, y_max
```

**Features:**
- Finds min/max coordinates from all targets + center
- Adds 30% padding by default (configurable)
- Ensures minimum 0.5 unit range for visibility
- Maintains square aspect ratio

### 2. Adaptive Circle Sizing (`_get_target_circle_size()`)

```python
def _get_target_circle_size():
    # Calculates average target distance from center
    # Scales circle size to ~8% of average distance
    # Applies reasonable bounds (0.02 to 0.15)
    return circle_radius
```

**Features:**
- Circle size scales with your experimental setup
- Prevents tiny circles (min 0.02) or huge circles (max 0.15)
- Automatically adjusts for close vs. distant targets

### 3. Smart Label Positioning

```python
# Scale label distance based on target proximity to center
target_distance = sqrt(x² + y²)
label_scale = 1.2 if target_distance > 0.1 else 2.0
```

**Features:**
- Labels positioned outside target circles
- Closer targets get more label offset
- Prevents label-circle overlap

## Usage

### Automatic (Recommended)
The system works automatically - no setup required!

```python
# Target positions are auto-detected from trajectory data
# Target positions are now hardcoded - no detection needed
target_positions = feature_extractor.target_positions

# Plotter automatically uses corrected positions and optimal scaling
plotter = BehavioralPlotter(all_trial_features, feature_extractor.target_positions)

# All plots now use optimal scaling
fig = plotter.plot_trajectories_by_target()
```

### Manual Adjustments (Optional)
Fine-tune scaling if needed:

```python
# Increase padding around plots
plotter.set_plot_scaling(padding_factor=0.5)

# Make target circles smaller
plotter.set_plot_scaling(target_circle_size=0.05)

# Adjust both
plotter.set_plot_scaling(0.4, 0.08)

# Reset to automatic
plotter.set_plot_scaling()
```

## Benefits

### 🎯 **Perfect Fit**
- No more wasted space around small target areas
- All trajectory detail clearly visible
- Optimal zoom level for your experimental setup

### 📏 **Consistent Scaling**
- All visualizations use the same intelligent scaling
- Target circles appropriately sized across all plots
- Consistent visual appearance

### 🔄 **Adaptive**
- Works with any target layout (close, far, few, many)
- Automatically adjusts to different experimental setups
- No configuration required

### 🎨 **Better Visualization**
- Target circles properly sized and positioned
- Labels don't overlap with targets
- Clear center-out patterns visible

## Plot Types Affected

All trajectory-based plots now use automatic scaling:

1. **`plot_cursor_trajectories_overlaid()`** - Overlaid trajectory plots
2. **`plot_trajectories_by_target()`** - Trajectories grouped by target
3. **`plot_correct_trials_overlay()`** - Successful trials overlay
4. **Diagnostic plots** - Target position comparison plots

## Examples

### Small Target Layout
For targets close to center (e.g., 0.3 units radius):
- Plot limits: approximately -0.5 to +0.5
- Target circles: ~0.024 radius
- Efficient use of plot space

### Large Target Layout  
For targets far from center (e.g., 1.5 units radius):
- Plot limits: approximately -2.0 to +2.0
- Target circles: ~0.12 radius
- Proper scaling for larger movements

### Mixed Distance Targets
For irregular target layouts:
- Automatically finds optimal bounding box
- Scales circles to average target spacing
- Maintains visibility of all elements

## Configuration Options

### Padding Factor
Controls whitespace around targets:
- `0.1` = Tight fit (10% padding)
- `0.3` = Default (30% padding) 
- `0.5` = Loose fit (50% padding)

### Target Circle Size
Controls target circle radius:
- `0.02` = Minimum size
- `auto` = 8% of average target distance (default)
- `0.15` = Maximum size

## Troubleshooting

### Plots Look Too Tight
```python
plotter.set_plot_scaling(padding_factor=0.5)  # More padding
```

### Target Circles Too Large/Small
```python
plotter.set_plot_scaling(target_circle_size=0.05)  # Custom size
```

### Reset to Automatic
```python
plotter.set_plot_scaling()  # Reset all to auto
```

### Check Current Settings
```python
x_min, x_max, y_min, y_max = plotter._get_plot_limits()
circle_size = plotter._get_target_circle_size()
print(f"Range: X({x_min:.3f}, {x_max:.3f}) Y({y_min:.3f}, {y_max:.3f})")
print(f"Circle size: {circle_size:.3f}")
```

## Future Enhancements

Potential improvements:
1. **Trajectory-aware scaling** - Consider trajectory extent, not just targets
2. **Density-based sizing** - Adjust circle size based on trajectory density
3. **Zoom regions** - Automatic detail views for clustered areas
4. **Animation support** - Smooth scaling transitions for interactive plots 