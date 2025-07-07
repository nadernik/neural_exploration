# Hardcoded Target Positions

The target positions are now hardcoded throughout the codebase with the following specific angles:

## Target Configuration
- **T0**: 90° (top/north)
- **T1**: 45° (northeast)
- **T2**: 0° (right/east)
- **T3**: -45° (southeast)
- **T4**: -90° (bottom/south)
- **T5**: -135° (southwest)
- **T6**: 180° (left/west)
- **T7**: 135° (northwest)

## Implementation
The hardcoded angles are implemented in both:
- `utils/behavioral_features.py` - `BehavioralFeatureExtractor._generate_target_positions()`
- `utils/behavioral_visualization.py` - `BehavioralPlotter._generate_target_positions()`

## Changes Made
1. **Removed automatic detection**: The `analyze_actual_target_positions()` and `auto_detect_and_fix_target_positions()` methods have been completely removed from the codebase.

2. **Updated target generation**: Both feature extractor and plotter now use the hardcoded angles instead of generating evenly spaced targets.

3. **Notebook updates**: The main analysis notebook should be updated to:
   - Remove the auto-detection step
   - Replace all `corrected_positions` references with `feature_extractor.target_positions`
   - Update any diagnostic code that depends on the old detection algorithm

## Benefits
- **Consistent**: All analyses use the same target layout
- **Reliable**: No dependency on trajectory data quality for target detection
- **Faster**: No need to analyze trajectory endpoints
- **Accurate**: Matches the exact experimental setup

## Code Examples

### Feature Extraction
```python
# Old approach (removed)
# corrected_positions = feature_extractor.auto_detect_and_fix_target_positions(all_trial_features)

# New approach
# Target positions are automatically set to hardcoded values
target_positions = feature_extractor.target_positions
```

### Visualization
```python
# Create plotter with hardcoded positions
plotter = BehavioralPlotter(all_trial_features, feature_extractor.target_positions)

# No need to update or fix positions - they're already correct
```

### Accessing Target Info
```python
# Access hardcoded target positions
for target_idx, pos in feature_extractor.target_positions.items():
    print(f"Target {target_idx}: ({pos['x']:.3f}, {pos['y']:.3f}) "
          f"at {pos['direction']}")
```

## Notebook Cell Updates Required

The following notebook cells need to be updated to remove `corrected_positions` references:

1. **Target Detection Cell** (Cell 5):
   - Replace auto-detection code with hardcoded position display
   - Remove `corrected_positions` variable

2. **Position Analysis Cell** (Cell 8):
   - Replace `corrected_positions` with `feature_extractor.target_positions`
   - Update all coordinate and distance calculations

3. **Any other cells** that reference `corrected_positions`:
   - Search for all instances and replace with `feature_extractor.target_positions`

## Manual Updates Needed

If you need to manually update the notebook, replace these patterns:

```python
# Replace these patterns:
corrected_positions.values() → feature_extractor.target_positions.values()
corrected_positions.items() → feature_extractor.target_positions.items()
corrected_positions.keys() → feature_extractor.target_positions.keys()
corrected_positions[target_idx] → feature_extractor.target_positions[target_idx]
```

This ensures all code uses the consistent hardcoded target positions throughout the analysis pipeline. 