# Neural-Behavioral Data Integration

A Python toolkit for integrating broadband neural recordings with behavioral data for neuroscience research.

## 🎯 Overview

This project provides tools to:
- Integrate neural recordings (.ns6 files) with behavioral data (CSV)
- Perform precise time alignment between neural and behavioral streams
- Save trial-segmented data to HDF5 format for analysis
- Verify data quality and visualize neural-behavioral correlations

## 📁 File Structure

### Core Files

```
neural_exploration/
├── precise_time_alignment.py          # Main integration script (RECOMMENDED)
├── neural_feature_extraction.py       # Advanced neural feature extraction
├── data_quality_verification.ipynb    # Data verification and visualization
├── neural_feature_exploration.ipynb   # Interactive feature exploration
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── ALIGNMENT_GUIDE.md                 # Detailed alignment methodology
├── NEURAL_FEATURE_EXTRACTION_GUIDE.md # Feature extraction guide
├── utils/                             # Utility modules
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading utilities
│   └── visualization.py               # Visualization utilities
└── development_archive/               # Experimental and development files
    ├── neural_behavioral_integration.py
    ├── neural_behavioral_integration_fixed.py
    ├── debug_integration.py
    ├── test_*.py                      # Various test scripts
    ├── *.ipynb                        # Exploratory notebooks
    └── test_neural.h5                 # Test data files
```

### Primary Scripts

1. **`precise_time_alignment.py`** - **MAIN INTEGRATION SCRIPT**
   - Robust time alignment between neural and behavioral data
   - Handles timezone conversions and timestamp parsing
   - Implements fallback methods for neural data extraction
   - Produces high-quality HDF5 output files

2. **`data_quality_verification.ipynb`** - **DATA VERIFICATION**
   - Comprehensive quality assessment of integrated data
   - Visualizes neural-behavioral correlations
   - Shows synchronized neural recordings with behavioral velocity
   - Provides quality metrics and troubleshooting guidance

3. **`neural_feature_extraction.py`** - **ADVANCED FEATURE EXTRACTION**
   - Extracts 4 types of neural features from spike channels
   - Signal features (spike band power 400-6000 Hz)
   - Local field potential (LFP) analysis with gamma bands
   - Voltage features (moving average, variance)
   - Thresholded spike counts (semi-spike detection)

4. **`neural_feature_exploration.ipynb`** - **INTERACTIVE FEATURE EXPLORATION**
   - Minimal interface for exploring extracted neural features
   - Visualization of all feature types across channels
   - Single-channel detailed analysis
   - Channel comparison and ranking tools
   - Save/load functionality for extracted features

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
- **Neural data**: `.ns6` file (Blackrock format)
- **Behavioral data**: CSV with columns: `timestamp`, `velocity_x`, `velocity_y`, `trial_start`, `trial_win`, `trial_lose`, `target_index`

### 3. Run Integration
```python
from precise_time_alignment import NeuralBehavioralIntegrator

# Initialize integrator
integrator = NeuralBehavioralIntegrator(
    neural_file="path/to/neural.ns6",
    behavioral_file="path/to/behavioral.csv",
    output_file="path/to/output.h5"
)

# Process data
integrator.process_data()
```

### 4. Verify Data Quality
Open and run `data_quality_verification.ipynb` to:
- Examine data structure and completeness
- Visualize neural-behavioral alignment
- Check correlation between neural activity and movement
- Generate quality assessment report

## 📊 Output Format

The integration produces HDF5 files with this structure:
```
output.h5
├── trial_1/
│   ├── neural              # Neural data (96 channels × time samples)
│   ├── velocity_x          # X velocity values
│   ├── velocity_y          # Y velocity values
│   ├── behavioral_timestamps # Behavioral timestamps
│   └── [attributes]        # Trial metadata (outcome, duration, etc.)
├── trial_2/
│   └── ...
└── [global_attributes]     # Session metadata
```

## 🔧 Technical Details

### Time Alignment
- **Neural timestamps**: Extracted from Blackrock file headers
- **Behavioral timestamps**: UNIX format in CSV
- **Alignment method**: Precise timezone-aware datetime matching
- **Fallback strategies**: Multiple methods for robust time origin estimation

### Data Quality Features
- **Neural data validation**: Checks for valid signal ranges and sampling rates
- **Behavioral data validation**: Ensures continuous timestamps and reasonable velocities
- **Trial segmentation**: Based on behavioral markers (trial_start, trial_win/lose)
- **Quality metrics**: Data completeness, temporal alignment, signal variability

### Performance Optimizations
- **Memory efficient**: Processes large files without loading entire datasets
- **Compression**: HDF5 with gzip compression for optimal storage
- **Sampling rate conversion**: Downsamples neural data from 30kHz to 1kHz
- **Robust error handling**: Graceful handling of missing or corrupted data

## 🧪 Development Archive

The `development_archive/` folder contains:
- **Experimental scripts**: Early versions and debugging tools
- **Test notebooks**: Exploratory data analysis
- **Debug utilities**: Tools used during development
- **Sample data**: Small test datasets

These files are preserved for reference but not needed for normal usage.

## 📚 Usage Examples

See `ALIGNMENT_GUIDE.md` for detailed methodology and troubleshooting.

## 🐛 Troubleshooting

### Common Issues
1. **Time alignment errors**: Check timezone settings and timestamp formats
2. **Missing neural data**: Verify file paths and neo library compatibility
3. **Memory errors**: Ensure sufficient RAM for large neural files
4. **Quality verification failures**: Run the verification notebook for diagnostics

### Support
- Check the `data_quality_verification.ipynb` for diagnostic information
- Review `ALIGNMENT_GUIDE.md` for detailed troubleshooting steps
- Examine files in `development_archive/` for debugging approaches

## 🔬 Research Applications

This toolkit is designed for:
- **Neural decoding experiments**: Predicting movement from neural activity
- **Brain-computer interfaces**: Real-time neural control applications
- **Neuroscience research**: Understanding neural-behavioral relationships
- **Data preprocessing**: Preparing neural data for machine learning

## 📈 Success Metrics

A successful integration typically produces:
- **File size**: ~6MB per trial (96 channels, ~10 seconds)
- **Data completeness**: >95% neural data coverage
- **Time alignment**: <1 second offset between neural and behavioral streams
- **Quality score**: >80/100 in verification notebook

---

**Note**: This is a research tool. Ensure proper validation of results for your specific experimental conditions.
