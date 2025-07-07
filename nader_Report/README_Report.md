# Neural Data Exploration - Final Report Package

**Author:** Nader Nikbakht (nikbakht@mit.edu)  
**Date:** January 2025  
**Project:** Neural Data Exploration Take-Home Challenge  

---

## 📋 Package Contents

This folder contains the complete analysis pipeline and findings for the neural data exploration project. The package includes:

### 📊 Main Findings Report
- **`Neural_Data_Exploration_Findings_Report.md`** - Comprehensive findings report with all results, methodology, and conclusions
- **`Neural_Data_Exploration_Report.html`** - Interactive HTML report with embedded visualizations and analysis figures

### 🔧 Core Processing Scripts
- **`precise_time_alignment.py`** - Main neural-behavioral data integration script (RECOMMENDED)
- **`neural_feature_extraction.py`** - Advanced neural feature extraction with multiple feature types

### 📓 Analysis Notebooks
- **`behavioral_feature_exploration.ipynb`** - Comprehensive behavioral analysis and task performance
- **`neural_feature_exploration.ipynb`** - Interactive neural feature exploration
- **`neural_feature_exploration_modular.ipynb`** - Modular neural analysis with spike detection
- **`decode_velocity_regression.ipynb`** - Ridge regression velocity decoding
- **`decoding_exploration.ipynb`** - Alternative decoding methods exploration
- **`spike_detection_comparison_pywaveclus.ipynb`** - PyWaveClus-style spike detection validation
- **`neural_decoding.ipynb`** - Neural signal analysis and visualization
- **`target_psth_analysis.ipynb`** - Target-specific neural response analysis
- **`data_quality_verification.ipynb`** - Data quality assessment and verification
- **`test_exploration.ipynb`** - Testing and validation notebook

### 📚 Documentation
- **`README.md`** - Project overview and usage guide
- **`ALIGNMENT_GUIDE.md`** - Detailed time alignment methodology
- **`NEURAL_FEATURE_EXTRACTION_GUIDE.md`** - Feature extraction guide
- **`PLOT_SCALING_IMPROVEMENTS.md`** - Visualization improvements documentation
- **`HARDCODED_TARGET_POSITIONS.md`** - Target position analysis

### 📁 Utility Modules (`utils/`)
- **`h5_data_loader.py`** - HDF5 data loading utilities
- **`spike_detection.py`** - PyWaveClus-style spike detection
- **`neural_behavioral_alignment.py`** - Neural-behavioral time alignment
- **`ridge_decoder.py`** - Ridge regression decoder implementation
- **`behavioral_features.py`** - Behavioral feature extraction
- **`visualization.py`** - Comprehensive plotting utilities
- **`analysis.py`** - Analysis helper functions
- **`diagnostics.py`** - Data quality diagnostics
- **`data_loader.py`** - General data loading utilities
- **`behavioral_visualization.py`** - Behavioral plotting utilities

### 📊 Generated Data Files
- **`behavioral_performance_summary.csv`** - Task performance metrics by target
- **`detailed_trial_features.csv`** - Comprehensive trial-by-trial behavioral features

### 🖼️ Figures
- **`neural_features_trial_1.png`** - Example neural feature extraction visualization

### ⚙️ Configuration
- **`requirements.txt`** - Python dependencies

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Key Dependencies
- Python 3.8+
- numpy, pandas, scipy, matplotlib, seaborn
- h5py, neo (for Blackrock file support)
- scikit-learn (for decoding)
- jupyter (for notebooks)

### 3. Data Processing Pipeline

**Step 1: Neural-Behavioral Integration**
```python
# Run the main integration script
python precise_time_alignment.py

# This creates: trials_aligned.h5
```

**Step 2: Data Quality Verification**
```python
# Open and run data_quality_verification.ipynb
# This validates the integration and shows quality metrics
```

**Step 3: Behavioral Analysis**
```python
# Open and run behavioral_feature_exploration.ipynb
# This analyzes task performance and generates summary statistics
```

**Step 4: Neural Feature Extraction**
```python
# Option A: Use the extraction script
python neural_feature_extraction.py

# Option B: Use the interactive notebook
# Open neural_feature_exploration.ipynb
```

**Step 5: Decoding Analysis**
```python
# For Ridge regression decoding:
# Open decode_velocity_regression.ipynb

# For comprehensive decoding exploration:
# Open decoding_exploration.ipynb
```

---

## 📈 Key Results Summary

### ✅ Successful Components
- **Data Integration:** 140 trials successfully integrated with precise time alignment
- **Behavioral Analysis:** 90.7% task success rate, clear movement patterns extracted
- **Neural Activity:** 21 channels with reliable spike detection, movement-related modulation
- **Feature Extraction:** Multiple neural feature types extracted (spike band power, LFP, etc.)

### ⚠️ Challenges Identified
- **Decoding Performance:** Ridge regression achieved R² ≈ -0.008 (poor linear decoding)
- **Data Sparsity:** Limited movement-related neural samples
- **Feature Engineering:** Simple firing rates may be insufficient for decoding

### 💡 Recommendations
- **Non-linear Methods:** Try Random Forest, Neural Networks, or Deep Learning
- **Enhanced Features:** Spectral features, cross-channel coupling, temporal dynamics
- **Alternative Targets:** Decode movement intent, target direction, or movement phases

---

## 📋 File Usage Guide

### For Data Processing:
1. **Start with:** `precise_time_alignment.py`
2. **Validate with:** `data_quality_verification.ipynb`
3. **Extract features with:** `neural_feature_extraction.py`

### For Analysis:
1. **Behavioral:** `behavioral_feature_exploration.ipynb`
2. **Neural:** `neural_feature_exploration.ipynb` or `neural_feature_exploration_modular.ipynb`
3. **Decoding:** `decode_velocity_regression.ipynb` or `decoding_exploration.ipynb`

### For Development:
- **Utils modules:** Reusable components for your own analysis
- **Documentation:** Detailed methodology and troubleshooting guides

---

## 🔬 Research Applications

This codebase can be adapted for:
- **Brain-Computer Interface (BCI) development**
- **Neural decoding research**
- **Motor cortex analysis**
- **Neural signal processing**
- **Behavioral neuroscience**

---

## 📧 Contact

**Nader Nikbakht**  
Email: nikbakht@mit.edu  

For questions about the methodology, implementation, or results, please refer to the comprehensive findings report or contact the author.

---

## 📜 License

This code is provided for research and educational purposes. Please cite appropriately if used in publications.

---

**Note:** This package represents a complete neural data exploration pipeline. The main findings report contains detailed analysis and interpretation of all results. 