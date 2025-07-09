#!/usr/bin/env python3
"""
Example Ridge Decoding using Existing Infrastructure
===================================================

This script demonstrates how to use the existing neural decoding infrastructure
in the codebase for ridge regression decoding of cursor velocity.

This example assumes you have H5 files with pre-processed neural and behavioral data,
as used in the existing codebase.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import existing utilities
from utils.spike_detection import SpikeDetector
from utils.neural_behavioral_alignment import NeuralBehavioralAligner
from utils.ridge_decoder import RidgeVelocityDecoder
from utils.h5_data_loader import H5DataLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
H5_FILE_PATH = r"D:\Data\ScienceCorp\trials_aligned.h5"  # Update this path

# Analysis parameters - MODIFIED TO USE ALL 96 CHANNELS
GOOD_CHANNELS = list(range(96))  # All 96 channels (0-95)
TRIAL_NUMBERS = list(range(1, 21))  # Use first 20 trials (same as working notebook)
BIN_SIZE = 0.1  # 100ms bins
THRESHOLD_FACTOR = 5.0
SPIKE_WINDOW = (-10, 32)

# Ridge parameters - EXACT SAME AS WORKING NOTEBOOK
ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TEST_SIZE = 0.2
CV_FOLDS = 5
INITIAL_ALPHA = 1.0  # Initial alpha for first decoder

print("🧠 Example Ridge Decoding with Existing Infrastructure")
print("=" * 60)

print(f"📋 Configuration:")
print(f"  • Trials: {len(TRIAL_NUMBERS)} (trials {TRIAL_NUMBERS[0]}-{TRIAL_NUMBERS[-1]})")
print(f"  • Bin size: {BIN_SIZE*1000:.0f}ms")
print(f"  • Channels: {len(GOOD_CHANNELS)} (ALL 96 CHANNELS)")
print(f"  • Alpha range: {ALPHA_RANGE[0]}-{ALPHA_RANGE[-1]}")
print(f"  • CV folds: {CV_FOLDS}")
print(f"  ⚠️  Note: Using all 96 channels will increase processing time and memory usage")

# =============================================================================
# STEP 1: Initialize Components
# =============================================================================

print("\n📊 Step 1: Initialize components")

# Initialize spike detector
spike_detector = SpikeDetector(
    sampling_rate=30000,
    threshold_factor=THRESHOLD_FACTOR,
    spike_window=SPIKE_WINDOW,
    good_channels=GOOD_CHANNELS
)

# Initialize neural-behavioral aligner
aligner = NeuralBehavioralAligner(
    bin_size=BIN_SIZE,
    interpolation_method='linear'
)

# Initialize H5 data loader
try:
    h5_loader = H5DataLoader(H5_FILE_PATH)
    print(f"✅ H5 data loader initialized with {H5_FILE_PATH}")
except Exception as e:
    print(f"❌ Error loading H5 file: {e}")
    print("Please check the file path and ensure the H5 file exists")
    exit(1)

# =============================================================================
# STEP 2: Process Multiple Trials
# =============================================================================

print(f"\n🔄 Step 2: Process {len(TRIAL_NUMBERS)} trials")

try:
    # Process multiple trials to create training dataset
    print("This may take several minutes - extracting spikes and aligning data...")
    
    neural_features_all, behavioral_targets_all = aligner.process_multiple_trials(
        H5_FILE_PATH, TRIAL_NUMBERS, spike_detector
    )
    
    print(f"✅ Dataset created:")
    print(f"  • Neural features: {neural_features_all.shape}")
    print(f"  • Behavioral targets: {behavioral_targets_all.shape}")
    print(f"  • Total time bins: {neural_features_all.shape[0]}")
    print(f"  • Total duration: {neural_features_all.shape[0] * BIN_SIZE:.1f} seconds")
    
    # Data quality check
    behavioral_magnitude = np.sqrt(behavioral_targets_all[:, 0]**2 + behavioral_targets_all[:, 1]**2)
    moving_samples = np.sum(behavioral_magnitude > 0.01)
    print(f"  • Moving samples (speed > 0.01): {moving_samples}/{len(behavioral_targets_all)} ({moving_samples/len(behavioral_targets_all)*100:.1f}%)")
    
except Exception as e:
    print(f"❌ Error processing trials: {e}")
    exit(1)

# =============================================================================
# STEP 3: Initial Training and Hyperparameter Optimization
# =============================================================================

print(f"\n🏋️ Step 3: Initial training and hyperparameter optimization")

# First, train with initial alpha to get baseline
print(f"Training initial model with alpha = {INITIAL_ALPHA}...")
initial_decoder = RidgeVelocityDecoder(
    alpha=INITIAL_ALPHA,
    normalize_features=True,
    normalize_targets=False
)

training_results = initial_decoder.train(
    neural_features_all, 
    behavioral_targets_all,
    test_size=TEST_SIZE,
    random_state=42
)

print(f"✅ Initial training complete:")
print(f"  • R² velocity_x: {training_results['r2_x']:.3f}")
print(f"  • R² velocity_y: {training_results['r2_y']:.3f}")
print(f"  • Overall correlation: {training_results['overall_correlation']:.3f}")

# Now search for optimal hyperparameters
print(f"\n🔍 Searching for optimal hyperparameters...")
hp_search_results = initial_decoder.hyperparameter_search(
    neural_features_all,
    behavioral_targets_all,
    alpha_range=ALPHA_RANGE,
    cv_folds=CV_FOLDS
)

best_alpha = hp_search_results['best_alpha']
print(f"✅ Best alpha found: {best_alpha} (R² = {hp_search_results['best_result']['overall_mean_r2']:.3f})")

# =============================================================================
# STEP 4: Train Final Model with Best Alpha
# =============================================================================

print(f"\n🏋️ Step 4: Train final model with best alpha")

# Create final decoder with best alpha
final_decoder = RidgeVelocityDecoder(
    alpha=best_alpha,
    normalize_features=True,
    normalize_targets=False
)

# Train the final model
final_training_results = final_decoder.train(
    neural_features_all, 
    behavioral_targets_all,
    test_size=TEST_SIZE,
    random_state=42
)

print(f"✅ Final model trained:")
print(f"  • R² velocity_x: {final_training_results['r2_x']:.3f}")
print(f"  • R² velocity_y: {final_training_results['r2_y']:.3f}")
print(f"  • Correlation velocity_x: {final_training_results['correlation_x']:.3f}")
print(f"  • Correlation velocity_y: {final_training_results['correlation_y']:.3f}")
print(f"  • Speed correlation: {final_training_results['overall_correlation']:.3f}")

# =============================================================================
# STEP 4.5: Channel Selection Based on Feature Importance
# =============================================================================

print(f"\n🔍 Step 4.5: Select top channels based on feature importance")

# Get regression coefficients for each velocity dimension
coef_x = np.abs(final_decoder.model_x.coef_)  # Absolute values for importance
coef_y = np.abs(final_decoder.model_y.coef_)

# Find top 20 channels for each dimension
top_channels_x = np.argsort(coef_x)[-20:][::-1]  # Top 20 for velocity_x
top_channels_y = np.argsort(coef_y)[-20:][::-1]  # Top 20 for velocity_y

# Combine and remove duplicates while preserving order
combined_channels = []
seen = set()

# Add channels from velocity_x first (in order of importance)
for ch in top_channels_x:
    if ch not in seen:
        combined_channels.append(ch)
        seen.add(ch)

# Add channels from velocity_y that weren't already included
for ch in top_channels_y:
    if ch not in seen:
        combined_channels.append(ch)
        seen.add(ch)

# Sort the final channel list for consistency
selected_channels = sorted(combined_channels)

print(f"📊 Channel Selection Results:")
print(f"  • Top 20 channels for velocity_x: {sorted(top_channels_x.tolist())}")
print(f"  • Top 20 channels for velocity_y: {sorted(top_channels_y.tolist())}")
print(f"  • Combined unique channels: {len(selected_channels)} channels")
print(f"  • Selected channels: {selected_channels}")

# Show feature importance for top channels
print(f"\n📈 Feature Importance (Top 10 channels):")
print(f"{'Channel':<8} {'Vel_X Coef':<12} {'Vel_Y Coef':<12} {'Combined':<12}")
print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")

for i, ch in enumerate(selected_channels[:10]):  # Show top 10
    coef_x_val = coef_x[ch]
    coef_y_val = coef_y[ch]
    combined_importance = coef_x_val + coef_y_val
    print(f"{ch:<8} {coef_x_val:<12.4f} {coef_y_val:<12.4f} {combined_importance:<12.4f}")

# =============================================================================
# STEP 5: Re-train with Selected Channels
# =============================================================================

print(f"\n🔄 Step 5: Re-train decoder with {len(selected_channels)} selected channels")

# Extract features for selected channels only
print("Extracting features for selected channels...")
selected_neural_features = neural_features_all[:, selected_channels]

print(f"📊 Selected feature matrix:")
print(f"  • Original shape: {neural_features_all.shape}")
print(f"  • Selected shape: {selected_neural_features.shape}")
print(f"  • Reduction: {neural_features_all.shape[1]} → {selected_neural_features.shape[1]} channels")

# Train new decoder with selected channels
print(f"Training optimized decoder...")
optimized_decoder = RidgeVelocityDecoder(
    alpha=best_alpha,
    normalize_features=True,
    normalize_targets=False
)

optimized_training_results = optimized_decoder.train(
    selected_neural_features, 
    behavioral_targets_all,
    test_size=TEST_SIZE,
    random_state=42
)

print(f"✅ Optimized model trained:")
print(f"  • R² velocity_x: {optimized_training_results['r2_x']:.3f}")
print(f"  • R² velocity_y: {optimized_training_results['r2_y']:.3f}")
print(f"  • Correlation velocity_x: {optimized_training_results['correlation_x']:.3f}")
print(f"  • Correlation velocity_y: {optimized_training_results['correlation_y']:.3f}")
print(f"  • Speed correlation: {optimized_training_results['overall_correlation']:.3f}")

# =============================================================================
# STEP 6: Performance Comparison
# =============================================================================

print(f"\n📈 Step 6: Performance comparison")

print(f"📊 Performance Comparison:")
selected_ch_header = f"Top {len(selected_channels)} Ch"
print(f"{'Metric':<20} {'All 96 Ch':<12} {selected_ch_header:<12} {'Improvement':<12}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")

# R² comparison
r2_x_improvement = optimized_training_results['r2_x'] - final_training_results['r2_x']
r2_y_improvement = optimized_training_results['r2_y'] - final_training_results['r2_y']
corr_x_improvement = optimized_training_results['correlation_x'] - final_training_results['correlation_x']
corr_y_improvement = optimized_training_results['correlation_y'] - final_training_results['correlation_y']
overall_improvement = optimized_training_results['overall_correlation'] - final_training_results['overall_correlation']

print(f"{'R² Velocity X':<20} {final_training_results['r2_x']:<12.3f} {optimized_training_results['r2_x']:<12.3f} {r2_x_improvement:+.3f}")
print(f"{'R² Velocity Y':<20} {final_training_results['r2_y']:<12.3f} {optimized_training_results['r2_y']:<12.3f} {r2_y_improvement:+.3f}")
print(f"{'Corr Velocity X':<20} {final_training_results['correlation_x']:<12.3f} {optimized_training_results['correlation_x']:<12.3f} {corr_x_improvement:+.3f}")
print(f"{'Corr Velocity Y':<20} {final_training_results['correlation_y']:<12.3f} {optimized_training_results['correlation_y']:<12.3f} {corr_y_improvement:+.3f}")
print(f"{'Overall Corr':<20} {final_training_results['overall_correlation']:<12.3f} {optimized_training_results['overall_correlation']:<12.3f} {overall_improvement:+.3f}")

# Cross-validation with selected channels
print(f"\n🔄 Cross-validation with selected channels...")
optimized_cv_results = optimized_decoder.cross_validate(
    selected_neural_features, 
    behavioral_targets_all, 
    cv_folds=CV_FOLDS
)

print(f"✅ Optimized cross-validation complete:")
print(f"  • CV R² velocity_x: {optimized_cv_results['mean_r2_x']:.3f} ± {optimized_cv_results['std_r2_x']:.3f}")
print(f"  • CV R² velocity_y: {optimized_cv_results['mean_r2_y']:.3f} ± {optimized_cv_results['std_r2_y']:.3f}")

# =============================================================================
# STEP 7: Cross-Validation Evaluation (Original)
# =============================================================================

print(f"\n📈 Step 7: Cross-validation evaluation (all channels)")

# Perform cross-validation on the final model
cv_results = final_decoder.cross_validate(
    neural_features_all, 
    behavioral_targets_all, 
    cv_folds=CV_FOLDS
)

print(f"✅ Cross-validation complete:")
print(f"  • CV R² velocity_x: {cv_results['mean_r2_x']:.3f} ± {cv_results['std_r2_x']:.3f}")
print(f"  • CV R² velocity_y: {cv_results['mean_r2_y']:.3f} ± {cv_results['std_r2_y']:.3f}")

# =============================================================================
# STEP 8: Visualization
# =============================================================================

print(f"\n📊 Step 8: Generate visualizations")

# Use the optimized model results for visualization
X_test, y_test, y_pred = optimized_training_results['test_data']

# Debug: Check data shapes and ranges
print(f"📋 Debug - Data shapes:")
print(f"  • X_test: {X_test.shape}")
print(f"  • y_test: {y_test.shape}")
print(f"  • y_pred: {y_pred.shape}")
print(f"  • y_test range: [{y_test.min():.3f}, {y_test.max():.3f}]")
print(f"  • y_pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

# Additional debugging for single point issue
print(f"📋 Debug - Data values:")
print(f"  • y_test unique values: {np.unique(y_test).shape[0]} unique values")
print(f"  • y_pred unique values: {np.unique(y_pred).shape[0]} unique values")
print(f"  • Non-zero y_test samples: {np.sum(np.any(y_test != 0, axis=1))}")
print(f"  • Non-zero y_pred samples: {np.sum(np.any(y_pred != 0, axis=1))}")

# Check if we're dealing with just one test sample
if y_test.shape[0] <= 5:
    print(f"⚠️  WARNING: Only {y_test.shape[0]} test samples! This will result in poor plots.")
    print("  Using all data for visualization instead...")
    
    # Fallback: Use all data for plotting
    print("  • Generating predictions for all data...")
    y_pred_all = optimized_decoder.predict(selected_neural_features)
    X_test, y_test, y_pred = selected_neural_features, behavioral_targets_all, y_pred_all
    
    print(f"  • Updated to use all {y_test.shape[0]} samples for plotting")
    
# Check if predictions are all zeros
if np.all(y_pred == 0):
    print("⚠️  WARNING: All predictions are zero! Model may not be trained properly.")
    
# Check if true values are all zeros
if np.all(y_test == 0):
    print("⚠️  WARNING: All true values are zero! Data may be incorrectly processed.")

# Show first few samples (after any fallback adjustments)
print(f"📋 Debug - First 5 samples (updated data):")
n_show = min(5, y_test.shape[0])
for i in range(n_show):
    print(f"  Sample {i}: true=[{y_test[i,0]:.3f}, {y_test[i,1]:.3f}], pred=[{y_pred[i,0]:.3f}, {y_pred[i,1]:.3f}]")

# Update debug info with final data
print(f"📋 Debug - Final data for plotting:")
print(f"  • Final y_test shape: {y_test.shape}")
print(f"  • Final y_pred shape: {y_pred.shape}")
print(f"  • Final y_test range: [{y_test.min():.3f}, {y_test.max():.3f}]")
print(f"  • Final y_pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

# Create visualizations
print("Creating matplotlib figure...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
print("Figure created successfully")

# Plot 1: Velocity X
axes[0, 0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6, s=20)
axes[0, 0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('True Velocity X')
axes[0, 0].set_ylabel('Predicted Velocity X')
axes[0, 0].set_title(f'Velocity X (R² = {optimized_training_results["r2_x"]:.3f}) - Top {len(selected_channels)} Ch')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Velocity Y
axes[0, 1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6, s=20)
axes[0, 1].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('True Velocity Y')
axes[0, 1].set_ylabel('Predicted Velocity Y')
axes[0, 1].set_title(f'Velocity Y (R² = {optimized_training_results["r2_y"]:.3f}) - Top {len(selected_channels)} Ch')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Speed comparison
true_speed = np.sqrt(y_test[:, 0]**2 + y_test[:, 1]**2)
pred_speed = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)

axes[1, 0].scatter(true_speed, pred_speed, alpha=0.6, s=20)
axes[1, 0].plot([true_speed.min(), true_speed.max()], 
                [true_speed.min(), true_speed.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True Speed')
axes[1, 0].set_ylabel('Predicted Speed')
axes[1, 0].set_title(f'Speed (r = {optimized_training_results["overall_correlation"]:.3f}) - Top {len(selected_channels)} Ch')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Hyperparameter results
hp_alphas = [r['alpha'] for r in hp_search_results['results']]
hp_scores = [r['overall_mean_r2'] for r in hp_search_results['results']]

axes[1, 1].semilogx(hp_alphas, hp_scores, 'b-o', linewidth=2, markersize=8)
axes[1, 1].axvline(best_alpha, color='red', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('Alpha (Regularization)')
axes[1, 1].set_ylabel('Cross-validation R²')
axes[1, 1].set_title('Hyperparameter Optimization')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
print("Saving plot...")
plt.savefig('example_ridge_decoding_optimized_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to example_ridge_decoding_optimized_results.png")
print("Displaying plot...")
plt.show(block=True)  # Use block=True to ensure plot displays
print("Plot displayed")

# =============================================================================
# STEP 9: Summary
# =============================================================================

print(f"\n" + "="*60)
print("DECODING SUMMARY")
print("="*60)

print(f"📊 Dataset:")
print(f"  • Trials processed: {len(TRIAL_NUMBERS)}")
print(f"  • Neural features: {neural_features_all.shape}")
print(f"  • Behavioral targets: {behavioral_targets_all.shape}")
print(f"  • Time bins: {neural_features_all.shape[0]}")
print(f"  • Total duration: {neural_features_all.shape[0] * BIN_SIZE:.1f} seconds")

print(f"\n🧠 Channel Selection:")
print(f"  • Started with: 96 channels")
print(f"  • Top 20 for velocity_x: {len(top_channels_x)} channels")
print(f"  • Top 20 for velocity_y: {len(top_channels_y)} channels")
print(f"  • Combined selected: {len(selected_channels)} channels")
print(f"  • Reduction: {96 - len(selected_channels)} channels removed")

print(f"\n🎯 Performance (Optimized Model):")
print(f"  • Best regularization: α = {best_alpha}")
print(f"  • R² velocity_x: {optimized_training_results['r2_x']:.3f}")
print(f"  • R² velocity_y: {optimized_training_results['r2_y']:.3f}")
print(f"  • Correlation velocity_x: {optimized_training_results['correlation_x']:.3f}")
print(f"  • Correlation velocity_y: {optimized_training_results['correlation_y']:.3f}")
print(f"  • Speed correlation: {optimized_training_results['overall_correlation']:.3f}")
print(f"  • CV R² velocity_x: {optimized_cv_results['mean_r2_x']:.3f} ± {optimized_cv_results['std_r2_x']:.3f}")
print(f"  • CV R² velocity_y: {optimized_cv_results['mean_r2_y']:.3f} ± {optimized_cv_results['std_r2_y']:.3f}")

print(f"\n📈 Comparison (96 Ch vs Top {len(selected_channels)} Ch):")
print(f"  • R² velocity_x: {final_training_results['r2_x']:.3f} → {optimized_training_results['r2_x']:.3f} ({r2_x_improvement:+.3f})")
print(f"  • R² velocity_y: {final_training_results['r2_y']:.3f} → {optimized_training_results['r2_y']:.3f} ({r2_y_improvement:+.3f})")
print(f"  • Overall correlation: {final_training_results['overall_correlation']:.3f} → {optimized_training_results['overall_correlation']:.3f} ({overall_improvement:+.3f})")

print(f"\n📈 Dataset Statistics:")
print(f"  • Neural activity range: {np.min(neural_features_all):.2f} to {np.max(neural_features_all):.2f} Hz")
print(f"  • Velocity X range: {np.min(behavioral_targets_all[:, 0]):.3f} to {np.max(behavioral_targets_all[:, 0]):.3f}")
print(f"  • Velocity Y range: {np.min(behavioral_targets_all[:, 1]):.3f} to {np.max(behavioral_targets_all[:, 1]):.3f}")

# Calculate movement statistics
speed_all = np.sqrt(behavioral_targets_all[:, 0]**2 + behavioral_targets_all[:, 1]**2)
moving_samples = np.sum(speed_all > 0.01)
print(f"  • Moving samples: {moving_samples}/{len(speed_all)} ({moving_samples/len(speed_all)*100:.1f}%)")

print(f"\n🎉 Ridge decoding analysis complete!")
print(f"📊 Results saved to: example_ridge_decoding_optimized_results.png")

# =============================================================================
# OPTIONAL: Feature Importance Analysis
# =============================================================================

print(f"\n🔍 Optional: Feature importance analysis")

# Get model coefficients from optimized decoder
coef_x_optimized = optimized_decoder.model_x.coef_
coef_y_optimized = optimized_decoder.model_y.coef_

# Calculate feature importance (absolute coefficients)
importance_x_optimized = np.abs(coef_x_optimized)
importance_y_optimized = np.abs(coef_y_optimized)

# Plot feature importance for selected channels
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Velocity X coefficients
axes[0].bar(range(len(importance_x_optimized)), importance_x_optimized)
axes[0].set_xlabel('Selected Channel Index')
axes[0].set_ylabel('|Coefficient|')
axes[0].set_title(f'Feature Importance - Velocity X (Top {len(selected_channels)} Channels)')
axes[0].grid(True, alpha=0.3)

# Velocity Y coefficients
axes[1].bar(range(len(importance_y_optimized)), importance_y_optimized)
axes[1].set_xlabel('Selected Channel Index')
axes[1].set_ylabel('|Coefficient|')
axes[1].set_title(f'Feature Importance - Velocity Y (Top {len(selected_channels)} Channels)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
print("Saving feature importance plot...")
plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved to feature_importance_optimized.png")
print("Displaying feature importance plot...")
plt.show(block=True)  # Use block=True to ensure plot displays
print("Feature importance plot displayed")

print(f"📊 Feature importance saved to: feature_importance_optimized.png")

# Print top channels from selected channels
top_idx_x = np.argsort(importance_x_optimized)[-5:][::-1]
top_idx_y = np.argsort(importance_y_optimized)[-5:][::-1]

print(f"\n🏆 Top 5 selected channels for velocity X: {[selected_channels[i] for i in top_idx_x]}")
print(f"🏆 Top 5 selected channels for velocity Y: {[selected_channels[i] for i in top_idx_y]}")

# Also show the original top channels from all 96 channels
print(f"🏆 Original top 5 channels for velocity X: {sorted(top_channels_x.tolist())[:5]}")
print(f"🏆 Original top 5 channels for velocity Y: {sorted(top_channels_y.tolist())[:5]}")

print(f"\n✅ Example ridge decoding complete!") 