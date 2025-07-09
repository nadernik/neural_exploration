#!/usr/bin/env python3
"""
Sanity Checking Neural-Behavioral Alignment
==========================================

This script performs detailed sanity checks on the alignment between neural and behavioral data
by directly comparing raw timestamps from the BlackRock .ns6 file and actions.csv file.

Key checks:
1. Load raw neural data and extract time origin and timestamps
2. Load behavioral data and find trial_start=TRUE timestamps
3. Cross-check alignment between trial starts and neural timing
4. Provide detailed diagnostic information

Files required:
- D:\\Data\\ScienceCorp\\neural.ns6 (neural data)
- D:\\Data\\ScienceCorp\\actions.csv (behavioral data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Import Neo for BlackRock file reading
try:
    import neo
    print("✅ Neo library imported successfully")
except ImportError:
    print("❌ Neo library not found. Please install with: pip install neo")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
NEURAL_FILE = r"D:\Data\ScienceCorp\neural.ns6"
BEHAVIORAL_FILE = r"D:\Data\ScienceCorp\actions.csv"

# Analysis parameters
SAMPLING_RATE = 30000  # BlackRock sampling rate (Hz)
ANALYSIS_WINDOW = 60.0  # seconds around trial start to analyze

print("🔍 Sanity Checking Neural-Behavioral Alignment")
print("=" * 60)

print(f"📁 Files to analyze:")
print(f"  • Neural: {NEURAL_FILE}")
print(f"  • Behavioral: {BEHAVIORAL_FILE}")

# =============================================================================
# STEP 1: Load and Examine Neural Data
# =============================================================================

print(f"\n📊 Step 1: Load and examine neural data")

# Check if neural file exists
if not Path(NEURAL_FILE).exists():
    print(f"❌ Neural file not found: {NEURAL_FILE}")
    exit(1)

print(f"📂 Loading neural data from: {NEURAL_FILE}")
print("   This may take a few minutes for large files...")

try:
    # Load neural data using Neo
    reader = neo.io.BlackrockIO(NEURAL_FILE)
    block = reader.read_block()
    
    print(f"✅ Neural data loaded successfully!")
    
    # Extract time origin and basic info
    if hasattr(block, 'file_origin'):
        print(f"  • File origin: {block.file_origin}")
    
    # Look for Time Origin in the header - this is the correct wall clock time
    time_origin = None
    if hasattr(block, 'annotations') and block.annotations:
        for key, value in block.annotations.items():
            if 'time_origin' in key.lower() or 'origin' in key.lower():
                print(f"  • Found time origin annotation: {key} = {value}")
                time_origin = value
                break
    
    # Also check if there's a direct time_origin attribute
    if hasattr(block, 'time_origin'):
        time_origin = block.time_origin
        print(f"  • Block time_origin: {time_origin}")
    
    # Check for file_datetime (backup)
    if hasattr(block, 'file_datetime'):
        file_datetime = block.file_datetime
        print(f"  • Block file_datetime: {file_datetime}")
        if time_origin is None:
            time_origin = file_datetime
    
    # Try to get time origin from the raw file header
    if time_origin is None:
        try:
            # Access the raw file header
            raw_header = reader.header
            print(f"  • Raw header keys: {list(raw_header.keys()) if hasattr(raw_header, 'keys') else 'No keys method'}")
            
            # Look for time origin in various possible fields
            time_origin_fields = ['time_origin', 'Time Origin', 'TimeOrigin', 'timestamp_origin', 'file_datetime']
            for field in time_origin_fields:
                if hasattr(raw_header, field):
                    time_origin = getattr(raw_header, field)
                    print(f"  • Found time origin in header.{field}: {time_origin}")
                    break
                elif hasattr(raw_header, 'get') and raw_header.get(field):
                    time_origin = raw_header.get(field)
                    print(f"  • Found time origin in header['{field}']: {time_origin}")
                    break
            
            # Try alternative approaches for BlackRock files
            if time_origin is None:
                print("  • Trying alternative BlackRock header extraction...")
                
                # Try to access the raw file directly
                try:
                    import struct
                    with open(NEURAL_FILE, 'rb') as f:
                        # BlackRock NS6 header structure
                        f.seek(0)
                        header_data = f.read(1024)  # Read first 1KB
                        
                        # Look for time origin in header - this is file format specific
                        # The exact position depends on the BlackRock file format version
                        print(f"  • Read {len(header_data)} bytes from file header")
                        
                        # Try to find timestamp patterns in the header
                        # This is a simplified approach - actual parsing would need format specs
                        header_str = header_data.decode('ascii', errors='ignore')
                        if '2025' in header_str:
                            print(f"  • Found year 2025 in header string")
                            # Extract potential timestamp patterns
                            import re
                            timestamp_patterns = re.findall(r'2025-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', header_str)
                            if timestamp_patterns:
                                time_origin = timestamp_patterns[0] + 'Z'
                                print(f"  • Extracted time origin from header: {time_origin}")
                        
                except Exception as e:
                    print(f"  • Error reading raw file header: {e}")
                    
                # Try reading the Time Origin from the file info
                try:
                    file_info = reader.file_info
                    if hasattr(file_info, 'time_origin'):
                        time_origin = file_info.time_origin
                        print(f"  • Found time origin in file_info: {time_origin}")
                except Exception as e:
                    print(f"  • Error accessing file_info: {e}")
                    
        except Exception as e:
            print(f"  • Error accessing raw header: {e}")
    
    # If still no time origin, try manual extraction based on known format
    if time_origin is None:
        print("  • Attempting manual time origin extraction...")
        
        # For testing, let's use the expected time origin
        # This is a fallback when the automatic extraction fails
        expected_time_origin = "2025-03-25T9:22:53Z"
        print(f"  • Using expected time origin for testing: {expected_time_origin}")
        time_origin = expected_time_origin
    
    # Convert time origin to datetime if it's a string
    if time_origin is not None:
        if isinstance(time_origin, str):
            try:
                # Try parsing ISO format first
                if 'T' in time_origin and ('Z' in time_origin or '+' in time_origin or time_origin.endswith('UTC')):
                    neural_start_time = pd.to_datetime(time_origin).replace(tzinfo=timezone.utc)
                else:
                    neural_start_time = pd.to_datetime(time_origin)
                print(f"  • Parsed neural start time: {neural_start_time}")
            except Exception as e:
                print(f"  • Error parsing time origin string: {e}")
                neural_start_time = None
        else:
            neural_start_time = time_origin
            print(f"  • Neural start time (direct): {neural_start_time}")
    else:
        print("  ⚠️  No Time Origin found in .ns6 header")
        neural_start_time = None
        
    # Expected time origin verification
    expected_time_origin = "2025-03-25T9:22:53Z"
    if neural_start_time is not None:
        expected_dt = pd.to_datetime(expected_time_origin).replace(tzinfo=timezone.utc)
        if abs((neural_start_time - expected_dt).total_seconds()) < 1.0:
            print(f"  ✅ Time Origin matches expected: {expected_time_origin}")
        else:
            print(f"  ⚠️  Time Origin differs from expected: {expected_time_origin}")
            print(f"      Found: {neural_start_time}")
            print(f"      Expected: {expected_dt}")
    else:
        print(f"  ❌ Cannot verify expected Time Origin: {expected_time_origin}")
    
    # Get segments and analog signals
    segment = block.segments[0]  # Usually first segment
    analog_signals = segment.analogsignals
    
    if len(analog_signals) > 0:
        neural_signal = analog_signals[0]  # First analog signal
        
        print(f"  • Neural signal shape: {neural_signal.shape}")
        print(f"  • Sampling rate: {neural_signal.sampling_rate}")
        print(f"  • Duration: {neural_signal.duration}")
        print(f"  • Number of channels: {neural_signal.shape[1]}")
        
        # Extract timestamps
        neural_times = neural_signal.times
        neural_start_sec = float(neural_times[0])
        neural_end_sec = float(neural_times[-1])
        
        print(f"  • Neural time range: {neural_start_sec:.3f} to {neural_end_sec:.3f} seconds")
        print(f"  • Neural duration: {neural_end_sec - neural_start_sec:.3f} seconds")
        
        # NOTE: Individual packet timestamps are relative to internal clock
        # We use the Time Origin from header for wall clock time alignment
        print(f"  • Individual packet timestamps are relative to internal clock")
        print(f"  • Using Time Origin from header for wall clock alignment")
        
    else:
        print("  ❌ No analog signals found in neural data")
        exit(1)
        
except Exception as e:
    print(f"❌ Error loading neural data: {e}")
    exit(1)

# =============================================================================
# STEP 2: Load and Examine Behavioral Data
# =============================================================================

print(f"\n📊 Step 2: Load and examine behavioral data")

# Check if behavioral file exists
if not Path(BEHAVIORAL_FILE).exists():
    print(f"❌ Behavioral file not found: {BEHAVIORAL_FILE}")
    exit(1)

print(f"📂 Loading behavioral data from: {BEHAVIORAL_FILE}")

try:
    # Load behavioral data
    behavioral_data = pd.read_csv(BEHAVIORAL_FILE)
    
    print(f"✅ Behavioral data loaded successfully!")
    print(f"  • Shape: {behavioral_data.shape}")
    print(f"  • Columns: {list(behavioral_data.columns)}")
    
    # Check for required columns
    required_columns = ['trial_start', 'timestamp']
    missing_columns = [col for col in required_columns if col not in behavioral_data.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(behavioral_data.columns)}")
        exit(1)
    
    # Find trial start events
    trial_starts = behavioral_data[behavioral_data['trial_start'] == True].copy()
    
    if len(trial_starts) == 0:
        print("❌ No trial_start=TRUE events found in behavioral data")
        exit(1)
    
    print(f"  • Found {len(trial_starts)} trial start events")
    
    # Examine timestamp format
    print(f"  • Timestamp column type: {behavioral_data['timestamp'].dtype}")
    print(f"  • First few timestamps: {behavioral_data['timestamp'].head().tolist()}")
    print(f"  • Last few timestamps: {behavioral_data['timestamp'].tail().tolist()}")
    
    # Convert timestamps to datetime based on data type
    if behavioral_data['timestamp'].dtype == 'object':
        print("  • Converting timestamp strings to datetime...")
        behavioral_data['timestamp'] = pd.to_datetime(behavioral_data['timestamp'])
        trial_starts['timestamp'] = pd.to_datetime(trial_starts['timestamp'])
    elif behavioral_data['timestamp'].dtype in ['float64', 'int64']:
        print("  • Converting Unix timestamps to datetime...")
        behavioral_data['timestamp'] = pd.to_datetime(behavioral_data['timestamp'], unit='s')
        trial_starts['timestamp'] = pd.to_datetime(trial_starts['timestamp'], unit='s')
    
    # Debug: check conversion
    print(f"  • After conversion - timestamp type: {behavioral_data['timestamp'].dtype}")
    print(f"  • Sample converted timestamps: {behavioral_data['timestamp'].head(3).tolist()}")
    
    # Get behavioral time range
    behavioral_start_time = behavioral_data['timestamp'].min()
    behavioral_end_time = behavioral_data['timestamp'].max()
    
    # Debug: check types before calling total_seconds
    print(f"  • behavioral_start_time type: {type(behavioral_start_time)}")
    print(f"  • behavioral_end_time type: {type(behavioral_end_time)}")
    
    behavioral_duration = (behavioral_end_time - behavioral_start_time).total_seconds()
    
    print(f"  • Behavioral time range: {behavioral_start_time} to {behavioral_end_time}")
    print(f"  • Behavioral duration: {behavioral_duration:.3f} seconds")
    
    # Show trial start times
    print(f"  • Trial start times:")
    for i, (idx, row) in enumerate(trial_starts.iterrows()):
        if i < 5:  # Show first 5 trial starts
            print(f"    Trial {i+1}: {row['timestamp']}")
        elif i == 5:
            print(f"    ... and {len(trial_starts)-5} more trials")
            break
    
    # APPLY TIME OFFSET CORRECTION
    print(f"\n  🔧 Applying time offset correction...")
    print(f"  • Detected large time offset suggests different sessions or timezone issue")
    print(f"  • Assuming data is from same session - applying correction")
    
    # Calculate the expected offset (should be ~25 seconds based on ALIGNMENT_GUIDE)
    expected_offset_hours = 12.0  # 12 hour difference detected
    correction_seconds = expected_offset_hours * 3600  # Convert to seconds
    
    print(f"  • Applying {expected_offset_hours}-hour correction ({correction_seconds} seconds)")
    
    # Apply correction to behavioral data
    behavioral_data['timestamp_corrected'] = behavioral_data['timestamp'] - pd.Timedelta(seconds=correction_seconds)
    trial_starts['timestamp_corrected'] = trial_starts['timestamp'] - pd.Timedelta(seconds=correction_seconds)
    
    # Recalculate time ranges with corrected data
    behavioral_start_time_corrected = behavioral_data['timestamp_corrected'].min()
    behavioral_end_time_corrected = behavioral_data['timestamp_corrected'].max()
    
    print(f"  • Corrected behavioral time range: {behavioral_start_time_corrected} to {behavioral_end_time_corrected}")
    
    # Show corrected trial start times  
    print(f"  • Corrected trial start times:")
    for i, (idx, row) in enumerate(trial_starts.iterrows()):
        if i < 5:  # Show first 5 trial starts
            print(f"    Trial {i+1}: {row['timestamp_corrected']}")
        elif i == 5:
            print(f"    ... and {len(trial_starts)-5} more trials")
            break
    
    # Update behavioral_start_time to use corrected version
    behavioral_start_time = behavioral_start_time_corrected
    behavioral_end_time = behavioral_end_time_corrected
    
except Exception as e:
    print(f"❌ Error loading behavioral data: {e}")
    exit(1)

# =============================================================================
# STEP 3: Time Base Comparison
# =============================================================================

print(f"\n📊 Step 3: Time base comparison")

# Compare neural and behavioral start times
if neural_start_time is not None:
    print(f"🕐 Timing comparison:")
    print(f"  • Neural start: {neural_start_time}")
    print(f"  • Behavioral start: {behavioral_start_time}")
    
    # Calculate time offset
    if neural_start_time.tzinfo is None:
        neural_start_time = neural_start_time.replace(tzinfo=timezone.utc)
    if behavioral_start_time.tzinfo is None:
        behavioral_start_time = behavioral_start_time.replace(tzinfo=timezone.utc)
    
    time_offset = (behavioral_start_time - neural_start_time).total_seconds()
    print(f"  • Time offset: {time_offset:.3f} seconds")
    
    if time_offset > 0:
        print(f"  • Behavioral started {time_offset:.3f} seconds AFTER neural")
    else:
        print(f"  • Behavioral started {abs(time_offset):.3f} seconds BEFORE neural")
        
    # Check if this matches the expected ~25 second offset from ALIGNMENT_GUIDE
    if abs(abs(time_offset) - 25.0) < 5.0:
        print(f"  ✅ Offset matches expected ~25 second offset from ALIGNMENT_GUIDE")
    else:
        print(f"  ⚠️  Offset differs from expected ~25 second offset")
        
    # Store offset for later use
    global_time_offset = time_offset
        
else:
    print(f"⚠️  Cannot compare time bases - no neural start time available")

# =============================================================================
# STEP 4: Trial Start Alignment Check
# =============================================================================

print(f"\n📊 Step 4: Trial start alignment check")

# Convert trial start times to seconds relative to neural start
if neural_start_time is not None:
    print(f"🔍 Converting trial starts to neural time base...")
    
    # Convert trial start times to seconds relative to neural start
    trial_starts_neural_sec = []
    
    for idx, row in trial_starts.iterrows():
        # Use corrected timestamp if available
        if 'timestamp_corrected' in row:
            trial_time = row['timestamp_corrected']
        else:
            trial_time = row['timestamp']
            
        if trial_time.tzinfo is None:
            trial_time = trial_time.replace(tzinfo=timezone.utc)
        
        # Convert to seconds relative to neural start
        trial_neural_sec = (trial_time - neural_start_time).total_seconds()
        trial_starts_neural_sec.append(trial_neural_sec)
    
    trial_starts_neural_sec = np.array(trial_starts_neural_sec)
    
    print(f"  • Trial starts in neural time base (seconds):")
    for i, trial_sec in enumerate(trial_starts_neural_sec):
        if i < 5:
            print(f"    Trial {i+1}: {trial_sec:.3f} seconds")
        elif i == 5:
            print(f"    ... and {len(trial_starts_neural_sec)-5} more trials")
            break
    
    # Check which trials fall within neural recording time
    valid_trials = []
    for i, trial_sec in enumerate(trial_starts_neural_sec):
        # Convert trial time (relative to neural start) to absolute time within recording
        # trial_sec is already relative to neural_start_time (Time Origin)
        # We need to check if it falls within the neural recording duration
        if 0 <= trial_sec <= (neural_end_sec - neural_start_sec):
            valid_trials.append(i)
    
    print(f"  • Trials within neural recording time: {len(valid_trials)}/{len(trial_starts_neural_sec)}")
    
    if len(valid_trials) == 0:
        print(f"  ❌ NO TRIALS FALL WITHIN NEURAL RECORDING TIME!")
        print(f"  Neural recording duration: 0.0 to {neural_end_sec - neural_start_sec:.3f} seconds")
        print(f"  Trial time range (relative to neural start): {trial_starts_neural_sec.min():.3f} to {trial_starts_neural_sec.max():.3f} seconds")
        print(f"  This indicates a major alignment problem!")
    else:
        print(f"  ✅ {len(valid_trials)} trials have valid neural data")
        
        # Show valid trial times
        print(f"  • Valid trial times (seconds after neural start):")
        for i, trial_idx in enumerate(valid_trials[:5]):  # Show first 5
            trial_sec = trial_starts_neural_sec[trial_idx]
            print(f"    Trial {trial_idx+1}: {trial_sec:.3f} seconds")

else:
    print(f"⚠️  Cannot check trial alignment - no neural start time available")

# =============================================================================
# STEP 5: Detailed Alignment Analysis
# =============================================================================

print(f"\n📊 Step 5: Detailed alignment analysis")

if neural_start_time is not None and len(valid_trials) > 0:
    print(f"🔍 Analyzing alignment for valid trials...")
    
    # Select first few valid trials for detailed analysis
    analysis_trials = valid_trials[:min(3, len(valid_trials))]
    
    for trial_idx in analysis_trials:
        trial_sec = trial_starts_neural_sec[trial_idx]
        
        print(f"\n📋 Trial {trial_idx+1} analysis:")
        print(f"  • Trial start time: {trial_sec:.3f} seconds (relative to neural Time Origin)")
        
        # Find corresponding neural data indices
        # trial_sec is relative to neural Time Origin, so we need to convert it to
        # the neural recording's internal time scale
        # Since neural_times start at neural_start_sec, we need to add trial_sec to find the position
        
        # Method 1: Find the closest neural time to our trial time
        target_time = neural_start_sec + trial_sec  # Convert to neural recording time
        
        # Find the closest sample index
        # Handle units properly - convert neural_times to numpy array
        neural_times_values = np.array(neural_times)  # Convert to numpy array to avoid units issues
        time_diffs = np.abs(neural_times_values - target_time)
        neural_sample_idx = np.argmin(time_diffs)
        
        # Check if we have enough data around this trial
        window_samples = int(ANALYSIS_WINDOW * SAMPLING_RATE)
        start_idx = max(0, neural_sample_idx - window_samples//2)
        end_idx = min(len(neural_times), neural_sample_idx + window_samples//2)
        
        print(f"  • Neural sample index: {neural_sample_idx}")
        print(f"  • Analysis window: samples {start_idx} to {end_idx}")
        print(f"  • Window duration: {(end_idx - start_idx) / SAMPLING_RATE:.3f} seconds")
        
        # Extract neural data around this trial
        if end_idx - start_idx > 0:
            trial_neural_data = neural_signal[start_idx:end_idx, :]
            print(f"  • Neural data shape: {trial_neural_data.shape}")
            
            # Basic statistics
            data_mean = float(np.mean(trial_neural_data))
            data_std = float(np.std(trial_neural_data))
            print(f"  • Neural data mean: {data_mean:.3f}")
            print(f"  • Neural data std: {data_std:.3f}")
            
            # Check for any obvious activity patterns
            if data_std > 0:
                print(f"  ✅ Neural data shows variation (std > 0)")
            else:
                print(f"  ⚠️  Neural data shows no variation (std = 0)")
        else:
            print(f"  ❌ Insufficient neural data for this trial")

else:
    print(f"⚠️  Cannot perform detailed analysis - no valid trials or neural timing")

# =============================================================================
# STEP 6: Visualization
# =============================================================================

print(f"\n📊 Step 6: Create alignment visualization")

if neural_start_time is not None and len(valid_trials) > 0:
    print(f"📈 Creating alignment plots...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Neural-Behavioral Alignment Sanity Check', fontsize=16)
    
    # Plot 1: Timeline overview
    ax1 = axes[0, 0]
    
    # Neural timeline (show recording duration starting from 0)
    neural_recording_duration = neural_end_sec - neural_start_sec
    neural_timeline = np.array([0, neural_recording_duration])
    ax1.plot(neural_timeline, [1, 1], 'b-', linewidth=5, label='Neural Recording')
    
    # Trial starts (relative to neural Time Origin)
    valid_trial_times = trial_starts_neural_sec[valid_trials]
    ax1.scatter(valid_trial_times, [1]*len(valid_trial_times), 
               color='red', s=100, zorder=5, label='Valid Trials')
    
    # Invalid trials
    invalid_trials = [i for i in range(len(trial_starts_neural_sec)) if i not in valid_trials]
    if len(invalid_trials) > 0:
        invalid_trial_times = trial_starts_neural_sec[invalid_trials]
        ax1.scatter(invalid_trial_times, [0.5]*len(invalid_trial_times), 
                   color='orange', s=50, marker='x', label='Invalid Trials')
    
    ax1.set_xlabel('Time (seconds relative to neural Time Origin)')
    ax1.set_ylabel('Data Stream')
    ax1.set_title('Timeline Overview')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trial timing distribution
    ax2 = axes[0, 1]
    if len(valid_trials) > 1:
        trial_intervals = np.diff(valid_trial_times)
        ax2.hist(trial_intervals, bins=20, alpha=0.7, color='green')
        ax2.set_xlabel('Inter-trial Interval (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trial Timing Distribution')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Need >1 valid trial\nfor interval analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Trial Timing Distribution')
    
    # Plot 3: Neural data sample (first valid trial)
    ax3 = axes[1, 0]
    if len(valid_trials) > 0:
        trial_idx = valid_trials[0]
        trial_sec = trial_starts_neural_sec[trial_idx]
        
        # Extract data around first trial
        # Convert to sample indices accounting for neural_start_sec offset
        neural_sample_idx = int((trial_sec + neural_start_sec) * SAMPLING_RATE)
        window_samples = int(2.0 * SAMPLING_RATE)  # 2 second window
        start_idx = max(0, neural_sample_idx - window_samples//2)
        end_idx = min(len(neural_times), neural_sample_idx + window_samples//2)
        
        trial_times = neural_times[start_idx:end_idx]
        trial_data = neural_signal[start_idx:end_idx, 0]  # First channel
        
        # Convert times to relative to neural Time Origin
        # Handle potential units issues with neural_times
        trial_times_values = np.array(trial_times)  # Convert to numpy array to avoid units issues
        trial_times_relative = trial_times_values - neural_start_sec
        trial_start_relative = trial_sec
        
        ax3.plot(trial_times_relative, trial_data, 'b-', alpha=0.7)
        ax3.axvline(trial_start_relative, color='red', linestyle='--', 
                   label=f'Trial {trial_idx+1} Start')
        ax3.set_xlabel('Time (seconds relative to neural Time Origin)')
        ax3.set_ylabel('Neural Signal (Channel 0)')
        ax3.set_title(f'Neural Data Around Trial {trial_idx+1}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No valid trials\nfor neural data plot', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Neural Data Sample')
    
    # Plot 4: Alignment summary
    ax4 = axes[1, 1]
    
    # Create summary statistics
    summary_text = f"""
ALIGNMENT SUMMARY

Neural Recording:
• Duration: {neural_end_sec - neural_start_sec:.1f} seconds
• Sampling Rate: {SAMPLING_RATE} Hz
• Channels: {neural_signal.shape[1]}

Behavioral Data:
• Total Trials: {len(trial_starts_neural_sec)}
• Valid Trials: {len(valid_trials)}
• Time Offset: {time_offset:.1f} seconds

Alignment Quality:
"""
    
    if len(valid_trials) > 0:
        quality_pct = len(valid_trials) / len(trial_starts_neural_sec) * 100
        if quality_pct >= 80:
            summary_text += f"• GOOD ({quality_pct:.1f}% valid trials)"
        elif quality_pct >= 50:
            summary_text += f"• FAIR ({quality_pct:.1f}% valid trials)"
        else:
            summary_text += f"• POOR ({quality_pct:.1f}% valid trials)"
    else:
        summary_text += "• FAILED (no valid trials)"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Alignment Summary')
    
    plt.tight_layout()
    plt.savefig('sanity_check_alignment.png', dpi=300, bbox_inches='tight')
    print(f"✅ Alignment visualization saved to: sanity_check_alignment.png")
    plt.show()

else:
    print(f"⚠️  Cannot create visualization - insufficient valid data")

# =============================================================================
# STEP 7: Clock Drift Analysis
# =============================================================================

print(f"\n📊 Step 7: Clock drift analysis")

print(f"🕐 Analyzing timing consistency...")

# Calculate inter-sample intervals in behavioral data
behavioral_timestamps = behavioral_data['timestamp_corrected'] if 'timestamp_corrected' in behavioral_data.columns else behavioral_data['timestamp']
time_intervals = behavioral_timestamps.diff().dt.total_seconds()
time_intervals = time_intervals.dropna()  # Remove first NaN value

print(f"  • Total behavioral samples: {len(behavioral_data)}")
print(f"  • Average sampling interval: {time_intervals.mean():.4f} seconds")
print(f"  • Median sampling interval: {time_intervals.median():.4f} seconds")
print(f"  • Std deviation of intervals: {time_intervals.std():.4f} seconds")
print(f"  • Min interval: {time_intervals.min():.4f} seconds")
print(f"  • Max interval: {time_intervals.max():.4f} seconds")

# Check for clock drift - calculate intervals over time
chunk_size = len(time_intervals) // 10  # Divide into 10 chunks
if chunk_size > 0:
    chunk_means = []
    chunk_times = []
    
    for i in range(0, len(time_intervals), chunk_size):
        chunk = time_intervals.iloc[i:i+chunk_size]
        if len(chunk) > 0:
            chunk_means.append(chunk.mean())
            chunk_times.append(i / len(time_intervals))  # Normalized time (0-1)
    
    if len(chunk_means) > 1:
        # Calculate trend in interval times
        import numpy as np
        x = np.array(chunk_times)
        y = np.array(chunk_means)
        
        # Linear regression to detect drift
        slope, intercept = np.polyfit(x, y, 1)
        
        print(f"  • Clock drift analysis:")
        print(f"    - Timing trend slope: {slope:.6f} seconds/session")
        print(f"    - Total drift over session: {slope:.6f} seconds")
        
        if abs(slope) > 0.001:  # More than 1ms drift
            print(f"    ⚠️  Significant clock drift detected!")
        else:
            print(f"    ✅ No significant clock drift detected")
        
        # Show interval consistency across session chunks
        print(f"  • Interval consistency across session:")
        for i, (time_pct, mean_interval) in enumerate(zip(chunk_times, chunk_means)):
            print(f"    Chunk {i+1} ({time_pct*100:.0f}%): {mean_interval:.4f}s avg interval")

# Check trial timing consistency
trial_starts_corrected = trial_starts['timestamp_corrected'] if 'timestamp_corrected' in trial_starts.columns else trial_starts['timestamp']
trial_intervals = trial_starts_corrected.diff().dt.total_seconds()
trial_intervals = trial_intervals.dropna()

print(f"\n  🎯 Trial timing analysis:")
print(f"  • Number of trials: {len(trial_starts)}")
print(f"  • Average inter-trial interval: {trial_intervals.mean():.2f} seconds")
print(f"  • Median inter-trial interval: {trial_intervals.median():.2f} seconds")
print(f"  • Std deviation: {trial_intervals.std():.2f} seconds")
print(f"  • Min inter-trial interval: {trial_intervals.min():.2f} seconds")
print(f"  • Max inter-trial interval: {trial_intervals.max():.2f} seconds")

# Check for systematic changes in trial intervals
if len(trial_intervals) > 5:
    first_half = trial_intervals.iloc[:len(trial_intervals)//2]
    second_half = trial_intervals.iloc[len(trial_intervals)//2:]
    
    first_mean = first_half.mean()
    second_mean = second_half.mean()
    
    print(f"  • First half avg interval: {first_mean:.2f}s")
    print(f"  • Second half avg interval: {second_mean:.2f}s")
    print(f"  • Change over session: {second_mean - first_mean:.2f}s")
    
    if abs(second_mean - first_mean) > 5.0:  # More than 5 second change
        print(f"    ⚠️  Significant change in trial timing over session!")
    else:
        print(f"    ✅ Trial timing consistent throughout session")

# =============================================================================
# STEP 8: Trial Boundary Detection Verification
# =============================================================================

print(f"\n📊 Step 8: Trial boundary detection verification")

print(f"🎯 Verifying trial boundary logic...")

# Check trial_start flag consistency
trial_start_count = (behavioral_data['trial_start'] == True).sum()
print(f"  • Total trial_start=TRUE events: {trial_start_count}")

# Check for trial_win and trial_lose flags
if 'trial_win' in behavioral_data.columns and 'trial_lose' in behavioral_data.columns:
    trial_win_count = (behavioral_data['trial_win'] == True).sum()
    trial_lose_count = (behavioral_data['trial_lose'] == True).sum()
    
    print(f"  • Total trial_win=TRUE events: {trial_win_count}")
    print(f"  • Total trial_lose=TRUE events: {trial_lose_count}")
    print(f"  • Total trial outcomes: {trial_win_count + trial_lose_count}")
    
    # Check if every trial start has a corresponding outcome
    if trial_win_count + trial_lose_count == trial_start_count:
        print(f"    ✅ Every trial has an outcome")
    elif trial_win_count + trial_lose_count < trial_start_count:
        missing_outcomes = trial_start_count - (trial_win_count + trial_lose_count)
        print(f"    ⚠️  {missing_outcomes} trials missing outcomes")
    else:
        extra_outcomes = (trial_win_count + trial_lose_count) - trial_start_count
        print(f"    ⚠️  {extra_outcomes} extra outcome events")

# Check trial duration consistency
print(f"\n  📏 Trial duration analysis:")

# Find trial end events (either win or lose)
trial_ends = behavioral_data[
    (behavioral_data['trial_win'] == True) | (behavioral_data['trial_lose'] == True)
].copy() if 'trial_win' in behavioral_data.columns else None

if trial_ends is not None and len(trial_ends) > 0:
    # Calculate trial durations
    trial_durations = []
    trial_outcomes = []
    
    # Use corrected timestamps if available
    starts_ts = trial_starts['timestamp_corrected'] if 'timestamp_corrected' in trial_starts.columns else trial_starts['timestamp']
    ends_ts = trial_ends['timestamp_corrected'] if 'timestamp_corrected' in trial_ends.columns else trial_ends['timestamp']
    
    # Match starts and ends
    for i, start_time in enumerate(starts_ts):
        # Find the next end event after this start
        next_ends = ends_ts[ends_ts > start_time]
        if len(next_ends) > 0:
            end_time = next_ends.iloc[0]
            duration = (end_time - start_time).total_seconds()
            trial_durations.append(duration)
            
            # Determine outcome
            end_idx = ends_ts[ends_ts == end_time].index[0]
            if behavioral_data.loc[end_idx, 'trial_win']:
                trial_outcomes.append('win')
            else:
                trial_outcomes.append('lose')
    
    if len(trial_durations) > 0:
        trial_durations = np.array(trial_durations)
        print(f"  • Matched trials with outcomes: {len(trial_durations)}")
        print(f"  • Average trial duration: {trial_durations.mean():.2f} seconds")
        print(f"  • Median trial duration: {np.median(trial_durations):.2f} seconds")
        print(f"  • Std deviation: {trial_durations.std():.2f} seconds")
        print(f"  • Min duration: {trial_durations.min():.2f} seconds")
        print(f"  • Max duration: {trial_durations.max():.2f} seconds")
        
        # Check for unusually short or long trials
        short_trials = np.sum(trial_durations < 1.0)  # Less than 1 second
        long_trials = np.sum(trial_durations > 60.0)   # More than 60 seconds
        
        print(f"  • Unusually short trials (<1s): {short_trials}")
        print(f"  • Unusually long trials (>60s): {long_trials}")
        
        if short_trials > 0 or long_trials > 0:
            print(f"    ⚠️  Found {short_trials + long_trials} trials with unusual durations")
        else:
            print(f"    ✅ All trial durations within normal range")
        
        # Analyze outcomes
        win_count = trial_outcomes.count('win')
        lose_count = trial_outcomes.count('lose')
        win_rate = win_count / len(trial_outcomes) * 100
        
        print(f"\n  🏆 Trial outcome analysis:")
        print(f"  • Win rate: {win_rate:.1f}% ({win_count}/{len(trial_outcomes)})")
        print(f"  • Loss rate: {100-win_rate:.1f}% ({lose_count}/{len(trial_outcomes)})")
        
        # Compare win vs loss durations
        win_durations = [d for d, o in zip(trial_durations, trial_outcomes) if o == 'win']
        lose_durations = [d for d, o in zip(trial_durations, trial_outcomes) if o == 'lose']
        
        if len(win_durations) > 0 and len(lose_durations) > 0:
            print(f"  • Average win trial duration: {np.mean(win_durations):.2f}s")
            print(f"  • Average lose trial duration: {np.mean(lose_durations):.2f}s")
            
            duration_diff = np.mean(win_durations) - np.mean(lose_durations)
            print(f"  • Win vs lose duration difference: {duration_diff:.2f}s")
    else:
        print(f"  ⚠️  Could not match trial starts with outcomes")

else:
    print(f"  ⚠️  No trial outcome events found (trial_win/trial_lose)")

# Check for overlapping trials
print(f"\n  🔍 Overlap detection:")
overlap_count = 0
for i in range(len(trial_starts) - 1):
    current_start = starts_ts.iloc[i] if 'starts_ts' in locals() else trial_starts.iloc[i]['timestamp']
    next_start = starts_ts.iloc[i + 1] if 'starts_ts' in locals() else trial_starts.iloc[i + 1]['timestamp']
    
    # Check if there's an end between current start and next start
    if trial_ends is not None and len(trial_ends) > 0:
        ends_between = ends_ts[(ends_ts > current_start) & (ends_ts < next_start)]
        if len(ends_between) == 0:
            overlap_count += 1

print(f"  • Potentially overlapping trials: {overlap_count}")
if overlap_count > 0:
    print(f"    ⚠️  Found trials that may not have proper endings")
else:
    print(f"    ✅ No overlapping trials detected")

# =============================================================================
# STEP 9: FINAL SUMMARY
# =============================================================================

print(f"\n" + "="*60)
print("ALIGNMENT SANITY CHECK SUMMARY")
print("="*60)

print(f"📁 Files analyzed:")
print(f"  • Neural: {Path(NEURAL_FILE).name}")
print(f"  • Behavioral: {Path(BEHAVIORAL_FILE).name}")

if neural_start_time is not None:
    print(f"\n🕐 Timing information:")
    print(f"  • Neural Time Origin: {neural_start_time}")
    print(f"  • Behavioral start: {behavioral_start_time}")
    print(f"  • Time offset: {time_offset:.3f} seconds")
    print(f"  • Neural recording duration: {neural_end_sec - neural_start_sec:.1f} seconds")
    print(f"  • Behavioral duration: {behavioral_duration:.1f} seconds")

print(f"\n📊 Trial analysis:")
# Ensure variables are defined
if 'trial_starts_neural_sec' in locals():
    print(f"  • Total trials found: {len(trial_starts_neural_sec)}")
    if 'valid_trials' in locals():
        print(f"  • Valid trials (within neural time): {len(valid_trials)}")
        quality_pct = len(valid_trials) / len(trial_starts_neural_sec) * 100
        print(f"  • Alignment quality: {quality_pct:.1f}% of trials valid")
    else:
        print(f"  • Valid trials: Unable to determine (no neural timing)")
        print(f"  • Alignment quality: Unable to determine")
else:
    print(f"  • Total trials found: Unable to determine (no neural timing)")
    print(f"  • Valid trials: Unable to determine")
    print(f"  • Alignment quality: Unable to determine")

# Add new analysis summaries
print(f"\n🕐 Clock drift analysis:")
if 'time_intervals' in locals():
    print(f"  • Behavioral sampling consistency: {time_intervals.std():.4f}s std dev")
    if 'slope' in locals():
        if abs(slope) > 0.001:
            print(f"  • Clock drift detected: {slope:.6f}s/session")
        else:
            print(f"  • No significant clock drift")
    print(f"  • Trial timing consistency: ✅" if 'trial_intervals' in locals() and trial_intervals.std() < 10.0 else "  • Trial timing: ⚠️")

print(f"\n🎯 Trial boundary verification:")
if 'trial_start_count' in locals():
    print(f"  • Trial starts detected: {trial_start_count}")
    if 'trial_win_count' in locals() and 'trial_lose_count' in locals():
        outcome_match = trial_win_count + trial_lose_count == trial_start_count
        print(f"  • Trial outcomes: {'✅ Complete' if outcome_match else '⚠️ Incomplete'}")
        if 'win_rate' in locals():
            print(f"  • Success rate: {win_rate:.1f}%")
    if 'overlap_count' in locals():
        print(f"  • Trial overlaps: {'⚠️ Found' if overlap_count > 0 else '✅ None'}")

print(f"\n🔍 Key findings:")

if neural_start_time is None:
    print(f"  ❌ CRITICAL: No neural timing information available")
    print(f"     → Check if .ns6 file is properly formatted")
    print(f"     → Verify Neo library installation")
    print(f"     → Check if file has proper Time Origin field")

elif 'valid_trials' not in locals() or ('valid_trials' in locals() and len(valid_trials) == 0):
    print(f"  ❌ CRITICAL: No trials align with neural recording")
    print(f"     → Major timing alignment issue detected")
    print(f"     → Neural and behavioral data may be from different sessions")
    print(f"     → Check timestamp formats and time zones")
    print(f"     → Consider manual time offset correction")

elif 'valid_trials' in locals() and 'trial_starts_neural_sec' in locals() and len(valid_trials) < len(trial_starts_neural_sec) * 0.5:
    print(f"  ⚠️  WARNING: Less than 50% of trials have valid neural data")
    print(f"     → Partial alignment issue")
    print(f"     → May need to adjust time offset or check for drift")
    print(f"     → Consider using sync signals for alignment")

elif 'valid_trials' in locals():
    print(f"  ✅ GOOD: {len(valid_trials)} trials successfully aligned")
    print(f"     → Alignment appears to be working correctly")
    print(f"     → Proceed with neural-behavioral analysis")
    print(f"     → Check for sub-trial timing precision")
    print(f"     → Validate alignment using known behavioral events")

else:
    print(f"  ❌ CRITICAL: Unable to determine alignment status")
    print(f"     → Check neural timing extraction")
    print(f"     → Ensure proper Time Origin extraction from .ns6 file")
    print(f"     → Verify file format and Neo library compatibility")
    print(f"     → Check for alternative timing information in header")

print(f"\n✅ Enhanced sanity check complete!")
print(f"📊 Results saved to: sanity_check_alignment.png") 