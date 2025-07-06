#!/usr/bin/env python3
"""
Example usage of the Neural-Behavioral Integration Script

This script shows how to use the NeuralBehavioralIntegrator class
to process neural and behavioral data.
"""

from neural_behavioral_integration import NeuralBehavioralIntegrator
import logging

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)

def main():
    """
    Example usage of the neural-behavioral integrator.
    """
    
    # File paths - modify these to match your actual files
    neural_file = r"D:\Data\ScienceCorp\neural.ns6"          # Your .ns6 neural data file
    behavioral_file = r"D:\Data\ScienceCorp\actions.csv"     # Your behavioral CSV file
    output_file = r"D:\Data\ScienceCorp\trials.h5"           # Output HDF5 file
    
    print("Neural-Behavioral Data Integration Example")
    print("=" * 50)
    
    # Create the integrator
    integrator = NeuralBehavioralIntegrator(
        neural_file=neural_file,
        behavioral_file=behavioral_file,
        output_file=output_file
    )
    
    # Option 1: Run the complete pipeline in one go
    print("\n1. Running complete pipeline...")
    try:
        integrator.process_all(downsample=True)
        print("✅ Complete pipeline finished successfully!")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return
    
    # Option 2: Run step by step for more control
    print("\n2. Alternative: Step-by-step processing...")
    
    # # Uncomment these lines if you want to run step-by-step instead
    # try:
    #     # Load behavioral data
    #     behavioral_data = integrator.load_behavioral_data()
    #     print(f"   - Loaded {len(behavioral_data)} behavioral samples")
    #     
    #     # Setup neural I/O
    #     neural_io = integrator.setup_neural_io()
    #     print(f"   - Neural I/O setup complete")
    #     
    #     # Segment trials
    #     trials = integrator.segment_trials()
    #     print(f"   - Found {len(trials)} trials")
    #     
    #     # Save to HDF5
    #     integrator.save_trials_to_hdf5(downsample=True)
    #     print("   - Saved trials to HDF5")
    #     
    # except Exception as e:
    #     print(f"❌ Step-by-step processing failed: {e}")
    
    # Show summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Neural file: {neural_file}")
    print(f"Behavioral file: {behavioral_file}")
    print(f"Output file: {output_file}")
    print(f"Trials processed: {len(integrator.trials)}")
    print("\nThe output HDF5 file contains:")
    print("- /trial_1/neural (neural data)")
    print("- /trial_1/behavioral_timestamps")
    print("- /trial_1/velocity_x, velocity_y")
    print("- Trial metadata as attributes")
    print("\nYou can now load this file for analysis!")

if __name__ == "__main__":
    main() 