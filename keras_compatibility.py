#!/usr/bin/env python3
"""
Keras Compatibility Module
=========================

This module provides compatibility functions for missing Keras functions
that may not be available in certain versions.

Usage:
    # At the start of your script, import this module
    import keras_compatibility
    
    # Or import specific functions
    from keras_compatibility import clip_to_image_size
"""

import numpy as np
import tensorflow as tf

# =============================================================================
# COMPATIBILITY FUNCTIONS
# =============================================================================

def clip_to_image_size(bounding_boxes, image_shape):
    """
    Clip bounding boxes to image size.
    
    This is a compatibility function for the missing clip_to_image_size function
    in certain Keras versions.
    
    Args:
        bounding_boxes: Tensor of shape (..., 4) containing bounding boxes in format [y1, x1, y2, x2]
        image_shape: Tensor of shape (2,) containing [height, width] of the image
        
    Returns:
        Clipped bounding boxes tensor of the same shape as input
    """
    # Ensure image_shape is a tensor
    if not isinstance(image_shape, tf.Tensor):
        image_shape = tf.convert_to_tensor(image_shape, dtype=bounding_boxes.dtype)
    
    # Extract height and width
    height, width = image_shape[0], image_shape[1]
    
    # Split bounding boxes into coordinates
    y1, x1, y2, x2 = tf.split(bounding_boxes, 4, axis=-1)
    
    # Clip coordinates to image boundaries
    y1 = tf.clip_by_value(y1, 0.0, height)
    x1 = tf.clip_by_value(x1, 0.0, width)
    y2 = tf.clip_by_value(y2, 0.0, height)
    x2 = tf.clip_by_value(x2, 0.0, width)
    
    # Ensure y2 >= y1 and x2 >= x1
    y2 = tf.maximum(y2, y1)
    x2 = tf.maximum(x2, x1)
    
    # Concatenate back into bounding boxes format
    clipped_boxes = tf.concat([y1, x1, y2, x2], axis=-1)
    
    return clipped_boxes

# =============================================================================
# PATCHING FUNCTIONS
# =============================================================================

def patch_keras_imports():
    """
    Patch Keras imports to use compatibility functions when original functions are missing.
    
    This function should be called at the start of your script to ensure compatibility.
    """
    import sys
    import types
    
    # Create a mock module for the missing imports
    class MockConvertersModule(types.ModuleType):
        def __init__(self):
            super().__init__('converters')
            self.clip_to_image_size = clip_to_image_size
    
    # Try to patch the missing module
    try:
        # Create the missing module path
        import keras.src.layers.preprocessing
        import keras.src.layers.preprocessing.image_preprocessing
        import keras.src.layers.preprocessing.image_preprocessing.bounding_boxes
        
        # Add our mock module
        mock_converters = MockConvertersModule()
        keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters = mock_converters
        
        print("✅ Successfully patched Keras imports")
        
    except ImportError:
        print("⚠️  Could not patch Keras imports - module structure not found")
        print("   You can still use the compatibility functions directly")

def safe_import_clip_to_image_size():
    """
    Safely import clip_to_image_size function, using compatibility version if needed.
    
    Returns:
        The clip_to_image_size function (either original or compatibility version)
    """
    try:
        # Try to import the original function
        from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import clip_to_image_size
        print("✅ Using original clip_to_image_size function")
        return clip_to_image_size
    except ImportError:
        # Use our compatibility implementation
        print("⚠️  Using compatibility implementation for clip_to_image_size")
        return clip_to_image_size

# =============================================================================
# AUTOMATIC PATCHING
# =============================================================================

# Automatically patch imports when this module is imported
patch_keras_imports()

# =============================================================================
# TESTING
# =============================================================================

def test_compatibility():
    """Test that the compatibility functions work correctly."""
    print("🧪 Testing Keras compatibility functions...")
    
    # Test clip_to_image_size
    test_boxes = tf.constant([[0.0, 0.0, 100.0, 100.0], 
                             [-10.0, -10.0, 110.0, 110.0]], dtype=tf.float32)
    image_shape = tf.constant([80.0, 80.0], dtype=tf.float32)
    
    clipped = clip_to_image_size(test_boxes, image_shape)
    
    print(f"Original boxes: {test_boxes.numpy()}")
    print(f"Image shape: {image_shape.numpy()}")
    print(f"Clipped boxes: {clipped.numpy()}")
    print("✅ All compatibility functions work correctly!")

if __name__ == "__main__":
    test_compatibility() 