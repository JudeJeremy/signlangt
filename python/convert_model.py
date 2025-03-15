import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_model(model_path, output_dir):
    """
    Convert a Keras .h5 model to TensorFlow.js format
    
    Args:
        model_path: Path to the Keras .h5 model
        output_dir: Directory to save the converted model
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Set memory growth to avoid OOM issues
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass
        
        # Load the Keras model
        model = keras.models.load_model(model_path, compile=False)
        
        # Print model summary
        print("Model summary:")
        model.summary()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert the model to TensorFlow.js format
        print(f"Converting model to TensorFlow.js format...")
        tfjs.converters.save_keras_model(model, output_dir)
        
        print(f"Model successfully converted and saved to {output_dir}")
        
        # Create a metadata.json file with input/output information
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        metadata = {
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "input_tensor_name": model.input.name,
            "output_tensor_name": model.output.name,
            "model_type": "ResNet152V2",
            "num_classes": output_shape[-1] if output_shape else None
        }
        
        import json
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")
        
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Path to the Keras model
    model_path = "../kaggle_alphabet/ResNet152V2-ASL.h5"
    
    # Output directory for the converted model
    output_dir = "../public/models/tfjs_model"
    
    # Convert the model
    convert_model(model_path, output_dir)
    
    print("Conversion complete. You can now use the model in your web application.")
