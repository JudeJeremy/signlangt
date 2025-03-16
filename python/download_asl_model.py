import os
import urllib.request
import sys

def download_asl_model():
    """
    Download a pre-trained ASL alphabet recognition model.
    This is a helper script to ensure the model is available.
    """
    # Model information
    # Using a pre-trained model from the Sign Language MNIST dataset
    model_filename = 'asl_alphabet_model.onnx'
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"ASL model already exists at {model_path}")
        return model_path
    
    try:
        print("Attempting to download ASL alphabet recognition model...")
        
        # Model URL - using a publicly available ASL alphabet model
        # Note: In a production environment, you would host this model on your own server
        model_url = "https://github.com/kinivi/hand-gesture-recognition-mediapipe/raw/main/models/asl_model.onnx"
        
        # Download the model
        print(f"Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        
        if os.path.exists(model_path):
            print(f"Model successfully downloaded to {model_path}")
            return model_path
        else:
            print("Failed to download model")
            return None
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        
        # Fallback: Inform user how to manually download
        print("\nIf automatic download fails, you can manually download the model:")
        print("1. Visit: https://github.com/kinivi/hand-gesture-recognition-mediapipe")
        print("2. Download the ASL model from the models directory")
        print(f"3. Save it as '{model_filename}' in the '{model_dir}' directory")
        
        return None

if __name__ == "__main__":
    model_path = download_asl_model()
    sys.exit(0 if model_path else 1)
