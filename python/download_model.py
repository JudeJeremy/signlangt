import os
import urllib.request
import zipfile
import shutil
import sys

def download_gesture_recognizer_model():
    """
    Download the MediaPipe gesture recognizer model if it doesn't exist.
    This is a helper script to ensure the model is available.
    """
    model_filename = 'gesture_recognizer.task'
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return True
    
    try:
        print("Attempting to download MediaPipe gesture recognizer model...")
        
        # MediaPipe model URL
        # Note: This URL might change in the future
        model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        
        # Download the model
        print(f"Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        
        if os.path.exists(model_path):
            print(f"Model successfully downloaded to {model_path}")
            return True
        else:
            print("Failed to download model")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        
        # Fallback: Inform user how to manually download
        print("\nIf automatic download fails, you can manually download the model:")
        print("1. Visit: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer")
        print("2. Download the gesture recognizer model")
        print(f"3. Save it as '{model_filename}' in the '{model_dir}' directory")
        
        return False

if __name__ == "__main__":
    success = download_gesture_recognizer_model()
    sys.exit(0 if success else 1)
