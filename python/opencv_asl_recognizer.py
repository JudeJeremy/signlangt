import cv2
import numpy as np
import os

class OpenCVASLRecognizer:
    """
    ASL alphabet recognition using OpenCV's DNN module.
    This class loads a pre-trained model for recognizing ASL alphabet letters.
    """
    
    def __init__(self, model_path):
        """
        Initialize the ASL recognizer with a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model file (ONNX format)
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the DNN model
        self.net = cv2.dnn.readNet(model_path)
        
        # Set up preprocessing parameters
        self.input_size = (64, 64)  # Common input size for hand models
        self.scale_factor = 1/255.0
        self.mean = (0, 0, 0)
        
        # Map of indices to ASL letters (0-25 for A-Z)
        self.idx_to_letter = {i: chr(ord('A') + i) for i in range(26)}
        
        print(f"OpenCV ASL Recognizer initialized with model: {model_path}")
    
    def preprocess_hand_image(self, hand_image):
        """
        Preprocess a hand image for the neural network.
        
        Args:
            hand_image: Image of the hand region
            
        Returns:
            Preprocessed image ready for the neural network
        """
        # Resize to the input size expected by the model
        resized = cv2.resize(hand_image, self.input_size)
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            resized, 
            self.scale_factor, 
            self.input_size, 
            self.mean, 
            swapRB=True
        )
        
        return blob
    
    def recognize(self, hand_image):
        """
        Recognize the ASL letter in a hand image.
        
        Args:
            hand_image: Image of the hand region
            
        Returns:
            Tuple of (letter, confidence)
        """
        # Check if image is valid
        if hand_image is None or hand_image.size == 0:
            return None, 0.0
        
        try:
            # Preprocess the image
            blob = self.preprocess_hand_image(hand_image)
            
            # Run inference
            self.net.setInput(blob)
            output = self.net.forward()
            
            # Get the prediction
            idx = np.argmax(output)
            confidence = float(output[0][idx])
            letter = self.idx_to_letter.get(idx, '?')
            
            return letter, confidence
            
        except Exception as e:
            print(f"Error in ASL recognition: {e}")
            return None, 0.0
    
    def isolate_hand(self, frame, landmarks):
        """
        Isolate the hand region from the frame using landmarks.
        
        Args:
            frame: Full video frame
            landmarks: Dictionary of hand landmarks
            
        Returns:
            Cropped and preprocessed hand image
        """
        try:
            # Get hand bounding box from landmarks
            height, width = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            x_coords = []
            y_coords = []
            
            for landmark in landmarks.values():
                x_coords.append(landmark['x'] * width)
                y_coords.append(landmark['y'] * height)
            
            # Add padding around the hand
            padding = 20
            x_min = max(0, int(min(x_coords)) - padding)
            y_min = max(0, int(min(y_coords)) - padding)
            x_max = min(width, int(max(x_coords)) + padding)
            y_max = min(height, int(max(y_coords)) + padding)
            
            # Ensure we have a valid region
            if x_min >= x_max or y_min >= y_max:
                return None
            
            # Crop the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            
            # Convert to RGB (if needed)
            if len(hand_img.shape) == 3 and hand_img.shape[2] == 3:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            
            return hand_img
            
        except Exception as e:
            print(f"Error isolating hand: {e}")
            return None
