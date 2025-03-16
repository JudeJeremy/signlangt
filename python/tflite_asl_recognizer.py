import os
import numpy as np
import tflite
from typing import Dict, Tuple, Optional, List

class TFLiteASLRecognizer:
    """
    ASL alphabet recognition using a TensorFlow Lite model.
    This class calculates 9 Euclidean distances between hand landmarks
    and uses a TensorFlow Lite model to predict the ASL letter.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ASL recognizer with a pre-trained TensorFlow Lite model.
        
        Args:
            model_path: Path to the pre-trained TensorFlow Lite model file
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the TensorFlow Lite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"TensorFlow Lite ASL Recognizer initialized with model: {model_path}")
        
        # Define the landmark pairs for distance calculation
        # Based on the repository description, we need 9 Euclidean distances:
        # (20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (20, 16), (16, 12), (12, 8), (8, 4)
        self.landmark_pairs = [
            (20, 0),  # pinky tip to wrist
            (16, 0),  # ring tip to wrist
            (12, 0),  # middle tip to wrist
            (8, 0),   # index tip to wrist
            (4, 0),   # thumb tip to wrist
            (20, 16), # pinky tip to ring tip
            (16, 12), # ring tip to middle tip
            (12, 8),  # middle tip to index tip
            (8, 4)    # index tip to thumb tip
        ]
        
        # Define the ASL letters (output classes)
        self.asl_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    
    def calculate_distances(self, landmarks: Dict) -> np.ndarray:
        """
        Calculate the 9 Euclidean distances between landmark pairs.
        
        Args:
            landmarks: Dictionary of hand landmarks
            
        Returns:
            Array of 9 Euclidean distances
        """
        distances = []
        
        for pair in self.landmark_pairs:
            p1, p2 = pair
            
            # Get 3D coordinates of the landmarks
            x1, y1, z1 = landmarks[p1]['x'], landmarks[p1]['y'], landmarks[p1]['z']
            x2, y2, z2 = landmarks[p2]['x'], landmarks[p2]['y'], landmarks[p2]['z']
            
            # Calculate Euclidean distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            distances.append(distance)
        
        return np.array(distances, dtype=np.float32).reshape(1, -1)
    
    def recognize(self, landmarks: Dict) -> Tuple[Optional[str], float]:
        """
        Recognize the ASL letter based on hand landmarks.
        
        Args:
            landmarks: Dictionary of hand landmarks
            
        Returns:
            Tuple of (letter, confidence)
        """
        try:
            # Calculate the distances
            distances = self.calculate_distances(landmarks)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], distances)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get the predicted class and confidence
            class_idx = np.argmax(output[0])
            confidence = output[0][class_idx]
            
            # Map the class index to the ASL letter
            letter = self.asl_letters[class_idx]
            
            return letter, float(confidence)
            
        except Exception as e:
            print(f"Error in ASL recognition: {e}")
            return None, 0.0
