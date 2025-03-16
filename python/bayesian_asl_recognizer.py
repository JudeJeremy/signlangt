import os
import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List

class BayesianASLRecognizer:
    """
    ASL alphabet recognition using a Bayesian classifier.
    This class uses the model from aqua1907/Gesture-Recognition repository.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ASL recognizer with a pre-trained Bayesian classifier.
        
        Args:
            model_path: Path to the pre-trained model file (pickle format)
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the Bayesian classifier
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        print(f"Bayesian ASL Recognizer initialized with model: {model_path}")
        
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
        
        return np.array(distances).reshape(1, -1)
    
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
            
            # Predict the letter using the Bayesian classifier
            letter = self.classifier.predict(distances)[0]
            
            # Get the probability/confidence
            proba = self.classifier.predict_proba(distances)[0]
            confidence = np.max(proba)
            
            return letter, confidence
            
        except Exception as e:
            print(f"Error in ASL recognition: {e}")
            return None, 0.0
