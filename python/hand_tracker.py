import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional, Deque
from collections import deque
import os
import time
import math

# Import our custom ASL recognizer
from tflite_asl_recognizer import TFLiteASLRecognizer

# Import the model downloader
from download_gesture_models import download_gesture_models

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands for landmark detection
        # We'll use this as a fallback and for visualization
        # Use 3D model for better accuracy
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Use more complex model for better 3D tracking
        )
        
        # Flag to indicate if we're using 3D landmarks
        self.use_3d_landmarks = True
        
        # Initialize MediaPipe GestureRecognizer if available
        self.use_gesture_recognizer = self._setup_gesture_recognizer()
        
        # Map for ASL gestures
        # This will be populated based on the model's output or used for custom mapping
        self.gesture_to_asl_map = {
            # Standard MediaPipe gestures that might correspond to ASL
            "Thumb_Up": "Y",  # Approximation
            "Thumb_Down": None,
            "Open_Palm": "B",  # Approximation
            "Victory": "V",
            "ILoveYou": None,
            "Closed_Fist": "S",  # Approximation
            "Pointing_Up": "D",  # Approximation
        }
        
        # ASL-specific gestures we'll detect with custom logic if needed
        self.asl_gestures = {
            'A': self._check_a_gesture,
            'B': self._check_b_gesture,
            'C': self._check_c_gesture,
            'D': self._check_d_gesture,
            'E': self._check_e_gesture,
            'F': self._check_f_gesture,
            'G': self._check_g_gesture,
            'H': self._check_h_gesture,
            'I': self._check_i_gesture,
            'J': self._check_j_gesture,
            'K': self._check_k_gesture,
            'L': self._check_l_gesture,
            'M': self._check_m_gesture,
            'N': self._check_n_gesture,
            'O': self._check_o_gesture,
            'P': self._check_p_gesture,
            'Q': self._check_q_gesture,
            'R': self._check_r_gesture,
            'S': self._check_s_gesture,
            'T': self._check_t_gesture,
            'U': self._check_u_gesture,
            'V': self._check_v_gesture,
            'W': self._check_w_gesture,
            'X': self._check_x_gesture,
            'Y': self._check_y_gesture,
            'Z': self._check_z_gesture
        }
        
        # Adaptive confidence thresholds for gesture detection
        self.base_confidence_threshold = 0.7
        
        # Individual thresholds for each gesture based on complexity and confusion potential
        self.gesture_thresholds = {
            'A': 0.75, 'B': 0.70, 'C': 0.75, 'D': 0.75, 'E': 0.75,
            'F': 0.75, 'G': 0.80, 'H': 0.75, 'I': 0.70, 'J': 0.85,  # J requires motion, higher threshold
            'K': 0.75, 'L': 0.70, 'M': 0.80, 'N': 0.80, 'O': 0.75,
            'P': 0.80, 'Q': 0.80, 'R': 0.75, 'S': 0.75, 'T': 0.75,
            'U': 0.75, 'V': 0.70, 'W': 0.70, 'X': 0.80, 'Y': 0.70,
            'Z': 0.85  # Z requires motion, higher threshold
        }
        
        # Commonly confused gesture pairs that need higher thresholds
        self.confused_pairs = [
            ('A', 'S'), ('A', 'E'), ('A', 'M'), ('A', 'N'), ('A', 'T'),
            ('S', 'E'), ('B', 'P'), ('D', 'Z'), ('U', 'V'), ('K', 'V'),
            ('M', 'N'), ('R', 'U')
        ]
        
        # Enhanced gesture stability system
        self.gesture_history = deque(maxlen=10)  # Increased from 3 to 10 for better stability
        self.last_gesture = None
        self.gesture_stability_count = 0
        self.required_stability = 3
        
        # Temporal consistency tracking
        self.last_detection_time = 0
        self.min_gesture_duration = 0.2  # Minimum time (seconds) a gesture should be held
        
        # State machine for gesture transitions
        self.current_state = "IDLE"  # States: IDLE, DETECTING, CONFIRMED, TRANSITIONING
        self.state_start_time = 0
        self.transition_cooldown = 0.5  # Time (seconds) to wait before accepting a new gesture

    def _setup_gesture_recognizer(self):
        """
        Set up the ASL gesture recognizer.
        Returns True if setup was successful, False otherwise.
        """
        try:
            # Download the gesture models if needed
            model_files = download_gesture_models()
            
            if not model_files:
                print("Gesture models not available. Falling back to basic hand tracking with custom gesture detection")
                return False
            
            # Get the path to the TensorFlow Lite model
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            model_path = os.path.join(models_dir, "asl_model.tflite")
            
            if not os.path.exists(model_path):
                print(f"ASL model not found at {model_path}. Falling back to basic hand tracking")
                return False
                
            try:
                # Initialize the TensorFlow Lite ASL Recognizer
                self.asl_recognizer = TFLiteASLRecognizer(model_path)
                
                # Store the recognizer as the gesture_recognizer for compatibility with existing code
                self.gesture_recognizer = self.asl_recognizer
                
                print("TensorFlow Lite ASL Recognizer initialized successfully")
                
                # Test the recognizer with a dummy input to make sure it works
                # This will catch issues with placeholder models
                dummy_landmarks = {i: {'x': 0.5, 'y': 0.5, 'z': 0.0} for i in range(21)}
                _, _ = self.asl_recognizer.recognize(dummy_landmarks)
                
                return True
            except Exception as e:
                print(f"Error testing ASL Recognizer: {e}")
                print("The model file may be a placeholder. Falling back to custom detection.")
                return False
            
        except Exception as e:
            print(f"Error setting up ASL Recognizer: {e}")
            print("Falling back to basic hand tracking with custom gesture detection")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """
        Process a video frame and detect hand landmarks and gestures.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple containing:
            - Processed frame with drawings
            - List of landmark dictionaries
            - List of detected gestures
        """
        # Convert BGR to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Lists to store results
        landmarks_list = []
        gestures = []
        
        # Get current time for temporal consistency
        current_time = time.time()
        
        if self.use_gesture_recognizer:
            # Use OpenCV ASL Recognizer
            try:
                # Process the frame with MediaPipe Hands first to get landmarks
                rgb_frame.flags.writeable = False
                results = self.hands.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks on frame
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                        
                        # Convert landmarks to dictionary
                        landmarks_dict = self._landmarks_to_dict(hand_landmarks)
                        landmarks_list.append(landmarks_dict)
                        
                        # Recognize the ASL letter directly from landmarks
                        asl_letter, confidence = self.asl_recognizer.recognize(landmarks_dict)
                        
                        # Get threshold for this gesture
                        threshold = self.base_confidence_threshold
                        
                        if asl_letter and confidence > threshold:
                            # Apply state machine logic
                            if self.current_state == "IDLE":
                                # New gesture detected
                                self.current_state = "DETECTING"
                                self.state_start_time = current_time
                                self.last_gesture = asl_letter
                                self.gesture_stability_count = 1
                                
                            elif self.current_state == "DETECTING":
                                if asl_letter == self.last_gesture:
                                    # Same gesture detected
                                    self.gesture_stability_count += 1
                                    
                                    # Check if gesture has been stable for long enough
                                    if (self.gesture_stability_count >= self.required_stability and 
                                        current_time - self.state_start_time >= self.min_gesture_duration):
                                        self.current_state = "CONFIRMED"
                                        self.gesture_history.append(asl_letter)
                                        gestures.append(asl_letter)
                                        
                                        # Add visual feedback for detected gesture
                                        cv2.putText(
                                            frame,
                                            f"Detected: {asl_letter} ({confidence:.2f})",
                                            (10, frame.shape[0] - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            (0, 255, 0),
                                            2
                                        )
                                else:
                                    # Different gesture detected
                                    # Check if the new gesture is in the recent history
                                    recent_gestures = list(self.gesture_history)
                                    if asl_letter in recent_gestures and recent_gestures.count(asl_letter) >= 2:
                                        # If this gesture has appeared recently, give it more weight
                                        self.gesture_stability_count = max(1, self.required_stability // 2)
                                    else:
                                        self.gesture_stability_count = 1
                                    
                                    self.last_gesture = asl_letter
                                    self.state_start_time = current_time
                                    
                            elif self.current_state == "CONFIRMED":
                                if asl_letter != self.last_gesture:
                                    # Transition to a new gesture
                                    self.current_state = "TRANSITIONING"
                                    self.state_start_time = current_time
                                    self.last_gesture = asl_letter
                                    self.gesture_stability_count = 1
                                
                            elif self.current_state == "TRANSITIONING":
                                # Wait for cooldown before accepting a new gesture
                                if current_time - self.state_start_time >= self.transition_cooldown:
                                    self.current_state = "DETECTING"
                                    self.state_start_time = current_time
                                    
                                    if asl_letter == self.last_gesture:
                                        self.gesture_stability_count += 1
                                    else:
                                        self.gesture_stability_count = 1
                                        self.last_gesture = asl_letter
                        else:
                            # No valid gesture detected
                            if self.current_state != "IDLE" and current_time - self.last_detection_time > self.min_gesture_duration * 2:
                                # Reset state if no gesture detected for a while
                                self.current_state = "IDLE"
                                self.gesture_stability_count = 0
                                self.last_gesture = None
                
                # Update last detection time
                self.last_detection_time = current_time
                
            except Exception as e:
                print(f"Error using ASL Recognizer: {e}")
                print("Falling back to basic hand tracking")
                self.use_gesture_recognizer = False
        
        # If GestureRecognizer failed or is not available, use basic hand tracking
        if not self.use_gesture_recognizer or not landmarks_list:
            # Process the frame with MediaPipe Hands
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Convert landmarks to dictionary
                    landmarks_dict = self._landmarks_to_dict(hand_landmarks)
                    landmarks_list.append(landmarks_dict)
                    
                    # Detect ASL gestures using custom logic
                    gesture, confidence = self._detect_asl_gesture(landmarks_dict)
                    
                    if gesture:
                        # Get threshold for this gesture
                        threshold = self.gesture_thresholds.get(gesture, self.base_confidence_threshold)
                        
                        # Check if this gesture is part of a confused pair
                        for pair in self.confused_pairs:
                            if gesture in pair:
                                # Get the other gesture in the pair
                                other_gesture = pair[0] if gesture == pair[1] else pair[1]
                                
                                # Check confidence of the other gesture
                                other_confidence = self._check_gesture_confidence(landmarks_dict, other_gesture)
                                
                                # If the other gesture has similar confidence, increase threshold
                                if confidence - other_confidence < 0.2:
                                    threshold += 0.05
                                    break
                        
                        if confidence > threshold:
                            # Apply state machine logic
                            if self.current_state == "IDLE":
                                # New gesture detected
                                self.current_state = "DETECTING"
                                self.state_start_time = current_time
                                self.last_gesture = gesture
                                self.gesture_stability_count = 1
                                
                            elif self.current_state == "DETECTING":
                                if gesture == self.last_gesture:
                                    # Same gesture detected
                                    self.gesture_stability_count += 1
                                    
                                    # Check if gesture has been stable for long enough
                                    if (self.gesture_stability_count >= self.required_stability and 
                                        current_time - self.state_start_time >= self.min_gesture_duration):
                                        self.current_state = "CONFIRMED"
                                        self.gesture_history.append(gesture)
                                        gestures.append(gesture)
                                        
                                        # Add visual feedback for detected gesture
                                        cv2.putText(
                                            frame,
                                            f"Detected: {gesture} ({confidence:.2f})",
                                            (10, frame.shape[0] - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            (0, 255, 0),
                                            2
                                        )
                                else:
                                    # Different gesture detected
                                    # Check if the new gesture is in the recent history
                                    recent_gestures = list(self.gesture_history)
                                    if gesture in recent_gestures and recent_gestures.count(gesture) >= 2:
                                        # If this gesture has appeared recently, give it more weight
                                        self.gesture_stability_count = max(1, self.required_stability // 2)
                                    else:
                                        self.gesture_stability_count = 1
                                    
                                    self.last_gesture = gesture
                                    self.state_start_time = current_time
                                    
                            elif self.current_state == "CONFIRMED":
                                if gesture != self.last_gesture:
                                    # Transition to a new gesture
                                    self.current_state = "TRANSITIONING"
                                    self.state_start_time = current_time
                                    self.last_gesture = gesture
                                    self.gesture_stability_count = 1
                                
                            elif self.current_state == "TRANSITIONING":
                                # Wait for cooldown before accepting a new gesture
                                if current_time - self.state_start_time >= self.transition_cooldown:
                                    self.current_state = "DETECTING"
                                    self.state_start_time = current_time
                                    
                                    if gesture == self.last_gesture:
                                        self.gesture_stability_count += 1
                                    else:
                                        self.gesture_stability_count = 1
                                        self.last_gesture = gesture
                        else:
                            # Confidence below threshold
                            if self.current_state != "IDLE" and current_time - self.last_detection_time > self.min_gesture_duration * 2:
                                # Reset state if no gesture detected for a while
                                self.current_state = "IDLE"
                                self.gesture_stability_count = 0
                                self.last_gesture = None
                    else:
                        # No gesture detected
                        if self.current_state != "IDLE" and current_time - self.last_detection_time > self.min_gesture_duration * 2:
                            # Reset state if no gesture detected for a while
                            self.current_state = "IDLE"
                            self.gesture_stability_count = 0
                            self.last_gesture = None
                    
                    # Update last detection time
                    self.last_detection_time = current_time
            else:
                # No hands detected
                if self.current_state != "IDLE" and current_time - self.last_detection_time > self.min_gesture_duration * 2:
                    # Reset state if no hands detected for a while
                    self.current_state = "IDLE"
                    self.gesture_stability_count = 0
                    self.last_gesture = None
        
        # Add state machine status to frame for debugging
        cv2.putText(
            frame,
            f"State: {self.current_state}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return frame, landmarks_list, gestures

    def _landmarks_to_dict(self, hand_landmarks) -> Dict:
        """Convert MediaPipe landmarks to dictionary format."""
        return {
            i: {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
            for i, landmark in enumerate(hand_landmarks.landmark)
        }

    def _check_gesture_confidence(self, landmarks: Dict, gesture_name: str) -> float:
        """Check the confidence of a specific gesture."""
        if gesture_name in self.asl_gestures:
            return self.asl_gestures[gesture_name](self._normalize_landmarks(landmarks))
        return 0.0

    def _detect_asl_gesture(self, landmarks: Dict) -> Tuple[Optional[str], float]:
        """
        Detect ASL gestures using custom logic.
        This is a fallback when the GestureRecognizer is not available.
        
        Args:
            landmarks: Dictionary of hand landmarks
            
        Returns:
            Tuple of (detected gesture name or None, confidence score)
        """
        # Normalize landmarks to make detection more robust
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # Check each ASL gesture
        best_gesture = None
        best_confidence = 0.0
        second_best_gesture = None
        second_best_confidence = 0.0
        
        for gesture_name, check_func in self.asl_gestures.items():
            confidence = check_func(normalized_landmarks)
            
            if confidence > best_confidence:
                second_best_confidence = best_confidence
                second_best_gesture = best_gesture
                best_confidence = confidence
                best_gesture = gesture_name
            elif confidence > second_best_confidence:
                second_best_confidence = confidence
                second_best_gesture = gesture_name
        
        # If the top two gestures have very similar confidence, increase the required threshold
        if second_best_gesture and (best_confidence - second_best_confidence < 0.15):
            # Check if these gestures are in our confused pairs
            is_confused_pair = False
            for pair in self.confused_pairs:
                if best_gesture in pair and second_best_gesture in pair:
                    is_confused_pair = True
                    break
            
            # If this is a known confused pair, require higher confidence
            if is_confused_pair and best_confidence < self.gesture_thresholds.get(best_gesture, self.base_confidence_threshold) + 0.1:
                return None, 0.0
        
        return best_gesture, best_confidence

    def _normalize_landmarks(self, landmarks: Dict) -> Dict:
        """
        Normalize hand landmarks to make detection more robust to different hand sizes and positions.
        Takes advantage of 3D information when available.
        
        Args:
            landmarks: Dictionary of hand landmarks
            
        Returns:
            Dictionary of normalized landmarks
        """
        # Find the bounding box of the hand
        min_x = min(landmark['x'] for landmark in landmarks.values())
        max_x = max(landmark['x'] for landmark in landmarks.values())
        min_y = min(landmark['y'] for landmark in landmarks.values())
        max_y = max(landmark['y'] for landmark in landmarks.values())
        
        # Also normalize Z if we're using 3D landmarks
        if self.use_3d_landmarks:
            min_z = min(landmark['z'] for landmark in landmarks.values())
            max_z = max(landmark['z'] for landmark in landmarks.values())
            depth = max(0.001, max_z - min_z)  # Avoid division by zero
        
        # Calculate the width and height of the bounding box
        width = max(0.001, max_x - min_x)  # Avoid division by zero
        height = max(0.001, max_y - min_y)
        
        # Normalize landmarks to [0, 1] range within the bounding box
        normalized_landmarks = {}
        for i, landmark in landmarks.items():
            normalized_z = (landmark['z'] - min_z) / depth if self.use_3d_landmarks else landmark['z']
            
            normalized_landmarks[i] = {
                'x': (landmark['x'] - min_x) / width,
                'y': (landmark['y'] - min_y) / height,
                'z': normalized_z  # Normalize Z if using 3D landmarks
            }
        
        return normalized_landmarks

    # Simplified gesture detection functions for fallback mode
    
    def _check_a_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'A'."""
        # Thumb up, all other fingers closed in a fist
        thumb_tip = landmarks[4]
        thumb_up = landmarks[2]['y'] - thumb_tip['y'] > 0.2
        
        # Check if other fingers are curled
        fingers_curled = True
        for i in [8, 12, 16, 20]:  # finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:  # if tip is above middle joint
                fingers_curled = False
                break
        
        return 0.9 if thumb_up and fingers_curled else 0.0

    def _check_b_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'B'."""
        # All fingers extended and together
        fingers_extended = True
        for i in [8, 12, 16, 20]:  # finger tips
            if landmarks[i]['y'] > landmarks[i-2]['y']:  # if tip is below middle joint
                fingers_extended = False
                break
        
        # Check if fingers are close together
        fingers_together = True
        for i in range(8, 20, 4):  # Check adjacent fingertips
            if abs(landmarks[i]['x'] - landmarks[i+4]['x']) > 0.2:
                fingers_together = False
                break
        
        return 0.9 if fingers_extended and fingers_together else 0.0

    def _check_c_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'C'."""
        # Curved hand position
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if thumb and index form a C shape
        c_shape = (abs(thumb_tip['y'] - index_tip['y']) < 0.3 and 
                  abs(thumb_tip['x'] - index_tip['x']) > 0.2)
        
        return 0.9 if c_shape else 0.0

    def _check_d_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'D'."""
        # Index finger up, others curled
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_up and others_curled else 0.0

    def _check_e_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'E'."""
        # All fingers curled, thumb across palm
        fingers_curled = True
        for i in [8, 12, 16, 20]:  # finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                fingers_curled = False
                break
        
        # Thumb across palm
        thumb_across = landmarks[4]['x'] > landmarks[5]['x']
        
        return 0.9 if fingers_curled and thumb_across else 0.0

    def _check_f_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'F'."""
        # Index and thumb touching, other fingers extended
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if thumb and index are touching
        touching = ((thumb_tip['x'] - index_tip['x'])**2 + 
                   (thumb_tip['y'] - index_tip['y'])**2) < 0.01
        
        # Check if other fingers are extended
        others_extended = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] > landmarks[i-2]['y']:
                others_extended = False
                break
        
        return 0.9 if touching and others_extended else 0.0

    def _check_g_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'G'."""
        # Index pointing to side, thumb extended
        index_side = abs(landmarks[8]['x'] - landmarks[5]['x']) > 0.2
        thumb_extended = landmarks[2]['y'] - landmarks[4]['y'] > 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_side and thumb_extended and others_curled else 0.0

    def _check_h_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'H'."""
        # Index and middle extended side by side
        index_extended = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_extended = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if fingers are side by side
        side_by_side = abs(landmarks[8]['x'] - landmarks[12]['x']) < 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_extended and middle_extended and side_by_side and others_curled else 0.0

    def _check_i_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'I'."""
        # Pinky extended, others curled
        pinky_extended = landmarks[17]['y'] - landmarks[20]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [8, 12, 16]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if pinky_extended and others_curled else 0.0

    def _check_j_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'J'."""
        # Similar to I but with movement (not easily detected in a single frame)
        # For simplicity, we'll check for pinky extended and curved
        pinky_extended = landmarks[17]['y'] - landmarks[20]['y'] > 0.2
        pinky_curved = abs(landmarks[20]['x'] - landmarks[17]['x']) > 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [8, 12, 16]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if pinky_extended and pinky_curved and others_curled else 0.0
        
    def _check_k_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'K'."""
        # Index and middle up, spread apart
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_up = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if fingers are spread apart
        spread_apart = abs(landmarks[8]['x'] - landmarks[12]['x']) > 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_up and middle_up and spread_apart and others_curled else 0.0

    def _check_l_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'L'."""
        # Thumb and index forming L shape
        thumb_extended = landmarks[2]['x'] - landmarks[4]['x'] > 0.1
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_extended and index_up and others_curled else 0.0

    def _check_m_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'M'."""
        # Thumb tucked under folded fingers
        thumb_tucked = landmarks[4]['y'] > landmarks[2]['y']
        
        # Check if fingers are folded
        fingers_folded = True
        for i in [8, 12, 16]:  # finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                fingers_folded = False
                break
        
        return 0.9 if thumb_tucked and fingers_folded else 0.0

    def _check_n_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'N'."""
        # Similar to M but with fewer fingers
        thumb_tucked = landmarks[4]['y'] > landmarks[2]['y']
        
        # Check if index and middle are folded
        index_folded = landmarks[8]['y'] > landmarks[6]['y']
        middle_folded = landmarks[12]['y'] > landmarks[10]['y']
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_tucked and index_folded and middle_folded and others_curled else 0.0

    def _check_o_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'O'."""
        # Thumb and index forming O shape
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if thumb and index are close
        o_shape = ((thumb_tip['x'] - index_tip['x'])**2 + 
                  (thumb_tip['y'] - index_tip['y'])**2) < 0.01
        
        # Check if other fingers are extended
        others_extended = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] > landmarks[i-2]['y']:
                others_extended = False
                break
        
        return 0.9 if o_shape and others_extended else 0.0

    def _check_p_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'P'."""
        # Thumb between index and middle pointing down
        thumb_down = landmarks[4]['y'] > landmarks[2]['y']
        index_extended = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_extended = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_down and index_extended and middle_extended and others_curled else 0.0

    def _check_q_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'Q'."""
        # Thumb and index pointing down
        thumb_down = landmarks[4]['y'] > landmarks[2]['y']
        index_down = landmarks[8]['y'] > landmarks[6]['y']
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_down and index_down and others_curled else 0.0

    def _check_r_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'R'."""
        # Index and middle crossed
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_up = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if fingers are crossed
        crossed = abs(landmarks[8]['x'] - landmarks[12]['x']) < 0.05
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_up and middle_up and crossed and others_curled else 0.0

    def _check_s_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'S'."""
        # Fist with thumb wrapped across front of fingers
        thumb_wrapped = landmarks[4]['x'] > landmarks[2]['x']
        
        # Check if all fingers are curled
        fingers_curled = True
        for i in [8, 12, 16, 20]:  # finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                fingers_curled = False
                break
        
        return 0.9 if thumb_wrapped and fingers_curled else 0.0

    def _check_t_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'T'."""
        # Thumb between index and middle
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if thumb is between index and middle
        thumb_between = abs(thumb_tip['x'] - index_tip['x']) < 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_between and others_curled else 0.0

    def _check_u_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'U'."""
        # Index and middle extended together
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_up = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if fingers are together
        together = abs(landmarks[8]['x'] - landmarks[12]['x']) < 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_up and middle_up and together and others_curled else 0.0

    def _check_v_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'V'."""
        # Index and middle in V shape
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_up = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        
        # Check if fingers are spread apart
        spread_apart = abs(landmarks[8]['x'] - landmarks[12]['x']) > 0.1
        
        # Check if other fingers are curled
        others_curled = True
        for i in [16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_up and middle_up and spread_apart and others_curled else 0.0

    def _check_w_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'W'."""
        # Three fingers spread
        index_up = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        middle_up = landmarks[9]['y'] - landmarks[12]['y'] > 0.2
        ring_up = landmarks[13]['y'] - landmarks[16]['y'] > 0.2
        
        # Check if fingers are spread apart
        spread_apart = (abs(landmarks[8]['x'] - landmarks[12]['x']) > 0.1 and
                       abs(landmarks[12]['x'] - landmarks[16]['x']) > 0.1)
        
        # Check if pinky is curled
        pinky_curled = landmarks[20]['y'] > landmarks[18]['y']
        
        return 0.9 if index_up and middle_up and ring_up and spread_apart and pinky_curled else 0.0

    def _check_x_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'X'."""
        # Index bent at middle joint
        index_bent = landmarks[8]['y'] > landmarks[6]['y'] and landmarks[6]['y'] < landmarks[5]['y']
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_bent and others_curled else 0.0

    def _check_y_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'Y'."""
        # Thumb and pinky extended, others curled
        thumb_extended = landmarks[2]['y'] - landmarks[4]['y'] > 0.2
        pinky_extended = landmarks[17]['y'] - landmarks[20]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [8, 12, 16]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if thumb_extended and pinky_extended and others_curled else 0.0

    def _check_z_gesture(self, landmarks: Dict) -> float:
        """Check if hand position matches ASL letter 'Z'."""
        # Index finger extended, others curled
        # Z is a motion letter, so this is an approximation for a single frame
        index_extended = landmarks[5]['y'] - landmarks[8]['y'] > 0.2
        
        # Check if other fingers are curled
        others_curled = True
        for i in [12, 16, 20]:  # other finger tips
            if landmarks[i]['y'] < landmarks[i-2]['y']:
                others_curled = False
                break
        
        return 0.9 if index_extended and others_curled else 0.0
