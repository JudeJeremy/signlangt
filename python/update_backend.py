import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def update_hand_tracker():
    """
    Update the hand_tracker.py file to use the trained model
    """
    # Check if the model exists
    model_path = '../models/best_finetuned_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_model.py first to train the model.")
        return False
    
    # Create a backup of the original hand_tracker.py
    original_file = 'hand_tracker.py'
    backup_file = 'hand_tracker_backup.py'
    
    if not os.path.exists(backup_file):
        shutil.copy(original_file, backup_file)
        print(f"Created backup of {original_file} as {backup_file}")
    
    # Create a new function to recognize ASL gestures using the trained model
    asl_recognition_function = """
    def recognize_asl_with_model(self, hand_image):
        \"\"\"
        Recognize ASL gestures using the trained model
        
        Args:
            hand_image: Image of the hand
            
        Returns:
            Tuple of (recognized letter, confidence)
        \"\"\"
        try:
            # Ensure the model is loaded
            if not hasattr(self, 'asl_model'):
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_finetuned_model.h5')
                if os.path.exists(model_path):
                    print(f"Loading ASL model from {model_path}")
                    self.asl_model = tf.keras.models.load_model(model_path)
                else:
                    print(f"ASL model not found at {model_path}")
                    return None, 0.0
            
            # Preprocess the image
            img = cv2.resize(hand_image, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.asl_model.predict(img, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Convert class index to letter (A=0, B=1, etc.)
            letter = chr(65 + predicted_class)
            
            return letter, float(confidence)
            
        except Exception as e:
            print(f"Error recognizing ASL with model: {e}")
            return None, 0.0
    """
    
    # Modify the process_frame method to use the new model
    process_frame_update = """
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
                    
                    # Extract hand image for model-based recognition
                    h, w, c = frame.shape
                    x_min, y_min, x_max, y_max = w, h, 0, 0
                    
                    # Find bounding box of hand
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    
                    # Add padding to bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Extract hand image
                    hand_image = frame[y_min:y_max, x_min:x_max]
                    
                    if hand_image.size > 0:
                        # Recognize ASL gesture using the trained model
                        gesture, confidence = self.recognize_asl_with_model(hand_image)
                        
                        if gesture and confidence > 0.7:
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
                        # No valid hand image
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
    """
    
    # Read the original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Add the new function before the last line (which should be the class closing bracket)
    lines = content.split('\n')
    
    # Find the position to insert the new function (before the last line)
    insert_position = len(lines) - 1
    
    # Insert the new function
    lines.insert(insert_position, asl_recognition_function)
    
    # Replace the process_frame method
    process_frame_start = "        # If GestureRecognizer failed or is not available, use basic hand tracking"
    process_frame_end = "        # Add state machine status to frame for debugging"
    
    start_index = -1
    end_index = -1
    
    for i, line in enumerate(lines):
        if process_frame_start in line:
            start_index = i + 1
        elif process_frame_end in line and start_index != -1:
            end_index = i
            break
    
    if start_index != -1 and end_index != -1:
        # Replace the content between start_index and end_index
        lines[start_index:end_index] = process_frame_update.split('\n')
    else:
        print("Warning: Could not find the process_frame method to update.")
    
    # Write the updated content back to the file
    with open(original_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Updated {original_file} to use the trained model.")
    return True

def update_app_py():
    """
    Update app.py to use the new model
    """
    # Create a backup of the original app.py
    original_file = 'app.py'
    backup_file = 'app_backup.py'
    
    if not os.path.exists(backup_file):
        shutil.copy(original_file, backup_file)
        print(f"Created backup of {original_file} as {backup_file}")
    
    # Read the original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Update the content to add a message about the trained model
    model_message = """
# Check if the trained ASL model exists
asl_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_finetuned_model.h5')
if os.path.exists(asl_model_path):
    print(f"Found trained ASL model at {asl_model_path}")
    print("The hand tracker will use this model for ASL recognition.")
else:
    print("Trained ASL model not found. Using fallback recognition methods.")
"""
    
    # Insert the message after the hand tracker initialization
    insert_after = "print(\"Hand tracker initialized. Using MediaPipe GestureRecognizer:\", "
    
    if insert_after in content:
        # Find the end of the line
        insert_pos = content.find('\n', content.find(insert_after)) + 1
        
        # Insert the message
        content = content[:insert_pos] + model_message + content[insert_pos:]
        
        # Write the updated content back to the file
        with open(original_file, 'w') as f:
            f.write(content)
        
        print(f"Updated {original_file} to check for the trained model.")
        return True
    else:
        print(f"Warning: Could not find the insertion point in {original_file}.")
        return False

if __name__ == "__main__":
    print("Updating backend to use the trained model...")
    
    # Update hand_tracker.py
    if update_hand_tracker():
        print("Successfully updated hand_tracker.py")
    else:
        print("Failed to update hand_tracker.py")
    
    # Update app.py
    if update_app_py():
        print("Successfully updated app.py")
    else:
        print("Failed to update app.py")
    
    print("\nBackend update complete!")
    print("You can now run app.py to use the trained model for ASL recognition.")
