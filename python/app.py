from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import os
from hand_tracker import HandTracker

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Import the model downloader
try:
    from download_model import download_gesture_recognizer_model
    
    # Download model if needed
    download_gesture_recognizer_model()
except Exception as e:
    print(f"Note: Model download script error: {e}")
    print("Will attempt to use MediaPipe's built-in model download or fall back to custom detection")

# Initialize hand tracker
hand_tracker = HandTracker()

print("Hand tracker initialized. Using MediaPipe GestureRecognizer:", 
      "Yes" if hand_tracker.use_gesture_recognizer else "No (falling back to custom detection)")

# Route to serve the main application
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to hand tracking server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_tracking')
def handle_start_tracking():
    # Try different camera indices if needed
    for camera_index in [1, 0, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        emit('error', {'message': 'Failed to open camera. Please check camera connection.'})
        return
    
    # FPS tracking variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Performance optimization variables
    frame_skip = 0
    max_frame_skip = 1  # Reduced to improve responsiveness
    jpeg_quality = 80   # Increased for better image quality
    processing_times = []
    
    # Debug information
    emit('status', {'message': f'Camera opened successfully. Starting hand tracking...'})
    
    try:
        while True:
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                
                # Adaptive frame skipping based on performance
                if len(processing_times) > 0:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    if avg_processing_time > 0.1:  # If processing takes more than 100ms
                        frame_skip = min(frame_skip + 1, max_frame_skip)
                    elif avg_processing_time < 0.05 and frame_skip > 0:  # If processing is fast
                        frame_skip = max(frame_skip - 1, 0)
                    processing_times = []
            
            # Process frame
            ret, frame = cap.read()
            if not ret:
                emit('error', {'message': 'Failed to read frame from camera'})
                break

            # Skip frames if needed for performance
            if frame_count % (frame_skip + 1) != 0:
                socketio.sleep(0.01)  # Short sleep on skipped frames
                continue

            try:
                # Measure processing time
                process_start = time.time()
                
                # Flip the frame horizontally for a more natural view (mirror-like)
                frame = cv2.flip(frame, 1)
                
                # Resize frame for faster processing (reduce to 60% size)
                frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                
                # Add a guide box to help with hand positioning
                height, width = frame.shape[:2]
                box_size = min(width, height) * 0.6
                x = int((width - box_size) / 2)
                y = int((height - box_size) / 2)
                cv2.rectangle(frame, (x, y), (x + int(box_size), y + int(box_size)), (255, 255, 255), 2)
                
                # Process frame and get results
                processed_frame, landmarks, gestures = hand_tracker.process_frame(frame)
                
                # Get gesture and confidence
                gesture_info = None
                if gestures and len(gestures) > 0:
                    # If using GestureRecognizer, gestures will be a list of letters
                    gesture = gestures[0]
                    
                    # Get confidence from the hand tracker
                    # For GestureRecognizer, this is already calculated
                    # For fallback mode, we'll calculate it here
                    confidence = 0.9  # Default high confidence for recognized gestures
                    
                    if landmarks and hand_tracker.use_gesture_recognizer == False:
                        # If using fallback mode, get confidence from the detector
                        _, conf = hand_tracker._detect_asl_gesture(landmarks[0])
                        confidence = conf
                    
                    gesture_info = {
                        'letter': gesture,
                        'confidence': round(confidence * 100, 2)
                    }
                
                # Add FPS text to the frame
                fps_text = f"Server FPS: {fps:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instructions to the frame
                cv2.putText(processed_frame, "Place hand in box", (width - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert frame to base64 for sending over WebSocket
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send results to client
                emit('frame', {
                    'frame': frame_base64,
                    'landmarks': landmarks,
                    'gesture': gesture_info,
                    'server_fps': round(fps, 1)
                })
                
                # Record processing time
                process_end = time.time()
                processing_times.append(process_end - process_start)
                
                # Adaptive sleep based on processing time
                process_time = process_end - process_start
                sleep_time = max(0.03, 0.08 - process_time)  # Target 12.5 FPS, but allow faster if possible
                socketio.sleep(sleep_time)
                
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                emit('error', {'message': 'Camera error: Failed to process video frame'})
                break
                
            except Exception as e:
                print(f"Error in tracking: {e}")
                emit('error', {'message': str(e)})
                break
                
    finally:
        try:
            cap.release()
        except:
            pass
        
        emit('status', {'message': 'Hand tracking stopped'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
