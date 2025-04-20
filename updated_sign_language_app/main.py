from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os

app = Flask(__name__)

# Load models
try:
    # Use custom_objects to handle the 'time_major' parameter in LSTM layers
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM
    
    # Custom LSTM layer that ignores the 'time_major' parameter
    class CustomLSTM(LSTM):
        def __init__(self, *args, **kwargs):
            # Remove 'time_major' from kwargs if present
            if 'time_major' in kwargs:
                kwargs.pop('time_major')
            super().__init__(*args, **kwargs)
    
    # Use the custom LSTM when loading the models
    custom_objects = {'LSTM': CustomLSTM}
    
    isl_model = load_model('optimized_sign_language_model9.h5', custom_objects=custom_objects)
    custom_model = load_model('optimized_sign_language_model9_1.h5', custom_objects=custom_objects)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # Create empty models for development if loading fails
    import tensorflow as tf
    isl_model = tf.keras.Sequential()
    custom_model = tf.keras.Sequential()

# Define actions for each model
isl_actions = np.array(['0','1','2','3','4','5','6','7','8','9',
                        'A','B','C','D','E','F','G','H','I','J',
                        'K','L','M','N','P','Q','R','S','T','U',
                        'V','W','X','Y','Z'])
custom_actions = np.array(['Good Bye', 'Hello how are you', 'I am fine', 'I am happy', 'Thank you', 'What is your Name'])

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global variable to store final detected text (used by both modes)
final_detected = ""

def mediapipe_detection(image, model):
    #"""Detects landmarks using MediaPipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
  #  """Extracts keypoints from MediaPipe results."""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def generate_frames(model, actions):
  #  """Captures video, processes frames, and performs sign detection with a 5-second delay for stable output."""
    global final_detected
    
    # Force the camera index to 0 (built-in webcam)
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    print(f"Camera opened status: {cap.isOpened()}")
    
    # Check camera properties
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {width}x{height}")
    
    # Initialize MediaPipe holistic model
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    sequence = []
    threshold = 0.8  
    previous_action = None  
    action_start_time = time.time()  # Track when the action was first detected

    while True:  # Changed from cap.isOpened() to always run
        # Try to read a frame
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            # Try to reopen the camera if it was closed
            cap = cv2.VideoCapture(0)
            ret = False
        
        if not ret:
            # If frame cannot be read, create a placeholder image
            image = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
            # Add text to inform user about camera issue
            cv2.putText(image, "Camera not available", (120, 240), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Please check camera connection", (80, 280), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Return the placeholder image
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)  # Wait a bit before trying again
            continue

        try:
            # Process the frame with MediaPipe
            image, results = mediapipe_detection(frame, holistic)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Extract keypoints and update sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep last 30 frames

            if len(sequence) == 30:
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    action = actions[np.argmax(res)]
                    confidence = np.max(res)

                    # Only update if confidence is high enough
                    if confidence > threshold:
                        if action == previous_action:
                            # If the same action is detected for 2.5 seconds, update final_detected
                            if time.time() - action_start_time >= 2.5:
                                final_detected = action  # Update final detected text
                                action_start_time = time.time()  # Reset timer after updating text field
                        else:
                            # Reset timer if a new action is detected
                            previous_action = action
                            action_start_time = time.time()

                        # Display the detected action on the video feed
                        cv2.putText(image, f'{action} ({confidence:.2f})', (15, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing frame: {e}")
            # If there's an error processing the frame, create a placeholder image
            image = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(image, "Error processing video", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the processed image and yield it as a response
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/isl')
def isl():
    return render_template('isl.html')

@app.route('/custom')
def custom():
    return render_template('custom.html')

@app.route('/video_feed_isl')
def video_feed_isl():
    return Response(generate_frames(isl_model, isl_actions), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_custom')
def video_feed_custom():
    return Response(generate_frames(custom_model, custom_actions), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to return the final detected text as JSON
@app.route('/get_text')
def get_text():
    global final_detected
    return jsonify({"text": final_detected})

# Endpoint to clear the final detected text
@app.route('/clear_text', methods=['POST'])
def clear_text():
    global final_detected
    final_detected = ""
    return jsonify({"message": "Text cleared"})

if __name__ == "__main__":
    app.run(debug=True)
