

# test_model.py
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# Load model
# model = load_model('action.h5')
# model = load_model('optimized_sign_language_model3.h5')#problem in j,many problem
# model = load_model('optimized_sign_language_model1.h5')#problem in 8;

# model = load_model('optimized_sign_language_model2.h5')#problem in j
# model = load_model('optimized_sign_language_model4.h5')#PROBLEM IN E
# model = load_model('optimized_sign_language_model5.h5')
# model = load_model('optimized_sign_language_model6.h5')#2,6,V
model = load_model('optimized_sign_language_model9.h5')
# actions = np.array([,'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
# actions = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E'])
actions = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'])


# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw left and right hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Keypoint extraction (reuse functions from data_collection.py)
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Real-time detection
sequence = []
threshold = 0.8  # Confidence threshold

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)
    
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Ensure sequence length is 30
    
    if len(sequence) == 30:
        # Check if any hand is detected
        if np.any(results.left_hand_landmarks) or np.any(results.right_hand_landmarks):
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = actions[np.argmax(res)]
            confidence = np.max(res)
            
            if confidence > threshold:
                cv2.putText(image, f'{action} ({confidence:.2f})', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            # Clear the sequence if no hand is detected
            sequence = []
    
    cv2.imshow('Action Detection', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

