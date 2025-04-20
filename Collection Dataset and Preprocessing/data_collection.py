# data_collection.py
import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Keypoint extraction functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw left and right hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Data collection parameters
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['R'])
no_sequences = 30
sequence_length = 30

# Create folders
for action in actions: 
    for sequence in range(no_sequences+1):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Capture data
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
                    cv2.waitKey(500)
                
                cv2.putText(image, f'Collecting {action} Video {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                cv2.imshow('OpenCV Feed', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()