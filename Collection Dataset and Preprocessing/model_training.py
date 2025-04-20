
# model_training.py (Optimized)
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load dataset
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H','I','J','K',
                    'L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'])
sequence_length = 30  

sequences, labels = [], []
min_sequences = float('inf')  # To balance dataset

# Load data and find the minimum available sequences per action
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Skipping missing class: {action}")
        continue
    
    num_sequences = len(os.listdir(action_path))
    min_sequences = min(min_sequences, num_sequences)

# Load sequences with balanced classes
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    
    for sequence in range(min_sequences):  # Ensure equal samples for all classes
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}, skipping sequence.")
                continue
            
            res = np.load(file_path)
            window.append(res)

        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(np.where(actions == action)[0][0])

# Convert data to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Build improved LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 126)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(128, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained model
model.save('optimized_sign_language_model13.h5')
print("Model training completed and saved.")






# # model_training.py
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.utils import to_categorical

# # Load data
# # actions = np.array(['0','1','2','3','4','5','6','7','8','9','A'])
# actions = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'])
# # actions = np.array([])

# DATA_PATH = os.path.join('MP_Data')
# sequence_length = 30

# sequences, labels = [], []
# for action in actions:
#     for sequence in range(len(os.listdir(os.path.join(DATA_PATH, action)))):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
#             window.append(res)
#         sequences.append(window)
#         labels.append(np.where(actions == action)[0][0])

# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# # Build model
# model = Sequential([
#     LSTM(128, return_sequences=True, activation='tanh', input_shape=(30, 126)),
#     Dropout(0.3),
#     LSTM(256, return_sequences=True, activation='tanh'),
#     Dropout(0.3),
#     LSTM(128, activation='tanh'),
#     Dropout(0.3),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     #batch  // loveable AI //AWS Sagemaker  //  
#     Dense(actions.shape[0], activation='softmax')
# ])

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[TensorBoard(log_dir='Logs')])

# # Save model
# model.save('action_detection_model.h5')








# # model_training.py (Optimized)
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical

# # Load dataset
# DATA_PATH = os.path.join('MP_Data')
# actions = np.array(['0','1','2','3','4','5','6','7','8','9',
#                     'A','B','C','D','E','F','G','H','I','J','K',
#                     'L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'])
# sequence_length = 30  

# sequences, labels = [], []
# min_sequences = float('inf')  # To balance dataset

# # Load data and find the minimum available sequences per action
# for action in actions:
#     action_path = os.path.join(DATA_PATH, action)
#     if not os.path.exists(action_path):
#         print(f"Skipping missing class: {action}")
#         continue
    
#     num_sequences = len(os.listdir(action_path))
#     min_sequences = min(min_sequences, num_sequences)

# # Load sequences with balanced classes
# for action in actions:
#     action_path = os.path.join(DATA_PATH, action)
#     if not os.path.exists(action_path):
#         continue
    
#     for sequence in range(min_sequences):  # Ensure equal samples for all classes
#         window = []
#         for frame_num in range(sequence_length):
#             file_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
#             if not os.path.exists(file_path):
#                 print(f"Missing file: {file_path}, skipping sequence.")
#                 continue
            
#             res = np.load(file_path)
#             window.append(res)

#         if len(window) == sequence_length:
#             sequences.append(window)
#             labels.append(np.where(actions == action)[0][0])

# # Convert data to NumPy arrays
# X = np.array(sequences)
# y = to_categorical(labels, num_classes=len(actions)).astype(int)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # Build improved LSTM model
# model = Sequential([
#     LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 126)),
#     BatchNormalization(),
#     Dropout(0.2),

#     LSTM(128, return_sequences=True, activation='tanh'),
#     BatchNormalization(),
#     Dropout(0.2),

#     LSTM(64, activation='tanh'),
#     BatchNormalization(),
#     Dropout(0.2),

#     # Dense(512, activation='relu'),
#     # BatchNormalization(),
#     # Dropout(0.2),

#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(len(actions), activation='softmax'),
   
# ])
# model.summary();

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# # Train the model
# model.fit(X_train, y_train, epochs=800, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# # Save the trained model
# model.save('optimized_sign_language_model11.h5')
# print("Model training completed and saved.")
