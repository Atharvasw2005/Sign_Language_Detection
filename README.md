# Sign_Language_Detection
# 🤟 Sign Language Detection System

An AI-powered real-time sign language recognition system that helps bridge communication gaps for the hearing and speech impaired. It supports both **Indian Sign Language (A-Z, 0-9)** and **Custom Phrases Mode** for personalized gesture recognition.

---

## 🚀 Features

- 🔤 **ISL Mode** – Recognize hand signs for alphabets (A–Z) and digits (0–9).
- 💬 **Custom Mode** – Recognize predefined full-sentence gestures (e.g., “Hello how are you”).
- 🎥 **Real-time Video Processing** – Uses webcam to capture live gestures.
- ✋ **Hand Landmark Detection** – Extracts 21 key points from the hand using MediaPipe.
- 🎯 **High Confidence Prediction** – Displays output only when model confidence > 80%.
- 🗣️ **Text and Speech Output** – Converts detected signs to both text and audio.
- 🌐 **User-Friendly Web Interface** – Simple, intuitive, and responsive UI.

---

## 🖥️ Demo Screenshots

### 🔵 Home Page
- Mode selection for ISL or Custom Sign Mode.
- Clean and modern UI.

### 🟢 ISL Mode
- Detects and displays single characters with landmark overlay.
- Shows confidence and predicted output in real-time.

### 🟣 Custom Mode
- Detects full phrases/sentences based on trained custom gestures.
- Enables expressive and practical communication.

---

## 🛠️ Tech Stack

| Layer           | Technology                         |
|----------------|-------------------------------------|
| Frontend       | HTML, CSS, JavaScript               |
| Backend        | Python, Flask                       |
| ML Framework   | TensorFlow / Keras                  |
| Hand Detection | MediaPipe                           |
| Text-to-Speech | pyttsx3 / gTTS                      |
| Deployment     | Localhost / Flask Server            |

---

## 📁 Project Structure

UPDATED_SIGN_LANGUAGE_APP/
│
├── static/                           # Static assets (CSS, JS, icons, images)
│
├── templates/                        # HTML template files (used by Flask)
│
├── main.py                           # Main Python script to run the Flask app
│
├── optimized_sign_language_model9_1.h5    # Trained ML model for ISL mode
│
├── optimized_sign_language_model9.h5      # Trained ML model for Custom mode
│
└── requirements.txt                 # List of dependencies for the project


Output:

Home Page:
![image](https://github.com/user-attachments/assets/4009b584-8083-4cbd-be8f-28e3675954a2)

Indian Sign Language
![image](https://github.com/user-attachments/assets/3c7005cc-4279-44fc-8491-223965d3c16b)

Custome Sign LAnguage
![image](https://github.com/user-attachments/assets/a280e704-d577-4120-a246-e81f6e654aae)

**Report Link:**
https://drive.google.com/file/d/1hQFNdblFF9xRdDokblXPWGcuvxrJeYzi/view?usp=drive_link




