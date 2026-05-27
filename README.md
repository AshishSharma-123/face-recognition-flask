# 🎭 FACE — Real-Time Face Recognition Web App

A Python + Flask web application that uses your webcam to **capture**, **train**, and **recognize** faces in real time using OpenCV's LBPH algorithm.

---

## 📸 Overview

FACE is a local face recognition system with a simple web interface. You can register a person by capturing their face via webcam, train a model on those images, and then run live recognition — all from your browser.

---

## ✨ Features

- 👤 **User Authentication** — Sign up and sign in with a session-based flow
- 📷 **Dataset Creation** — Capture up to 200 face images per person via webcam
- 🧠 **Model Training** — Train an LBPH face recognizer on collected images
- 🔍 **Live Recognition** — Real-time face detection and identification with confidence scores
- ❓ **Unknown Detection** — Faces below the confidence threshold are labeled as UNKNOWN

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask (Python) |
| Face Detection | OpenCV Haar Cascades |
| Face Recognition | OpenCV LBPH Recognizer |
| Image Processing | PIL (Pillow), NumPy |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
FACE/
├── app.py                            # Flask routes and application entry point
├── face_recognition.py               # Core logic: dataset, training, recognition
├── classifier.yml                    # Saved trained model (generated after training)
├── label_mappings.txt                # Person name → numeric ID mapping
├── haarcascade_frontalface_default.xml  # Pre-trained Haar Cascade face detector
├── data/                             # Captured face images (organized by person)
│   └── <person_name>/
│       └── *.jpg
├── templates/
│   ├── index.html                    # Landing page
│   ├── signup.html                   # Registration page
│   ├── signin.html                   # Login page
│   └── dashboard.html                # Main dashboard with action buttons
└── static/
    ├── css/
    │   ├── style.css
    │   ├── auth.css
    │   └── dashboard.css
    └── js/
        └── dashboard.js
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- A working webcam
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/FACE.git
cd FACE

# 2. Install dependencies
pip install flask opencv-python opencv-contrib-python pillow numpy

# 3. Run the app
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

> ⚠️ `opencv-contrib-python` is required for the LBPH face recognizer. Do not use `opencv-python` alone.

---

## 🚀 How to Use

### Step 1 — Sign Up / Sign In
Go to the home page, create an account, or sign in with any username and password.

### Step 2 — Create Dataset
On the dashboard, click **"Create Dataset"** and enter a person's name. Your webcam will open and automatically capture up to **200 face images**, saving them to `data/<name>/`.

- Move your face slightly during capture for better variety
- Ensure good lighting (dark faces are skipped automatically)
- Press **ESC** to stop early

### Step 3 — Train Model
Click **"Train Model"** on the dashboard. The app will scan all images in `data/`, train the LBPH recognizer, and save:
- `classifier.yml` — the trained model
- `label_mappings.txt` — name-to-ID mappings

Repeat Steps 2–3 for each person you want to recognize.

### Step 4 — Recognize Face
Click **"Recognize Face"** to open a live webcam feed. The app will draw bounding boxes around detected faces and display:
- **Name + Confidence %** if confidence is above 77%
- **UNKNOWN** if the face doesn't match anyone in the model

---

## 🔄 How It Works

```
Webcam Input
     │
     ▼
Haar Cascade Detector  ──→  Detects face region (bounding box)
     │
     ▼
Grayscale + Resize (200×200)
     │
     ▼
  [Training Mode]          [Recognition Mode]
Save to data/<name>/       LBPH Recognizer.predict()
         │                         │
         ▼                         ▼
  LBPH Recognizer.train()   Confidence Score > 77%?
         │                     Yes → Show Name
         ▼                     No  → Show UNKNOWN
  classifier.yml saved
```

---

## 🧩 API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/signup` | GET, POST | User registration |
| `/signin` | GET, POST | User login |
| `/dashboard` | GET | Main dashboard |
| `/create-dataset` | POST | Starts webcam face capture |
| `/train-model` | GET | Trains the LBPH classifier |
| `/recognize-face` | GET | Starts live face recognition |

---

## ⚠️ Known Limitations

- **No real database** — credentials are not verified or persisted; any username/password combination works on sign-in.
- **Webcam required** — all features depend on a connected webcam. This app cannot be deployed as a remote web service.
- **Single machine only** — training and recognition run on the server's local webcam, so this is designed for local use only.
- **Minimum training data** — at least 10 usable images per person are recommended for reliable recognition.

---

## 💡 Tips for Better Accuracy

- Capture faces in **different lighting conditions**
- Include **slight head rotations** (left, right, up, down)
- Aim for at least **100+ samples** per person
- Ensure your face is **close enough** to the camera (takes up at least 20% of frame height)
- Retrain the model after adding new people

---

## 📄 License

This project is for educational and personal use. Feel free to fork and extend it.
