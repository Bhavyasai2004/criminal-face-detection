# Real-Time Criminal Face Detection & Alert Web App

A high-performance security operations web application built with a **Python Flask** backend running on port `5002`, utilizing **OpenCV Haar Cascade Classifiers**, **NumPy** matrix math, and a **Tailwind CSS** dark-mode dashboard for live suspect watchlist identification.

---

## 🏗️ Architecture & Pipeline Overview

1. **Face Detection (`cv2.CascadeClassifier`):**
   - Utilizes `haarcascade_frontalface_default.xml` to detect human facial regions in real-time camera frames.
2. **Feature Extraction & Normalization:**
   - Detected face bounding boxes are cropped and converted to grayscale.
   - Resized to a standard `64x64` resolution.
   - Processed with histogram equalization (`cv2.equalizeHist`) to compensate for lighting variations.
   - Flattened into a 1D vector and **L2-normalized** to ensure invariant vector comparisons.
3. **Cosine Similarity Matching:**
   - Compares query vectors against all enrolled suspect vectors in the in-memory `WATCHLIST` using the dot product (`float(np.dot(a, b))`).
   - If similarity exceeds the match threshold (`0.65`), a security alert is triggered with a confidence score.

---

## ✨ Key Features

* **Live Video Feed & Canvas Overlay:** Draws green bounding boxes for unmatched/clear faces and red alert boxes with suspect metadata for watchlist matches.
* **Enroll Suspect from Webcam Form:** Instantly capture and register a new suspect's name and case ID into the active watchlist.
* **Real-Time Detection & Alert Log:** Live updating log recording detection timestamps, matched suspect names, and confidence percentages.
* **Latency & System Stats:** Live pipeline latency metrics (`ms`) and active face count displays.

---

## 🚀 Setup & Installation Instructions

### 1. Project Directory Structure
Ensure both files are placed in your root directory:

    project-root/
    ├── app.py
    ├── index.html
    └── README.md

### 2. Install Dependencies
Make sure Python is installed, then install Flask, OpenCV, and NumPy:

    pip install flask opencv-python numpy

### 3. Run the Application
Start the Flask backend server on port `5002`:

    python app.py

### 4. Open in Browser
Navigate to the local server URL in any modern browser with webcam permissions enabled:

    http://127.0.0.1:5002