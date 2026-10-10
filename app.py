import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder=".")

# In-memory Watchlist / Matchlist
# List of dicts: { "name": str, "case_id": str, "vector": np.ndarray }
WATCHLIST = []

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def decode_base64_image(data_url):
    """Decodes a base64 data URL into an OpenCV BGR image."""
    try:
        if "," in data_url:
            header, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_normalized_feature(face_roi_gray):
    """Crops, equalizes, flattens, and L2-normalizes a face into a feature vector."""
    try:
        # Resize to fixed standard shape 64x64
        resized = cv2.resize(face_roi_gray, (64, 64))
        # Histogram Equalization for lighting normalization
        equalized = cv2.equalizeHist(resized)
        # Flatten and convert to float32
        flat = equalized.flatten().astype(np.float32)
        # L2 Normalization
        norm = np.linalg.norm(flat)
        if norm > 0:
            normalized = flat / norm
        else:
            normalized = flat
        return normalized
    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/enroll", methods=["POST"])
def enroll():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON payload provided."}), 400

    name = data.get("name", "").strip()
    case_id = data.get("case_id", "").strip()
    image_data = data.get("image", "")

    if not name or not case_id or not image_data:
        return jsonify({"status": "error", "message": "Missing name, case_id, or image."}), 400

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image data."}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected in the frame."}), 400

    # Pick the largest face by area (w * h)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    face_roi = gray[y:y+h, x:x+w]

    vector = extract_normalized_feature(face_roi)
    if vector is None:
        return jsonify({"status": "error", "message": "Failed to extract face features."}), 500

    # Store in Watchlist
    WATCHLIST.append({
        "name": name,
        "case_id": case_id,
        "vector": vector
    })

    return jsonify({
        "status": "enrolled",
        "count": len(WATCHLIST),
        "message": f"Successfully enrolled {name} (Case ID: {case_id})."
    })

@app.route("/api/detect", methods=["POST"])
def detect():
    start_time = time.time()
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"faces": [], "latency_ms": 0.0}), 400

    img = decode_base64_image(data["image"])
    if img is None:
        return jsonify({"faces": [], "latency_ms": 0.0}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result_faces = []
    match_threshold = 0.65  # Cosine similarity threshold

    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y+h, x:x+w]
        query_vector = extract_normalized_feature(face_roi)

        matched = False
        matched_name = "Unknown"
        matched_case_id = ""
        highest_similarity = 0.0

        if query_vector is not None and len(WATCHLIST) > 0:
            for suspect in WATCHLIST:
                # Cosine similarity between L2 normalized vectors is simply their dot product
                sim = float(np.dot(query_vector, suspect["vector"]))
                if sim > highest_similarity:
                    highest_similarity = sim

            if highest_similarity >= match_threshold:
                # Find best matching record
                best_match = max(WATCHLIST, key=lambda s: float(np.dot(query_vector, s["vector"])))
                matched = True
                matched_name = best_match["name"]
                matched_case_id = best_match["case_id"]

        result_faces.append({
            "box": [int(x), int(y), int(w), int(h)],
            "matched": bool(matched),
            "name": str(matched_name),
            "case_id": str(matched_case_id),
            "confidence": float(round(highest_similarity * 100, 2))
        })

    latency_ms = (time.time() - start_time) * 1000.0

    return jsonify({
        "faces": result_faces,
        "latency_ms": round(latency_ms, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)