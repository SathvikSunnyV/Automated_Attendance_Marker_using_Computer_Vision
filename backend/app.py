import os
from flask import Flask, request, jsonify
from config import UPLOAD_FOLDER, EMBEDDINGS_FOLDER, ATTENDANCE_FOLDER
from utils.image_utils import allowed_file, load_image
from detection.detector import detect_faces
from alignment.aligner import align_face
from recognition.embedder import generate_embedding
from recognition.matcher import match_face
from attendance.attendance import mark_attendance
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = load_image(filepath)
    detections = detect_faces(image)

    results = []

    for det in detections:
        face = align_face(image, det)
        embedding = generate_embedding(face)
        identity, score = match_face(embedding)

        if identity:
            mark_attendance(identity)

        results.append({
            "bbox": {
                "x1": int(det["x1"]),
                "y1": int(det["y1"]),
                "x2": int(det["x2"]),
                "y2": int(det["y2"]),
                "confidence": float(det["confidence"])
            },
            "identity": identity,
            "similarity": float(score) if score is not None else 0.0
        })


    return jsonify({
        "faces_detected": len(results),
        "results": results
    })

@app.route("/enroll", methods=["POST"])
def enroll_face():
    if "image" not in request.files or "identity" not in request.form:
        return jsonify({"error": "Image and identity are required"}), 400

    identity = request.form["identity"].strip()

    if identity == "":
        return jsonify({"error": "Identity cannot be empty"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = load_image(filepath)
    detections = detect_faces(image)

    if len(detections) == 0:
        return jsonify({"error": "No face detected"}), 400

    embeddings = []

    for det in detections:
        face = align_face(image, det)
        embedding = generate_embedding(face)
        embeddings.append(embedding)

    final_embedding = np.mean(embeddings, axis=0)

    save_path = os.path.join(EMBEDDINGS_FOLDER, f"{identity}.npy")
    np.save(save_path, final_embedding)

    return jsonify({
        "message": f"{identity} enrolled successfully",
        "faces_used": len(embeddings)
    })



if __name__ == "__main__":
    app.run(debug=True)
