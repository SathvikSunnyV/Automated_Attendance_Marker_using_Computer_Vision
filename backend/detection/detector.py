from retinaface import RetinaFace
import numpy as np
from config import MIN_FACE_SIZE

def detect_faces(image):
    """
    RetinaFace-based real face detector.
    Returns list of bounding boxes + landmarks.
    """

    detections = RetinaFace.detect_faces(image)

    results = []

    if detections is None:
        return results

    for _, face in detections.items():
        x1, y1, x2, y2 = face["facial_area"]

        # size filter
        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        landmarks = face.get("landmarks", None)

        results.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(face["score"]),
            "landmarks": landmarks
        })

    return results
