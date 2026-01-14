import numpy as np

def detect_faces(image):
    """
    Temporary placeholder detector.
    Returns fake bounding boxes to test pipeline.
    """

    h, w, _ = image.shape

    # Fake: assume one face in center
    bbox = {
        "x1": int(w * 0.3),
        "y1": int(h * 0.3),
        "x2": int(w * 0.7),
        "y2": int(h * 0.7),
        "confidence": 0.99,
        "landmarks": None
    }

    return [bbox]
