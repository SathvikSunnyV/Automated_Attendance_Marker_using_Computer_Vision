import numpy as np
import os
from config import EMBEDDINGS_FOLDER

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def match_face(embedding, threshold=0.8):
    """
    Open-set face recognition.
    Returns (identity, score) or (None, best_score)
    """

    if embedding is None:
        return None, 0.0

    best_match = None
    best_score = -1.0

    if not os.path.isdir(EMBEDDINGS_FOLDER):
        return None, 0.0

    for item in os.listdir(EMBEDDINGS_FOLDER):
        if not item.endswith(".npy"):
            continue

        path = os.path.join(EMBEDDINGS_FOLDER, item)
        if not os.path.isfile(path):
            continue

        stored_embedding = np.load(path)

        score = cosine_similarity(embedding, stored_embedding)

        if score > best_score:
            best_score = score
            best_match = item.replace(".npy", "")

    # CRITICAL UNKNOWN REJECTION
    if best_score < threshold:
        return None, float(best_score)

    return best_match, float(best_score)
