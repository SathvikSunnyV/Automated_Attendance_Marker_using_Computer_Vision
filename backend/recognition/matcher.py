import numpy as np
import os
from config import EMBEDDINGS_FOLDER

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def match_face(embedding, threshold=0.7):
    best_match = None
    best_score = 0.0

    if not os.path.isdir(EMBEDDINGS_FOLDER):
        return None, 0.0

    for item in os.listdir(EMBEDDINGS_FOLDER):
        path = os.path.join(EMBEDDINGS_FOLDER, item)

        # 🔒 ABSOLUTE SAFETY
        if not os.path.isfile(path):
            continue
        if not item.endswith(".npy"):
            continue

        try:
            stored_embedding = np.load(path)
        except Exception:
            continue

        score = cosine_similarity(embedding, stored_embedding)

        if score >= threshold and score > best_score:
            best_score = score
            best_match = item.replace(".npy", "")

    return best_match, best_score
