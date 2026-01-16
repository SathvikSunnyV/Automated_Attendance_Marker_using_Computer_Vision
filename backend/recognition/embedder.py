import numpy as np
import cv2
from keras_facenet import FaceNet

# Load FaceNet once (VERY IMPORTANT)
embedder = FaceNet()

def generate_embedding(face_image):
    """
    Generate a 512-D FaceNet embedding from an aligned face.
    """

    if face_image is None:
        return None

    # FaceNet expects RGB, 160x160
    face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))

    # normalize
    face = face.astype("float32")
    mean, std = face.mean(), face.std()
    face = (face - mean) / (std + 1e-8)

    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)[0]

    # L2 normalize (CRITICAL)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding
