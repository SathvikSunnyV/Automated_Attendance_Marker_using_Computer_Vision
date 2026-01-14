import numpy as np

def generate_embedding(face_image):
    """
    Dummy embedding generator.
    Returns fixed-size vector.
    """

    np.random.seed(face_image.size % 12345)
    embedding = np.random.rand(128)
    return embedding
