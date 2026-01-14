import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage", "uploads")
EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, "storage", "embeddings")
ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "storage", "attendance_logs")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MIN_FACE_SIZE = 60  # pixels
