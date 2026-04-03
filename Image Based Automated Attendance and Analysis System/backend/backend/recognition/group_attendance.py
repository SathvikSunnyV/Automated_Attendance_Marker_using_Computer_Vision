import cv2
import numpy as np
import json

from database.db import SessionLocal
from database.models import Embedding, Student
from recognition.face_matching import recognize_face
from recognition.model_loader import get_face_app


# cache embeddings in memory
embedding_cache = None


def load_embeddings_from_db(class_id):

    db = SessionLocal()

    records = db.query(Embedding).all()
    print("Embeddings loaded:", len(records))

    database = {}

    for r in records:
        database[r.student_id] = np.array(json.loads(r.embedding_vector),dtype=np.float32)

    

    db.close()

    return database


def get_all_students(class_id):
    db = SessionLocal()
    students = db.query(Student).all()
    db.close()
    return students


def process_group_image_bytes(image_bytes, class_id):

    face_app = get_face_app()   # ✅ correct place

    # convert bytes -> image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    faces = face_app.get(img)
    

    database = load_embeddings_from_db(class_id)
    

    recognized = {}

    # detect faces
    for face in faces:

        embedding = face.normed_embedding.reshape(1, -1)

        student_id, score = recognize_face(embedding, database)
       

        if student_id is not None:

            if student_id not in recognized or score > recognized[student_id]:
                recognized[student_id] = float(score)

    students = get_all_students(class_id)

    attendance_list = []

    for student in students:

        if student.id in recognized:

            attendance_list.append({
                "student_id": student.id,
                "roll_no": student.roll_no,
                "name": student.name,
                "status": "Present"
            })

        else:

            attendance_list.append({
                "student_id": student.id,
                "roll_no": student.roll_no,
                "name": student.name,
                "status": "Absent"
            })

    return attendance_list