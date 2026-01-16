import cv2
import numpy as np

def align_face(image, bbox, output_size=(160, 160)):
    """
    Align face using eye landmarks.
    Falls back to crop if landmarks are missing.
    """

    landmarks = bbox.get("landmarks", None)

    # fallback: simple crop
    if landmarks is None:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        face = image[y1:y2, x1:x2]
        return cv2.resize(face, output_size)

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    # convert to numpy
    left_eye = np.array(left_eye, dtype="float32")
    right_eye = np.array(right_eye, dtype="float32")

    # compute angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # center between eyes
    eyes_center = (
        int((left_eye[0] + right_eye[0]) // 2),
        int((left_eye[1] + right_eye[1]) // 2)
    )

    # rotation
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # transform bounding box
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    face = rotated[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return cv2.resize(face, output_size)
