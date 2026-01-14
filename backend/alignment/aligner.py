def align_face(image, bbox):
    """
    Placeholder alignment.
    Crops face using bounding box.
    """

    x1, y1 = bbox["x1"], bbox["y1"]
    x2, y2 = bbox["x2"], bbox["y2"]

    face = image[y1:y2, x1:x2]
    return face
