# attendance.py
import os
import argparse
import logging
import cv2
import numpy as np
import pandas as pd
import urllib.parse
import json

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    has_gpu_provider = any('CUDAExecutionProvider' in p for p in providers)
except Exception:
    providers = None
    has_gpu_provider = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class StaticAttendanceSystem:
    def __init__(self, db_folder, det_size=(640, 640), match_threshold=0.40, model_name="buffalo_l"):
        self.db_folder = os.path.abspath(os.path.expanduser(db_folder))
        self.det_size = tuple(det_size)
        self.match_threshold = float(match_threshold)
        self.model_name = model_name
        self.embeddings_file = os.path.join(self.db_folder, "students.json")   # ← persistent DB

        # insightface provider selection or fallback
        if has_gpu_provider:
            providers_arg = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            providers_arg = ['CPUExecutionProvider']
            ctx_id = -1

        # lazy import of insightface to avoid import-time heavy failures if model missing
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.insightface = insightface
            self.FaceAnalysis = FaceAnalysis
        except Exception:
            self.insightface = None
            self.FaceAnalysis = None

        # placeholder for the FaceAnalysis app; actual init is done in make_system
        self.app = None
        self.known_faces = {}
        self.load_database()

    def make_system(self, det_size=(640, 640), model_name="buffalo_l"):
        # initialize the insightface FaceAnalysis app if available
        if not self.FaceAnalysis:
            raise RuntimeError("insightface not available in the environment")

        self.det_size = tuple(det_size)
        self.model_name = model_name

        try:
            self.app = self.FaceAnalysis(name=self.model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_gpu_provider else ['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0 if has_gpu_provider else -1, det_size=self.det_size)
            logging.info("Initialized FaceAnalysis with model %s and det_size %s", self.model_name, self.det_size)
        except Exception as e:
            logging.warning("Failed to initialize FaceAnalysis: %s", e)
            # try CPU-only fallback
            try:
                self.app = self.FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=-1, det_size=self.det_size)
                logging.info("Initialized FaceAnalysis (CPU fallback) %s", self.model_name)
            except Exception as e2:
                logging.error("Cannot initialize FaceAnalysis: %s", e2)
                raise

    def load_database(self):
        os.makedirs(self.db_folder, exist_ok=True)
        self.known_faces = {}

        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for entry in data:
                    name = entry["name"]
                    emb = np.array(entry["embedding"], dtype=np.float32)
                    filename = entry.get("filename", f"{name}.jpg")
                    self.known_faces[name] = {"embedding": emb, "filename": filename}
                logging.info("Loaded %d students from students.json", len(self.known_faces))
            except Exception as e:
                logging.warning("Failed to load embeddings file: %s", e)

    def save_database(self):
        try:
            data = []
            for name, v in self.known_faces.items():
                data.append({"name": name, "embedding": v["embedding"].tolist(), "filename": v.get("filename")})
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logging.info("Saved %d students to students.json", len(self.known_faces))
        except Exception as e:
            logging.warning("Failed to save embeddings: %s", e)

    def register_student(self, name: str, image_path: str) -> str:
        """Called from frontend – extracts embedding + saves photo + persists"""
        name = name.strip()
        if not name:
            raise ValueError("Student name cannot be empty")

        # safe filename
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
        filename = f"{safe_name}.jpg"
        dest_path = os.path.join(self.db_folder, filename)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read uploaded photo")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the provided photo")

        faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces_sorted[0].embedding
        if np.linalg.norm(emb) != 0:
            emb = emb / np.linalg.norm(emb)

        # save photo to db folder (overwrite allowed)
        cv2.imwrite(dest_path, img)

        # persist embedding
        self.known_faces[name] = {"embedding": emb.astype(np.float32), "filename": filename}
        self.save_database()
        logging.info("Registered student %s", name)
        return filename

    def process_group_photo(self, image_path, output_path="output_attendance.jpg", csv_path=None, show=False):
        """
        Process a group photo, identify known students, draw boxes/labels and save
        the annotated image to output_path. Returns attendance list.

        This version performs a simple IoU-based deduplication (non-max style)
        of overlapping detections to avoid drawing multiple boxes for the same face.
        """
        def iou(boxA, boxB):
            # boxes are (x1, y1, x2, y2)
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
            areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
            union = areaA + areaB - interArea
            return (interArea / union) if union > 0 else 0.0

        # normalize & resolve paths
        image_path = os.path.abspath(os.path.expanduser(image_path))
        output_path = os.path.abspath(os.path.expanduser(output_path))
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(output_path), "attendance_report.csv")
        else:
            csv_path = os.path.abspath(os.path.expanduser(csv_path))

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        faces = self.app.get(img)
        logging.info("Detected %d faces in the group photo.", len(faces))

        # If detector returns no faces, just save and return empty attendance
        if not faces:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            return []

        # --- Deduplicate overlapping detections (simple NMS-like by IoU) ---
        face_items = []
        for f in faces:
            try:
                bbox = f.bbox.astype(int)
            except Exception:
                bbox = np.array(f.bbox, dtype=int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            face_items.append({"face": f, "bbox": (x1, y1, x2, y2), "area": area})

        # sort by area descending (keep larger box first)
        face_items.sort(key=lambda x: x["area"], reverse=True)

        kept = []
        IOU_THRESHOLD = 0.35   # conservative threshold to merge near-duplicates
        for item in face_items:
            bb = item["bbox"]
            skip = False
            for k in kept:
                if iou(bb, k["bbox"]) > IOU_THRESHOLD:
                    skip = True
                    break
            if not skip:
                kept.append(item)

        # Use kept list as the final faces to evaluate/annotate
        faces_to_use = [it["face"] for it in kept]
        logging.info("After deduplication: %d faces remain.", len(faces_to_use))

        attendance = []
        for face in faces_to_use:
            try:
                bbox = face.bbox.astype(int)
            except Exception:
                bbox = np.array(face.bbox, dtype=int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # normalize embedding
            emb = face.embedding
            if np.linalg.norm(emb) != 0:
                emb = emb / np.linalg.norm(emb)

            identified = "Unknown"
            best_score = -1.0
            best_filename = None

            for name, data in self.known_faces.items():
                try:
                    score = float(np.dot(emb, data["embedding"]))
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    identified = name
                    best_filename = data.get("filename")

            # choose color/label based on matching threshold
            if best_score >= self.match_threshold and best_filename:
                safe_url = f"/db/{urllib.parse.quote(best_filename)}"
                label = f"{identified} ({best_score:.2f})"
                color = (0, 200, 0)  # green-ish for matched
                attendance.append({
                    "Name": identified,
                    "Confidence": float(best_score),
                    "db_image_url": safe_url
                })
            else:
                label = f"Unknown ({best_score:.2f})"
                color = (0, 0, 255)  # red for unknown

            # draw the final, non-duplicated boxes and labels
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

        if attendance:
            try:
                df = pd.DataFrame([{"Name": a["Name"], "Confidence": a["Confidence"]} for a in attendance])
                df.to_csv(csv_path, index=False)
            except Exception as e:
                logging.warning("Failed to write CSV report: %s", e)

        if show:
            try:
                cv2.imshow("Attendance", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                logging.warning("Cannot display image window: %s", e)

        return attendance