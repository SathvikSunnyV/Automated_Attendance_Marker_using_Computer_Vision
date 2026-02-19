# attendance.py
import os
import argparse
import logging
import cv2
import numpy as np
import pandas as pd

# Attempt to detect onnxruntime GPU provider (optional)
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    has_gpu_provider = any('CUDAExecutionProvider' in p for p in providers)
except Exception:
    providers = None
    has_gpu_provider = False

# insightface
import insightface
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class StaticAttendanceSystem:
    def __init__(self, db_folder, det_size=(640, 640), match_threshold=0.40, model_name="buffalo_l"):
        """
        db_folder: folder containing registration images (filename without extension -> identity)
        det_size: detection size; reduce if VRAM is limited
        match_threshold: cosine similarity threshold (0-1)
        model_name: insightface model name (keep as original if it worked for you)
        """
        self.db_folder = os.path.abspath(os.path.expanduser(db_folder))
        self.det_size = tuple(det_size)
        self.match_threshold = float(match_threshold)
        self.model_name = model_name

        # Decide providers & context id for insightface
        if has_gpu_provider:
            providers_arg = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            providers_arg = ['CPUExecutionProvider']
            ctx_id = -1

        logging.info("Initializing FaceAnalysis (model=%s) with providers=%s ctx_id=%s",
                     self.model_name, providers_arg, ctx_id)
        # Create FaceAnalysis app
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=providers_arg)
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        except Exception as e:
            logging.error("Failed to initialize FaceAnalysis: %s", e)
            raise

        self.known_embeddings = {}
        self.load_database(self.db_folder)

    def load_database(self, db_folder):
        logging.info("Loading database from: %s", db_folder)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder, exist_ok=True)
            logging.warning("Created folder '%s'. Put registered face images there and rerun.", db_folder)
            return

        for fname in sorted(os.listdir(db_folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(db_folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning("Cannot read image: %s", img_path)
                continue

            faces = self.app.get(img)
            if len(faces) == 0:
                logging.warning("No face detected in %s", fname)
                continue

            # choose largest detected face (safer in multi-face registration images)
            faces_sorted = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            emb = faces_sorted[0].embedding
            norm = np.linalg.norm(emb)
            if norm == 0:
                logging.warning("Zero-length embedding for %s - skipping", fname)
                continue
            emb = emb / norm
            self.known_embeddings[name] = emb.astype(np.float32)
            logging.info("Loaded: %s (embedding length %d)", name, len(emb))

        logging.info("Total registered identities loaded: %d", len(self.known_embeddings))

    def process_group_photo(self, image_path, output_path="output_attendance.jpg", csv_path=None, show=False):
        image_path = os.path.abspath(os.path.expanduser(image_path))
        output_path = os.path.abspath(os.path.expanduser(output_path))
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(output_path), "attendance_report.csv")
        else:
            csv_path = os.path.abspath(os.path.expanduser(csv_path))

        img = cv2.imread(image_path)
        if img is None:
            logging.error("Could not read image: %s", image_path)
            raise FileNotFoundError(f"Could not read image: {image_path}")

        faces = self.app.get(img)
        logging.info("Detected %d faces in the group photo.", len(faces))

        attendance = []
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding
            norm = np.linalg.norm(emb)
            if norm == 0:
                logging.debug("Skipping face with zero embedding.")
                continue
            emb = emb / norm

            identified = "Unknown"
            best_score = -1.0

            for name, db_emb in self.known_embeddings.items():
                score = float(np.dot(emb, db_emb))
                if score > best_score:
                    best_score = score
                    identified = name

            if best_score >= self.match_threshold:
                label = f"{identified} ({best_score:.2f})"
                color = (0, 255, 0)
                attendance.append({"Name": identified, "Confidence": float(best_score)})
            else:
                label = f"Unknown ({best_score:.2f})"
                color = (0, 0, 255)

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        logging.info("Annotated image saved to: %s", output_path)

        if attendance:
            df = pd.DataFrame(attendance)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            logging.info("Attendance saved to: %s", csv_path)
            # Print small summary to console
            logging.info("Attendance summary:\n%s", df.to_string(index=False))
        else:
            logging.info("No registered users recognized in the group photo.")

        if show:
            # Local display (requires a GUI environment)
            cv2.imshow("Attendance", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return attendance


def parse_args():
    p = argparse.ArgumentParser(description="Static face-attendance using insightface (local/VSCode friendly).")
    p.add_argument("--db", "-d", required=True, help="Path to database folder with one image per person (filename -> name).")
    p.add_argument("--image", "-i", required=True, help="Path to group photo to analyze.")
    p.add_argument("--out", "-o", default="output_attendance.jpg", help="Output annotated image path.")
    p.add_argument("--csv", "-c", default=None, help="Output attendance CSV path (default: <out_dir>/attendance_report.csv).")
    p.add_argument("--det-size", nargs=2, type=int, metavar=("W", "H"), default=(640, 640),
                   help="Detection size (width height). Reduce if low VRAM.")
    p.add_argument("--threshold", "-t", type=float, default=0.50, help="Cosine similarity threshold (0-1).")
    p.add_argument("--model", default="buffalo_l", help="insightface model name (if you used a custom model).")
    p.add_argument("--show", action="store_true", help="Show annotated image in a window (requires GUI).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    system = StaticAttendanceSystem(
        db_folder=args.db,
        det_size=(args.det_size[0], args.det_size[1]),
        match_threshold=args.threshold,
        model_name=args.model
    )

    try:
        system.process_group_photo(args.image, output_path=args.out, csv_path=args.csv, show=args.show)
    except Exception as e:
        logging.exception("Processing failed: %s", e)
        raise