import os
import uuid
import argparse
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
from attendance import StaticAttendanceSystem

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_DIR = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

system = None

def make_system(db_folder, det_size=(640,640), threshold=0.50, model_name="buffalo_l"):
    global system
    system = StaticAttendanceSystem(db_folder=db_folder, det_size=det_size, match_threshold=threshold, model_name=model_name)
    return system

@app.route("/", methods=["GET"])
def root():
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route('/api/upload', methods=['POST'])
def upload():
    global system
    if not system:
        return jsonify({"success": False, "error": "System not initialized"}), 503

    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400

        filename = secure_filename(f.filename)
        tmp_name = f"tmp_{uuid.uuid4().hex}_{filename}"
        tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
        f.save(tmp_path)

        output_filename = "output_attendance.jpg"
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        attendance = system.process_group_photo(tmp_path, output_path=output_path, csv_path=None, show=False)

        try:
            os.remove(tmp_path)
        except:
            pass

        faces_count = len(attendance) if attendance else 0
        image_url = url_for('static', filename=f'uploads/{output_filename}', _external=True)

        return jsonify({
            "success": True,
            "image_url": image_url,
            "faces_count": faces_count,
            "attendance": attendance or []
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", "-d", default="./db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--det-w", type=int, default=640)
    parser.add_argument("--det-h", type=int, default=640)
    parser.add_argument("--model", default="buffalo_l")
    args = parser.parse_args()

    make_system(args.db, (args.det_w, args.det_h), args.threshold, args.model)
    app.run(host=args.host, port=args.port, debug=True)