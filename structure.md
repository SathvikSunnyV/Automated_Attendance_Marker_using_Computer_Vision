Face-Attendance-System/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   │
│   ├── detection/
│   │   └── retinaface_detector.py
│   │
│   ├── alignment/
│   │   └── face_alignment.py
│   │
│   ├── recognition/
│   │   ├── cnn_model.py
│   │   ├── embedder.py
│   │   └── matcher.py
│   │
│   ├── attendance/
│   │   └── mark_attendance.py
│   │
│   ├── storage/
│   │   ├── embeddings/
│   │   ├── enrolled_faces/
│   │   └── attendance_logs/
│   │
│   └── utils/
│       ├── image_utils.py
│       └── similarity_utils.py
│
├── frontend/
│   ├── index.html        ← image upload UI
│   ├── upload.js         ← sends image to backend
│   ├── result_view.js    ← draws boxes + names
│   └── style.css
│
├── training/
│   ├── dataset/
│   ├── train_model.py
│   ├── loss_functions.py
│   └── evaluate.py
│
├── docs/
│   ├── architecture.md
│   ├── limitations.md
│   └── workflow.md
│
└── README.md
