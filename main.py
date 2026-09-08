import os
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)


# =========================
# Download model once
# =========================

def download_model():
    if os.path.exists(MODEL_PATH):
        print("MediaPipe model already exists.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Downloading MediaPipe model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("MediaPipe model downloaded successfully.")


download_model()


# =========================
# MediaPipe Face Landmarker
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Real-Time VTO Engine",
    version="1.0.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Health Check
# =========================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "vto"
    }


# =========================
# Process Frame
# =========================

@app.post("/api/v1/vto/process-frame")
async def process_frame(
    file: UploadFile = File(...)
):

    contents = await file.read()

    if not contents:
        return {
            "detected": False
        }

    # Convert bytes -> NumPy array
    nparr = np.frombuffer(
        contents,
        np.uint8
    )

    # Decode image
    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return {
            "detected": False
        }

    h, w, _ = img.shape

    # BGR -> RGB
    rgb_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_img
    )

    # Face detection
    detection_result = detector.detect(
        mp_image
    )

    if not detection_result.face_landmarks:
        return {
            "detected": False
        }

    landmarks = detection_result.face_landmarks[0]

    # =========================
    # Eye points
    # =========================

    left_eye = (
        int(landmarks[33].x * w),
        int(landmarks[33].y * h)
    )

    right_eye = (
        int(landmarks[263].x * w),
        int(landmarks[263].y * h)
    )

    # Nose bridge
    nose_bridge = (
        int(landmarks[6].x * w),
        int(landmarks[6].y * h)
    )

    # =========================
    # Glasses center
    # =========================

    center_x = int(
        (left_eye[0] + right_eye[0]) / 2
    )

    eye_mid_y = (
        left_eye[1] + right_eye[1]
    ) / 2

    center_y = int(
        nose_bridge[1] * 0.7
        +
        eye_mid_y * 0.3
    )

    # =========================
    # Glasses width
    # =========================

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    dist = math.sqrt(
        dx**2 + dy**2
    )

    glasses_width = int(
        dist * 2.05
    )

    # =========================
    # Glasses angle
    # =========================

    angle = math.degrees(
        math.atan2(dy, dx)
    )
    # =========================
    # Response
    # =========================

    return {
        "detected": True,
        "center_x": center_x,
        "center_y": center_y,
        "glasses_width": glasses_width,
        "angle": angle
    }


# =========================
# Run Server
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
