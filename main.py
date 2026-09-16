import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""=========================
 Paths & Model Download
========================="""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()

"""=========================
 MediaPipe Face Landmarker
========================="""
# استخدام IMAGE mode الخفيف للـ Single Frame processing
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

"""=========================
 FastAPI App Setup
========================="""
app = FastAPI(title="Real-Time VTO Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "vto"}

"""=========================
 Optimized Processing Worker
========================="""
def process_landmarks(image_bytes: bytes):
    # Decode المباشر كـ RGB لتوفير تحويلcvtColor الزائد
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"detected": False}

    h, w = img.shape[:2]

    # تصغير الحجم فوراً لو الفرونت إند بعت صورة كبيرة
    if w > 320:
        scale = 320.0 / w
        new_w = 320
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    # تحويل لـ RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return {"detected": False}

    landmarks = detection_result.face_landmarks[0]

    # استخراج النقاط المطلوبة للنظارة
    left_x, left_y = int(landmarks[33].x * w), int(landmarks[33].y * h)
    right_x, right_y = int(landmarks[263].x * w), int(landmarks[263].y * h)
    nose_y = int(landmarks[6].y * h)

    center_x = (left_x + right_x) // 2
    eye_mid_y = (left_y + right_y) / 2.0
    center_y = int(nose_y * 0.7 + eye_mid_y * 0.3)

    dx = right_x - left_x
    dy = right_y - left_y
    dist = math.hypot(dx, dy)

    glasses_width = int(dist * 2.05)
    angle = math.degrees(math.atan2(dy, dx))

    return {
        "detected": True,
        "center_x": center_x,
        "center_y": center_y,
        "glasses_width": glasses_width,
        "angle": angle
    }

"""=========================
 Direct File Upload Endpoint
========================="""
@app.post("/api/v1/vto/process-frame")
async def process_frame(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return {"detected": False}

        return await run_in_threadpool(process_landmarks, image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing frame: {str(exc)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
