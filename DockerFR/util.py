"""
Utility functions for the face recognition pipeline.
All processing is centralized here.
"""

import io
import math
import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2
from insightface.app import FaceAnalysis

# ---------------------- GLOBALS ----------------------
MAX_DIM = 800

# InsightFace model (embedding + landmarks)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))


# ---------------------- BASIC HELPERS ----------------------
def detect_faces(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    scale = min(1.0, MAX_DIM / max(image.size))
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    faces = face_app.get(image_bgr)

    if not faces:
        return [], []

    bboxes, crops = [], []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        bboxes.append([x1, y1, x2, y2])
        crop = image_np[y1:y2, x1:x2]
        crops.append(Image.fromarray(crop))

    return bboxes, crops


def compute_face_embedding(face_image_bytes):
    nparr = np.frombuffer(face_image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Invalid image bytes")

    faces = face_app.get(img_bgr)
    if not faces:
        raise ValueError("No face detected for embedding")

    emb = faces[0].embedding
    emb = emb / norm(emb)

    return emb


def detect_face_keypoints(face):
    """
    face = object returned by InsightFace
    """
    return {
        "left_eye": face.kps[0],
        "right_eye": face.kps[1],
        "nose": face.kps[2],
        "mouth_left": face.kps[3],
        "mouth_right": face.kps[4],
    }


def warp_face(image_bytes, keypoints, output_size=(112, 112)):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    ref_landmarks = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)

    if output_size != (112, 112):
        sx = output_size[0] / 112.0
        sy = output_size[1] / 112.0
        ref_landmarks *= np.array([sx, sy])

    src_points = np.array([
        keypoints["left_eye"],
        keypoints["right_eye"],
        keypoints["nose"],
        keypoints["mouth_left"],
        keypoints["mouth_right"],
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_points, ref_landmarks, cv2.LMEDS)

    if M is None:
        raise ValueError("Affine transform failed")

    warped = cv2.warpAffine(image_rgb, M, output_size, flags=cv2.INTER_LINEAR)

    return warped


def _cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def smooth_squash(x):
    center = 0.65
    k = 4.0
    return 1 / (1 + math.exp(-k * (x - center)))


# ---------------------- FULL PIPELINE ----------------------
def calculate_face_similarity(image_a_bytes, image_b_bytes) -> float:
    """
    Full face comparison pipeline:
    - Detect faces
    - Get landmarks
    - Warp faces
    - Extract embeddings
    - Cosine similarity
    - Map [-1,1] → [0,1]
    """

    # Load images
    img_a = cv2.imdecode(np.frombuffer(image_a_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(image_b_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces_a = face_app.get(img_a)
    faces_b = face_app.get(img_b)

    if not faces_a or not faces_b:
        raise ValueError("No face detected in one of the images")

    fa = faces_a[0]
    fb = faces_b[0]

    kp_a = detect_face_keypoints(fa)
    kp_b = detect_face_keypoints(fb)

    warp_face(image_a_bytes, kp_a)
    warp_face(image_b_bytes, kp_b)

    emb_a = fa.embedding / norm(fa.embedding)
    emb_b = fb.embedding / norm(fb.embedding)

    cosine_sim = float(np.dot(emb_a, emb_b))
    sim_raw = (cosine_sim + 1) / 2

    min_diff = 0.3
    max_same = 0.85
    calibrated = (sim_raw - min_diff) / (max_same - min_diff)
    calibrated = np.clip(calibrated, 0.0, 1.0) ** 2.2

    return float(calibrated)
