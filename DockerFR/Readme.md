from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import numpy as np
from PIL import Image
import cv2
import torch
from numpy.linalg import norm

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis

# ---------------------- FastAPI Setup ----------------------
app = FastAPI(
    title="Face Recognition API",
    description="Face similarity API using InsightFace",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

class Echo(BaseModel):
    text: str

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello, FastAPI + InsightFace!"}

@app.get("/torch-version", tags=["Info"])
def torch_version():
    return {"torch_version": torch.__version__}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/echo", tags=["Demo"])
def echo(body: Echo):
    return {"you_sent": body.text}

# ---------------------- InsightFace Setup ----------------------
# Initialize the face analyzer with detection, landmarks, and embedding
face_app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_5', 'recognition'])
face_app.prepare(ctx_id=0, nms=0.4)  # ctx_id=0 uses GPU if available, else CPU

# ---------------------- Utility Functions ----------------------
def detect_faces(image_bytes, show_crops=False):
    """
    Detect faces in image bytes.
    Returns list of dicts with 'bbox', 'landmarks', 'embedding'
    """
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    faces = face_app.get(image)

    if len(faces) == 0:
        return []

    results = []
    for f in faces:
        bbox = f.bbox.astype(int)  # [x1, y1, x2, y2]
        landmarks = f.landmark_2d_5  # 5-point landmarks
        embedding = f.embedding      # 512D embedding

        # Optionally show cropped face
        if show_crops:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)
            crop_pil.show()

        results.append({
            "bbox": bbox,
            "landmarks": landmarks,
            "embedding": embedding
        })

    return results

def warp_face(image_bytes, landmarks, output_size=(112, 112)):
    """
    Align a face using 5-point landmarks (eyes, nose, mouth corners)
    """
    src_pts = np.array(landmarks, dtype=np.float32)

    # Standard 5-point template for 112x112
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # Compute similarity transform
    tfm = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)[0]

    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    aligned = cv2.warpAffine(image, tfm, output_size, flags=cv2.INTER_LINEAR)

    return aligned

def _cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def calculate_face_similarity(image_a_bytes: bytes, image_b_bytes: bytes):
    """
    Full pipeline: detect, align, embed, and compute similarity
    """
    faces_a = detect_faces(image_a_bytes)
    faces_b = detect_faces(image_b_bytes)

    if not faces_a or not faces_b:
        raise ValueError("No face detected in one of the images.")

    # Take first face detected in each image
    emb_a = faces_a[0]['embedding']
    emb_b = faces_b[0]['embedding']

    similarity = _cosine_similarity(emb_a, emb_b)
    similarity = max(0.0, min(1.0, similarity))
    return similarity

# ---------------------- API Endpoint ----------------------
@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image file"),
    image_b: UploadFile = File(..., description="Second face image file"),
):
    try:
        content_a = await image_a.read()
        content_b = await image_b.read()
        similarity = calculate_face_similarity(content_a, content_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"similarity": similarity}

@app.post("/detect-show", tags=["Face Recognition"])
async def detect_show(image: UploadFile = File(...)):
    """
    Detect faces and return an image with bounding boxes drawn.
    """
    content = await image.read()
    image_np = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
    faces = detect_faces(content)

    for f in faces:
        x1, y1, x2, y2 = f['bbox']
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    buf = pil_bytes_from_np(image_np)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.post("/warp-show", tags=["Face Recognition"])
async def warp_show(image: UploadFile = File(...)):
    """
    Detect and align the first face, then return the aligned image.
    """
    content = await image.read()
    faces = detect_faces(content)

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")

    aligned = warp_face(content, faces[0]['landmarks'])
    buf = pil_bytes_from_np(aligned)
    return StreamingResponse(buf, media_type="image/jpeg")