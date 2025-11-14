from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import io
import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2
import torch
from fastapi.responses import StreamingResponse
# Libraries for face recognition
from retinaface import RetinaFace
from deepface import DeepFace
from insightface.app import FaceAnalysis


app = FastAPI(
    title="My FastAPI Service",
    description="A simple demo API running in Docker. Swagger is at /docs and ReDoc at /redoc.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

class Echo(BaseModel):
    text: str

# ---------------------- Basic Endpoints ----------------------
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello, FastAPI in Docker!"}

@app.get("/torch-version", tags=["Info"])
def torch_version():
    return {"torch_version": torch.__version__}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/echo", tags=["Demo"])
def echo(body: Echo):
    return {"you_sent": body.text}

# ---------------------- Utility Functions ----------------------
MAX_DIM = 800  # max dimension to resize images for proper warping

def detect_faces(image_bytes):
    """
    Detect faces in raw image bytes using InsightFace.
    Returns a list of bounding boxes [x1, y1, x2, y2] and cropped face images.
    Resizes image if too large to improve detection.
    """

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize if too large
    scale = min(1.0, MAX_DIM / max(image.size))
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy (RGB)
    image_np = np.array(image)

    # InsightFace expects BGR
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run detection using InsightFace
    faces = face_app.get(image_bgr)

    if not faces:
        return [], []

    bboxes = []
    crops = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        bboxes.append([x1, y1, x2, y2])

        face_crop = image_np[y1:y2, x1:x2]
        crops.append(Image.fromarray(face_crop))

    return bboxes, crops


# --- Initialize the InsightFace model once globally ---
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU, det_size=(width, height)

def compute_face_embedding(face_image_bytes):
    """
    Convert a face image (in bytes) to a 512D embedding using InsightFace.
    Uses a pre-initialized FaceAnalysis model.
    Raises ValueError if no face is detected.
    """

    # Convert bytes to BGR image
    nparr = np.frombuffer(face_image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Invalid image bytes provided")

    # Detect faces and get embeddings
    faces = face_app.get(img_bgr)

    if not faces:
        raise ValueError("No face detected for embedding")

    # Get the first face's embedding
    emb = faces[0].embedding

    # Normalize the embedding (L2 norm) to have norm = 1
    emb = emb / np.linalg.norm(emb)

    print(type(emb), emb.shape, np.linalg.norm(emb))  # pour debug

    # Return the normalized embedding
    return emb


def warp_face(image_bytes, keypoints, output_size=(112,112)):
    # --- Convert bytes to OpenCV image (BGR) ---
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

    # --- Convert to RGB for landmarks (optional) ---
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # --- Reference landmarks (ArcFace style) ---
    ref_landmarks = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    if output_size != (112,112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        ref_landmarks *= np.array([scale_x, scale_y])

    # --- Source points ---
    src_points = np.array([
        keypoints["left_eye"],
        keypoints["right_eye"],
        keypoints["nose"],
        keypoints["mouth_left"],
        keypoints["mouth_right"]
    ], dtype=np.float32)

    # --- Estimate affine transformation ---
    M, _ = cv2.estimateAffinePartial2D(src_points, ref_landmarks, method=cv2.LMEDS)
    if M is None:
        raise ValueError("Affine transform failed")

    # --- Warp image ---
    warped = cv2.warpAffine(image_rgb, M, output_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

    return warped


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _to_rgb_np(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil)

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def calculate_face_similarity(image_a_bytes: bytes, image_b_bytes: bytes) -> float:
    """
    Calculate similarity between two face images using InsightFace embeddings.
    The images are passed as raw bytes. No manual warping is applied.
    """
    try:
        emb_a = compute_face_embedding(image_a_bytes)
        emb_b = compute_face_embedding(image_b_bytes)
    except ValueError as e:
        raise ValueError("No face detected in one of the images.") from e
    
    
    similarity = (float(np.dot(emb_a, emb_b)) + 1.0) / 2.0
    return similarity
# ---------------------- Endpoints ----------------------
@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...)
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


