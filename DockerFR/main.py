from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import io
import math
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

def smooth_squash(x):
    center = 0.65   # centre du plateau 0.6–0.7
    k = 4.0         # raideur, ajuste pour lisser/raidir
    return 1 / (1 + math.exp(-k * (x - center)))

def calculate_face_similarity(image_a_bytes: bytes, image_b_bytes: bytes) -> float:
    """
    End-to-end face similarity pipeline with calibrated similarity 0..1:
    - Detect faces
    - Align faces
    - Use embeddings
    - Map similarity to intuitive 0..1 scale
    """

    # 1. Detection
    img_a = cv2.imdecode(np.frombuffer(image_a_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(image_b_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces_a = face_app.get(img_a)
    faces_b = face_app.get(img_b)

    if not faces_a or not faces_b:
        raise ValueError("No face detected in one of the images")

    face_a = faces_a[0]
    face_b = faces_b[0]

    # 2. Keypoints
    kp_a = {"left_eye": face_a.kps[0], "right_eye": face_a.kps[1],
            "nose": face_a.kps[2], "mouth_left": face_a.kps[3], "mouth_right": face_a.kps[4]}
    kp_b = {"left_eye": face_b.kps[0], "right_eye": face_b.kps[1],
            "nose": face_b.kps[2], "mouth_left": face_b.kps[3], "mouth_right": face_b.kps[4]}

    # 3. Warp faces (utile si tu veux afficher ou autre)
    warped_a = warp_face(image_a_bytes, kp_a)
    warped_b = warp_face(image_b_bytes, kp_b)

    # 4. Embeddings
    emb_a = face_a.embedding / np.linalg.norm(face_a.embedding)
    emb_b = face_b.embedding / np.linalg.norm(face_b.embedding)

    # 5. Cosine similarity
    cosine_sim = float(np.dot(emb_a, emb_b))  # [-1,1]

    # 6. Map to 0..1 with sigmoid-like calibration
    #    - simulate 0 = très différent, 1 = même personne
    sim_raw = (cosine_sim + 1) / 2  # basic map [-1,1] -> [0,1]

    # 7. Calibration simple “min-max” interne
    #    - valeurs typiques de Buffalo_L : différent ~0.3, même personne ~0.8
    min_diff = 0.3
    max_same = 0.85
    sim_calibrated = (sim_raw - min_diff) / (max_same - min_diff)
    sim_calibrated = np.clip(sim_calibrated, 0.0, 1.0)**2.2


    return float(sim_calibrated)




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


