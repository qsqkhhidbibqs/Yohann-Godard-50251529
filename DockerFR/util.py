"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List


def detect_faces(image_bytes):
    """
    Detect faces in raw image bytes.
    Returns a list of bounding boxes [x1, y1, x2, y2] and cropped face images.
    Resizes image if it's too large to improve warping.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize if too large
    scale = min(1.0, MAX_DIM / max(image.size))
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    image_np = np.array(image)

    detections = RetinaFace.detect_faces(image_np)
    if not detections:
        return [], []

    bboxes = []
    crops = []

    for face in detections.values():
        bbox = face["facial_area"]
        bboxes.append(bbox)

        x1, y1, x2, y2 = bbox
        face_crop = image_np[y1:y2, x1:x2]
        crops.append(Image.fromarray(face_crop))

    return bboxes, crops


def compute_face_embedding(face_image: Any) -> Any:
    """
    Compute a numerical embedding vector for the provided face image.

    The embedding should capture discriminative facial features for comparison.
    """
    raise NotImplementedError("Student implementation required for face embedding")


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    raise NotImplementedError("Student implementation required for keypoint detection")


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


def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    raise NotImplementedError("Student implementation required for face similarity calculation")
