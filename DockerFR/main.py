from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch

# Import all utils
from util import calculate_face_similarity

app = FastAPI(
    title="Face Similarity API",
    version="1.0",
    description="Upload two images and get a similarity score."
)

class Echo(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/torch-version")
def torch_version():
    return {"torch_version": torch.__version__}


@app.post("/echo")
def echo(body: Echo):
    return {"you_sent": body.text}


# ---------------------- FACE SIMILARITY ENDPOINT ----------------------
@app.post("/face-similarity")
async def face_similarity(image_a: UploadFile = File(...),
                          image_b: UploadFile = File(...)):
    try:
        bytes_a = await image_a.read()
        bytes_b = await image_b.read()

        score = calculate_face_similarity(bytes_a, bytes_b)

        return {"similarity": score}

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
