import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH       = Path("model/plant_disease_model.tflite")
CLASS_NAMES_PATH = Path("model/class_names.json")
HF_REPO_ID       = "tanishq-jaiswal/plant-disease-model"
IMG_SIZE         = (224, 224)
MAX_FILE_SIZE_MB = 10

# ── Global model state ────────────────────────────────────────────────────────
interpreter      = None
input_details    = None
output_details   = None
class_names: list[str] = []

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details, class_names

    # ── Download from Hugging Face if not present ─────────────────────────────
    if not MODEL_PATH.exists():
        logger.info("Downloading model from Hugging Face...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=HF_REPO_ID, filename="plant_disease_model.tflite", local_dir="model")
        hf_hub_download(repo_id=HF_REPO_ID, filename="class_names.json", local_dir="model")
        logger.info("✅ Downloaded from Hugging Face")
    else:
        logger.info("Model found locally. Skipping download.")

    # ── Load TFLite model ─────────────────────────────────────────────────────
    logger.info("Loading TFLite model...")
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("✅ TFLite model loaded | Input shape: %s", input_details[0]['shape'])

    # ── Load class names ──────────────────────────────────────────────────────
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    logger.info("✅ Loaded %d class names", len(class_names))

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info("Running warmup prediction...")
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    logger.info("✅ Warmup complete. API is ready.")

    yield

    logger.info("Shutting down.")
    del interpreter


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌿 Plant Disease Detection API",
    description=(
        "Upload a plant leaf image (JPG/PNG) to detect diseases.\n\n"
        "Powered by EfficientNetB0 trained on 38 plant disease classes.\n\n"
        "**Usage:** POST an image to `/predict` and receive the disease name + confidence."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    return response

# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode → RGB → resize to 224x224 → normalize to [0,1] for TFLite.
    Returns shape (1, 224, 224, 3) as float32.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPG or PNG.")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # TFLite expects [0, 1]
    return np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_disease_name(raw: str) -> dict:
    parts   = raw.split("___")
    plant   = parts[0].replace("_", " ") if len(parts) > 0 else raw
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return {"plant": plant, "disease": disease, "raw": raw}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "🌿 Plant Disease Detection API is running.",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "num_classes": len(class_names),
    }


@app.get("/classes", tags=["General"])
def get_classes():
    return {
        "total": len(class_names),
        "classes": [format_disease_name(c) for c in class_names]
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Plant leaf image (JPG or PNG)")):
    """
    Predict plant disease from a leaf image.
    Returns predicted disease, confidence score, and top 3 predictions.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG or PNG."
        )

    # Validate file size
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB."
        )

    # Preprocess
    try:
        input_arr = preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # Predict using TFLite interpreter
    try:
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_arr)
        interpreter.invoke()
        predictions  = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (38,)
        inference_ms = (time.perf_counter() - start) * 1000
        logger.info("Prediction done in %.1fms", inference_ms)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed.")

    # Format results
    best_idx     = int(np.argmax(predictions))
    top3_indices = np.argsort(predictions)[::-1][:3]
    best_name    = format_disease_name(class_names[best_idx])
    is_healthy   = "healthy" in class_names[best_idx].lower()

    top3 = [
        {
            **format_disease_name(class_names[i]),
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top3_indices
    ]

    return {
        "predicted_disease" : class_names[best_idx],
        "plant"             : best_name["plant"],
        "disease"           : best_name["disease"],
        "confidence"        : round(float(predictions[best_idx]) * 100, 2),
        "is_healthy"        : is_healthy,
        "top_3"             : top3,
        "inference_ms"      : round(inference_ms, 1),
        "filename"          : file.filename,
    }


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)