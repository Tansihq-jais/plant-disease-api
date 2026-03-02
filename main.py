import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH       = Path("model/plant_disease_efficientnet.keras")
CLASS_NAMES_PATH = Path("model/class_names.json")
IMG_SIZE         = (224, 224)   # EfficientNetB0 native size
MAX_FILE_SIZE_MB = 10

# ── Global model state ────────────────────────────────────────────────────────
ml_model: tf.keras.Model = None
class_names: list[str]   = []

# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ml_model, class_names
    logger.info("Loading model from %s ...", MODEL_PATH)
    try:
        ml_model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("✅ Model loaded successfully")
        logger.info("   Input shape  : %s", ml_model.input_shape)
        logger.info("   Output shape : %s", ml_model.output_shape)
    except Exception as e:
        logger.error("❌ Model load failed: %s", e)
        raise RuntimeError(f"Could not load model: {e}")

    logger.info("Loading class names from %s ...", CLASS_NAMES_PATH)
    try:
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        logger.info("✅ Loaded %d class names", len(class_names))
    except Exception as e:
        logger.error("❌ Class names load failed: %s", e)
        raise RuntimeError(f"Could not load class names: {e}")

    # Warmup — run one dummy prediction so the first real request isn't slow
    logger.info("Running warmup prediction ...")
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    ml_model.predict(dummy, verbose=0)
    logger.info("✅ Warmup complete. API is ready.")

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down. Clearing model from memory.")
    del ml_model

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
    Decode → convert to RGB → resize to 224x224 → batch.
    NOTE: Do NOT divide by 255. EfficientNetB0 normalizes internally.
    Returns shape (1, 224, 224, 3) with pixel values in [0, 255].
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPG or PNG.")

    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)          # [0, 255] — no normalization
    return np.expand_dims(arr, axis=0)             # (1, 224, 224, 3)

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_disease_name(raw: str) -> dict:
    """Split 'Tomato___Late_blight' into plant and disease parts."""
    parts = raw.split("___")
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
    """Health check endpoint for Render / Docker / load balancers."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "num_classes": len(class_names),
    }


@app.get("/classes", tags=["General"])
def get_classes():
    """Return all 38 supported disease classes."""
    return {
        "total": len(class_names),
        "classes": [format_disease_name(c) for c in class_names]
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Plant leaf image (JPG or PNG)")):
    """
    Predict plant disease from a leaf image.

    - **file**: JPG or PNG image of a plant leaf (max 10MB)

    Returns:
    - `predicted_disease`: top predicted class
    - `plant`: plant name
    - `disease`: disease name
    - `confidence`: confidence percentage
    - `top_3`: top 3 predictions with confidence scores
    - `is_healthy`: whether the plant is predicted healthy
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
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed is {MAX_FILE_SIZE_MB}MB."
        )

    # Preprocess
    try:
        input_arr = preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # Predict
    try:
        start = time.perf_counter()
        predictions = ml_model.predict(input_arr, verbose=0)[0]  # shape: (38,)
        inference_ms = (time.perf_counter() - start) * 1000
        logger.info("Prediction completed in %.1fms", inference_ms)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed.")

    # Format results
    best_idx      = int(np.argmax(predictions))
    top3_indices  = np.argsort(predictions)[::-1][:3]
    best_name     = format_disease_name(class_names[best_idx])
    is_healthy    = "healthy" in class_names[best_idx].lower()

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
