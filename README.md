# 🌿 Plant Disease Detection API — Production Deployment Guide
### FastAPI + Docker + Render

---

## 📁 Required Project Structure

Set up this exact folder structure on your Windows PC before doing anything:

```
plant-disease-api/
├── main.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── model/
    ├── plant_disease_efficientnet.keras   ← download from Google Drive
    └── class_names.json                   ← download from Google Drive
```

---

## STEP 1 — Download Your Model from Google Drive

In Google Colab, run:
```python
from google.colab import files
files.download('/content/drive/MyDrive/plant_disease_efficientnet.keras')
files.download('/content/drive/MyDrive/class_names.json')
```

On your PC:
1. Create the folder: `plant-disease-api/model/`
2. Move both downloaded files into it

---

## STEP 2 — Install Docker on Windows

1. Go to https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Windows
3. Install and restart your PC
4. Open Docker Desktop — wait until it says "Engine running"
5. Verify in Command Prompt:
```bash
docker --version
# Docker version 25.x.x
```

---

## STEP 3 — Build & Test Locally

Open Command Prompt in your `plant-disease-api/` folder:

```bash
# Build the Docker image (~5-10 minutes first time)
docker build -t plant-disease-api .

# Run the container
docker run -p 8000:8000 plant-disease-api
```

You should see:
```
INFO | Loading model from model/plant_disease_efficientnet.keras ...
INFO | ✅ Model loaded successfully
INFO | ✅ Loaded 38 class names
INFO | ✅ Warmup complete. API is ready.
INFO | Uvicorn running on http://0.0.0.0:8000
```

Now open your browser:
- Swagger UI (interactive docs): http://localhost:8000/docs
- Health check:                  http://localhost:8000/health
- List all classes:              http://localhost:8000/classes

Test with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_leaf_image.jpg"
```

Expected response:
```json
{
  "predicted_disease": "Tomato___Late_blight",
  "plant": "Tomato",
  "disease": "Late blight",
  "confidence": 96.42,
  "is_healthy": false,
  "top_3": [
    {"plant": "Tomato", "disease": "Late blight", "confidence": 96.42},
    {"plant": "Tomato", "disease": "Early blight", "confidence": 2.11},
    {"plant": "Potato", "disease": "Late blight", "confidence": 0.98}
  ],
  "inference_ms": 142.3,
  "filename": "your_leaf_image.jpg"
}
```

---

## STEP 4 — Push to GitHub

The model file is too large for GitHub. Push only code.

```bash
git init
git add main.py Dockerfile requirements.txt .dockerignore .gitignore
git commit -m "Plant disease API - production ready"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-api.git
git push -u origin main
```

---

## STEP 5 — Upload Model to Hugging Face (Free Model Hosting)

```bash
pip install huggingface_hub
huggingface-cli login
```

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="YOUR_USERNAME/plant-disease-model", repo_type="model")
api.upload_file(path_or_fileobj="model/plant_disease_efficientnet.keras",
                path_in_repo="plant_disease_efficientnet.keras",
                repo_id="YOUR_USERNAME/plant-disease-model", repo_type="model")
api.upload_file(path_or_fileobj="model/class_names.json",
                path_in_repo="class_names.json",
                repo_id="YOUR_USERNAME/plant-disease-model", repo_type="model")
print("Model uploaded!")
```

Then update lifespan in main.py to auto-download from Hugging Face on startup:
```python
# Add huggingface_hub==0.24.0 to requirements.txt first

from huggingface_hub import hf_hub_download

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id="YOUR_USERNAME/plant-disease-model",
                    filename="plant_disease_efficientnet.keras", local_dir="model")
    hf_hub_download(repo_id="YOUR_USERNAME/plant-disease-model",
                    filename="class_names.json", local_dir="model")
```

---

## STEP 6 — Deploy on Render

1. Go to https://render.com and sign up
2. New + → Web Service → Connect GitHub repo
3. Settings:

| Setting | Value |
|---|---|
| Environment | Docker |
| Region | Singapore (closest to India) |
| Plan | Free or Starter ($7/mo for always-on) |

4. Click Create Web Service
5. Wait 5-10 minutes
6. Live at: https://plant-disease-api.onrender.com

---

## STEP 7 — Python Client Example

```python
import requests

url = "https://plant-disease-api.onrender.com/predict"
with open("leaf.jpg", "rb") as f:
    response = requests.post(url, files={"file": ("leaf.jpg", f, "image/jpeg")})

result = response.json()
print(f"Disease   : {result['disease']}")
print(f"Plant     : {result['plant']}")
print(f"Confidence: {result['confidence']}%")
print(f"Healthy   : {result['is_healthy']}")
```

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| model/ directory not found | Create model/ folder with both files before docker build |
| Cannot connect to Docker daemon | Open Docker Desktop and wait for engine to start |
| Port 8000 already in use | Run docker ps then docker stop container_id |
| Render build fails | Check build logs for missing files |
| Free Render sleeps | First request ~30s wake up. Upgrade to Starter for always-on |

---

## Quick Reference

```bash
# Build
docker build -t plant-disease-api .

# Run
docker run -p 8000:8000 plant-disease-api

# Stop
docker ps
docker stop <container_id>

# Rebuild after changes
docker build -t plant-disease-api . && docker run -p 8000:8000 plant-disease-api
```
