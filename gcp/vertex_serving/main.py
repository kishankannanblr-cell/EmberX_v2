# gcp/vertex_serving/main.py
"""
Google Vertex AI Custom Prediction Container for GuardianX EmberX
================================================================
Conforms to Vertex AI Custom Container Specification:
- Listens on $AIP_HTTP_PORT (default: 8080)
- GET  /health -> Returns 200 OK for liveness probe
- POST /predict -> Receives instances, runs EmberX (+ optional SAHI), returns triage results
"""

import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO

# Import local SAHI inference module if available
try:
    from sahi_obb_inference import SahiOBBInference
    sahi_available = True
except ImportError:
    sahi_available = False

app = FastAPI(title="GuardianX EmberX Vertex AI Prediction Service")

# Model Loading & Warmup
MODEL_PATH = os.getenv("AIP_STORAGE_URI", "/models")
LOCAL_WEIGHTS = os.path.join(MODEL_PATH, "emberx_v2.pt") if os.path.exists(os.path.join(MODEL_PATH, "emberx_v2.pt")) else "emberx_v2.pt"

print(f"📦 Loading EmberX weights from: {LOCAL_WEIGHTS}")
model = None
sahi_engine = None

@app.on_event("startup")
def load_models():
    global model, sahi_engine
    if os.path.exists(LOCAL_WEIGHTS):
        model = YOLO(LOCAL_WEIGHTS)
        if sahi_available:
            sahi_engine = SahiOBBInference(LOCAL_WEIGHTS)
        print("✅ EmberX model loaded successfully on startup.")
    else:
        print(f"⚠️ Warning: Weights not found at {LOCAL_WEIGHTS}. Fallback will be loaded on demand.")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Vertex AI liveness / readiness check."""
    return {"status": "healthy", "model_loaded": model is not None}

def decode_image(instance: dict) -> Image.Image:
    """Decode image from base64 string or download from URL."""
    if "b64" in instance:
        b64_str = instance["b64"]
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif "image_url" in instance:
        resp = requests.get(instance["image_url"], timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        raise ValueError("Instance must contain either 'b64' or 'image_url'.")

@app.post("/predict")
async def predict(request: Request):
    """Vertex AI Prediction Handler."""
    global model, sahi_engine
    body = await request.json()
    instances = body.get("instances", [])
    
    if not instances:
        return JSONResponse(status_code=400, content={"error": "No instances provided in request."})

    predictions = []
    for inst in instances:
        try:
            img = decode_image(inst)
            run_sahi = inst.get("run_sahi", False)

            if run_sahi and sahi_engine:
                # High-res SAHI sliced inference
                result = sahi_engine.predict(img)
                predictions.append(result)
            else:
                # Standard low-latency full-frame inference
                img_np = np.array(img)
                res = model.predict(img_np, verbose=False, imgsz=640)[0]
                
                detections = []
                human_count = 0
                has_fire = False
                has_smoke = False

                if hasattr(res, 'obb') and res.obb is not None:
                    for box in res.obb:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names.get(cls_id, f"class_{cls_id}")
                        conf = float(box.conf[0].item())
                        poly = box.xyxyxyxy[0].cpu().numpy().tolist()

                        if cls_name == "human" and conf >= 0.22:
                            human_count += 1
                        elif cls_name == "fire" and conf >= 0.42:
                            has_fire = True
                        elif cls_name == "smoke" and conf >= 0.35:
                            has_smoke = True

                        detections.append({
                            "class_name": cls_name,
                            "class_id": cls_id,
                            "confidence": conf,
                            "polygon": poly
                        })

                # Triage calculation
                h_det = 1.0 if human_count > 0 else 0.0
                f_det = 1.0 if has_fire else 0.0
                s_det = 1.0 if has_smoke else 0.0
                raw_score = (0.55 * h_det) + (0.25 * f_det) + (0.15 * s_det) + (0.05 * min(human_count, 5))
                priority_score = min(round(raw_score, 3), 1.0)

                predictions.append({
                    "detections": detections,
                    "counts": {"human": human_count, "fire": int(has_fire), "smoke": int(has_smoke)},
                    "triage": {
                        "priority_score": priority_score,
                        "human_count": human_count,
                        "has_fire": has_fire,
                        "has_smoke": has_smoke
                    }
                })
        except Exception as e:
            predictions.append({"error": str(e)})

    return {"predictions": predictions}

if __name__ == "__main__":
    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
