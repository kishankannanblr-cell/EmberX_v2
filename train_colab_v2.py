# train_colab_v2.py
"""
Google Colab Training Script for EmberX v2 (Intensive Training Run)
Usage in Google Colab:
1. Upload this script, 'emberx_v1.pt', and 'Fire Smoke and Human Detector.v32i.yolov8-obb.zip' to Colab's files side panel.
2. Run: !python train_colab_v2.py
"""
import os
import zipfile
import yaml
from pathlib import Path
from ultralytics import YOLO

# 1. Configuration
BASE_DIR = Path('/content')
DATASET_ZIP = BASE_DIR / "Fire Smoke and Human Detector.v32i.yolov8-obb.zip"
EXTRACT_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "emberx_v1.pt"
YAML_PATH = EXTRACT_DIR / "data.yaml"

# 2. Extract Dataset
if not DATASET_ZIP.exists():
    print(f"ERROR: Dataset zip not found at {DATASET_ZIP}.")
    print("Please upload 'Fire Smoke and Human Detector.v32i.yolov8-obb.zip' to the Colab files panel.")
    exit(1)

if not EXTRACT_DIR.exists():
    print(f"Extracting dataset from {DATASET_ZIP} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset extracted successfully.")
else:
    print(f"Dataset already extracted at {EXTRACT_DIR}")

# 3. Configure data.yaml with Colab paths
if YAML_PATH.exists():
    with open(YAML_PATH, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Update paths to absolute locations in Google Colab
    data_cfg['path'] = str(EXTRACT_DIR.resolve()).replace("\\", "/")
    data_cfg['train'] = 'train/images'
    data_cfg['val'] = 'valid/images'
    data_cfg['test'] = 'test/images'
    
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_cfg, f)
    print(f"Updated data.yaml structure at {YAML_PATH}:")
    print(data_cfg)
else:
    print(f"ERROR: data.yaml not found at {YAML_PATH}")
    exit(1)

# 4. Load weights (Fallback to pretrained if v1.pt not uploaded)
if MODEL_PATH.exists():
    print(f"Loading existing model weights from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))
else:
    print(f"WARNING: {MODEL_PATH} not found. Starting from pretrained 'yolov8s-obb.pt' weights.")
    model = YOLO('yolov8s-obb.pt')

# 5. Run Intensive Training (EmberX v2)
print("Starting intensive training for EmberX v2...")
results = model.train(
    data=str(YAML_PATH.resolve()).replace("\\", "/"),
    epochs=120,                  # Train longer for full convergence
    imgsz=640,                   # Image size matched to dataset
    batch=16,                    # Standard batch size for Colab T4 GPU
    device=0,                    # Use Colab GPU
    name='emberx_v2_intensive',  # Name of the output model run
    
    # --- Intensive Training Hyperparameters ---
    lr0=0.005,                   # Smaller initial learning rate for fine-tuning
    lrf=0.01,                    # Final learning rate is 1% of lr0
    cos_lr=True,                 # Cosine learning rate decay for smoother convergence
    patience=50,                 # Stop early if validation doesn't improve for 50 epochs
    
    # --- Advanced Augmentation to prevent overfitting & boost smoke/victim detection ---
    mosaic=1.0,                  # Combine 4 images into one (helps with small objects)
    mixup=0.15,                  # Blend two images (increases robustness)
    scale=0.5,                   # Randomly scale images (+/- 50%)
    fliplr=0.5,                  # Horizontal flip (50% probability)
    degrees=10.0,                # Small random rotation (+/- 10 degrees)
    hsv_h=0.015,                 # Hue adjustment
    hsv_s=0.7,                   # Saturation jitter (vital for discriminating diffuse smoke plumes)
    hsv_v=0.4,                   # Value/brightness jitter (vital for low-light & glare variations)
    translate=0.1,               # Translation jitter
    close_mosaic=10,             # Disable mosaic for final 10 epochs for crisp boundary convergence
    
    # --- Technical settings ---
    val=True,                    # Validate after each epoch
    cache=False                  # Turn off RAM caching to avoid Colab RAM crash
)

print("\n=======================================================")
print("Training completed successfully!")
print("Your best weights are saved at: /content/runs/obb/emberx_v2_intensive/weights/best.pt")
print("Download this file and rename it to 'emberx_v2.pt' for deployment.")
print("=======================================================")
