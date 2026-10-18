import os
import zipfile
import yaml
from pathlib import Path
from ultralytics import YOLO

# 1. Paths configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_ZIP = Path(r"C:\Users\Kishan\Downloads\Fire Smoke and Human Detector.v32i.yolov8-obb.zip")
EXTRACT_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "emberx_v1.pt"

# 2. Extract dataset if not already done
if not EXTRACT_DIR.exists():
    print(f"Extracting dataset from {DATASET_ZIP} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset extracted successfully.")
else:
    print(f"Dataset already extracted at {EXTRACT_DIR}")

# 3. Configure data.yaml with absolute paths
yaml_path = EXTRACT_DIR / "data.yaml"
if yaml_path.exists():
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Update paths to absolute locations in this environment
    data_cfg['path'] = str(EXTRACT_DIR.resolve()).replace("\\", "/")
    data_cfg['train'] = 'train/images'
    data_cfg['val'] = 'valid/images'
    data_cfg['test'] = 'test/images'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)
    print(f"Updated data.yaml structure at {yaml_path}:")
    print(data_cfg)
else:
    print(f"ERROR: data.yaml not found at {yaml_path}")

# 4. Load the existing emberx_v1.pt weights
print(f"Loading existing model weights from {MODEL_PATH}...")
model = YOLO(str(MODEL_PATH))

# 5. Run training
# Note: Using device='cpu' since no CUDA GPU is available locally.
# Batch size is set to 8 to avoid memory issues.
# Modify EPOCHS to train for more/fewer epochs.
EPOCHS = 50
print(f"Starting training for {EPOCHS} epochs on CPU...")
results = model.train(
    data=str(yaml_path.resolve()).replace("\\", "/"),
    epochs=EPOCHS,
    imgsz=640,
    batch=8,
    name='emberx_v1_continued',
    device='cpu'
)
print("Training completed successfully!")
