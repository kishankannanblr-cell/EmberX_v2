# train_emberx_v3_yolo11.py
"""
GuardianX - EmberX v3 Intensive Training Pipeline
==================================================
Key Enhancements:
1. Backbone Upgrade: Migrates from YOLOv8-OBB to YOLO11-OBB (yolo11s-obb / yolo11m-obb).
2. Life-Safety Loss Reweighting: Amplified box & class loss to aggressively penalize missed humans (high recall).
3. Smoke, Low-Visibility & Occlusion Augmentations: High mosaic, mixup, and copy-paste for occluded targets.
4. Sliced/Multi-Scale Support: Optimized for small-object search-and-rescue detection.

Works seamlessly on Google Colab (with GPU) or local workstations.
"""

import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_training(
    base_dir: str = "/content",
    dataset_zip_name: str = "Fire Smoke and Human Detector.v32i.yolov8-obb.zip",
    model_variant: str = "yolo11s-obb.pt",
    epochs: int = 150,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0"
):
    base_path = Path(base_dir)
    extract_dir = base_path / "dataset"
    zip_path = base_path / dataset_zip_name
    yaml_path = extract_dir / "data.yaml"

    print("===================================================================")
    print("🚀 GuardianX: Starting EmberX v3 (YOLO11-OBB) High-Recall Training")
    print("===================================================================")

    # 1. Check or extract dataset
    if not extract_dir.exists():
        if zip_path.exists():
            import zipfile
            print(f"📦 Extracting dataset from {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("✅ Dataset extracted successfully.")
        else:
            print(f"⚠️ Warning: Neither {extract_dir} nor {zip_path} found.")
            print("Ensure dataset is present before continuing.")

    # 2. Update data.yaml paths
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)

        data_cfg['path'] = str(extract_dir.resolve()).replace("\\", "/")
        data_cfg['train'] = 'train/images'
        data_cfg['val'] = 'valid/images'
        data_cfg['test'] = 'test/images'

        with open(yaml_path, 'w') as f:
            yaml.dump(data_cfg, f)
        print(f"✅ Verified data.yaml configuration: {yaml_path}")
    else:
        print(f"❌ Error: data.yaml not found at {yaml_path}")
        return

    # 3. Initialize Model (YOLO11-OBB)
    print(f"🔄 Initializing YOLO11-OBB architecture ({model_variant})...")
    model = YOLO(model_variant)

    # 4. Train with High-Recall SAR Hyperparameters
    print("🔥 Launching intensive training run...")
    results = model.train(
        data=str(yaml_path.resolve()).replace("\\", "/"),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        name="emberx_v3_yolo11_high_recall",
        
        # --- Optimizer & Learning Rate Schedule ---
        optimizer="AdamW",
        lr0=0.003,            # Fine-tuned initial learning rate for AdamW
        lrf=0.01,             # Cosine decay down to 1% of lr0
        cos_lr=True,          # Cosine annealing for smooth convergence
        warmup_epochs=4.0,    # Warmup periods for stable gradient propagation
        patience=45,          # Early stopping patience
        
        # --- Loss Weights (Life-Safety Prioritization) ---
        box=9.5,              # High box regression loss for exact OBB angle localization
        cls=1.8,              # Boosted classification loss to penalize missed victims
        dfl=1.5,              # Distribution Focal Loss for sharper bounding box boundaries
        
        # --- Advanced Augmentations for Low-Visibility / Dense Smoke ---
        mosaic=1.0,           # 4-image mosaic: forces detector to locate small objects
        mixup=0.20,           # Alpha-blend images to simulate translucent smoke plumes
        copy_paste=0.15,      # Segment copy-paste to train on heavily cluttered scenes
        scale=0.6,            # Random scale (+/- 60%) to learn both distant & close victims
        fliplr=0.5,           # Horizontal flip
        degrees=15.0,         # Rotation variation for arbitrary victim posture
        perspective=0.0005,   # Drone perspective distortion
        hsv_h=0.015,          # Tone variations
        hsv_s=0.7,            # Smoke saturation drops
        hsv_v=0.4,            # Shadow and extreme darkness variation
        
        # --- Execution Settings ---
        val=True,
        save=True,
        save_period=10,
        cache=False,          # Set to True if RAM >= 32GB
        workers=4
    )

    print("\n===================================================================")
    print("🎉 Training Complete!")
    print(f"Best weights saved at: runs/obb/emberx_v3_yolo11_high_recall/weights/best.pt")
    print("===================================================================")
    return results

if __name__ == "__main__":
    # Check if running in Google Colab or local environment
    base = "/content" if os.path.exists("/content") else os.path.dirname(os.path.abspath(__file__))
    setup_training(base_dir=base)
