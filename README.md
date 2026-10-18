# 🔥 GuardianX: Autonomous AI-Driven Fireground Rescue Triage & Tactical Dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8/YOLO11](https://img.shields.io/badge/YOLO-v8%20%2F%2011%20OBB-0052FF.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GuardianX** is an AI-driven decision-support platform designed to optimize Search-and-Rescue (SAR) triage and situational awareness during high-rise and structural building fires. Utilizing custom **EmberX (YOLOv8-OBB / YOLO11-OBB)** computer vision models and Slicing Aided Hyper Inference (SAHI), GuardianX converts raw indoor visual feeds (UAV/Drone, POV cameras, thermal/RGB CCTV) into an action-oriented tactical command dashboard.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Model Performance & Metrics](#-model-performance--metrics)
- [Triage Prioritization Algorithm](#-triage-prioritization-algorithm)
- [SAHI & High-Recall Inference](#-sahi--high-recall-inference)
- [Installation](#-installation)
- [Usage & Quick Start](#-usage--quick-start)
  - [1. Tactical Dashboard (Streamlit)](#1-tactical-dashboard-streamlit)
  - [2. Flask REST Server](#2-flask-rest-server)
- [Training Pipeline](#-training-pipeline)
- [Repository Structure](#-repository-structure)
- [Safety & Operational Disclaimer](#-safety--operational-disclaimer)
- [Citation & Acknowledgements](#-citation--acknowledgements)

---

## 🚀 Overview

High-rise and complex structural fires present severe egress and reconnaissance challenges. Traditional visual sweeps under extreme smoke density, heat, and falling debris force incident commanders to make high-stakes decisions with incomplete information.

**GuardianX** bridges the gap between raw object detection confidence scores and operational decision support. Rather than presenting raw numbers to emergency personnel, GuardianX:
1. Detects **humans**, **fire**, and **smoke** using Oriented Bounding Boxes (OBB).
2. Computes a continuous, life-safety-weighted **Priority Score**.
3. Classifies rooms and streaming video feeds into a structured **6-Tier Emergency Triage Hierarchy**.
4. Generates automated **Dispatch Recommendations** directly for Search & Rescue (SAR) and Fire Suppression units.

---

## ✨ Key Features

- **Oriented Bounding Box (OBB) Detection**: Built on Ultralytics YOLOv8-OBB and YOLO11-OBB (`emberx_v2.pt`, `emberx_v3`) to precisely detect rotated objects and victim postures in complex, cluttered scenes.
- **Asymmetric Confidence Thresholding**: Customized thresholds prioritizing life-safety:
  - `Human`: **0.22** (aggresively catches distant/occluded victims)
  - `Smoke`: **0.20** (detects diffuse, amorphous smoke plumes)
  - `Fire`: **0.35** (precision detection for active combustion)
- **Multi-Room & Multi-Feed Triage**: Monitors multiple interior room streams simultaneously (e.g., GoPro, Drone, POV feeds), ranking locations by rescue urgency.
- **Stratified Temporal Jitter Frame Sampling**: Samples 12–24 jittered time-windows per video stream to catch dynamic hazard peaks without video processing bottlenecks.
- **Slicing Aided Hyper Inference (SAHI)**: Integrates 640x640 tile slicing with Rotated Non-Maximum Suppression (NMS) in `sahi_obb_inference.py` to isolate small or distant victims in high-resolution UAV aerial frames.
- **Dual User Interfaces**:
  - **Streamlit Tactical Dashboard** (`app.py`): Glassmorphism incident command dashboard with live room ranking and dispatch prompts.
  - **Flask REST API Server** (`server.py`): Light web service with REST endpoints for remote deployment and feed analysis.

---

## 🏗 System Architecture & Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │   Video / Image Input Feeds (UAVs, CCTVs, POV Feeds)  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stratified Temporal Sampling & SAHI Frame Slicing     │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │   EmberX Model Inference (YOLOv8-OBB / YOLO11-OBB)     │
  │   Class Detections: Human | Fire | Smoke                │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │   GuardianX Triage Algorithm                            │
  │   Priority Score = 0.55(H_det) + 0.25(F_det)            │
  │                  + 0.15(S_det) + 0.05(H_count)          │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │   Command & Dispatch Dashboard (Streamlit / Flask)     │
  │   Urgency Ranking | Bounding Box Overlay | Dispatch UI  │
  └────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance & Metrics

The custom-trained **EmberX** model was trained on a dataset of **9,749** fireground images (8,001 training, 1,017 validation, 731 testing).

| Split / Metric | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Set (1,017 imgs)** | **86.4%** | **79.2%** | **84.3%** | **71.5%** |
| **Test Set (731 imgs)** | **84.7%** | **75.0%** | **82.7%** | **61.3%** |

### Computational Throughput (NVIDIA® T4 GPU)
- **Preprocessing Latency**: 2.1 - 5.2 ms
- **Inference Latency (YOLOv8-OBB @ 640x640)**: 18.9 - 19.3 ms
- **Postprocessing & Rotated NMS**: 2.2 - 3.4 ms
- **Total Pipeline Latency**: **~25 ms per frame (~40 FPS)**

---

## 🧮 Triage Prioritization Algorithm

GuardianX converts detection counts into a continuous priority score and assigns rooms to an emergency triage hierarchy:

$$\text{Priority Score} = 0.55(H_{\text{det}}) + 0.25(F_{\text{det}}) + 0.15(S_{\text{det}}) + 0.05(H_{\text{count}})$$

Where $H_{\text{det}}$, $F_{\text{det}}$, and $S_{\text{det}}$ are binary presence indicators ($1$ if count $\ge 1$, else $0$), and $H_{\text{count}}$ is the victim count.

### Emergency Triage Hierarchy & Dispatch Rules

| Tier | Priority Tier Badge | Conditions | Tactical Dispatch Instructions |
| :---: | :--- | :--- | :--- |
| **P1** | **Critical (Multi-Victim)** | $\ge 2$ Humans, $\ge 1$ Fire | **IMMEDIATE DISPATCH**: Search & Rescue Team (SAR) + Fire Suppression. |
| **P2** | **Critical (Single-Victim)** | $1$ Human, $\ge 1$ Fire | **IMMEDIATE DISPATCH**: Search & Rescue Team (SAR) + Fire Suppression. |
| **P3** | **Lower Category A (Multi-Victim)** | $\ge 2$ Humans, $0$ Fire | **DISPATCH**: Search & Rescue Team (SAR) (Multiple Victims, No Fire Visible). |
| **P4** | **Lower Category A (Single-Victim)** | $1$ Human, $0$ Fire | **DISPATCH**: Search & Rescue Team (SAR) (Single Victim, No Fire Visible). |
| **P5** | **Lower Category B / Smoke Hazard** | $0$ Humans, $\ge 1$ Fire / Smoke | **DISPATCH**: Fire Suppression / Tactical Ventilation Team. |
| **P6** | **Standard Monitoring** | $0$ Humans, $0$ Fire, $0$ Smoke | **MONITOR**: Standard surveillance; no active life threat detected. |

---

## 🔍 SAHI & High-Recall Inference

For distant victims or low-visibility high-resolution aerial frames from indoor UAVs, standard downscaling to 640x640 can cause small objects to blur or disappear. GuardianX provides `sahi_obb_inference.py`, implementing **Slicing Aided Hyper Inference**:
- Slices input images into overlapping 640x640 patches.
- Runs inference on both full-frame and sliced patches.
- Translates crop-local coordinates to global canvas space.
- Merges detections using **Rotated Non-Maximum Suppression (NMS)** with Shapely polygon IoU checks.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU acceleration recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/kishankannanblr-cell/ai-human-detection-model.git
cd ai-human-detection-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥 Usage & Quick Start

### 1. Tactical Dashboard (Streamlit)
Launch the interactive decision-support interface:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501`.

**Dashboard Features**:
- **Multi-Room SAR Triage**: Simultaneously analyzes live/local room streams (e.g., Rooms 1–4) and displays top incident moments along with SAR rankings.
- **Image Triage & Ranking**: Upload custom fireground photos or test pre-loaded samples 1–6 to generate annotated bounding box overlays and dispatch prompts.

### 2. Flask REST Server
Alternatively, launch the lightweight Flask backend:
```bash
python server.py
```
Access the web dashboard at `http://localhost:5000`.

---

## 🏋️ Training Pipeline

GuardianX provides modular scripts for training both local and cloud-based models:

- **Local Training** (`train_local.py` / `train_emberx_v3_yolo11.py`):
  Trains YOLOv8-OBB or YOLO11-OBB with loss reweighting (amplified box & class loss) and augmentations tailored for smoke and low-visibility (Mosaic 1.0, Mixup 0.2, Copy-Paste 0.15).
  ```bash
  python train_emberx_v3_yolo11.py
  ```
- **Google Colab Notebooks**:
  - `Deploy_EmberX_Colab.ipynb`: Interactive deployment notebook.
  - `Train_EmberX_Colab.ipynb`: Cloud training notebook with GPU support.

---

## 📁 Repository Structure

```
.
├── app.py                         # Streamlit Tactical Fireground Triage Dashboard
├── server.py                      # Flask REST Web Application & Endpoints
├── sahi_obb_inference.py          # Slicing Aided Hyper Inference (SAHI) with Rotated NMS
├── train_emberx_v3_yolo11.py      # High-Recall YOLO11-OBB Training Pipeline
├── train_local.py                 # Local YOLOv8-OBB Training Script
├── train_colab_v2.py              # Cloud Colab Training Script
├── emberx_v2.pt                   # Pre-trained EmberX v2 YOLOv8-OBB Weights
├── Deploy_EmberX_Colab.ipynb      # Deployment Colab Notebook
├── Train_EmberX_Colab.ipynb       # Training Colab Notebook
├── extracted_paper_text.txt       # GuardianX Research Paper Text
├── results_and_discussion.md      # Quantitative Results & Discussion Report
├── requirements.txt               # Dependencies
├── static/                        # Web Assets & Sample Videos (Room 1–4)
└── templates/                     # Flask HTML Templates (index.html)
```

---

## ⚠️ Safety & Operational Disclaimer

> **CAUTION FOR FIRST RESPONDERS**:
> GuardianX is designed strictly as a **secondary decision-support tool** to assist incident command and triage operators. First responders must not rely solely on automated computer vision outputs. Structural collapse, heavy particulate smoke occlusion, camera lens contamination, or signal degradation can result in undetected victims (false negatives). All automated dispatches must operate under a **human-in-the-loop** protocol.

---

## 📜 Citation & Acknowledgements

### Author
**Kishan Kannan** — *Greenwood High International, Bangalore*

### Acknowledgements & References
- **The Innovation Story (TIS)**: Computational advice, dataset insights, and problem identification for EmberX.
- **Ultralytics YOLO**: YOLOv8 & YOLO11 OBB architecture support.
- **Roboflow**: Fire, Smoke, and Human dataset hosting and augmentation pipeline.

If you use GuardianX in your research, please cite:
```bibtex
@article{kannan2026guardianx,
  title={GuardianX: Autonomous AI-Driven Human Detection and Room-Priority Rescue Triage in High-Rise Building Fires},
  author={Kannan, Kishan},
  year={2026}
}
```
