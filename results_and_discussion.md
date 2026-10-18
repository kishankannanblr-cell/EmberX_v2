# 3. Results and Discussion

## 3.1. Dataset Characteristics and Experimental Setup
The custom object detection model, designated as **EmberX**, was trained on a structural fire-ground dataset containing three distinct annotated classes: *fire*, *human*, and *smoke*. The dataset comprises a total of **9,749** images, which were partitioned into a training set of **8,001** images ($82.1\%$), a validation set of **1,017** images ($10.4\%$), and an independent test set of **731** images ($7.5\%$). 

Training was performed using a transfer learning approach based on the **Ultralytics YOLOv8-OBB (Oriented Bounding Box)** architecture. The OBB framework is critical for emergency scene detection, as standard axis-aligned bounding boxes fail to accurately localize humans and fire propagation paths in arbitrary orientations. Training was accelerated on a Google Colab environment utilizing an **NVIDIA® T4 GPU (16 GB VRAM)** with Automatic Mixed Precision (AMP) enabled. The learning rate was governed by a cosine decay scheduler, starting at an initial rate ($\text{lr}_0$) of $0.005$ and decaying to a final rate ($\text{lr}_f$) of $1\%$ of the initial value. Advanced data augmentations were applied to prevent overfitting, including Mosaic augmentation ($1.0$), Mixup ($0.15$), random scaling ($\pm 50\%$), and horizontal flips.

---

## 3.2. Quantitative Detection Performance
The model's performance was evaluated using standard computer vision metrics, including Precision (P), Recall (R), mean Average Precision at an IoU threshold of 0.5 ($\text{mAP}_{50}$), and mean Average Precision across the IoU range from 0.5 to 0.95 ($\text{mAP}_{50-95}$). Table 1 summarizes the performance of the model on the validation and test splits.

### Table 1: EmberX Object Detection Performance Metrics

| Dataset Split | Precision (P) | Recall (R) | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ |
| :--- | :---: | :---: | :---: | :---: |
| **Validation ($1,017$ images)** | $0.864$ ($86.4\%$) | $0.792$ ($79.2\%$) | **$0.843$ ($84.3\%$)** | $0.715$ ($71.5\%$) |
| **Test ($731$ images)** | $0.847$ ($84.7\%$) | $0.750$ ($75.0\%$) | **$0.827$ ($82.7\%$)** | $0.613$ ($61.3\%$) |

The model achieved a test-set $\text{mAP}_{50}$ of **$0.827$**, representing highly robust detection capabilities across the three hazard classes in unseen test environments. The slight decrease in performance from validation to test splits is typical of real-world datasets, where testing images represent distinct, out-of-distribution visual environments.

---

## 3.3. Inference Speed and Real-Time Suitability
For operational search-and-rescue (SAR) deployment, the system must process streaming video feeds at real-time speeds with minimal latency. Table 2 details the computational latency per frame when processing images of size $640 \times 640$ pixels on the test hardware.

### Table 2: Computational Latency and Throughput (NVIDIA® T4 GPU)

| Processing Stage | Latency per Frame ($\text{ms}$) |
| :--- | :---: |
| Preprocessing | $2.1 - 5.2$ |
| Inference (YOLOv8-OBB) | $18.9 - 19.3$ |
| Postprocessing | $2.2 - 3.4$ |
| **Total Pipeline Latency** | **$23.2 - 27.9$ (~40 FPS)** |

With a total latency of approximately **$25\text{ ms}$ per frame**, the complete detection pipeline achieves a throughput of approximately **$40$ frames per second (FPS)**. This throughput satisfies the real-time threshold ($>30\text{ FPS}$) and guarantees that the GuardianX software can process high-definition video streams from unmanned aerial vehicles (UAVs) or stationary thermal cameras without introducing processing bottlenecks.

---

## 3.4. Triage Prioritization Logic and Decision Support
The primary technical contribution of the GuardianX platform is the conversion of raw, uncontextualized bounding box coordinates into actionable room-priority classifications. The priority score for each monitored room is computed using a weighted rule-based triage formula:

$$\text{Priority Score} = 0.55(H_{\text{det}}) + 0.25(F_{\text{det}}) + 0.15(S_{\text{det}}) + 0.05(H_{\text{count}})$$

where $H_{\text{det}}$, $F_{\text{det}}$, and $S_{\text{det}}$ are binary indicators ($0$ or $1$) representing the detection of humans, fire, and smoke, respectively, and $H_{\text{count}}$ represents the absolute count of victims detected. 

Using this calculated priority score and the presence of hazards, the GuardianX system classifies each scene into one of six emergency triage tiers, routing direct dispatch guidelines to the tactical command dashboard:

1. **Priority 1: Critical (Multi-Victim)** — Active fire combined with two or more trapped victims. 
   * *Action: Immediate Search & Rescue (SAR) dispatch + Fire Suppression.*
2. **Priority 2: Critical (Single-Victim)** — Active fire combined with one trapped victim. 
   * *Action: Immediate SAR dispatch + Fire Suppression.*
3. **Priority 3: Lower Category A (Multi-Victim)** — Two or more trapped victims detected with no visible fire. 
   * *Action: Emergency SAR dispatch.*
4. **Priority 4: Lower Category A (Single-Victim)** — One trapped victim detected with no visible fire. 
   * *Action: Emergency SAR dispatch.*
5. **Priority 5: Lower Category B** — Active fire detected with no visible victims. 
   * *Action: Fire Suppression dispatch.*
6. **Priority 6: Lowest / Monitoring** — No immediate fire, smoke, or victims detected. 
   * *Action: Routine monitoring.*

This structured approach bridges the gap between raw machine learning output and real-world tactical decisions, translating detection statistics into immediate, unambiguous command protocols.

---

## 3.5. Technical Limitations and Edge Cases
> **CAUTION:** First responders must not rely solely on automated visual detections during search-and-rescue operations. Obstacles such as structural collapse, heavy smoke occlusion, or camera feed failure can result in undetected victims (false negatives). This system must function as a secondary decision-support tool within a human-in-the-loop triage protocol.

Three main technical limitations were identified during experimental testing:
* **Visual Mismatch in Low-Visibility Conditions**: In environments characterized by dense particulate matter or heavy smoke opacity, standard RGB camera feeds experience severe visual degradation. The model's recall drops as human silhouettes become occluded. Future work will integrate long-wave infrared (LWIR) thermal camera feeds, enabling the model to detect thermal signatures of victims through thick smoke.
* **Class Confusion and False Positives**: In complex building fire scenes, debris, shadows, and reflections can occasionally be misclassified as human shapes or smoke. For instance, high-intensity flickering light from fires may be falsely detected as multiple distinct fire sources, artificially inflating the scene's calculated priority score. Implementing temporal smoothing across consecutive video frames can mitigate these transient false detections.
* **Spatial Constraints**: The visual model ranks rooms based on visible, frame-by-frame cues. However, it cannot evaluate structural collapse risks, toxic gas accumulation, oxygen depletion, or blocked exit paths. Consequently, a room prioritized as "Priority 5" (active fire, no victims) might house an adjacent room containing trapped individuals who are hidden from the camera's line of sight.
