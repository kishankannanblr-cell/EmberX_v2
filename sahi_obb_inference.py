# sahi_obb_inference.py
"""
GuardianX - Slicing Aided Hyper Inference (SAHI) for Oriented Bounding Boxes (OBB)
==================================================================================
Solves small-object occlusion and distant victim detection in high-resolution drone frames.
Slices high-res frames into overlapping 640x640 patches, runs EmberX OBB inference,
translates coordinates back to global canvas, and merges using Rotated NMS.
"""

import math
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Any
from ultralytics import YOLO

class SahiOBBInference:
    def __init__(
        self,
        model_path: str = "emberx_v2.pt",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        conf_thresholds: Dict[str, float] = None,
        iou_threshold: float = 0.45
    ):
        self.model = YOLO(model_path)
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.iou_threshold = iou_threshold
        
        # Asymmetric confidence thresholds prioritizing human life-safety
        self.conf_thresholds = conf_thresholds or {
            "human": 0.22,  # Low threshold to capture occluded/distant victims
            "fire": 0.35,   # High precision threshold for active combustion
            "smoke": 0.20   # Lower threshold for diffuse, amorphous smoke plumes
        }
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'fire', 1: 'human', 2: 'smoke'}

    def _generate_slices(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping bounding box coordinates for slicing."""
        step_x = int(self.slice_width * (1.0 - self.overlap_width_ratio))
        step_y = int(self.slice_height * (1.0 - self.overlap_height_ratio))
        
        slices = []
        y = 0
        while y < img_h:
            y_end = min(y + self.slice_height, img_h)
            y_start = max(0, y_end - self.slice_height)
            
            x = 0
            while x < img_w:
                x_end = min(x + self.slice_width, img_w)
                x_start = max(0, x_end - self.slice_width)
                slices.append((x_start, y_start, x_end, y_end))
                if x_end >= img_w:
                    break
                x += step_x
                
            if y_end >= img_h:
                break
            y += step_y
            
        return slices

    def _rotated_box_iou(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two 4-point OBB polygons."""
        from shapely.geometry import Polygon
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            if not p1.is_valid or not p2.is_valid:
                return 0.0
            inter_area = p1.intersection(p2).area
            union_area = p1.area + p2.area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception:
            return 0.0

    def _rotated_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Rotated Non-Maximum Suppression per class."""
        if not detections:
            return []
            
        # Group by class
        by_class = {}
        for det in detections:
            c = det["class_name"]
            by_class.setdefault(c, []).append(det)

        keep_detections = []
        for c, dets in by_class.items():
            # Sort descending by confidence
            dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
            kept = []
            for candidate in dets:
                cand_poly = np.array(candidate["polygon"], dtype=np.float32)
                overlap = False
                for existing in kept:
                    exist_poly = np.array(existing["polygon"], dtype=np.float32)
                    if self._rotated_box_iou(cand_poly, exist_poly) > self.iou_threshold:
                        overlap = True
                        break
                if not overlap:
                    kept.append(candidate)
            keep_detections.extend(kept)

        return keep_detections

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run SAHI + Full-Frame inference on a PIL image.
        Returns:
            Dictionary with aggregated detections, counts, and GuardianX triage priority.
        """
        img_np = np.array(image)
        img_h, img_w = img_np.shape[:2]
        
        all_raw_detections = []

        # 1. Full-frame inference (captures large fire and smoke plumes)
        full_res = self.model.predict(img_np, verbose=False, imgsz=640)[0]
        if hasattr(full_res, 'obb') and full_res.obb is not None:
            for i, box in enumerate(full_res.obb):
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                
                # Check asymmetric confidence threshold
                if conf < self.conf_thresholds.get(cls_name, 0.3):
                    continue
                    
                poly = box.xyxyxyxy[0].cpu().numpy().tolist()
                all_raw_detections.append({
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "polygon": poly,
                    "source": "full_frame"
                })

        # 2. Sliced patch inference (recovers small and occluded victims)
        slices = self._generate_slices(img_w, img_h)
        for (x_start, y_start, x_end, y_end) in slices:
            crop = img_np[y_start:y_end, x_start:x_end]
            crop_res = self.model.predict(crop, verbose=False, imgsz=640)[0]
            
            if hasattr(crop_res, 'obb') and crop_res.obb is not None:
                for i, box in enumerate(crop_res.obb):
                    cls_id = int(box.cls[0].item())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    
                    if conf < self.conf_thresholds.get(cls_name, 0.3):
                        continue
                        
                    poly_local = box.xyxyxyxy[0].cpu().numpy()
                    # Offset local crop coordinates back to global canvas
                    poly_global = poly_local + np.array([x_start, y_start])
                    all_raw_detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "polygon": poly_global.tolist(),
                        "source": "sliced_patch"
                    })

        # 3. Rotated NMS to eliminate duplicate detections across slices
        final_detections = self._rotated_nms(all_raw_detections)

        # 4. GuardianX Triage Priority Formula Calculation
        # Priority Score = 0.55*(H_det) + 0.25*(F_det) + 0.15*(S_det) + 0.05*(H_count)
        human_count = sum(1 for d in final_detections if d["class_name"] == "human")
        has_fire = any(d["class_name"] == "fire" for d in final_detections)
        has_smoke = any(d["class_name"] == "smoke" for d in final_detections)
        has_human = human_count > 0

        h_det = 1.0 if has_human else 0.0
        f_det = 1.0 if has_fire else 0.0
        s_det = 1.0 if has_smoke else 0.0

        raw_score = (0.55 * h_det) + (0.25 * f_det) + (0.15 * s_det) + (0.05 * min(human_count, 5))
        priority_score = min(round(raw_score, 3), 1.0)

        # Classify into GuardianX 6-Tier Triage Hierarchy
        if has_fire and human_count >= 2:
            triage_tier = "Priority 1: Critical (Multi-Victim)"
            action = "Immediate SAR dispatch + Fire Suppression"
            tier_num = 1
        elif has_fire and human_count == 1:
            triage_tier = "Priority 2: Critical (Single-Victim)"
            action = "Immediate SAR dispatch + Fire Suppression"
            tier_num = 2
        elif (not has_fire) and human_count >= 2:
            triage_tier = "Priority 3: Lower Category A (Multi-Victim)"
            action = "Emergency SAR dispatch"
            tier_num = 3
        elif (not has_fire) and human_count == 1:
            triage_tier = "Priority 4: Lower Category A (Single-Victim)"
            action = "Emergency SAR dispatch"
            tier_num = 4
        elif has_fire and human_count == 0:
            triage_tier = "Priority 5: Lower Category B (Fire Only)"
            action = "Fire Suppression dispatch"
            tier_num = 5
        else:
            triage_tier = "Priority 6: Lowest / Monitoring"
            action = "Routine tactical monitoring"
            tier_num = 6

        return {
            "detections": final_detections,
            "counts": {
                "human": human_count,
                "fire": sum(1 for d in final_detections if d["class_name"] == "fire"),
                "smoke": sum(1 for d in final_detections if d["class_name"] == "smoke")
            },
            "triage": {
                "priority_score": priority_score,
                "tier_number": tier_num,
                "triage_tier": triage_tier,
                "recommended_action": action
            },
            "slice_count": len(slices)
        }

if __name__ == "__main__":
    print("Testing SAHI OBB Inference...")
    import os
    if os.path.exists("emberx_v2.pt"):
        sahi = SahiOBBInference("emberx_v2.pt")
        print("SAHI OBB Inference initialized successfully with emberx_v2.pt!")
    else:
        print("emberx_v2.pt not found in current directory.")
