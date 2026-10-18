import os
import io
import math
import shutil
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import yt_dlp
import cv2

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
MODEL_ID = "fire-smoke-and-human-detector-cokiv/1"
LOCAL_MODEL_PATH = "emberx_v1.pt"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
SAMPLES_DIR = os.path.join(STATIC_DIR, 'samples')
ANNOTATED_DIR = os.path.join(STATIC_DIR, 'annotated')
DATASET_TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test', 'images')

# Colors for bounding boxes
CLASS_COLORS = {
    "fire": "#ef4444",   # Vivid Red
    "human": "#3b82f6",  # Electric Blue
    "smoke": "#9ca3af"   # Cool Gray
}

# Ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Auto-copy sample images from test dataset if samples folder is empty
def initialize_samples():
    if not os.path.exists(DATASET_TEST_DIR):
        print(f"Dataset test directory not found at {DATASET_TEST_DIR}. Skipping auto-copy.")
        return
        
    existing_samples = os.listdir(SAMPLES_DIR)
    if len(existing_samples) < 5:
        print("Initializing sample images from test dataset...")
        test_images = [f for f in os.listdir(DATASET_TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Choose a variety of files to showcase different scenarios
        # We want some containing humans, some containing fire, some containing smoke
        selected_files = []
        
        # Try to find specific patterns or just grab diverse files
        for img in test_images:
            if len(selected_files) >= 8:
                break
            # Look for distinctive names (like new_fire_*, middle_*, image_*, img-*)
            selected_files.append(img)
            
        for f in selected_files:
            src = os.path.join(DATASET_TEST_DIR, f)
            dest = os.path.join(SAMPLES_DIR, f)
            shutil.copy2(src, dest)
        print(f"Copied {len(selected_files)} sample files to {SAMPLES_DIR}")

initialize_samples()

# --- Local YOLO Model ---
from ultralytics import YOLO

_yolo_model = None

def get_yolo_model():
    """Load the local YOLO model (cached after first load)."""
    global _yolo_model
    if _yolo_model is None:
        model_path = os.path.join(BASE_DIR, LOCAL_MODEL_PATH)
        if not os.path.exists(model_path) and os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
        print(f"Loading local YOLO model from {model_path}...")
        _yolo_model = YOLO(model_path)
    return _yolo_model

def query_model_local(image_data):
    """
    Run inference using the local YOLO model.
    Returns a list of prediction dicts matching the expected format.
    """
    model = get_yolo_model()
    img = Image.open(io.BytesIO(image_data))
    results = model(img, verbose=False)[0]
    predictions = []

    # Process OBB (Oriented Bounding Boxes)
    if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
        xywhr = results.obb.xywhr.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()
        clss = results.obb.cls.cpu().numpy()
        names = results.names

        for i in range(len(xywhr)):
            cx, cy, w, h, angle_rad = xywhr[i]
            predictions.append({
                "class": names.get(int(clss[i]), "unknown"),
                "confidence": float(confs[i]),
                "x": float(cx),
                "y": float(cy),
                "width": float(w),
                "height": float(h),
                "angle": float(math.degrees(angle_rad))
            })
    # Process standard Bounding Boxes
    elif hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
        xywh = results.boxes.xywh.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        names = results.names

        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            predictions.append({
                "class": names.get(int(clss[i]), "unknown"),
                "confidence": float(confs[i]),
                "x": float(cx),
                "y": float(cy),
                "width": float(w),
                "height": float(h),
                "angle": 0.0
            })

    return predictions

def get_obb_corners(cx, cy, w, h, angle_deg):
    """Calculates coordinates of the 4 corner points of a rotated bounding box."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    hw = w / 2.0
    hh = h / 2.0
    
    # Unrotated corner offsets relative to center
    dxs = [-hw, hw, hw, -hw]
    dys = [-hh, -hh, hh, hh]
    
    corners = []
    for dx, dy in zip(dxs, dys):
        # Rotate offsets and add center coordinates
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        corners.append((rx, ry))
    return corners

def annotate_image(img_path, predictions, output_path):
    """Annotates an image with bounding boxes (rotated or standard) and labels."""
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if in indexed/grayscale mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            draw = ImageDraw.Draw(img)
            
            # Try to load a clean default font or fallback
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()

            for pred in predictions:
                cls_name = pred.get("class", "unknown").lower()
                conf = pred.get("confidence", 0.0)
                cx = pred.get("x", 0)
                cy = pred.get("y", 0)
                w = pred.get("width", 0)
                h = pred.get("height", 0)
                angle = pred.get("angle", 0.0) # OBB rotation in degrees
                
                color_hex = CLASS_COLORS.get(cls_name, "#10b981")
                # Draw OBB (Oriented Bounding Box)
                corners = get_obb_corners(cx, cy, w, h, angle)
                
                # Draw thick boundary polygon
                draw.polygon(corners, outline=color_hex, width=4)
                
                # Draw label background box
                label_text = f"{cls_name} {int(conf * 100)}%"
                
                # Determine text size using textbbox if available (Pillow 10+), otherwise textsize
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
                    text_w = right - left
                    text_h = bottom - top
                except AttributeError:
                    text_w, text_h = draw.textsize(label_text, font=font)
                
                # Draw label near the top-left corner of the bounding box
                tx, ty = corners[0]
                draw.rectangle([tx, ty - text_h - 4, tx + text_w + 6, ty], fill=color_hex)
                draw.text((tx + 3, ty - text_h - 2), label_text, fill="white", font=font)
                
            img.save(output_path, "JPEG", quality=90)
            return True
    except Exception as e:
        print(f"Error annotating image {img_path}: {e}")
        return False

def calculate_priority(human_count, fire_count):
    """
    Ranks the scene priority based on counts of humans and fire:
    Priority 1: Critical (Multi-Victim) -> Life Threat + Fire, 2+ trapped victims
    Priority 2: Critical (Single-Victim) -> Life Threat + Fire, 1 trapped victim
    Priority 3: Lower Category A (Multi-Victim) -> Trapped Victims (2+), No Visible Fire
    Priority 4: Lower Category A (Single-Victim) -> Trapped Victim (1), No Visible Fire
    Priority 5: Lower Category B -> Active Fire, No Victims Spotted
    Priority 6: Lowest / Monitoring -> No immediate fire or victims
    """
    if human_count >= 2 and fire_count >= 1:
        return {
            "tier": 1,
            "badge": "Critical (Multi-Victim)",
            "color": "danger",
            "dispatch": "IMMEDIATE DISPATCH: Search & Rescue Team (SAR) (Priority Multi-Victim) + Fire Suppression Team."
        }
    elif human_count == 1 and fire_count >= 1:
        return {
            "tier": 2,
            "badge": "Critical (Single-Victim)",
            "color": "danger-single",
            "dispatch": "IMMEDIATE DISPATCH: Search & Rescue Team (SAR) + Fire Suppression Team."
        }
    elif human_count >= 2 and fire_count == 0:
        return {
            "tier": 3,
            "badge": "Lower Category A (Multi-Victim)",
            "color": "warning",
            "dispatch": "DISPATCH: Search & Rescue Team (SAR) (Multiple Victims, No Fire Visible)."
        }
    elif human_count == 1 and fire_count == 0:
        return {
            "tier": 4,
            "badge": "Lower Category A (Single-Victim)",
            "color": "warning-single",
            "dispatch": "DISPATCH: Search & Rescue Team (SAR) (Single Victim, No Fire Visible)."
        }
    elif fire_count >= 1 and human_count == 0:
        return {
            "tier": 5,
            "badge": "Lower Category B (Suppression Only)",
            "color": "info",
            "dispatch": "DISPATCH: Fire Suppression Team (Hose Group - Active Fire, No Victims Spotted)."
        }
    else:
        return {
            "tier": 6,
            "badge": "Standard Monitoring",
            "color": "success",
            "dispatch": "MONITOR: No immediate active fire or life threat spotted. Normal surveillance."
        }

def extract_youtube_frames(url, num_frames=6, output_dir=None):
    """
    Downloads a low-resolution version of a YouTube video to a temp file,
    OR opens a local video file directly if it exists,
    extracts evenly-spaced frames, converts them to RGB/PIL, and returns them
    along with the video title.
    """
    import uuid
    if not output_dir:
        output_dir = ANNOTATED_DIR
        
    is_local = os.path.exists(url)
    
    if is_local:
        temp_video_path = url
        title = os.path.splitext(os.path.basename(url))[0]
    else:
        temp_filename = f"temp_youtube_video_{uuid.uuid4().hex}.mp4"
        temp_video_path = os.path.join(output_dir, temp_filename)
        
        # Clean up any potential files
        for ext in ["", ".part", ".ytdl"]:
            p = temp_video_path + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"Could not remove existing temp file {p}: {e}")
                
        ydl_opts = {
            'format': 'worst[ext=mp4]/worst', # Lowest resolution mp4 for fast download
            'outtmpl': temp_video_path,
            'quiet': True,
            'no_warnings': True,
            'continuedl': False, # Disable resume to prevent range errors
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'YouTube Video')
        except Exception as e:
            print(f"yt-dlp first attempt failed: {e}. Trying fallback formats...")
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'YouTube Video')
                
        if not os.path.exists(temp_video_path):
            raise FileNotFoundError("Failed to download YouTube video stream.")
            
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        if not is_local:
            # Clean up files on error
            for ext in ["", ".part", ".ytdl"]:
                p = temp_video_path + ext
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass
        raise ValueError("Could not open video file with OpenCV.")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    # Safe title for file names
    safe_title = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in title])
    
    if total_frames > 0:
        step = total_frames // (num_frames + 1)
        for i in range(1, num_frames + 1):
            idx = step * i
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                frame_filename = f"yt_frame_{safe_title}_{i}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                pil_img.save(frame_path, "JPEG")
                
                frames.append({
                    "name": f"{title} (Frame {i})",
                    "path": frame_path,
                    "filename": frame_filename
                })
    else:
        count = 0
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 120 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                count += 1
                frame_filename = f"yt_frame_{safe_title}_{count}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                pil_img.save(frame_path, "JPEG")
                
                frames.append({
                    "name": f"{title} (Frame {count})",
                    "path": frame_path,
                    "filename": frame_filename
                })
            idx += 1
            
    cap.release()
    
    if not is_local:
        for ext in ["", ".part", ".ytdl"]:
            p = temp_video_path + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"Could not clean up temp file {p}: {e}")
        
    return frames, title

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Accepts a YouTube URL, extracts frames, performs inference on each,
    ranks them by priority, and returns the sorted results.
    """
    data = request.get_json() or {}
    video_url = data.get("url")
    if not video_url:
        return jsonify({"error": "No YouTube URL provided."}), 400
        
    try:
        frames, title = extract_youtube_frames(video_url, num_frames=6, output_dir=ANNOTATED_DIR)
    except Exception as e:
        print(f"Error extracting YouTube frames: {e}")
        return jsonify({"error": f"Failed to extract frames from YouTube video: {str(e)}"}), 500
        
    results = []
    
    for item in frames:
        img_path = item["path"]
        img_name = item["name"]
        filename = item["filename"]
        
        try:
            with open(img_path, "rb") as image_file:
                img_data = image_file.read()
                
            predictions = query_model_local(img_data)
                
            fire_count = 0
            human_count = 0
            smoke_count = 0
            
            filtered_preds = []
            for pred in predictions:
                if pred.get("confidence", 0.0) >= 0.40:
                    cls_name = pred.get("class", "").lower()
                    filtered_preds.append(pred)
                    if cls_name == "fire":
                        fire_count += 1
                    elif cls_name == "human":
                        human_count += 1
                    elif cls_name == "smoke":
                        smoke_count += 1
                        
            out_filename = f"annotated_{filename}"
            annotated_path = os.path.join(ANNOTATED_DIR, out_filename)
            
            success = annotate_image(img_path, filtered_preds, annotated_path)
            if not success:
                shutil.copy2(img_path, annotated_path)
                
            p_details = calculate_priority(human_count, fire_count)
            
            results.append({
                "original_name": img_name,
                "annotated_url": f"/static/annotated/{out_filename}",
                "counts": {
                    "fire": fire_count,
                    "human": human_count,
                    "smoke": smoke_count
                },
                "priority_tier": p_details["tier"],
                "priority_badge": p_details["badge"],
                "color_class": p_details["color"],
                "dispatch_info": p_details["dispatch"]
            })
            
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
            out_filename = f"annotated_{filename}"
            annotated_path = os.path.join(ANNOTATED_DIR, out_filename)
            if os.path.exists(img_path):
                shutil.copy2(img_path, annotated_path)
            results.append({
                "original_name": img_name,
                "annotated_url": f"/static/annotated/{out_filename}",
                "counts": {"fire": 0, "human": 0, "smoke": 0},
                "priority_tier": 6,
                "priority_badge": "Error Processing",
                "color_class": "success",
                "dispatch_info": f"Error running inference: {str(e)}. Monitoring recommended."
            })
        finally:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass
                    
    results.sort(key=lambda x: x["priority_tier"])
    return jsonify({"results": results, "title": title})

@app.route('/api/analyze-multi-room', methods=['POST'])
def analyze_multi_room():
    """
    Analyzes 4 predefined room videos. For each room, extracts frames, performs
    inference to find the single most critical frame (highest priority), and then
    ranks the 4 selected room frames against each other from most SAR to least SAR.
    """
    rooms = [
        {"id": "Room 1", "url": os.path.join(STATIC_DIR, "Room_1.mp4")},
        {"id": "Room 2", "url": os.path.join(STATIC_DIR, "Room_2.mp4")},
        {"id": "Room 3", "url": os.path.join(STATIC_DIR, "Room_3.mp4")},
        {"id": "Room 4", "url": os.path.join(STATIC_DIR, "Room_4.mp4")}
    ]
    
    room_results = []
    
    for room in rooms:
        url = room["url"]
        room_id = room["id"]
        
        try:
            # Extract 5 frames per room for analysis
            frames, title = extract_youtube_frames(url, num_frames=5, output_dir=ANNOTATED_DIR)
        except Exception as e:
            print(f"Error extracting frames for {room_id}: {e}")
            # Append fallback representation on error
            room_results.append({
                "room_id": room_id,
                "video_title": f"Video ({room_id})",
                "original_name": "N/A",
                "annotated_url": "",
                "counts": {"fire": 0, "human": 0, "smoke": 0},
                "priority_tier": 6,
                "priority_badge": "Connection Error",
                "color_class": "success",
                "dispatch_info": f"Failed to download or parse room video: {str(e)}"
            })
            continue

        best_frame_pred = None
        
        for item in frames:
            img_path = item["path"]
            img_name = item["name"]
            filename = item["filename"]
            
            try:
                with open(img_path, "rb") as image_file:
                    img_data = image_file.read()
                    
                predictions = query_model_local(img_data)
            except Exception as e:
                print(f"Error running local inference for {img_name}: {e}")
                predictions = []
                
            fire_count = 0
            human_count = 0
            smoke_count = 0
            
            filtered_preds = []
            for pred in predictions:
                if pred.get("confidence", 0.0) >= 0.40:
                    cls_name = pred.get("class", "").lower()
                    filtered_preds.append(pred)
                    if cls_name == "fire":
                        fire_count += 1
                    elif cls_name == "human":
                        human_count += 1
                    elif cls_name == "smoke":
                        smoke_count += 1
            
            p_details = calculate_priority(human_count, fire_count)
            
            # Select the most critical frame for this room:
            # We want the lowest priority_tier (1 is most critical, 6 is least).
            # Break ties by choosing more humans, then more fire, then first frame.
            is_better = False
            if best_frame_pred is None:
                is_better = True
            else:
                current_tier = p_details["tier"]
                best_tier = best_frame_pred["priority_tier"]
                if current_tier < best_tier:
                    is_better = True
                elif current_tier == best_tier:
                    current_humans = human_count
                    best_humans = best_frame_pred["counts"]["human"]
                    if current_humans > best_humans:
                        is_better = True
                    elif current_humans == best_humans:
                        current_fire = fire_count
                        best_fire = best_frame_pred["counts"]["fire"]
                        if current_fire > best_fire:
                            is_better = True
            
            if is_better:
                best_frame_pred = {
                    "item": item,
                    "predictions": filtered_preds,
                    "counts": {
                        "fire": fire_count,
                        "human": human_count,
                        "smoke": smoke_count
                    },
                    "priority_tier": p_details["tier"],
                    "priority_badge": p_details["badge"],
                    "color_class": p_details["color"],
                    "dispatch_info": p_details["dispatch"]
                }
        
        # Annotate and save ONLY the chosen best frame for this room
        if best_frame_pred:
            best_item = best_frame_pred["item"]
            img_path = best_item["path"]
            filename = best_item["filename"]
            out_filename = f"annotated_room_{room_id.replace(' ', '_')}_{filename}"
            annotated_path = os.path.join(ANNOTATED_DIR, out_filename)
            
            success = annotate_image(img_path, best_frame_pred["predictions"], annotated_path)
            if not success:
                shutil.copy2(img_path, annotated_path)
                
            room_results.append({
                "room_id": room_id,
                "video_title": title,
                "original_name": best_item["name"],
                "annotated_url": f"/static/annotated/{out_filename}",
                "counts": best_frame_pred["counts"],
                "priority_tier": best_frame_pred["priority_tier"],
                "priority_badge": best_frame_pred["priority_badge"],
                "color_class": best_frame_pred["color_class"],
                "dispatch_info": best_frame_pred["dispatch_info"]
            })
        else:
            room_results.append({
                "room_id": room_id,
                "video_title": title,
                "original_name": "N/A",
                "annotated_url": "",
                "counts": {"fire": 0, "human": 0, "smoke": 0},
                "priority_tier": 6,
                "priority_badge": "Standard Monitoring",
                "color_class": "success",
                "dispatch_info": "No critical frames extracted. Standard monitoring recommended."
            })
            
        # Clean up all extracted frames
        for item in frames:
            img_path = item["path"]
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass
                    
    # Rank rooms from most SAR to least SAR:
    # 1. priority_tier ascending (lower tier value = higher priority)
    # 2. counts.human descending (more humans = higher SAR priority)
    # 3. counts.fire descending (more fire = higher urgency)
    ranked_results = sorted(room_results, key=lambda x: (
        x["priority_tier"],
        -x["counts"]["human"],
        -x["counts"]["fire"]
    ))
    
    return jsonify({
        "results": room_results,
        "ranked_results": ranked_results
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def list_samples():
    """Returns list of available sample images for the quick test interface."""
    try:
        samples = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return jsonify({"samples": sorted(samples)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Accepts uploads (or sample filenames) and performs inference on each.
    Ranks them from highest to lowest priority and returns the annotated images and details.
    """
    files_to_process = []
    
    # Check if files were uploaded
    uploaded_files = request.files.getlist('images')
    if uploaded_files and uploaded_files[0].filename != '':
        for i, file in enumerate(uploaded_files):
            filename = f"upload_{i}_{file.filename}"
            temp_path = os.path.join(ANNOTATED_DIR, f"temp_{filename}")
            file.save(temp_path)
            files_to_process.append({
                "name": file.filename,
                "path": temp_path,
                "is_temp": True
            })
            
    # Check if sample image filenames were provided
    sample_names = request.form.getlist('samples')
    if sample_names:
        for name in sample_names:
            sample_path = os.path.join(SAMPLES_DIR, name)
            if os.path.exists(sample_path):
                temp_path = os.path.join(ANNOTATED_DIR, f"temp_sample_{name}")
                shutil.copy2(sample_path, temp_path)
                files_to_process.append({
                    "name": name,
                    "path": temp_path,
                    "is_temp": True
                })

    if len(files_to_process) < 3:
        for f in files_to_process:
            if f.get("is_temp") and os.path.exists(f["path"]):
                os.remove(f["path"])
        return jsonify({"error": "Please provide 3 or more images for triage analysis."}), 400

    results = []

    for item in files_to_process:
        img_path = item["path"]
        img_name = item["name"]
        
        try:
            with open(img_path, "rb") as image_file:
                img_data = image_file.read()
                
            predictions = query_model_local(img_data)
                
            fire_count = 0
            human_count = 0
            smoke_count = 0
            
            filtered_preds = []
            for pred in predictions:
                if pred.get("confidence", 0.0) >= 0.40:
                    cls_name = pred.get("class", "").lower()
                    filtered_preds.append(pred)
                    if cls_name == "fire":
                        fire_count += 1
                    elif cls_name == "human":
                        human_count += 1
                    elif cls_name == "smoke":
                        smoke_count += 1

            out_filename = f"annotated_{os.path.basename(img_path).replace('temp_', '')}"
            annotated_path = os.path.join(ANNOTATED_DIR, out_filename)
            
            annotate_image(img_path, filtered_preds, annotated_path)
            p_details = calculate_priority(human_count, fire_count)
            
            results.append({
                "original_name": img_name,
                "annotated_url": f"/static/annotated/{out_filename}",
                "counts": {
                    "fire": fire_count,
                    "human": human_count,
                    "smoke": smoke_count
                },
                "priority_tier": p_details["tier"],
                "priority_badge": p_details["badge"],
                "color_class": p_details["color"],
                "dispatch_info": p_details["dispatch"]
            })
            
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
            results.append({
                "original_name": img_name,
                "annotated_url": "",
                "counts": {"fire": 0, "human": 0, "smoke": 0},
                "priority_tier": 6,
                "priority_badge": "Error Processing",
                "color_class": "success",
                "dispatch_info": f"Error running inference: {str(e)}. Monitoring recommended."
            })
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    results.sort(key=lambda x: x["priority_tier"])
    return jsonify({"results": results})

if __name__ == '__main__':
    print("Starting Fireground Triage Dashboard Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
