import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time


VIDEO_PATHS = ["Video1.mp4","video2.mp4","video3.mp4","video4.mp4"]
YOLO_MODEL_PATH = "yolov10s.pt"
PIXELS_PER_METER = 50.0
SPEED_LIMIT_KMPH = 50
MIN_GREEN_TIME = 10   # for testing
MAX_GREEN_TIME = 30   # for testing
SHOW_GUI = True
WINDOW_SIZE = 1200  # final display size

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
AMBULANCE_CLASS = 0

# ---------------- INIT ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

model = YOLO(YOLO_MODEL_PATH)
model.to(device)
model.fuse()

# Open videos
caps = []
for p in VIDEO_PATHS:
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        print(f"Warning: Cannot open {p}")
        continue
    caps.append(cap)
if len(caps) == 0:
    raise Exception("No valid videos found!")

num_lanes = len(caps)
last_frames = [np.zeros((WINDOW_SIZE//2, WINDOW_SIZE//2, 3), dtype=np.uint8) for _ in range(num_lanes)]
tracker_history = [{} for _ in range(num_lanes)]
lane_wait_time = [0 for _ in range(num_lanes)]  # track stop countdown

active_lane = 0
green_start_time = time.time()
frame_idx = 0

# ---------------- FUNCTIONS ----------------
def resize_keep_aspect(frame, target_size):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    return cv2.resize(frame, (int(w*scale), int(h*scale)))

def merge_frames(frames):
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (max_w, max_h)) for f in frames]
    top = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:]) if len(resized) > 2 else np.zeros_like(top)
    return np.vstack((top, bottom))

def estimate_speed(prev, curr, dt, ppm=PIXELS_PER_METER):
    dx = curr[0]-prev[0]
    dy = curr[1]-prev[1]
    dist_m = np.sqrt(dx*dx + dy*dy)/ppm
    return (dist_m/dt)*3.6 if dt>0 else 0.0

def assign_ids(prev_centroids, curr_centroids, max_dist=50):
    assigned = {}
    used_prev = set()
    vid = max(prev_centroids.keys(), default=0)+1
    for c in curr_centroids:
        best_id = None
        best_d = max_dist
        for pid, pc in prev_centroids.items():
            if pid in used_prev:
                continue
            d = np.linalg.norm(np.array(c)-np.array(pc))
            if d < best_d:
                best_d = d
                best_id = pid
        if best_id is not None:
            assigned[best_id] = c
            used_prev.add(best_id)
        else:
            assigned[vid] = c
            vid += 1
    return assigned

def get_dynamic_green_time(num_vehicles):
    t = int((num_vehicles/10)*MAX_GREEN_TIME)
    return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, t))

# ---------------- MAIN LOOP ----------------
while True:
    frames = []
    frame_idx += 1
    t_now = time.time()
    t_elapsed = t_now - green_start_time

    # --- Read active lane frame ---
    ret, frame = caps[active_lane].read()
    if ret:
        last_frames[active_lane] = resize_keep_aspect(frame, WINDOW_SIZE//2)
    frame_active = last_frames[active_lane]

    # --- Ambulance detection in all lanes ---
    ambulance_detected = False
    for idx, f in enumerate(last_frames):
        results_check = model.predict(f, imgsz=640, conf=0.3, verbose=False)[0]
        classes_check = results_check.boxes.cls.cpu().numpy() if hasattr(results_check.boxes,"cls") else []
        if any(c == AMBULANCE_CLASS for c in classes_check):
            if idx != active_lane:
                active_lane = idx
                green_start_time = t_now
            ambulance_detected = True
            break

    # --- Process active lane ---
    results = model.predict(frame_active, imgsz=640, conf=0.3, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes,"xyxy") else []
    classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes,"cls") else []

    curr_centroids = []
    valid_boxes = []
    valid_classes = []
    for i, box in enumerate(boxes):
        cls = int(classes[i])
        if cls not in VEHICLE_CLASSES and cls != AMBULANCE_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box)
        centroid = ((x1+x2)//2,(y1+y2)//2)
        curr_centroids.append(centroid)
        valid_boxes.append((x1,y1,x2,y2))
        valid_classes.append(cls)

    assigned = assign_ids({k:v["centroid"] for k,v in tracker_history[active_lane].items()},
                          curr_centroids)
    new_hist = {}
    for vid, centroid in assigned.items():
        speed = 0.0
        if vid in tracker_history[active_lane]:
            dt = t_now - tracker_history[active_lane][vid]["time"]
            speed = estimate_speed(tracker_history[active_lane][vid]["centroid"], centroid, dt)
        new_hist[vid] = {"centroid": centroid, "time": t_now}

        # Draw bounding boxes
        for i,(box,cls) in enumerate(zip(valid_boxes, valid_classes)):
            if curr_centroids[i] != centroid:
                continue
            x1, y1, x2, y2 = box
            color = (0,0,255) if cls == AMBULANCE_CLASS else (0,255,0)
            cv2.rectangle(frame_active,(x1,y1),(x2,y2),color,1)
            cv2.putText(frame_active,f"ID:{vid} {int(speed)}km/h",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    tracker_history[active_lane] = new_hist

    # --- Display frames and timers for all lanes ---
    for idx, f in enumerate(last_frames):
        display_frame = f.copy()
        count = len(tracker_history[idx])
        if idx == active_lane:
            remaining = int(get_dynamic_green_time(count) - t_elapsed)
            text = f"Lane {idx} GO: {remaining}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8,25), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            frames.append(display_frame)
        else:
            remaining = int(get_dynamic_green_time(len(tracker_history[active_lane])) - t_elapsed)
            text = f"Lane {idx} STOP: {max(0, remaining)}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8,25), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            frames.append(display_frame)

    merged_frame = merge_frames(frames)
    if SHOW_GUI:
        cv2.imshow("Traffic Simulation", merged_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Switch lane based on dynamic green time ---
    t_max = get_dynamic_green_time(len(tracker_history[active_lane]))
    if t_elapsed > t_max and not ambulance_detected:
        active_lane = (active_lane + 1) % num_lanes
        green_start_time = t_now
