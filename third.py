import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import os

# ---------------- USER SETTINGS ----------------
VIDEO_PATHS = ["Video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
YOLO_MODEL_PATH = "yolov10s.pt"
PIXELS_PER_METER = 50.0
SPEED_LIMIT_KMPH = 50
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30
SHOW_GUI = True
WINDOW_SIZE = 1200

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
AMBULANCE_CLASS = 0

# ---------------- INIT ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

model = YOLO(YOLO_MODEL_PATH)
model.to(device)
try:
    model.fuse()
except Exception:
    pass

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
lane_next_id = [1 for _ in range(num_lanes)]
lane_wait_time = [0 for _ in range(num_lanes)]
active_lane = 0
green_start_time = time.time()
frame_idx = 0

# ---------------- CSV & Snapshot ----------------
VIOLATION_CSV = "violations_log.csv"
CSV_HEADERS = ["Time", "Lane", "FromSide", "Direction", "VehicleID",
               "Type", "Color", "Plate", "ViolationType", "SnapshotFile"]

SNAPSHOT_DIR = "violation_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if not os.path.exists(VIOLATION_CSV):
    with open(VIOLATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

def save_snapshot(frame, box, lane_idx, vid, vtype, violation_type):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_vtype = str(vtype).replace(" ", "_")
    safe_violation = violation_type.replace(" ", "_").replace("|", "_")
    filename = f"{SNAPSHOT_DIR}/lane{lane_idx}_id{vid}_{safe_vtype}_{safe_violation}_{timestamp}.png"
    try:
        cv2.imwrite(filename, crop)
        return filename
    except Exception as e:
        print(f"Error saving snapshot {filename}: {e}")
        return ""

def log_violation(record):
    with open(VIOLATION_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if isinstance(record, dict):
            row = {k: record.get(k, "") for k in CSV_HEADERS}
            writer.writerow(row)
        else:
            row = dict(zip(CSV_HEADERS, record))
            writer.writerow(row)

# ---------------- UTILITY FUNCTIONS ----------------
def resize_keep_aspect(frame, target_size):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = target_size / max(h, w)
    return cv2.resize(frame, (int(w*scale), int(h*scale)))

def merge_frames(frames):
    if len(frames) == 0:
        return np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (max_w, max_h)) for f in frames]
    if len(resized) == 1:
        top = np.hstack((resized[0], np.zeros_like(resized[0])))
        bottom = np.zeros_like(top)
    elif len(resized) == 2:
        top = np.hstack(resized[:2])
        bottom = np.zeros_like(top)
    else:
        if len(resized) == 3:
            top = np.hstack(resized[:2])
            bottom = np.hstack((resized[2], np.zeros_like(resized[0])))
        else:
            top = np.hstack(resized[:2])
            bottom = np.hstack(resized[2:4])
    return np.vstack((top, bottom))

def estimate_speed(prev, curr, dt, ppm=PIXELS_PER_METER):
    dx = curr[0]-prev[0]
    dy = curr[1]-prev[1]
    dist_m = np.sqrt(dx*dx + dy*dy)/ppm
    return (dist_m/dt)*3.6 if dt>0 else 0.0

def assign_ids(prev_centroids, curr_centroids, starting_vid=1, max_dist=50):
    assigned = {}
    used_prev = set()
    vid = max(prev_centroids.keys(), default=starting_vid-1) + 1
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
    return assigned, vid

def get_dynamic_green_time(num_vehicles):
    t = int((num_vehicles/10)*MAX_GREEN_TIME)
    return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, t))

def get_vehicle_color(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"
    avg_color = crop.mean(axis=(0,1)).astype(int)
    return f"RGB({int(avg_color[2])},{int(avg_color[1])},{int(avg_color[0])})"

def recognize_plate(frame, box):
    return "PLATE-UNKNOWN"

def get_direction(prev, curr):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) > abs(dy):
        return "L→R" if dx > 0 else "R→L"
    else:
        return "T→B" if dy > 0 else "B→T"

def side_of_point(pt, frame_shape, tol=0.2):
    h, w = frame_shape[:2]
    x, y = pt
    left_bound = w * tol
    right_bound = w * (1 - tol)
    top_bound = h * tol
    bottom_bound = h * (1 - tol)
    if x <= left_bound: return "LEFT"
    if x >= right_bound: return "RIGHT"
    if y <= top_bound: return "TOP"
    if y >= bottom_bound: return "BOTTOM"
    dists = {"LEFT": x, "RIGHT": w-x, "TOP": y, "BOTTOM": h-y}
    return min(dists, key=dists.get)

# ---------------- MAIN LOOP ----------------
while True:
    frames = []
    frame_idx += 1
    t_now = time.time()
    t_elapsed = t_now - green_start_time

    # Read frames from all lanes
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            last_frames[idx] = resize_keep_aspect(frame, WINDOW_SIZE//2)

    ambulance_detected = False
    for lane_idx, frame_lane in enumerate(last_frames):
        try:
            results = model.predict(frame_lane, imgsz=640, conf=0.3, verbose=False)[0]
        except Exception as e:
            print(f"YOLO predict error on lane {lane_idx}: {e}")
            results = None

        boxes, classes = [], []
        if results is not None:
            boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, "xyxy") else []
            classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, "cls") else []
        curr_centroids, valid_boxes, valid_classes = [], [], []

        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls not in VEHICLE_CLASSES and cls != AMBULANCE_CLASS: continue
            x1, y1, x2, y2 = map(int, box)
            centroid = ((x1+x2)//2, (y1+y2)//2)
            curr_centroids.append(centroid)
            valid_boxes.append((x1,y1,x2,y2))
            valid_classes.append(cls)

        prev_centroids = {k: v["centroid"] for k,v in tracker_history[lane_idx].items()}
        assigned, next_vid = assign_ids(prev_centroids, curr_centroids, starting_vid=lane_next_id[lane_idx])
        lane_next_id[lane_idx] = next_vid

        new_hist = {}
        for vid, centroid in assigned.items():
            try:
                cidx = curr_centroids.index(centroid)
            except ValueError:
                continue
            box = valid_boxes[cidx]
            cls = valid_classes[cidx]
            speed, direction, origin_side = 0.0, "UNKNOWN", None
            if vid in tracker_history[lane_idx]:
                prev = tracker_history[lane_idx][vid]["centroid"]
                dt = t_now - tracker_history[lane_idx][vid]["time"]
                speed = estimate_speed(prev, centroid, dt)
                direction = get_direction(prev, centroid)
                origin_side = tracker_history[lane_idx][vid].get("origin_side", None)
            else:
                origin_side = side_of_point(centroid, frame_lane.shape)

            new_hist[vid] = {"centroid": centroid, "time": t_now, "origin_side": origin_side,
                              "first_seen": tracker_history[lane_idx].get(vid, {}).get("first_seen", t_now),
                              "last_box": box}

            vehicle_type = "AMBULANCE" if cls==AMBULANCE_CLASS else ("CAR" if cls==2 else ("MOTOR" if cls==3 else ("BUS" if cls==5 else "TRUCK")))
            color = get_vehicle_color(frame_lane, box)
            plate = recognize_plate(frame_lane, box)

            # Ambulance check
            if cls == AMBULANCE_CLASS:
                ambulance_detected = True
                if lane_idx != active_lane:
                    active_lane = lane_idx
                    green_start_time = t_now
                snap_file = save_snapshot(frame_lane, box, lane_idx, vid, vehicle_type, "AMBULANCE_PRESENT_SWITCH_GREEN")
                log_violation({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               "Lane": lane_idx, "FromSide": origin_side if origin_side else "UNKNOWN",
                               "Direction": direction, "VehicleID": vid, "Type": vehicle_type,
                               "Color": color, "Plate": plate, "ViolationType": "AMBULANCE_PRESENT_SWITCH_GREEN",
                               "SnapshotFile": snap_file})

            # Violations
            violation_type = None
            if speed > SPEED_LIMIT_KMPH:
                violation_type = f"Overspeed_{int(speed)}>{SPEED_LIMIT_KMPH}"
            if lane_idx != active_lane and speed > 2.0:
                violation_type = (violation_type + "|RedLight") if violation_type else "RedLight"
            if violation_type:
                snap_file = save_snapshot(frame_lane, box, lane_idx, vid, vehicle_type, violation_type)
                log_violation({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               "Lane": lane_idx, "FromSide": origin_side if origin_side else "UNKNOWN",
                               "Direction": direction, "VehicleID": vid, "Type": vehicle_type,
                               "Color": color, "Plate": plate, "ViolationType": violation_type,
                               "SnapshotFile": snap_file})

            x1,y1,x2,y2 = box
            color_box = (0,0,255) if cls==AMBULANCE_CLASS else (0,255,0)
            cv2.rectangle(frame_lane, (x1,y1), (x2,y2), color_box, 2)
            cv2.putText(frame_lane, f"ID:{vid} {int(speed)}km/h {plate}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,0),1)
            cv2.putText(frame_lane, vehicle_type, (x1, y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200),1)

        tracker_history[lane_idx] = new_hist

    # Display
    frames = []
    for idx,f in enumerate(last_frames):
        display_frame = f.copy()
        count = len(tracker_history[idx])
        if idx == active_lane:
            remaining = int(get_dynamic_green_time(count) - t_elapsed)
            text = f"Lane {idx} GO: {remaining}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.rectangle(display_frame, (2,2), (display_frame.shape[1]-2, display_frame.shape[0]-2), (0,255,0),3)
        else:
            remaining = int(get_dynamic_green_time(len(tracker_history[active_lane])) - t_elapsed)
            text = f"Lane {idx} STOP: {max(0,remaining)}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            cv2.rectangle(display_frame, (2,2), (display_frame.shape[1]-2, display_frame.shape[0]-2), (0,0,255),2)
        frames.append(display_frame)

    merged_frame = merge_frames(frames)
    if SHOW_GUI:
        cv2.imshow("Traffic Simulation", merged_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Correct lane switching ---
    t_max = get_dynamic_green_time(len(tracker_history[active_lane]))
    if not ambulance_detected and t_elapsed > t_max:
        active_lane = (active_lane + 1) % num_lanes
        green_start_time = t_now

# Cleanup
for c in caps:
    c.release()
cv2.destroyAllWindows()
