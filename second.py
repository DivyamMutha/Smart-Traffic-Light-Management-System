import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import os

# ---------------- USER SETTINGS (kept from your original file) ----------------
VIDEO_PATHS = [0, "video2.mp4", "video3.mp4", "video4.mp4"]
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
try:
    model.fuse()
except Exception:
    # some ultralytics versions may not have fuse or require different use; ignore if fails
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

# tracker_history[lane] = { vid: { "centroid": (x,y), "time": t, "origin_side": "LEFT", "first_seen": t, "last_box":(x1,y1,x2,y2) } }
tracker_history = [{} for _ in range(num_lanes)]

# For unique IDs per lane we will keep them local to lane (so IDs don't collide across lanes)
lane_next_id = [1 for _ in range(num_lanes)]

lane_wait_time = [0 for _ in range(num_lanes)]  # track stop countdown

active_lane = 0
green_start_time = time.time()
frame_idx = 0

# ---------------- CSV & Snapshot CONFIG ----------------
VIOLATION_CSV = "violations_log.csv"
CSV_HEADERS = ["Time", "Lane", "FromSide", "Direction", "VehicleID",
               "Type", "Color", "Plate", "ViolationType", "SnapshotFile"]

SNAPSHOT_DIR = "violation_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Ensure CSV has headers before writing (UTF-8 to avoid UnicodeEncodeError)
if not os.path.exists(VIOLATION_CSV):
    with open(VIOLATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

def save_snapshot(frame, box, lane_idx, vid, vtype, violation_type):
    """
    Crop and save snapshot image for a violation.
    Returns the saved filename or empty string on failure.
    """
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
    """
    record: dict with keys = CSV_HEADERS OR list in same order as CSV_HEADERS (last element SnapshotFile may be "")
    This function appends a row to the CSV using UTF-8 encoding.
    """
    with open(VIOLATION_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if isinstance(record, dict):
            # ensure all headers present
            row = {k: record.get(k, "") for k in CSV_HEADERS}
            writer.writerow(row)
        else:
            # assume list in CSV_HEADERS order
            row = dict(zip(CSV_HEADERS, record))
            writer.writerow(row)

# ---------------- FUNCTIONS ----------------
def resize_keep_aspect(frame, target_size):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = target_size / max(h, w)
    return cv2.resize(frame, (int(w*scale), int(h*scale)))

def merge_frames(frames):
    # Accepts up to 4 frames and arranges them 2x2 (or 1xN)
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
        # 3 or 4
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
    """
    prev_centroids: dict {vid: (x,y)}
    curr_centroids: list of (x,y)
    Returns: assigned dict {vid: centroid}, next_vid
    """
    assigned = {}
    used_prev = set()
    vid = max(prev_centroids.keys(), default=starting_vid-1) + 1
    # Try greedy nearest neighbor
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
    # clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"
    avg_color = crop.mean(axis=(0,1)).astype(int)  # BGR
    # convert to human-friendly string
    return f"RGB({int(avg_color[2])},{int(avg_color[1])},{int(avg_color[0])})"

def recognize_plate(frame, box):
    # Placeholder for an ALPR model. Replace this function with a real ALPR call.
    # For now, return a dummy plate or empty if uncertain.
    return "PLATE-UNKNOWN"

def get_direction(prev, curr):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    # prefer major axis
    if abs(dx) > abs(dy):
        return "L→R" if dx > 0 else "R→L"
    else:
        return "T→B" if dy > 0 else "B→T"

def side_of_point(pt, frame_shape, tol=0.2):
    # frame_shape = (h,w,3)
    h, w = frame_shape[:2]
    x, y = pt
    left_bound = w * tol
    right_bound = w * (1 - tol)
    top_bound = h * tol
    bottom_bound = h * (1 - tol)
    # If near left edge
    if x <= left_bound:
        return "LEFT"
    if x >= right_bound:
        return "RIGHT"
    if y <= top_bound:
        return "TOP"
    if y >= bottom_bound:
        return "BOTTOM"
    # otherwise, near center; pick closest edge
    dists = {
        "LEFT": x,
        "RIGHT": w - x,
        "TOP": y,
        "BOTTOM": h - y
    }
    return min(dists, key=dists.get)

# ---------------- MAIN LOOP (updated to process all lanes every iteration) ----------------
while True:
    frames = []
    frame_idx += 1
    t_now = time.time()
    t_elapsed = t_now - green_start_time

    # --- Read all lane frames (this replaces single active read) ---
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            last_frames[idx] = resize_keep_aspect(frame, WINDOW_SIZE//2)

    # --- Run detection & tracking for ALL lanes ---
    ambulance_detected = False
    for lane_idx, frame_lane in enumerate(last_frames):
        # Run YOLO detect per lane frame
        # Some ultralytics versions accept numpy arrays directly
        try:
            results = model.predict(frame_lane, imgsz=640, conf=0.3, verbose=False)[0]
        except Exception as e:
            # If prediction fails for some reason, skip this lane this frame and continue
            print(f"YOLO predict error on lane {lane_idx}: {e}")
            results = None

        boxes = []
        classes = []
        if results is not None:
            boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, "xyxy") else []
            classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, "cls") else []
        curr_centroids = []
        valid_boxes = []
        valid_classes = []

        # Filter detections for vehicles/ambulance
        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls not in VEHICLE_CLASSES and cls != AMBULANCE_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            curr_centroids.append(centroid)
            valid_boxes.append((x1, y1, x2, y2))
            valid_classes.append(cls)

        # Build prev_centroids dict for assign_ids
        prev_centroids = {k: v["centroid"] for k, v in tracker_history[lane_idx].items()}
        # Assign IDs; use lane_next_id to get next vid baseline
        assigned, next_vid = assign_ids(prev_centroids, curr_centroids, starting_vid=lane_next_id[lane_idx])
        lane_next_id[lane_idx] = next_vid

        # Build new history for this lane
        new_hist = {}
        # For matching centroid -> box/class we will use index lookup
        for vid, centroid in assigned.items():
            # Match centroid to index in curr_centroids (safe because values are exact tuples)
            try:
                cidx = curr_centroids.index(centroid)
            except ValueError:
                # should not happen, but skip if cannot match
                continue

            box = valid_boxes[cidx]
            cls = valid_classes[cidx]
            speed = 0.0
            direction = "UNKNOWN"
            origin_side = None

            if vid in tracker_history[lane_idx]:
                prev = tracker_history[lane_idx][vid]["centroid"]
                dt = t_now - tracker_history[lane_idx][vid]["time"]
                speed = estimate_speed(prev, centroid, dt)
                direction = get_direction(prev, centroid)
                origin_side = tracker_history[lane_idx][vid].get("origin_side", None)
            else:
                # new ID, determine origin side using centroid and frame size
                origin_side = side_of_point(centroid, frame_lane.shape)

            # Save history info (update last_box for later reference)
            new_hist[vid] = {
                "centroid": centroid,
                "time": t_now,
                "origin_side": origin_side,
                "first_seen": tracker_history[lane_idx].get(vid, {}).get("first_seen", t_now),
                "last_box": box
            }

            # Visual annotation variables
            vehicle_type = "AMBULANCE" if cls == AMBULANCE_CLASS else ("CAR" if cls==2 else ("MOTOR" if cls==3 else ("BUS" if cls==5 else "TRUCK")))
            color = get_vehicle_color(frame_lane, box)
            plate = recognize_plate(frame_lane, box)

            # Check for ambulance
            if cls == AMBULANCE_CLASS:
                ambulance_detected = True
                if lane_idx != active_lane:
                    active_lane = lane_idx
                    green_start_time = t_now
                # prepare ambulance log and snapshot
                snap_file = save_snapshot(frame_lane, box, lane_idx, vid, vehicle_type, "AMBULANCE_PRESENT_SWITCH_GREEN")
                log_violation({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Lane": lane_idx,
                    "FromSide": origin_side if origin_side else "UNKNOWN",
                    "Direction": direction,
                    "VehicleID": vid,
                    "Type": vehicle_type,
                    "Color": color,
                    "Plate": plate,
                    "ViolationType": "AMBULANCE_PRESENT_SWITCH_GREEN",
                    "SnapshotFile": snap_file
                })

            # Violation checks
            violation_type = None
            # Overspeed
            if speed > SPEED_LIMIT_KMPH:
                violation_type = f"Overspeed_{int(speed)}>{SPEED_LIMIT_KMPH}"

            # Red-light: if this lane is not active (STOP) and vehicle moving (speed > small threshold)
            if lane_idx != active_lane and speed > 2.0:
                # Mark as red-light if moving while not active
                if violation_type:
                    violation_type += "|RedLight"
                else:
                    violation_type = "RedLight"

            # If any violation happened, log it with one line and save snapshot
            if violation_type:
                snap_file = save_snapshot(frame_lane, box, lane_idx, vid, vehicle_type, violation_type)
                record = {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Lane": lane_idx,
                    "FromSide": origin_side if origin_side else "UNKNOWN",
                    "Direction": direction,
                    "VehicleID": vid,
                    "Type": vehicle_type,
                    "Color": color,
                    "Plate": plate,
                    "ViolationType": violation_type,
                    "SnapshotFile": snap_file
                }
                log_violation(record)

            # Draw bounding box & info on lane frame
            x1, y1, x2, y2 = box
            color_box = (0, 0, 255) if cls == AMBULANCE_CLASS else (0, 255, 0)
            cv2.rectangle(frame_lane, (x1, y1), (x2, y2), color_box, 2)
            info_text = f"ID:{vid} {int(speed)}km/h {plate}"
            cv2.putText(frame_lane, info_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)
            cv2.putText(frame_lane, f"{vehicle_type}", (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Update tracker history after processing detections for this lane
        tracker_history[lane_idx] = new_hist

    # --- Display frames and timers for all lanes (keeps your original display semantics) ---
    frames = []
    for idx, f in enumerate(last_frames):
        display_frame = f.copy()
        count = len(tracker_history[idx])
        if idx == active_lane:
            remaining = int(get_dynamic_green_time(count) - t_elapsed)
            text = f"Lane {idx} GO: {remaining}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # optionally mark active lane visually
            cv2.rectangle(display_frame, (2,2), (display_frame.shape[1]-2, display_frame.shape[0]-2), (0,255,0), 3)
            frames.append(display_frame)
        else:
            remaining = int(get_dynamic_green_time(len(tracker_history[active_lane])) - t_elapsed)
            text = f"Lane {idx} STOP: {max(0, remaining)}s | Vehicles: {count}"
            cv2.putText(display_frame, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(display_frame, (2,2), (display_frame.shape[1]-2, display_frame.shape[0]-2), (0,0,255), 2)
            frames.append(display_frame)

    merged_frame = merge_frames(frames)
    if SHOW_GUI:
        cv2.imshow("Traffic Simulation", merged_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Switch lane based on dynamic green time (unchanged logic, but now uses dynamic counts from all lanes) ---
    t_max = get_dynamic_green_time(len(tracker_history[active_lane]))
    if t_elapsed > t_max and not ambulance_detected:
        active_lane = (active_lane + 1) % num_lanes
        green_start_time = t_now

# Cleanup
for c in caps:
    c.release()
cv2.destroyAllWindows()
