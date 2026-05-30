# services/processing.py

import cv2
from ultralytics import YOLO

from services.alert_service import handle_alert
from tracker import CentroidTracker
from detectors.zone_intrusion import ZoneIntrusionDetector


# ---------------- INIT MODELS (LOAD ONCE) ---------------- #
model = YOLO("yolov8n.pt")  # lightweight model

tracker = CentroidTracker()
zone_detector = ZoneIntrusionDetector()


# ---------------- MAIN PROCESS FUNCTION ---------------- #
def process_frame(frame, camera_id="store_front", user_email=None):
    """
    Full pipeline:
    - Detect people (YOLO)
    - Track objects (Centroid Tracker)
    - Detect zone intrusion
    - Trigger alerts
    - Draw everything
    """

    # ---------------- DETECTION ---------------- #
    results = model(frame, verbose=False)[0]

    boxes = []
    people_count = 0

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                people_count += 1

    # ---------------- TRACKING ---------------- #
    objects = tracker.update(boxes)

    # Convert to required format for zone detection
    tracked_objects = {
        obj_id: {"centroid": centroid}
        for obj_id, centroid in objects.items()
    }

    # ---------------- ZONE INTRUSION ---------------- #
    intrusions = zone_detector.detect_intrusions(camera_id, tracked_objects)

    # ---------------- DRAW BOXES ---------------- #
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------- DRAW TRACK IDs ---------------- #
    for obj_id, centroid in objects.items():
        cx, cy = centroid
        cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # ---------------- DRAW ZONES ---------------- #
    frame = zone_detector.draw_zones(frame, camera_id)

    # ---------------- CROWD ALERT ---------------- #
    if people_count > 5:
        alert_text = f"Crowd detected ({people_count} people)"
        handle_alert("crowd", alert_text)

        cv2.putText(frame, "⚠️ CROWD ALERT", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------- INTRUSION ALERT ---------------- #
    for intrusion in intrusions:
        cx, cy = intrusion["centroid"]

        handle_alert(
            "intrusion",
            f"Object {intrusion['object_id']} entered zone {intrusion['zone_id']}"
        )

        cv2.putText(frame, "🚨 INTRUSION", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ---------------- DISPLAY COUNT ---------------- #
    cv2.putText(frame, f"People: {people_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame