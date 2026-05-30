import cv2
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Returns list of person bounding boxes [x1,y1,x2,y2]."""
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = r
            if int(cls_id) == 0 and score > 0.5:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes

    def detect_all(self, frame):
        """Returns (person_boxes, raw_yolo_result) — raw result exposes all classes."""
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = r
            if int(cls_id) == 0 and score > 0.5:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes, results


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces
