# import cv2
# import numpy as np
# import os

# class ObjectDetector:
#     def __init__(self):
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#         proto = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt")
#         weights = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")

#         if not os.path.exists(proto) or not os.path.exists(weights):
#             raise FileNotFoundError("❌ Model files not found in /models folder")

#         self.net = cv2.dnn.readNetFromCaffe(proto, weights)

#         self.classes = [
#             "background", "aeroplane", "bicycle", "bird", "boat",
#             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#             "dog", "horse", "motorbike", "person", "pottedplant",
#             "sheep", "sofa", "train", "tvmonitor"
#         ]

#     def detect_objects(self, frame, conf_thresh=0.6):
#         h, w = frame.shape[:2]

#         blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
#         self.net.setInput(blob)

#         detections = self.net.forward()
#         results = []

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             class_id = int(detections[0, 0, i, 1])

#             if confidence > conf_thresh:
#                 label = self.classes[class_id]

#                 # ⚠️ MobileNetSSD DOES NOT detect bags reliably
#                 if label in ["bottle", "chair", "tvmonitor"]:  # example objects
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (x1, y1, x2, y2) = box.astype("int")

#                     results.append(((x1, y1, x2, y2), label, confidence))

#         return results

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect_objects(self, frame, conf_thresh=0.5):
        results = self.model(frame, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = self.model.names[cls]

                if conf > conf_thresh and label in ["backpack", "handbag", "suitcase"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(((x1, y1, x2, y2), label, conf))

        return detections