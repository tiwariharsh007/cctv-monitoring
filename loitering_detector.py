from datetime import datetime

class LoiteringDetector:
    def __init__(self, loitering_threshold=300):
        self.loitering_threshold = loitering_threshold  # Time in seconds
        self.loitering_objects = {}

    def update(self, tracked_objects):
        alert_text = []
        for object_id, (cx, cy) in tracked_objects.items():
            # If object already being tracked for loitering
            if object_id in self.loitering_objects:
                last_seen, timer = self.loitering_objects[object_id]
                if (datetime.now() - last_seen).total_seconds() < self.loitering_threshold:
                    self.loitering_objects[object_id] = (datetime.now(), timer + 1)
                else:
                    self.loitering_objects[object_id] = (datetime.now(), 0)
            else:
                self.loitering_objects[object_id] = (datetime.now(), 0)

            # Check loitering condition
            if self.loitering_objects[object_id][1] >= self.loitering_threshold:
                alert_text.append(f"⚠️ Loitering Detected for Object {object_id}")

        return alert_text
