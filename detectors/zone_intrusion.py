import cv2
import numpy as np
import json
import os


class ZoneIntrusionDetector:
    def __init__(self, zone_config_path="zones/zone_config.json"):
        self.zones = self.load_zones(zone_config_path)

    def load_zones(self, path):
        if not os.path.exists(path):
            print(f"❌ Zone config not found: {path}")
            return {}

        with open(path, 'r') as f:
            return json.load(f)

    def point_in_polygon(self, point, polygon):
        polygon_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_np, point, False) >= 0

    def detect_intrusions(self, camera_name, tracked_objects):
        intrusions = []

        if camera_name not in self.zones:
            return intrusions

        for obj_id, obj in tracked_objects.items():
            centroid = obj.get("centroid", None)

            # ✅ FIX: proper check
            if centroid is None:
                continue

            # ✅ ensure tuple (important)
            centroid = tuple(map(int, centroid))

            for zone in self.zones[camera_name]:
                if zone.get("type") == "restricted":
                    if self.point_in_polygon(centroid, zone["points"]):
                        intrusions.append({
                            "object_id": obj_id,
                            "zone_id": zone["id"],
                            "centroid": centroid
                        })

        return intrusions

    def draw_zones(self, frame, camera_name):
        if camera_name not in self.zones:
            return frame

        for zone in self.zones[camera_name]:
            color = (0, 0, 255) if zone.get("type") == "restricted" else (255, 255, 0)

            pts = np.array(zone["points"], dtype=np.int32)

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # ✅ safer text position
            text_pos = tuple(map(int, zone["points"][0]))
            cv2.putText(frame, zone["id"], text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame