import math


# YOLO COCO class IDs for unattended-item detection
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


class AbandonedObjectDetector:
    """
    Detects bags/suitcases/backpacks that remain stationary while
    no person is nearby — classic 'unattended item' security scenario.

    Requires passing raw YOLO results (not just person boxes) so it can
    inspect all detected classes.

    Real-world use: airports, metro stations, malls, government buildings.
    """

    def __init__(self, stationary_frames=150, person_proximity_px=160):
        self.stationary_frames     = stationary_frames
        self.person_proximity_px   = person_proximity_px
        self._tracked_bags         = {}   # key -> {cx, cy, cls, frames}

    def update(self, raw_yolo_result, tracked_persons: dict) -> list:
        """
        Args:
            raw_yolo_result : result[0] from model(frame, verbose=False)
            tracked_persons : {obj_id: (cx, cy)} from CentroidTracker
        Returns:
            list of alert dicts with 'box', 'type', 'frames'
        """
        alerts         = []
        current_bags   = {}

        if raw_yolo_result is None or raw_yolo_result.boxes is None:
            self._tracked_bags = {}
            return alerts

        for row in raw_yolo_result.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = row
            cls_id = int(cls_id)
            if cls_id not in BAG_CLASSES or score < 0.4:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Match to an existing tracked bag (within 60px)
            matched_key = None
            for key, bag in self._tracked_bags.items():
                if math.hypot(cx - bag['cx'], cy - bag['cy']) < 60:
                    matched_key = key
                    break

            if matched_key is None:
                matched_key = f"{cls_id}_{cx}_{cy}"

            prev_frames = self._tracked_bags.get(matched_key, {}).get('frames', 0)
            current_bags[matched_key] = {
                'box':    (x1, y1, x2, y2),
                'cx':     cx,
                'cy':     cy,
                'cls':    cls_id,
                'frames': prev_frames + 1,
            }

        # Check for unattended bags
        for key, bag in current_bags.items():
            if bag['frames'] < self.stationary_frames:
                continue
            person_nearby = any(
                math.hypot(bag['cx'] - px, bag['cy'] - py) < self.person_proximity_px
                for px, py in tracked_persons.values()
            )
            if not person_nearby:
                alerts.append({
                    'box':    bag['box'],
                    'type':   BAG_CLASSES[bag['cls']],
                    'frames': bag['frames'],
                })

        self._tracked_bags = current_bags
        return alerts
