import math

# YOLO COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class AccidentDetector:
    """Heuristic vehicle-accident detector.

    A generic object detector cannot *recognise* a crash, so this infers one from
    motion. It tracks vehicles (car/truck/bus/motorcycle) frame-to-frame and flags:

      • Collision  — two vehicle boxes overlap (IoU ≥ iou_threshold) for a few
                     frames while at least one was recently moving and then
                     sharply decelerated (impact).
      • Sudden stop — a single vehicle moving > min_speed that abruptly halts
                     (speed drops by ≥ speed_drop and falls near zero).

    Requires the raw YOLO result (all classes), like AbandonedObjectDetector.
    Tune the thresholds in config.yaml → detection. Treat output as a hint for
    review, not a certified crash classifier.
    """

    def __init__(self, iou_threshold=0.30, speed_drop=0.55, min_speed=6.0,
                 overlap_frames=3, match_dist=90, min_score=0.4, vehicle_classes=None):
        self.iou_threshold = iou_threshold
        self.speed_drop    = speed_drop       # fractional drop counted as "impact"
        self.min_speed     = min_speed        # px/frame to count as "was moving"
        self.overlap_frames= overlap_frames
        self.match_dist    = match_dist
        self.min_score     = min_score
        self.vehicle_classes = vehicle_classes or VEHICLE_CLASSES

        self._tracks  = {}    # vid -> dict(cx, cy, box, cls, speed, prev_speed, peak, age)
        self._overlap = {}    # frozenset(pair) -> consecutive overlap frames
        self._next_id = 0

    def _decelerated(self, t) -> bool:
        return (t["prev_speed"] >= self.min_speed
                and t["speed"] <= t["prev_speed"] * (1 - self.speed_drop)) or t["speed"] < 2.0

    def update(self, raw_yolo_result) -> list:
        dets = []
        if raw_yolo_result is not None and raw_yolo_result.boxes is not None:
            for x1, y1, x2, y2, score, cls in raw_yolo_result.boxes.data.tolist():
                cls = int(cls)
                if cls in self.vehicle_classes and score >= self.min_score:
                    dets.append((int(x1), int(y1), int(x2), int(y2), cls))

        # ── match detections to existing vehicle tracks (nearest centroid) ──────
        new_tracks, used = {}, set()
        for (x1, y1, x2, y2, cls) in dets:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best, best_d = None, self.match_dist
            for tid, t in self._tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self._next_id
                self._next_id += 1
                prev = None
            else:
                used.add(best)
                prev = self._tracks[best]

            speed = math.hypot(cx - prev["cx"], cy - prev["cy"]) if prev else 0.0
            peak  = max(speed, prev["peak"] * 0.9) if prev else speed   # decaying recent max
            new_tracks[best] = {
                "cx": cx, "cy": cy, "box": (x1, y1, x2, y2), "cls": cls,
                "speed": speed, "prev_speed": prev["speed"] if prev else 0.0,
                "peak": peak, "age": (prev["age"] + 1) if prev else 1,
            }
        self._tracks = new_tracks

        alerts = []

        # ── sudden stop / impact (single vehicle) ───────────────────────────────
        for t in self._tracks.values():
            if t["age"] >= 3 and t["prev_speed"] >= self.min_speed \
                    and t["speed"] <= t["prev_speed"] * (1 - self.speed_drop) \
                    and t["speed"] < self.min_speed * 0.5:
                alerts.append({
                    "box":  t["box"],
                    "type": self.vehicle_classes.get(t["cls"], "vehicle"),
                    "detail": f"Sudden stop ({t['prev_speed']:.0f}→{t['speed']:.0f} px/frame)",
                })

        # ── collision (overlapping vehicle pair + impact) ───────────────────────
        ids = list(self._tracks.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self._tracks[ids[i]], self._tracks[ids[j]]
                key  = frozenset((ids[i], ids[j]))
                iou  = _iou(a["box"], b["box"])
                if iou >= self.iou_threshold:
                    self._overlap[key] = self._overlap.get(key, 0) + 1
                    recently_moving = a["peak"] >= self.min_speed or b["peak"] >= self.min_speed
                    impact = self._decelerated(a) or self._decelerated(b)
                    if self._overlap[key] >= self.overlap_frames and recently_moving and impact:
                        ax1, ay1, ax2, ay2 = a["box"]
                        bx1, by1, bx2, by2 = b["box"]
                        union = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
                        alerts.append({
                            "box":  union,
                            "type": f"{self.vehicle_classes.get(a['cls'],'vehicle')}+"
                                    f"{self.vehicle_classes.get(b['cls'],'vehicle')}",
                            "detail": f"Collision (overlap IoU {iou:.2f})",
                        })
                else:
                    self._overlap.pop(key, None)

        # drop overlap counters for vehicles that left the scene
        self._overlap = {k: v for k, v in self._overlap.items()
                         if all(tid in self._tracks for tid in k)}
        return alerts
