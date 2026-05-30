import math


class FightingDetector:
    """
    Detects fighting/aggression by identifying pairs of people who are:
      1. Close to each other (centroids within proximity_threshold pixels)
      2. Both moving rapidly (velocity > motion_threshold pixels/frame)
      3. This persists for at least confirm_frames consecutive frames

    Works on tracked centroids — no extra ML model needed.
    """

    def __init__(self,
                 proximity_threshold=90,
                 motion_threshold=12,
                 confirm_frames=8):
        self.proximity_threshold = proximity_threshold
        self.motion_threshold    = motion_threshold
        self.confirm_frames      = confirm_frames

        self._prev_positions = {}   # obj_id -> (cx, cy)
        self._pair_counters  = {}   # (id_a, id_b) -> consecutive frames count

    def update(self, tracked: dict) -> list:
        """
        Args:
            tracked: dict {object_id: (cx, cy)}
        Returns:
            list of alert strings, empty if no fighting detected
        """
        alerts = []
        ids = list(tracked.keys())

        # Compute per-object speed (pixels moved since last frame)
        speeds = {}
        for obj_id, (cx, cy) in tracked.items():
            if obj_id in self._prev_positions:
                px, py = self._prev_positions[obj_id]
                speeds[obj_id] = math.hypot(cx - px, cy - py)
            else:
                speeds[obj_id] = 0.0

        active_pairs = set()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                cx_a, cy_a = tracked[id_a]
                cx_b, cy_b = tracked[id_b]

                dist          = math.hypot(cx_a - cx_b, cy_a - cy_b)
                both_fast     = (speeds.get(id_a, 0) > self.motion_threshold and
                                 speeds.get(id_b, 0) > self.motion_threshold)
                close_together = dist < self.proximity_threshold

                pair = (min(id_a, id_b), max(id_a, id_b))
                active_pairs.add(pair)

                if close_together and both_fast:
                    self._pair_counters[pair] = self._pair_counters.get(pair, 0) + 1
                else:
                    self._pair_counters[pair] = 0

                if self._pair_counters.get(pair, 0) >= self.confirm_frames:
                    alerts.append(f"Fighting detected (objects {id_a} & {id_b})")

        # Clean up pairs that are no longer visible
        gone = set(self._pair_counters) - active_pairs
        for p in gone:
            del self._pair_counters[p]

        self._prev_positions = dict(tracked)
        return alerts
