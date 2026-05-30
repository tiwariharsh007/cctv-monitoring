import math


class SpeedDetector:
    """
    Detects running / rushing behaviour.
    Fires when a tracked person moves faster than `speed_threshold` pixels/frame
    for at least `confirm_frames` consecutive frames.

    Real-world use: panic, fleeing, emergency response, security breach.
    """

    def __init__(self, speed_threshold=20, confirm_frames=6):
        self.speed_threshold = speed_threshold
        self.confirm_frames  = confirm_frames
        self._prev      = {}   # obj_id -> (cx, cy)
        self._counters  = {}   # obj_id -> consecutive fast frames
        self._speeds    = {}   # obj_id -> current speed (for display)

    def update(self, tracked: dict) -> list:
        alerts = []

        for obj_id, (cx, cy) in tracked.items():
            if obj_id in self._prev:
                px, py = self._prev[obj_id]
                speed = math.hypot(cx - px, cy - py)
                self._speeds[obj_id] = round(speed, 1)

                if speed > self.speed_threshold:
                    self._counters[obj_id] = self._counters.get(obj_id, 0) + 1
                    if self._counters[obj_id] >= self.confirm_frames:
                        alerts.append(f"Running detected: object {obj_id} ({speed:.0f} px/frame)")
                else:
                    self._counters[obj_id] = 0
            else:
                self._speeds[obj_id] = 0.0

            self._prev[obj_id] = (cx, cy)

        # Clean up objects no longer tracked
        gone = set(self._prev) - set(tracked)
        for obj_id in gone:
            self._prev.pop(obj_id, None)
            self._counters.pop(obj_id, None)
            self._speeds.pop(obj_id, None)

        return alerts

    def get_speeds(self) -> dict:
        """Returns {obj_id: speed_px_per_frame} for overlay display."""
        return dict(self._speeds)
