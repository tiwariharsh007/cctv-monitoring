import time


class DwellTimeTracker:
    """
    Tracks how long each person has been visible in the scene.

    Real-world use:
    - Retail: identify customers spending too long (potential shoplifting)
    - Security: alert when someone lingers in a restricted area
    - Healthcare: detect patient left unattended too long
    """

    def __init__(self, alert_threshold_secs=300):
        self.alert_threshold = alert_threshold_secs
        self._entry_time = {}   # obj_id -> epoch time of first appearance
        self._alerted    = {}   # obj_id -> bool (avoid repeat alert same session)

    def update(self, tracked: dict):
        now    = time.time()
        alerts = []

        for obj_id in tracked:
            if obj_id not in self._entry_time:
                self._entry_time[obj_id] = now

            dwell = now - self._entry_time[obj_id]

            if dwell >= self.alert_threshold and not self._alerted.get(obj_id, False):
                alerts.append(f"Long dwell: object {obj_id} present for {int(dwell)}s")
                self._alerted[obj_id] = True

        # Clean up objects that left the scene
        gone = set(self._entry_time) - set(tracked)
        for obj_id in gone:
            del self._entry_time[obj_id]
            self._alerted.pop(obj_id, None)

        return alerts

    def get_dwell_times(self) -> dict:
        """Returns {obj_id: seconds_in_scene} for all currently visible objects."""
        now = time.time()
        return {obj_id: int(now - t) for obj_id, t in self._entry_time.items()}
