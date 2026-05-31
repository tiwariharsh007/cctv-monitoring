"""Single source of truth for per-frame surveillance analysis + alerting.

Both `main.py` (headless) and the Streamlit dashboard drive THIS engine, so every
source — recorded video, live phone stream, webcam — runs identical detection AND
fires identical alerts (snapshot + email) with the same cooldown. Keeping one engine
prevents the two code paths from drifting apart (e.g. "crowd detected but no email").
"""
import os
import time
import threading
import numpy as np
import cv2
from datetime import datetime

from detector import PersonDetector
from tracker import CentroidTracker
from line_counter import LineCounter
from detectors.zone_intrusion   import ZoneIntrusionDetector
from detectors.speed            import SpeedDetector
from detectors.abandoned_object import AbandonedObjectDetector
from detectors.accident         import AccidentDetector
from detectors.dwell_time        import DwellTimeTracker
from alerts import send_email_alert
from services.alert_service import handle_alert

try:
    from caption_generator import init_caption_generator, generate_smart_alert_message
except Exception:                       # caption deps optional
    init_caption_generator = None
    generate_smart_alert_message = None

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Must match the identical function in zone_draw_tool.py so that zone
# coordinates recorded by the editor always align with analysis frames.
def _fit1280(frame: np.ndarray) -> np.ndarray:
    """Letterbox-resize to 1280×720, preserving aspect ratio."""
    if frame.shape[:2] == (720, 1280):
        return frame
    h, w = frame.shape[:2]
    scale = min(1280 / w, 720 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[(720-nh)//2:(720-nh)//2+nh, (1280-nw)//2:(1280-nw)//2+nw] = resized
    return canvas


def _box_color_by_dwell(dwell_secs: int):
    if dwell_secs > 60:
        return (0, 0, 220)
    if dwell_secs > 20:
        return (0, 140, 255)
    return (0, 210, 0)


class SurveillanceEngine:
    def __init__(self, cfg: dict, cam_name="Main CCTV", detector=None,
                 captions=False, email_alerts=True):
        d  = cfg.get("detection", {})
        self.CROWD        = d.get("crowd_threshold", 5)
        self.LINE_POS     = d.get("line_position", 300)
        self.RUN_SPEED    = d.get("running_speed_px", 20)
        self.ABAND_FRAMES = d.get("abandoned_frames", 150)
        self.INACT_FRAMES = d.get("inactivity_frames", 30)
        self.FALL_CONFIRM = d.get("fall_confirm_frames", 3)
        self.TAILGATE     = d.get("tailgate_secs", 3)
        self.ACCIDENT_ON  = d.get("accident_enabled", True)
        self.COOLDOWN     = cfg.get("alerts", {}).get("cooldown_seconds", 60)

        self.cam_name     = cam_name
        self.email_alerts = email_alerts
        self.captions     = bool(captions and init_caption_generator)
        if self.captions:
            init_caption_generator()

        self.detector  = detector or PersonDetector()
        self.tracker   = CentroidTracker(max_disappeared=15, max_history=20)
        self.counter   = LineCounter(line_position=self.LINE_POS)
        self.speed     = SpeedDetector(speed_threshold=self.RUN_SPEED)
        self.abandoned = AbandonedObjectDetector(stationary_frames=self.ABAND_FRAMES)
        self.zones     = ZoneIntrusionDetector()
        self.dwell     = DwellTimeTracker()
        self._zone_cfg_mtime = self._zone_cfg_mtime_now()
        self.accident  = AccidentDetector(
            iou_threshold=d.get("accident_iou", 0.30),
            speed_drop=d.get("accident_speed_drop", 0.55),
            min_speed=d.get("accident_min_speed", 6.0),
        ) if self.ACCIDENT_ON else None

        self.heatmap        = None
        self._fall_frames   = 0
        self._last_crossing = 0.0
        self._last_alert    = {}
        self.zone_frame_counts = {}        # zone_id → cumulative occupant-frames (this session)
        self.monitored_activities = set()  # Activities to monitor (controlled by dashboard)
        os.makedirs("snapshots", exist_ok=True)

    @staticmethod
    def _zone_cfg_mtime_now() -> float:
        try:
            return os.path.getmtime("zones/zone_config.json")
        except OSError:
            return 0.0

    def set_monitored_activities(self, activities: list):
        """Set which activities to monitor. Only these will trigger alerts."""
        self.monitored_activities = set(activities) if activities else set()

    def _is_activity_monitored(self, activity: str) -> bool:
        """Check if an activity should be monitored."""
        # Intrusion is always monitored (zone-based)
        if activity == "Intrusion":
            return True
        # Other activities only monitored if explicitly selected
        return activity in self.monitored_activities

    def reload_zones(self):
        """Hot-reload zone config from disk (call after zone_draw_tool saves)."""
        self.zones.reload()
        self._zone_cfg_mtime = self._zone_cfg_mtime_now()

    def set_active_zones(self, zone_ids: list):
        """Restrict zone detection to the supplied zone IDs for this camera."""
        self.zones.filter_to_active(self.cam_name, zone_ids)

    # ── zone-gate helper ────────────────────────────────────────────────────────

    def _zone_gate(self, activity: str, tracked: dict) -> bool:
        """
        Returns True when an alert for `activity` should fire.

        Logic:
          - No zones configured for this activity → True (global, existing behaviour).
          - Zones configured AND at least one person is currently inside → True.
          - Zones configured but nobody inside any relevant zone → False.
        """
        filtered = self.zones.filter_tracked_for_activity(
            self.cam_name, tracked, activity)
        return filtered is None or bool(filtered)

    # ── alert plumbing ──────────────────────────────────────────────────────────
    def _should_alert(self, key: str) -> bool:
        now = time.time()
        if key not in self._last_alert or now - self._last_alert[key] > self.COOLDOWN:
            self._last_alert[key] = now
            return True
        return False

    def _save_snapshot(self, frame, event_type: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshots/{event_type}_{ts}.jpg"
            cv2.imwrite(path, frame)
            return path
        except Exception:
            return None

    def _fire(self, frame, event_type, detail, subject):
        snap = self._save_snapshot(frame, event_type)   # fast local write
        handle_alert(event_type, detail, snap)          # fast console log

        # Email with AI-generated message from image analysis are network calls that can take seconds —
        # run them off-thread so a fired alert never freezes the frame loop.
        if self.email_alerts:
            def _notify():
                msg = (generate_smart_alert_message(event_type, snap)
                       if self.captions else detail)
                send_email_alert(subject, msg, snap)    # self-gated by EMAIL_ALERTS in .env
            threading.Thread(target=_notify, daemon=True).start()

    # ── per-frame analysis ──────────────────────────────────────────────────────
    def process(self, frame):
        """Annotate `frame` in place, fire any alerts, and return a log row dict."""
        # Auto-reload zones if zone_config.json was updated on disk
        mtime = self._zone_cfg_mtime_now()
        if mtime > self._zone_cfg_mtime:
            self.zones.reload()
            self._zone_cfg_mtime = mtime

        # Normalise to 1280×720 using letterbox (aspect-ratio-preserving).
        # Zone editor uses the same transform, so coordinates always align.
        frame = _fit1280(frame)

        if self.heatmap is None:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        alert_parts = []

        boxes, raw_yolo   = self.detector.detect_all(frame)
        tracked           = self.tracker.update(boxes)
        prev_in, prev_out = self.counter.count_in, self.counter.count_out
        self.counter.update(tracked)
        self.dwell.update(tracked)
        visible = len(tracked)

        # trails
        for _, pts in self.tracker.object_history.items():
            if len(pts) < 2:
                continue
            for i in range(1, min(len(pts), 12)):
                a = int(180 * i / 12)
                cv2.line(frame, tuple(map(int, pts[-i - 1])), tuple(map(int, pts[-i])),
                         (a, a, 255 - a), 1)

        # boxes colored by dwell
        dwell_times = self.dwell.get_dwell_times()
        for (x1, y1, x2, y2) in boxes:
            bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
            cid = (min(tracked.keys(),
                       key=lambda oid: abs(tracked[oid][0] - bcx) + abs(tracked[oid][1] - bcy))
                   if tracked else None)
            dwell = dwell_times.get(cid, 0) if cid is not None else 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), _box_color_by_dwell(dwell), 2)

        # labels + heatmap accumulation
        for oid, (cx, cy) in tracked.items():
            cv2.circle(frame, (cx, cy), 3, (255, 200, 0), -1)
            dwell = dwell_times.get(oid, 0)
            lbl = f"ID{oid}" + (f" {dwell}s" if dwell >= 10 else "")
            cv2.putText(frame, lbl, (cx - 10, cy - 12), FONT, 0.36, (255, 255, 255), 1)
            if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
                cv2.circle(self.heatmap, (cx, cy), 16, 1, -1)

        # fall (confirmed over N frames)
        posture = "Standing"
        if boxes:
            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if max(1, x2 - x1) / max(1, y2 - y1) > 1.4:
                posture = "Lying"
        self._fall_frames = self._fall_frames + 1 if posture == "Lying" else 0
        if self._fall_frames >= self.FALL_CONFIRM and self._zone_gate("fall", tracked) and self._is_activity_monitored("Fall"):
            cv2.putText(frame, "FALL DETECTED", (10, 95), FONT, 0.85, (0, 0, 255), 2)
            alert_parts.append("Fall")
            if self._should_alert("fall"):
                self._fire(frame, "fall", "Person fallen and not moving", "🚨 FALL DETECTED")

        # tailgating
        if self.counter.count_in != prev_in or self.counter.count_out != prev_out:
            now_t = time.time()
            if (self._last_crossing > 0
                    and (now_t - self._last_crossing) < self.TAILGATE
                    and self._zone_gate("tailgating", tracked)
                    and self._is_activity_monitored("Tailgating")):
                alert_parts.append("Tailgating")
                if self._should_alert("tailgating"):
                    self._fire(frame, "tailgating",
                               "Two people crossed line in quick succession", "⚠ TAILGATING ALERT")
            self._last_crossing = now_t

        # inactivity
        inactive = [oid for oid, pts in self.tracker.object_history.items()
                    if len(pts) >= self.INACT_FRAMES
                    and max(p[0] for p in pts) - min(p[0] for p in pts) < 8
                    and max(p[1] for p in pts) - min(p[1] for p in pts) < 8]
        if any(oid in inactive for oid in tracked) and self._is_activity_monitored("Inactivity"):
            alert_parts.append("Inactivity")
            if self._should_alert("inactivity"):
                self._fire(frame, "inactivity", "Person motionless", "⚠ INACTIVITY ALERT")

        # running
        for run_alert in self.speed.update(tracked):
            if self._zone_gate("running", tracked) and self._is_activity_monitored("Running"):
                cv2.putText(frame, "RUNNING", (10, 125), FONT, 0.75, (0, 140, 255), 2)
                alert_parts.append("Running")
                if self._should_alert("running"):
                    self._fire(frame, "running", run_alert, "⚡ RUNNING DETECTED")

        # crowd — use zone-resident count when crowd zones are defined
        zone_crowd = self.zones.count_in_activity_zones(self.cam_name, tracked, "crowd")
        crowd_count = visible if zone_crowd is None else zone_crowd
        if crowd_count > self.CROWD and self._is_activity_monitored("Crowd"):
            lbl = f"CROWD ({crowd_count})" + ("" if zone_crowd is None else " in zone")
            cv2.putText(frame, lbl, (10, 155), FONT, 0.8, (0, 60, 255), 2)
            alert_parts.append("Crowd")
            if self._should_alert("crowd"):
                self._fire(frame, "crowd", f"{crowd_count} people"
                           + (" in crowd zone" if zone_crowd is not None else " in frame"),
                           "⚠ CROWD ALERT")

        # zone intrusion - supports multiple zone types (restricted, entry_exit, high_value, loitering)
        frame = self.zones.draw_zones(frame, self.cam_name, tracked)
        wrapped = {oid: {"centroid": pos} for oid, pos in tracked.items()}

        # Accumulate per-zone occupancy across the session so the dashboard's
        # "Zone Activity" breakdown reflects THIS run (written out on stop).
        for _z in self.zones.zones.get(self.cam_name, []):
            _n = len(self.zones._zone_occupants(wrapped, _z))
            if _n:
                _zid = _z.get("id", "zone")
                self.zone_frame_counts[_zid] = self.zone_frame_counts.get(_zid, 0) + _n

        zone_alerts = self.zones.detect_intrusions(self.cam_name, wrapped)
        
        for alert in zone_alerts:
            zone_type = alert.get("zone_type", "restricted")
            zone_id = alert.get("zone_id", "zone")
            severity = alert.get("severity", "LOW")
            message = alert.get("message", "Zone alert")
            
            # Only fire HIGH severity alerts (restrict intrusion warnings)
            if zone_type == "restricted" and severity == "HIGH":
                alert_parts.append("Intrusion")
                if self._should_alert(f"intrusion_{zone_id}"):
                    self._fire(frame, "intrusion", 
                             f"Unauthorized access: {message}",
                             f"🚨 ZONE INTRUSION - {zone_id}")
            
            elif zone_type == "high_value" and severity == "MEDIUM":
                alert_parts.append("SuspiciousActivity")
                if self._should_alert(f"highvalue_{zone_id}"):
                    self._fire(frame, "suspicious",
                             f"High-value area alert: {message}",
                             f"⚠️ SUSPICIOUS ACTIVITY - {zone_id}")

            # entry_exit is informational only (LOW severity)

        # abandoned object — zone-gate checks whether the item's centroid is in an
        # abandoned_object zone (or fires globally if no such zones are defined).
        for ab in self.abandoned.update(raw_yolo, tracked):
            x1, y1, x2, y2 = ab["box"]
            cx_ab, cy_ab = (x1 + x2) // 2, (y1 + y2) // 2
            in_zone = self.zones.point_in_activity_zone(
                self.cam_name, (cx_ab, cy_ab), "abandoned_object")
            if (in_zone is None or in_zone) and self._is_activity_monitored("AbandonedObject"):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"UNATTENDED {ab['type'].upper()}", (x1, y1 - 8),
                            FONT, 0.55, (0, 0, 255), 2)
                alert_parts.append("AbandonedObject")
                if self._should_alert("abandoned"):
                    self._fire(frame, "abandoned", f"Unattended {ab['type']}", "🧳 UNATTENDED ITEM")

        # vehicle accident (collision / sudden stop)
        if self.accident is not None and self._is_activity_monitored("Accident"):
            accidents = self.accident.update(raw_yolo)
            if accidents:
                if "Accident" not in alert_parts:
                    alert_parts.append("Accident")
                for acc in accidents:
                    x1, y1, x2, y2 = acc["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"ACCIDENT: {acc['detail']}", (x1, max(20, y1 - 8)),
                                FONT, 0.55, (0, 0, 255), 2)
                if self._should_alert("accident"):
                    top = accidents[0]
                    self._fire(frame, "accident",
                               f"{top['type']} — {top['detail']}", "🚑 VEHICLE ACCIDENT")

        self._draw_hud(frame, visible, alert_parts)

        return {
            "frame":         frame,
            "visible_count": visible,
            "in_count":      self.counter.count_in,
            "out_count":     self.counter.count_out,
            "posture":       posture,
            "alerts":        alert_parts,
            "alert":         " ".join(alert_parts),
        }

    # ── overlays / exports ──────────────────────────────────────────────────────
    def _draw_hud(self, frame, visible, alert_parts):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 72), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        occ = max(0, self.counter.count_in - self.counter.count_out)
        cv2.putText(frame,
                    f"IN:{self.counter.count_in}  OUT:{self.counter.count_out}  Occupancy:{occ}  Now:{visible}",
                    (10, 24), FONT, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.cam_name}   {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
                    (10, 52), FONT, 0.50, (160, 160, 160), 1)

        if alert_parts:
            labels = list(dict.fromkeys(alert_parts))
            cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "ALERT: " + "  |  ".join(labels), (10, h - 10),
                        FONT, 0.60, (0, 60, 255), 2)

        cv2.line(frame, (0, self.LINE_POS), (w, self.LINE_POS), (0, 240, 240), 2)
        cv2.putText(frame, "  IN >>", (10, self.LINE_POS - 6), FONT, 0.38, (0, 240, 240), 1)

    def zone_activity(self):
        """Per-zone share of total occupant-time this session.

        Returns a list of {zone, frames, percent} for every configured zone on
        this camera (zones with no activity report 0%). Empty if no zones exist.
        """
        zones = self.zones.zones.get(self.cam_name, [])
        total = sum(self.zone_frame_counts.values())
        rows = []
        for z in zones:
            zid = z.get("id", "zone")
            frames = self.zone_frame_counts.get(zid, 0)
            pct = (100.0 * frames / total) if total else 0.0
            rows.append({"zone": zid, "frames": frames, "percent": round(pct, 1)})
        return rows

    def colored_heatmap(self, frame=None):
        if self.heatmap is None or self.heatmap.max() == 0:
            return None
        norm    = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        if frame is not None and frame.shape[:2] == colored.shape[:2]:
            colored = cv2.addWeighted(frame, 0.4, colored, 0.6, 0)
        return colored
