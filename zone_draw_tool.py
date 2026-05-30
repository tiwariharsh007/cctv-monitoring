"""
Interactive Zone Drawing Tool for Surveillance System

Allows operators to draw polygon zones on a video frame and assign specific
detection activities to monitor within each zone.

Usage (standalone):
    python zone_draw_tool.py

Usage (programmatic — called from main.py before monitoring starts):
    from zone_draw_tool import ZoneDrawingApp
    app = ZoneDrawingApp(camera_name="Main CCTV", frame=my_numpy_frame)
    app.run()

Drawing controls:
    LEFT-CLICK   : Add polygon point
    RIGHT-CLICK  : Finish polygon → opens activity selection panel
    Z            : Undo last point
    C            : Clear current in-progress polygon
    D            : Delete the last saved zone
    S            : Save all zones and exit
    Q            : Quit (prompts to save)

Activity selection (after right-click, panel shown on right):
    1-9          : Toggle activity on/off
    ENTER/SPACE  : Confirm selected activities and save zone
    ESC          : Cancel — discard the polygon, return to drawing
"""

import cv2
import json
import os
import sys
import numpy as np

# ── DPI fix for Windows (must run before any cv2.namedWindow call) ─────────────
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()    # system-DPI aware fallback
        except Exception:
            pass


# ── Activity catalogue ─────────────────────────────────────────────────────────
# Each entry: (activity_key, short_label, description)
ACTIVITIES = [
    ("intrusion",        "Intrusion",        "Alert on any entry into zone"),
    ("loitering",        "Loitering",        "Long stays beyond threshold"),
    ("running",          "Running",          "Fast movement detected"),
    ("crowd",            "Crowd",            "Too many people in zone"),
    ("fall",             "Fall",             "Person falls in zone"),
    ("fighting",         "Fighting",         "Aggressive movement in zone"),
    ("abandoned_object", "Abandoned Object", "Unattended item left in zone"),
    ("after_hours",      "After Hours",      "Motion during off-hours"),
    ("tailgating",       "Tailgating",       "Two entries in quick succession"),
]

# Backward-compat zone type derived from the first selected activity
_ACTIVITY_TO_ZONE_TYPE = {
    "intrusion":        "restricted",
    "loitering":        "loitering",
    "running":          "restricted",
    "crowd":            "restricted",
    "fall":             "restricted",
    "fighting":         "restricted",
    "abandoned_object": "high_value",
    "after_hours":      "restricted",
    "tailgating":       "entry_exit",
}

# Zone overlay colours (BGR) for display
_ZONE_COLORS = {
    "restricted": (0,   0,   255),
    "entry_exit":  (0,   255, 255),
    "high_value":  (255, 50,  0  ),
    "loitering":   (0,   220, 255),
}


class ZoneDrawingApp:
    """
    Interactive zone drawing with per-zone activity selection.

    Modes
    -----
    Static frame  (frame=<ndarray>)   — a single captured frame is shown.
                                        Ideal for pre-monitoring setup from main.py.
    Video file    (video_path=<str>)  — the first frame is read and frozen.
                                        N / P keys step through the video to pick
                                        a different reference frame.
    """

    CONFIG_PATH = "zones/zone_config.json"

    def __init__(self, camera_name: str, frame=None, video_path: str = None):
        if frame is None and video_path is None:
            raise ValueError("Provide either frame= or video_path=")

        self.camera_name = camera_name
        self.zones: list       = []
        self.current_zone: list = []    # points being drawn this session
        self._selecting        = False  # True while activity panel is visible
        self._frozen_frame     = None   # frame frozen during activity selection
        self._selected_activities: set = set()
        self._mouse_pos        = (0, 0) # tracks live cursor for rubber-band line

        self.load_existing_zones()

        # ── source setup ──────────────────────────────────────────────────────
        if frame is not None:
            # Static-frame mode: resize once, re-use forever
            self._static_frame = cv2.resize(frame, (1280, 720))
            self.cap            = None
            self.frame_idx      = 0
            self.total_frames   = 1
            self._video_frame   = None  # not used in static mode
        else:
            self._static_frame = None
            self.cap           = cv2.VideoCapture(video_path)
            self.total_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx     = 0
            self._video_frame  = None   # cached — only re-read on N / P key

    # ── persistence ───────────────────────────────────────────────────────────

    def load_existing_zones(self):
        if not os.path.exists(self.CONFIG_PATH):
            return
        try:
            with open(self.CONFIG_PATH) as f:
                data = json.load(f)
            self.zones = data.get(self.camera_name, [])
            if self.zones:
                print(f"  Loaded {len(self.zones)} existing zone(s) for '{self.camera_name}'")
        except Exception as e:
            print(f"  Could not load zones: {e}")

    def save_zones(self):
        os.makedirs("zones", exist_ok=True)
        data = {}
        if os.path.exists(self.CONFIG_PATH):
            try:
                with open(self.CONFIG_PATH) as f:
                    data = json.load(f)
            except Exception:
                pass
        data[self.camera_name] = self.zones
        with open(self.CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n  Saved {len(self.zones)} zone(s) for '{self.camera_name}' → {self.CONFIG_PATH}")
        for z in self.zones:
            acts = ", ".join(z.get("monitored_activities", [z.get("type", "?")]))
            print(f"    • {z['id']} | activities: {acts} | points: {len(z['points'])}")

    # ── frame helpers ──────────────────────────────────────────────────────────

    def _get_background(self) -> np.ndarray:
        """
        Return a fresh 1280×720 copy of the background frame.

        Static mode : always the same captured frame.
        Video  mode : the cached frame is returned; it is only refreshed when
                      the user presses N or P (so the background stays frozen
                      while drawing points).
        """
        if self._static_frame is not None:
            return self._static_frame.copy()

        # Video mode — read first frame if we have nothing cached yet
        if self._video_frame is None and self.cap is not None:
            ret, raw = self.cap.read()
            if ret:
                self._video_frame = cv2.resize(raw, (1280, 720))
                self.frame_idx   += 1

        if self._video_frame is not None:
            return self._video_frame.copy()

        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def _advance_video(self, delta: int):
        """Step the video by `delta` frames and refresh the cache."""
        if self.cap is None:
            return
        new_idx = max(0, min(self.frame_idx + delta, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
        ret, raw = self.cap.read()
        if ret:
            self._video_frame = cv2.resize(raw, (1280, 720))
            self.frame_idx    = new_idx
            print(f"  Frame {self.frame_idx} / {self.total_frames}")

    # ── drawing helpers ────────────────────────────────────────────────────────

    def _draw_saved_zones(self, frame: np.ndarray) -> np.ndarray:
        for zone in self.zones:
            zone_type = zone.get("type", "restricted")
            color     = _ZONE_COLORS.get(zone_type, (128, 128, 128))
            pts       = np.array(zone["points"], dtype=np.int32)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            tx, ty = map(int, zone["points"][0])
            cv2.putText(frame, zone.get("id", "zone"),
                        (tx + 4, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Activity badges (up to 4)
            acts = zone.get("monitored_activities", [])
            bx, by = tx + 4, ty + 18
            for act in acts[:4]:
                cv2.putText(frame, act[:4].upper(), (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                bx += 42
        return frame

    def _draw_current_polygon(self, frame: np.ndarray) -> np.ndarray:
        """Draw the in-progress polygon with a rubber-band preview line."""
        if not self.current_zone:
            return frame

        # Draw filled dots with black outlines for each committed point
        for i, pt in enumerate(self.current_zone):
            cv2.circle(frame, pt, 8,  (0, 255,  0), -1)         # green fill
            cv2.circle(frame, pt, 9,  (0,   0,  0),  1)         # black border
            cv2.circle(frame, pt, 10, (255, 255, 0),  1)         # yellow ring
            cv2.putText(frame, str(i + 1), (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lines connecting committed points
        if len(self.current_zone) >= 2:
            pts_arr = np.array(self.current_zone, dtype=np.int32)
            cv2.polylines(frame, [pts_arr], isClosed=False,
                          color=(0, 255, 0), thickness=2)

        # Rubber-band line: last committed point → live cursor
        mx, my = self._mouse_pos
        if mx > 0 or my > 0:
            cv2.line(frame, self.current_zone[-1], (mx, my),
                     (0, 200, 80), 1, cv2.LINE_AA)

            # Close-preview dashes: first point → cursor (when ≥ 3 points)
            if len(self.current_zone) >= 3:
                cv2.line(frame, self.current_zone[0], (mx, my),
                         (100, 200, 0), 1, cv2.LINE_AA)

        # Status label near cursor
        n = len(self.current_zone)
        tip = f"{n} pt{'s' if n != 1 else ''}  R-click to finish"
        cv2.putText(frame, tip, (mx + 14, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return frame

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Top info bar and control hints."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 68), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        info = (f"Camera: {self.camera_name}  |  "
                f"Saved zones: {len(self.zones)}  |  "
                f"Points drawn: {len(self.current_zone)}")
        cv2.putText(frame, info, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        keys = ("L-CLICK: add point  |  R-CLICK: finish polygon  |  "
                "Z: undo  |  C: clear  |  D: del last  |  S: save & exit  |  Q: quit")
        cv2.putText(frame, keys, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

        if self.cap is not None:
            nav = f"N: +30 frames  |  P: -30 frames  |  frame {self.frame_idx}/{self.total_frames}"
            cv2.putText(frame, nav, (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        return frame

    # ── activity selection overlay ─────────────────────────────────────────────

    def _draw_activity_panel(self, frame: np.ndarray) -> np.ndarray:
        _, w = frame.shape[:2]
        panel_x = w - 450

        overlay  = frame.copy()
        panel_h  = 55 + len(ACTIVITIES) * 42 + 70
        cv2.rectangle(overlay, (panel_x - 14, 55), (w - 8, 55 + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (panel_x - 14, 55), (w - 8, 55 + panel_h), (60, 60, 60), 1)

        cv2.putText(frame, "ASSIGN ACTIVITIES TO ZONE",
                    (panel_x, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
        cv2.putText(frame,
                    "Number key = toggle   ENTER = confirm   ESC = cancel",
                    (panel_x, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        for i, (key, label, desc) in enumerate(ACTIVITIES):
            y        = 140 + i * 42
            selected = key in self._selected_activities

            box_col  = (0, 200, 80) if selected else (80, 80, 80)
            cv2.rectangle(frame, (panel_x, y - 18), (panel_x + 22, y + 4),
                          box_col, -1 if selected else 1)
            if selected:
                cv2.putText(frame, "v", (panel_x + 4, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            cv2.putText(frame, f"[{i + 1}]", (panel_x + 28, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)

            lbl_col  = (50, 255, 120) if selected else (200, 200, 200)
            cv2.putText(frame, label, (panel_x + 72, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, lbl_col, 1 + int(selected))
            cv2.putText(frame, desc, (panel_x + 72, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (110, 110, 110), 1)

        bot_y = 140 + len(ACTIVITIES) * 42 + 20
        if self._selected_activities:
            cv2.putText(frame,
                        f"ENTER — confirm ({len(self._selected_activities)} selected)",
                        (panel_x, bot_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 80), 2)
        else:
            cv2.putText(frame, "Select at least one activity",
                        (panel_x, bot_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 80, 200), 1)
        cv2.putText(frame, "ESC — cancel zone", (panel_x, bot_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 200), 1)

        # Darken the polygon area so the panel stands out
        if self.current_zone and len(self.current_zone) >= 3:
            pts_arr = np.array(self.current_zone, dtype=np.int32)
            cv2.polylines(frame, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)

        return frame

    # ── mouse callback ─────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, _flags, _param):
        # Always track mouse position (used for rubber-band preview line)
        self._mouse_pos = (x, y)

        if self._selecting:
            return  # ignore clicks while activity panel is open

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            print(f"  Point {len(self.current_zone)} added at ({x}, {y})"
                  + (f"  — right-click to finish"
                     if len(self.current_zone) >= 3 else ""))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone) >= 3:
                # Freeze the current composite frame for the panel background
                self._frozen_frame     = self._last_display.copy()
                self._selected_activities = set()
                self._selecting        = True
            else:
                print(f"  Need >= 3 points to close a polygon "
                      f"(have {len(self.current_zone)})")

    # ── zone finalisation ──────────────────────────────────────────────────────

    def _finalise_zone(self):
        activities = list(self._selected_activities)
        zone_type  = _ACTIVITY_TO_ZONE_TYPE.get(activities[0], "restricted")
        zone_id    = f"zone_{len(self.zones) + 1}"

        self.zones.append({
            "id":                   zone_id,
            "type":                 zone_type,
            "label":                zone_id,
            "points":               self.current_zone.copy(),
            "monitored_activities": activities,
        })
        print(f"  Zone '{zone_id}' saved — activities: {', '.join(activities)}")

        self.current_zone     = []
        self._selecting       = False
        self._frozen_frame    = None

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        # ASCII-only window name: Unicode dashes create two ghost windows on Windows
        win = f"Zone Editor - {self.camera_name}"

        # WINDOW_AUTOSIZE: window dimensions exactly match the frame —
        # guarantees mouse coordinates are never offset by DPI scaling.
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._mouse_callback)

        # Initialise _last_display so the right-click freeze always has a frame
        self._last_display = self._get_background()

        print(f"\n  Zone Editor  |  Camera: {self.camera_name}")
        print("  LEFT-CLICK to add polygon points.")
        print("  RIGHT-CLICK (3+ points) to finish a polygon and select activities.")
        print("  Press S to save and exit,  Q to quit without saving.\n")

        while True:
            # ── build the display frame ────────────────────────────────────────
            if self._selecting:
                display = self._frozen_frame.copy()
                display = self._draw_activity_panel(display)
            else:
                display = self._get_background()
                display = self._draw_saved_zones(display)
                display = self._draw_current_polygon(display)
                display = self._draw_hud(display)
                self._last_display = display.copy()  # keep for right-click freeze

            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF

            # ── activity selection mode ────────────────────────────────────────
            if self._selecting:
                for i in range(len(ACTIVITIES)):
                    if key == ord(str(i + 1)):
                        act = ACTIVITIES[i][0]
                        if act in self._selected_activities:
                            self._selected_activities.discard(act)
                            print(f"  [ ] {ACTIVITIES[i][1]} deselected")
                        else:
                            self._selected_activities.add(act)
                            print(f"  [x] {ACTIVITIES[i][1]} selected")

                if key in (13, ord(' ')):         # Enter or Space → confirm
                    if self._selected_activities:
                        self._finalise_zone()
                    else:
                        print("  No activities selected — zone discarded.")
                        self.current_zone = []
                        self._selecting   = False

                elif key == 27:                   # ESC → cancel
                    print("  Zone cancelled.")
                    self.current_zone = []
                    self._selecting   = False

            # ── drawing mode ───────────────────────────────────────────────────
            else:
                if key == ord('s'):
                    self.save_zones()
                    break

                elif key == ord('q'):
                    ans = input("\n  Save zones before quitting? [Y/n]: ").strip().lower()
                    if ans != 'n':
                        self.save_zones()
                    break

                elif key == ord('z'):
                    if self.current_zone:
                        removed = self.current_zone.pop()
                        print(f"  Undo — removed point {removed}")
                    else:
                        print("  Nothing to undo.")

                elif key == ord('c'):
                    self.current_zone = []
                    print("  Current polygon cleared.")

                elif key == ord('d'):
                    if self.zones:
                        removed = self.zones.pop()
                        print(f"  Deleted zone: {removed['id']}")
                    else:
                        print("  No zones to delete.")

                # Video-only navigation — refreshes the frozen background frame
                elif self.cap is not None:
                    if key == ord('n'):
                        self._advance_video(+30)
                    elif key == ord('p'):
                        self._advance_video(-30)

        # Destroy by the same ASCII name used at creation
        cv2.destroyWindow(win)
        if self.cap:
            self.cap.release()


# ── standalone entry-point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob as _glob

    print("\n" + "=" * 50)
    print("   ZONE DRAWING & ACTIVITY CONFIGURATION")
    print("=" * 50)

    camera_name = input("\nCamera ID (e.g. 'Main CCTV'): ").strip() or "Main CCTV"

    print("\nSource:")
    print("  [1] Video file")
    print("  [2] Webcam snapshot (index 0 by default)")
    mode = input("Mode [1]: ").strip() or "1"

    if mode == "2":
        idx = input("Webcam index [0]: ").strip() or "0"
        cap = cv2.VideoCapture(int(idx))
        if not cap.isOpened():
            print("Cannot open webcam.")
            sys.exit(1)
        for _ in range(5):          # discard first frames (exposure settling)
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Could not read frame from webcam.")
            sys.exit(1)
        ZoneDrawingApp(camera_name=camera_name, frame=frame).run()

    else:
        videos = sorted(_glob.glob("data/test_videos/*.mp4"))
        if videos:
            for i, v in enumerate(videos, 1):
                print(f"  ({i}) {v}")
            sel = input("Pick video [1]: ").strip() or "1"
            video_path = (videos[int(sel) - 1]
                          if sel.isdigit() and 1 <= int(sel) <= len(videos)
                          else input("Enter full video path: ").strip())
        else:
            video_path = input("Enter video path: ").strip()

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            sys.exit(1)
        ZoneDrawingApp(camera_name=camera_name, video_path=video_path).run()
