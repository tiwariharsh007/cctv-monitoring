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
    RIGHT-CLICK  : Finish polygon and save zone (intrusion detection)
    Z            : Undo last point
    C            : Clear current in-progress polygon
    D            : Delete the last saved zone
    S            : Save all zones and exit
    Q            : Quit (prompts to save)
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


# ── Shared frame helper ────────────────────────────────────────────────────────
def _fit1280(frame: np.ndarray) -> np.ndarray:
    """
    Resize to 1280×720 preserving aspect ratio.
    Ultra-wide or non-16:9 sources are padded with black bars so that
    zone coordinates are tied to scene content, not a squeezed projection.
    """
    if frame.shape[:2] == (720, 1280):
        return frame
    h, w = frame.shape[:2]
    scale = min(1280 / w, 720 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    x0 = (1280 - nw) // 2
    y0 = (720  - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas



# Zone overlay colours (BGR) for display
_ZONE_COLORS = {
    "restricted": (0,   0,   255),
    "entry_exit":  (0,   255, 255),
    "high_value":  (255, 50,  0  ),
    "loitering":   (0,   220, 255),
}


class ZoneDrawingApp:
    """
    Interactive zone drawing tool for intrusion detection.

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
        self._mouse_pos        = (0, 0) # tracks live cursor for rubber-band line

        self.load_existing_zones()

        # ── source setup ──────────────────────────────────────────────────────
        if frame is not None:
            # Static-frame mode: resize once, re-use forever
            self._static_frame = _fit1280(frame)
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
                self._video_frame = _fit1280(raw)
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
            self._video_frame = _fit1280(raw)
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

    # ── mouse callback ─────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, _flags, _param):
        # Convert fullscreen canvas coordinates → 1280×720 image coordinates.
        # _fs_map = (scale, x_offset, y_offset) set in run() before the loop.
        scale, x0, y0 = getattr(self, "_fs_map", (1.0, 0, 0))
        ix = max(0, min(1279, int((x - x0) / scale)))
        iy = max(0, min(719,  int((y - y0) / scale)))

        self._mouse_pos = (ix, iy)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((ix, iy))
            print(f"  Point {len(self.current_zone)} added at ({ix}, {iy})"
                  + ("  — right-click to finish"
                     if len(self.current_zone) >= 3 else ""))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone) >= 3:
                self._finalise_zone()
            else:
                print(f"  Need >= 3 points to close a polygon "
                      f"(have {len(self.current_zone)})")

    # ── zone finalisation ──────────────────────────────────────────────────────

    def _finalise_zone(self):
        zone_id = f"zone_{len(self.zones) + 1}"
        self.zones.append({
            "id":                   zone_id,
            "type":                 "restricted",
            "label":                zone_id,
            "points":               self.current_zone.copy(),
            "monitored_activities": ["intrusion"],
        })
        print(f"  Zone '{zone_id}' saved (intrusion detection)")
        self.current_zone = []

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        win = f"Zone Editor - {self.camera_name}"

        # ── detect screen resolution ───────────────────────────────────────────
        try:
            import ctypes as _ct
            _u32 = _ct.windll.user32
            SW = int(_u32.GetSystemMetrics(0))
            SH = int(_u32.GetSystemMetrics(1))
        except Exception:
            SW, SH = 1920, 1080

        # ── letterbox: fit 1280×720 into the screen without stretching ─────────
        _scale = min(SW / 1280, SH / 720)
        _dw    = int(1280 * _scale)
        _dh    = int(720  * _scale)
        _x0    = (SW - _dw) // 2
        _y0    = (SH - _dh) // 2

        # Store mapping used by _mouse_callback to convert canvas → image coords
        self._fs_map = (_scale, _x0, _y0)

        # ── create fullscreen borderless window ────────────────────────────────
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(win, self._mouse_callback)

        self._last_display = self._get_background()

        print(f"\n  Zone Editor  |  Camera: {self.camera_name}  (FULLSCREEN {SW}x{SH})")
        print("  LEFT-CLICK to add polygon points.")
        print("  RIGHT-CLICK (3+ points) to finish and save zone.")
        print("  Press S to save and exit,  Q to quit without saving.\n")

        while True:
            # ── build the 1280×720 logic frame ────────────────────────────────
            display = self._get_background()
            display = self._draw_saved_zones(display)
            display = self._draw_current_polygon(display)
            display = self._draw_hud(display)
            self._last_display = display.copy()

            # ── scale to fullscreen canvas and show ────────────────────────────
            _scaled = cv2.resize(display, (_dw, _dh), interpolation=cv2.INTER_LINEAR)
            _canvas = np.zeros((SH, SW, 3), dtype=np.uint8)
            _canvas[_y0:_y0+_dh, _x0:_x0+_dw] = _scaled
            cv2.imshow(win, _canvas)
            key = cv2.waitKey(30) & 0xFF

            # ── key handling ───────────────────────────────────────────────────
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
    import argparse as _ap

    # When launched from the Streamlit dashboard the --camera / --source args
    # are passed automatically so no interactive prompts are needed.
    parser = _ap.ArgumentParser(description="Zone Drawing Tool")
    parser.add_argument("--camera", default="", help="Camera name (e.g. 'Main CCTV')")
    parser.add_argument("--source", default="", help="Video path or webcam index")
    args = parser.parse_args()

    # ── Non-interactive mode (launched from Streamlit with args) ───────────────
    if args.camera and args.source:
        camera_name = args.camera
        src = args.source

        # Webcam index?
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
            if not cap.isOpened():
                print(f"Cannot open webcam {src}.")
                sys.exit(1)
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Could not read frame from webcam.")
                sys.exit(1)
            ZoneDrawingApp(camera_name=camera_name, frame=frame).run()

        elif os.path.exists(src):
            ZoneDrawingApp(camera_name=camera_name, video_path=src).run()

        else:
            print(f"Source not found: {src}")
            sys.exit(1)

        sys.exit(0)

    # ── Interactive mode (run directly from terminal) ─────────────────────────
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
        for _ in range(5):
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
