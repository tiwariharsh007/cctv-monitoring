"""
Smart CCTV Surveillance System

Camera source → config.yaml:
  source: 0                            → default webcam
  source: "data/test_videos/front.mp4" → local video file
  source: "rtsp://user:pass@ip:554"    → IP/CCTV camera (auto-reconnects)
"""
import os, sys, csv, glob, time, yaml
import cv2
from datetime import datetime, timedelta
from collections import Counter

from db import init_db, insert_log
from analysis_engine import SurveillanceEngine

os.makedirs("snapshots", exist_ok=True)
os.makedirs("logs",      exist_ok=True)
os.makedirs("reports",   exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Detection/alert thresholds are read by SurveillanceEngine straight from CFG, so
# main.py only needs the source/report bits here.
CFG = yaml.safe_load(open("config.yaml"))

CAM_SOURCE   = CFG["camera"]["source"]
LIVE_SOURCE  = CFG["camera"].get("live_source", 0)
CAM_NAME     = CFG["camera"].get("name", "Main CCTV")
SKIP_FRAMES  = CFG["camera"].get("skip_frames", 2)
CAP_ENABLED  = CFG["alerts"].get("captions_enabled", False)
REPORT_FOLDER= CFG["reports"]["folder"]
SAVE_CSV     = CFG["reports"]["save_daily_csv"]


def print_session_summary(session_logs: list, elapsed_sec: float):
    if not session_logs:
        return
    occ     = [r["visible_count"] for r in session_logs if r["visible_count"] > 0]
    alerts  = [p for r in session_logs for p in r["alert"].split() if p]
    print("\n" + "═" * 52)
    print(f"  SESSION SUMMARY — {CAM_NAME}")
    print(f"  Duration     : {int(elapsed_sec//60)}m {int(elapsed_sec%60)}s")
    print(f"  Total IN     : {session_logs[-1]['in_count']}")
    print(f"  Total OUT    : {session_logs[-1]['out_count']}")
    if occ:
        print(f"  Peak occ.    : {max(occ)} people")
        print(f"  Avg occ.     : {sum(occ)/len(occ):.1f} people")
    if alerts:
        print(f"  Alerts       : {dict(Counter(alerts).most_common())}")
    print("═" * 52 + "\n")


def save_daily_report(session_logs: list):
    if not session_logs:
        return
    date = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(REPORT_FOLDER,
                        f"report_{CAM_NAME.replace(' ','_')}_{date}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["time","camera_id","in_count","out_count",
                           "visible_count","posture","alert"])
        writer.writeheader()
        writer.writerows(session_logs)
    print(f"📄 Report saved → {path}")


# ── CAMERA SOURCE ──────────────────────────────────────────────────────────────
def choose_source():
    """Pick what to analyse: recorded video or live camera.

    CLI (non-interactive):
        python main.py video [path]   → recorded file (default config.source)
        python main.py live  [url]    → live stream  (default config.live_source)
        python main.py webcam [index] → built-in / USB webcam
        python main.py <path|url|idx> → use that source verbatim
    No args → interactive menu.
    """
    args = sys.argv[1:]
    if args:
        a = args[0].lower()
        if a in ("live", "phone", "stream"):
            return args[1] if len(args) > 1 else LIVE_SOURCE
        if a in ("video", "file", "recorded"):
            return args[1] if len(args) > 1 else CAM_SOURCE
        if a in ("webcam", "cam"):
            return int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
        return args[0]   # treat as a literal path / URL / index

    videos = sorted(glob.glob("data/test_videos/*.mp4"))
    print("\n┌─ Select source ─────────────────────────────────")
    print("│  [1] Recorded video")
    print("│  [2] Live camera / phone stream")
    print("│  [3] Webcam (built-in / USB)")
    print("└──────────────────────────────────────────────────")
    choice = input("Choice [1]: ").strip() or "1"

    if choice == "2":
        url = input(f"Stream URL [{LIVE_SOURCE}]: ").strip() or LIVE_SOURCE
        return url
    if choice == "3":
        idx = input("Webcam index [0]: ").strip() or "0"
        return int(idx) if idx.isdigit() else 0

    # Recorded video — let the user pick from data/test_videos if any exist
    if videos:
        for i, v in enumerate(videos, 1):
            print(f"   ({i}) {v}")
        sel = input(f"Pick video [1]: ").strip() or "1"
        if sel.isdigit() and 1 <= int(sel) <= len(videos):
            return videos[int(sel) - 1]
    return CAM_SOURCE if isinstance(CAM_SOURCE, str) else (videos[0] if videos else 0)


def resolve_source(src):
    """Classify the configured source.
      int / "0"                       → webcam      (live, wall-clock)
      rtsp:// http:// https://         → stream      (live, wall-clock, auto-reconnect)
      anything else (a path)           → file        (video clock)
    A phone camera served by an app (IP Webcam, DroidCam, RTSP Camera, …) is just a
    stream URL — e.g. http://192.168.1.5:8080/video  or  rtsp://192.168.1.5:8554/live.
    """
    if isinstance(src, int):
        return src, "webcam"
    s = str(src).strip()
    if s.lower().startswith(("rtsp://", "http://", "https://")):
        return s, "stream"
    if s.isdigit():
        return int(s), "webcam"
    return s, "file"


def open_capture(source, is_stream):
    cap = cv2.VideoCapture(source)
    if is_stream:
        # Keep only the latest frame so analysis tracks the live feed instead of
        # falling behind on a buffered backlog (critical for phone/IP streams).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def run(chosen_source=None):
    source, src_kind = resolve_source(CAM_SOURCE if chosen_source is None else chosen_source)
    is_file   = src_kind == "file"
    is_stream = src_kind == "stream"

    print(f"🎥  Opening [{src_kind}]: {source}")
    cap = open_capture(source, is_stream)
    if not cap.isOpened():
        hint = ("→ Phone stream? Check the URL, that the app is streaming, and that "
                "phone + PC share the same Wi-Fi." if is_stream else
                "→ Check config.yaml → camera → source")
        print(f"❌  Cannot open: {source}\n    {hint}")
        return

    # Files play faster than real-time → derive timestamps from the video clock so
    # analytics spread across the real duration. Live sources use wall-clock.
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    clock_origin = datetime.now()

    # The whole analysis + alerting pipeline lives in SurveillanceEngine, shared
    # with the dashboard so every source behaves identically.
    engine = SurveillanceEngine(CFG, cam_name=CAM_NAME, captions=CAP_ENABLED)

    db_conn = init_db("logs/analytics.db")

    # Clear previous session
    db_conn.execute("DELETE FROM traffic_logs")
    db_conn.commit()
    for f in ["logs/live_feed.jpg", "logs/heatmap_main.jpg"]:
        if os.path.exists(f):
            os.remove(f)
    print("🗑️  Previous session cleared — starting fresh.")

    # State
    last_frame    = None
    frame_count   = 0
    last_db_time  = 0.0
    last_in       = 0
    last_out      = 0
    session_logs  = []
    session_start = time.time()

    print(f"✅  {CAM_NAME} running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_stream:
                print("⚠️  Stream lost — reconnecting in 3 s …")
                time.sleep(3)
                cap = open_capture(source, is_stream)
                continue
            print("📹  Video ended.")
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        if is_file:
            clock_sec = frame_count / fps                    # seconds into the video
            stamp_dt  = clock_origin + timedelta(seconds=clock_sec)
        else:
            clock_sec = time.time()                          # wall-clock for live cams
            stamp_dt  = datetime.now()
        now_str = stamp_dt.strftime("%Y-%m-%d %H:%M:%S")

        # ── FULL ANALYSIS + ALERTING (shared engine) ─────────────────────────
        result        = engine.process(frame)
        frame         = result["frame"]
        visible_count = result["visible_count"]
        posture       = result["posture"]
        alert_str     = result["alert"]

        if frame_count % 10 == 0:
            cv2.imwrite("logs/live_feed.jpg", frame)
        if frame_count % 200 == 0:
            hm = engine.colored_heatmap(frame)
            if hm is not None:
                cv2.imwrite("logs/heatmap_main.jpg", hm)

        last_frame = frame

        try:
            h, w  = frame.shape[:2]
            scale = min(1.0, 1280 / w)
            disp  = cv2.resize(frame, (int(w*scale), int(h*scale)))
            cv2.imshow(CAM_NAME, disp)
        except Exception:
            pass

        # ── DB LOG (throttle by the same clock that stamps rows) ───────────────
        cin, cout = result["in_count"], result["out_count"]
        crossing  = (cin != last_in or cout != last_out)

        if alert_str or crossing or (clock_sec - last_db_time) >= 2:
            row = {
                "time":          now_str,
                "camera_id":     CAM_NAME,
                "in_count":      cin,
                "out_count":     cout,
                "visible_count": visible_count,
                "posture":       posture,
                "alert":         alert_str,
            }
            insert_log(db_conn, row)
            session_logs.append(row)
            last_db_time = clock_sec
            last_in      = cin
            last_out     = cout

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    hm = engine.colored_heatmap(last_frame)
    if hm is not None:
        cv2.imwrite("logs/heatmap_main.jpg", hm)
    if last_frame is not None:
        cv2.imwrite("logs/live_feed.jpg", last_frame)
    if SAVE_CSV:
        save_daily_report(session_logs)
    print_session_summary(session_logs, time.time() - session_start)
    cap.release()
    db_conn.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(choose_source())
