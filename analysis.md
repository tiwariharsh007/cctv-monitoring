# Project Analysis — Real-Time Surveillance System

A computer-vision surveillance system that ingests video (recorded file, webcam, or
live IP/RTSP stream), runs YOLO-based person/object detection, tracks people, detects
events (fall, crowd, running, intrusion, abandoned object, accident, etc.), logs
analytics to SQLite, saves snapshots, and fires alerts (console + email + optional
AI-generated captions). It can be driven headless (`main.py`) or through a Streamlit
dashboard.

---

## 1. Files & Folder Structure (with use)

Only the files and folders that participate in the actual running of the system are
listed below (docs, license, dead code, and manual utility scripts are omitted).

```text

RealTimeSurveillanceSystem/
│
│  ── Entry points (run the system) ──
├── main.py                     # Primary headless runner; orchestrates the full pipeline
├── dashboard_streamlit.py      # Streamlit UI; drives the same SurveillanceEngine
│
│  ── Core analysis pipeline ──
├── analysis_engine.py          # SurveillanceEngine — per-frame analysis + alert coordination
├── detector.py                 # Person/object detection using YOLOv8 (PersonDetector)
├── tracker.py                  # Centroid-based multi-object tracking + ID history
├── zone_draw_tool.py           # OpenCV GUI to draw/edit detection zones (ZoneDrawingApp)
│
├── detectors/                  # Behavioral event detectors used by the engine
│   ├── zone_intrusion.py       # Zone logic (intrusion / entry-exit / high-value / loitering)
│   ├── accident.py             # Vehicle accident (collision / sudden-stop) detection
│   ├── speed.py                # Running detection (centroid speed)
│   ├── abandoned_object.py     # Unattended / abandoned-object detection
│   └── dwell_time.py           # Per-ID dwell-time tracking (loitering, box coloring)
│
│  ── Alerts, logging & reporting ──
├── services/
│   └── alert_service.py        # Central alert handler (handle_alert) used by the engine
├── alerts.py                   # Email (+ optional Twilio) alert dispatch with snapshot
├── caption_generator.py        # AI alert-caption generation via Google Gemini (optional)
├── db.py                       # SQLite database initialization and logging
├── reporting.py                # CSV traffic-report generation
│
│  ── Configuration & model ──
├── config.yaml                 # Central config (camera source, thresholds, cooldown)
├── .env                        # Secrets / flags (email, Gemini, Twilio)
├── yolov8n.pt                  # YOLOv8-nano model weights
│
│  ── Data & runtime outputs ──
├── zones/zone_config.json      # Saved zone polygons per camera (written by zone tool)
├── data/test_videos/           # Input video clips
├── logs/                       # analytics.db, heatmaps, live-feed jpg, traffic CSVs
├── snapshots/                  # Auto-saved event snapshot images
└── reports/                    # Generated daily CSV reports

```

> **Note:** `main.py` and `dashboard_streamlit.py` both drive the **same**
> `SurveillanceEngine`, so every video source behaves identically.

---

## 2. Project Flow

### A. Headless flow (`main.py` — primary)

```
config.yaml  ─────────────┐
                          ▼
  choose_source()  →  resolve_source()  →  open_capture()
                          │
              setup_zones() ──(optional)── ZoneDrawingApp → zones/zone_config.json
                          │
                  SurveillanceEngine(cfg)        init_db("logs/analytics.db")
                          │
          ┌───────────────┴──────────────────── per-frame loop ─────────────┐
          │  cap.read()  → skip-frame throttle → engine.process(frame)       │
          │                                                                  │
          │   engine.process():                                              │
          │     _fit1280 (letterbox 1280×720, aligns with zone coords)       │
          │     PersonDetector.detect_all()        → boxes + raw YOLO        │
          │     CentroidTracker.update()           → tracked IDs + history   │
          │     LineCounter.update()               → IN / OUT counts         │
          │     DwellTimeTracker.update()          → dwell secs              │
          │     heatmap accumulation + trails + ID/dwell labels              │
          │     EVENT DETECTORS (zone-gated + activity-gated):               │
          │        fall · tailgating · inactivity · running · crowd ·        │
          │        zone intrusion · abandoned object · vehicle accident      │
          │     for each fired event → _should_alert() cooldown →            │
          │        _fire(): save snapshot → handle_alert() (console) →        │
          │                 off-thread: caption (Gemini) + send_email_alert  │
          │     _draw_hud() overlay                                          │
          │                                                                  │
          │   back in loop: cv2.imshow, write logs/live_feed.jpg,            │
          │                 periodic heatmap, throttled insert_log() to DB,  │
          │                 append to session_logs                           │
          └──────────────────────────────────────────────────────────────────┘
                          │  (Q pressed / video ends)
                          ▼
        save heatmap + live feed · save_daily_report() CSV ·
        print_session_summary() · release capture · close DB
```

### B. Dashboard flow (`dashboard_streamlit.py`)

```
Streamlit UI  →  user selects activities to monitor
              →  SurveillanceEngine.set_monitored_activities(...)
              →  same engine.process() loop as above
              →  live feed + heatmap shown in browser
              →  analytics read from logs/analytics.db (SQLite)
```

### C. Alternate Flask flow (`app.py` — independent)

```
Flask + SocketIO
  /video        → MJPEG stream via services/processing.process_frame()
                  (uses CentroidTracker + ZoneIntrusionDetector)
  /api/alert(s) → handle_alert()
  /api/reports  → reporting.generate_report() → CSV
  socket thread → send_real_time_data() emits periodic updates
```

### Data / alert sinks

- **SQLite** (`logs/analytics.db`, `traffic_logs` table) — analytics rows.
- **Snapshots** (`snapshots/*.jpg`) — per-event captures.
- **Heatmap / live feed** (`logs/*.jpg`) — visual outputs.
- **Reports** (`reports/*.csv`) — daily session summaries.
- **Email / SMS** — via `alerts.py` (gated by `.env`), message optionally generated by Gemini.

### Key design note

`main.py` and the Streamlit dashboard both drive the **same `SurveillanceEngine`**, so
every source behaves identically. `app.py` and `dashboard_flask.py` are separate,
parallel implementations that do not share that engine.
