import streamlit as st
import pandas as pd
import numpy as np
import cv2
import yaml
import sqlite3
import os, glob, io, time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import datetime

from db import init_db, insert_log
from analysis_engine import SurveillanceEngine

st.set_page_config(layout="wide", page_title="Smart CCTV Dashboard")

DB_PATH    = "logs/analytics.db"
LOG_FOLDER = "logs"
SNAP_DIR   = "snapshots"
LIVE_PATH  = os.path.join(LOG_FOLDER, "live_feed.jpg")
HEATMAP    = os.path.join(LOG_FOLDER, "heatmap_main.jpg")
ZONE_CSV   = os.path.join(LOG_FOLDER, "heatmap_zones.csv")

ALERT_TYPES = [
    "Accident", "Fall", "Tailgating", "Running", "Crowd", "CapacityWarning",
    "Loitering", "Inactivity", "Intrusion", "AbandonedObject", "AfterHours",
]
SEVERITY = {
    "Accident": "HIGH", "Fall": "HIGH", "Intrusion": "HIGH", "AbandonedObject": "HIGH",
    "AfterHours": "HIGH", "CapacityWarning": "HIGH", "Tailgating": "HIGH",
    "Crowd": "MEDIUM", "Loitering": "MEDIUM", "Running": "MEDIUM", "Inactivity": "MEDIUM",
}
SEV_COLOR = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#adb5bd"}
LIVE_THRESHOLD_SEC = 20   # frame newer than this → system considered ONLINE

# ── ANALYSIS ENGINE (same pipeline on a RECORDED video OR a LIVE camera) ────────
try:
    _CFG = yaml.safe_load(open("config.yaml")) or {}
except Exception:
    _CFG = {}
CROWD_THRESH     = int(_CFG.get("detection", {}).get("crowd_threshold", 4))
DEFAULT_LIVE_URL = str(_CFG.get("camera", {}).get("live_source", "http://192.168.1.37:8080/video"))

for _k, _v in {"an_active": False, "an_source": None, "an_label": "",
               "an_detect": True, "an_loop": True}.items():
    st.session_state.setdefault(_k, _v)


@st.cache_resource(show_spinner="Loading detection model…")
def get_detector():
    from detector import PersonDetector
    return PersonDetector()


def _is_stream(src):
    return isinstance(src, str) and src.lower().startswith(("rtsp://", "http://", "https://"))


def open_source(src):
    cap = cv2.VideoCapture(src)
    if _is_stream(src):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # live → keep only the newest frame
        except Exception:
            pass
    return cap


def _start_analysis(source, label, detect, loop):
    _stop_analysis()
    try:
        st.cache_data.clear()          # drop stale DB/chart cache from prior runs
    except Exception:
        pass
    st.session_state._an_events = []   # fresh alert feed for this session
    st.session_state.update(an_source=source, an_label=label,
                            an_detect=detect, an_loop=loop, an_active=True)


def _stop_analysis():
    cap = st.session_state.get("_an_cap")
    if cap is not None:
        try: cap.release()
        except Exception: pass
    conn = st.session_state.get("_an_db")
    if conn is not None:
        try: conn.close()
        except Exception: pass
    for k in ("_an_cap", "_an_engine", "_an_db"):
        st.session_state.pop(k, None)
    st.session_state.an_active = False


def render_analysis():
    """Standalone page: one frame per rerun → responsive & stoppable. Recorded & live
    drive the SAME SurveillanceEngine as main.py, so detection, alert firing
    (snapshot + email) and DB logging are identical for every source."""
    st.title("Live Analysis")
    src = st.session_state.an_source

    c_stop, c_info = st.columns([1, 4])
    if c_stop.button("Stop", use_container_width=True):
        _stop_analysis(); st.rerun()
    c_info.caption(f"Analysing: **{st.session_state.an_label}**  ·  "
                   f"detection {'ON' if st.session_state.an_detect else 'OFF'}")

    if st.session_state.get("_an_cap") is None:
        cap = open_source(src)
        if not cap.isOpened():
            st.error(f"Could not open source: `{src}`")
            if _is_stream(src):
                st.info("Live stream unreachable — is the phone app streaming on the same Wi-Fi? "
                        "Use the **/video** endpoint, e.g. `http://192.168.1.37:8080/video`.")
            _stop_analysis(); return
        st.session_state._an_cap    = cap
        # Use the configured camera name (matches zone_config.json keys) so that
        # zones drawn with zone_draw_tool are applied during dashboard analysis.
        _cam_name = _CFG.get("camera", {}).get("name", "Main CCTV")
        st.session_state._an_engine = SurveillanceEngine(
            _CFG, cam_name=_cam_name,
            detector=get_detector(), captions=False)   # email self-gated by .env
        st.session_state._an_db     = init_db("logs/analytics.db")
        st.session_state._an_events = []
        st.session_state._an_last   = (0, 0, 0)        # (frame#, last_in, last_out)

    cap    = st.session_state._an_cap
    engine = st.session_state._an_engine
    ok, frame = cap.read()
    if not ok:
        if _is_stream(src):
            st.warning("Stream interrupted — stopping. Restart it from the controls.")
            _stop_analysis(); return
        if st.session_state.an_loop:                    # recorded file ended → loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            st.success("Recorded video finished.")
            _stop_analysis(); return

    visible = cin = cout = 0
    alerts = []
    if st.session_state.an_detect:
        try:
            result = engine.process(frame)            # full pipeline + alert firing
        except Exception as _exc:
            st.error(f"Analysis error: {_exc}")
            _stop_analysis(); return
        frame   = result["frame"]
        visible = result["visible_count"]
        cin, cout = result["in_count"], result["out_count"]
        alerts  = result["alerts"]
        if alerts:
            ev = st.session_state.setdefault("_an_events", [])
            ev.append((datetime.now().strftime("%H:%M:%S"), " ".join(alerts)))
            del ev[:-300]

        # Log to the same DB that powers the charts (throttled, like main.py)
        fnum, l_in, l_out = st.session_state._an_last
        fnum += 1
        if result["alert"] or cin != l_in or cout != l_out or fnum % 8 == 0:
            insert_log(st.session_state._an_db, {
                "time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera_id":     engine.cam_name,
                "in_count":      cin, "out_count": cout,
                "visible_count": visible, "posture": result["posture"],
                "alert":         result["alert"],
            })
            l_in, l_out = cin, cout
        st.session_state._an_last = (fnum, l_in, l_out)

    # ── Same layout as the Live Feed: frame on the left, status panel on the right ──
    f_col, s_col = st.columns([2, 1])
    with f_col:
        st.subheader("Analyzed Frame")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"Captured {datetime.now().strftime('%H:%M:%S')}")

    with s_col:
        st.subheader("Live Status")
        st.metric("People In Scene", visible)
        cc1, cc2 = st.columns(2)
        cc1.metric("Entered", cin)
        cc2.metric("Exited",  cout)

        events = st.session_state.get("_an_events", [])
        if events:
            st.markdown("**Most recent alert**")
            t_last, a_last = events[-1]
            st.error(f"{a_last}  ·  {t_last}")
            st.markdown("**Recent events**")
            for t, a in reversed(events[-10:]):
                st.write(f"`{t}`  {a}")
        else:
            st.success("No alerts detected")

    time.sleep(0.01)
    st.rerun()


# While analysing, render ONLY this page and skip the heavy dashboard below.
if st.session_state.an_active:
    render_analysis()
    st.stop()

# ── TOP BAR (title + controls only — no analysis here) ────────────────────────
t_col, r_col, a_col = st.columns([6, 1, 1])
with t_col:
    st.title("Smart CCTV Dashboard")
    st.caption("Real-time monitoring & analytics")
with r_col:
    auto_refresh = st.toggle("Auto", value=False, help="Auto-refresh every 5 s")
with a_col:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=[
            "time","camera_id","in_count","out_count","visible_count","posture","alert"
        ])
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM traffic_logs ORDER BY time", conn)
    conn.close()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "visible_count" not in df.columns:
        df["visible_count"] = 0
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

df = load_data()
if df is None or df.empty:
    st.warning("No data yet. Run `python3 main.py` or `python3 populate_test_data.py` first.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
all_cameras = sorted(df["camera_id"].dropna().unique().tolist())
cam_choice  = st.sidebar.selectbox("Camera", ["All Cameras"] + all_cameras)
sel_cameras = all_cameras if cam_choice == "All Cameras" else [cam_choice]

sel_events  = st.sidebar.multiselect(
    "Alert type  (empty = all)",
    options=ALERT_TYPES, default=[],
    placeholder="Filter alert log by type…",
)

min_t, max_t = df["time"].min(), df["time"].max()
if min_t != max_t:
    s, e = st.sidebar.slider(
        "Time range",
        min_value=min_t.to_pydatetime(), max_value=max_t.to_pydatetime(),
        value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
        format="MM/DD HH:mm",
    )
    start_ts, end_ts = pd.Timestamp(s), pd.Timestamp(e)
else:
    start_ts, end_ts = min_t, max_t

# ── FILTERED VIEWS ────────────────────────────────────────────────────────────
occ_filt = df[
    df["camera_id"].isin(sel_cameras) &
    (df["time"] >= start_ts) & (df["time"] <= end_ts)
].copy()
occ_filt["hour"]      = occ_filt["time"].dt.hour
occ_filt["date"]      = occ_filt["time"].dt.date
occ_filt["alert_str"] = occ_filt["alert"].fillna("").str.strip()

def alert_matches(val):
    if not sel_events:
        return True
    s = str(val).strip() if pd.notna(val) else ""
    return bool(s) and any(p in sel_events for p in s.split())

alert_rows = occ_filt[occ_filt["alert_str"] != ""]
alert_log  = occ_filt[occ_filt["alert"].apply(alert_matches) & (occ_filt["alert_str"] != "")]

# in_count / out_count are CUMULATIVE session counters (they only ever rise).
# Activity inside a filtered window is therefore (max − min) per camera — never the
# raw max, which still counts everything since main.py started. Summing per camera
# keeps the figure correct when several cameras are pooled under "All Cameras".
def window_entries(frame: pd.DataFrame, col: str) -> int:
    if frame.empty:
        return 0
    per_cam = frame.groupby("camera_id")[col].agg(lambda s: max(0, int(s.max() - s.min())))
    return int(per_cam.sum())

# Shared aggregates (computed once, used inside tabs only)
total_in    = window_entries(occ_filt, "in_count")
total_out   = window_entries(occ_filt, "out_count")
peak_occ    = int(occ_filt["visible_count"].max())     if not occ_filt.empty else 0
# "Now" = sum of each camera's most-recent in-scene count, not one stray last row.
current_occ = (
    int(occ_filt.sort_values("time").groupby("camera_id")["visible_count"].last().sum())
    if not occ_filt.empty else 0
)

def sev_of(alert_str):
    sevs = [SEVERITY.get(p, "LOW") for p in str(alert_str).split()]
    return "HIGH" if "HIGH" in sevs else "MEDIUM" if "MEDIUM" in sevs else "LOW"

# ── TABS (no metrics row directly under the title) ────────────────────────────
tab_live, tab_flow, tab_insights, tab_alerts, tab_snaps, tab_heat = st.tabs([
    "Live Feed", "People Flow", "Insights",
    "Alerts", "Snapshots", "Heatmap",
])

# ══ TAB 1 — LIVE FEED ══════════════════════════════════════════════════════════
with tab_live:
    with st.expander("Analyse a video source — recorded or live", expanded=True):
        mode   = st.radio("Source", ["Recorded video", "Live camera / phone", "Webcam"],
                          horizontal=True)
        detect = st.checkbox("Run analysis overlay (detection + counting + alerts)", value=True)
        loop   = True
        source, label = None, ""

        if mode == "Recorded video":
            vids = sorted(glob.glob("data/test_videos/*.mp4"))
            if vids:
                pick = st.selectbox("Video file", vids,
                                    format_func=lambda p: os.path.basename(p))
                source, label = pick, os.path.basename(pick)
            else:
                source = st.text_input("Video path", value="data/test_videos/front.mp4")
                label  = os.path.basename(source)
            loop = st.checkbox("Loop when finished", value=True)
        elif mode == "Live camera / phone":
            source = st.text_input("Stream URL (use the /video endpoint)", value=DEFAULT_LIVE_URL,
                                   help="Android 'IP Webcam' → Start server → http://PHONE_IP:8080/video")
            label  = "Live camera"
        else:
            idx    = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
            source = int(idx); label = f"Webcam {int(idx)}"

        if st.button("Start analysis", type="primary"):
            _start_analysis(source, label, detect, loop)
            st.rerun()
        st.caption("Recorded files and live streams play here the same way, with the same overlay. "
                   "`python main.py` additionally logs to the database that powers the charts.")

    _zone_cfg_path = "zones/zone_config.json"
    _cam_name_cfg  = _CFG.get("camera", {}).get("name", "Main CCTV")
    with st.expander("Zone Configuration", expanded=False):
        st.markdown(f"**Camera:** `{_cam_name_cfg}`")

        _zone_data = {}
        if os.path.exists(_zone_cfg_path):
            try:
                import json as _json
                with open(_zone_cfg_path) as _zf:
                    _zone_data = _json.load(_zf)
            except Exception:
                pass

        _zones_for_cam = _zone_data.get(_cam_name_cfg, [])
        if _zones_for_cam:
            st.success(f"{len(_zones_for_cam)} zone(s) configured")
            for _z in _zones_for_cam:
                acts = ", ".join(_z.get("monitored_activities", [_z.get("type", "?")]))
                pts  = len(_z.get("points", []))
                st.write(f"- **{_z['id']}** | type: `{_z.get('type','?')}` "
                         f"| activities: `{acts}` | {pts} pts")
        else:
            st.info("No zones configured yet for this camera.")

        zc1, zc2 = st.columns(2)
        with zc1:
            if st.button("Reload zones", help="Pick up zones saved by zone_draw_tool"):
                if st.session_state.get("_an_engine"):
                    st.session_state._an_engine.reload_zones()
                st.rerun()
        with zc2:
            if st.button("Launch Zone Editor",
                         help="Opens zone_draw_tool.py in a new window (requires local display)"):
                import subprocess as _sp
                _sp.Popen(["python", "zone_draw_tool.py"],
                          creationflags=getattr(_sp, "CREATE_NEW_CONSOLE", 0))
                st.info("Zone editor launched — draw your zones, press S to save, "
                        "then click Reload zones above.")

        st.caption("Zones drawn here apply to the *analysis overlay* above and to "
                   "`python main.py`. Activities selected per zone control which "
                   "detectors are scoped to that region.")

    st.divider()

    # System online/offline based on how fresh the live frame is
    if os.path.exists(LIVE_PATH):
        frame_age = time.time() - os.path.getmtime(LIVE_PATH)
        online    = frame_age < LIVE_THRESHOLD_SEC
    else:
        frame_age, online = None, False

    if online:
        st.success(f"System online — live frame updated {int(frame_age)}s ago")
    elif frame_age is not None:
        st.error(f"System offline — last frame {int(frame_age)}s ago. Is `main.py` running?")
    else:
        st.info("No live frame yet — start `python3 main.py`.")

    f_col, s_col = st.columns([2, 1])

    with f_col:
        st.subheader("Live Camera Frame")
        if os.path.exists(LIVE_PATH):
            try:
                mt = os.path.getmtime(LIVE_PATH)
                st.image(Image.open(LIVE_PATH),
                         caption=f"Captured {datetime.fromtimestamp(mt).strftime('%H:%M:%S')}",
                         use_container_width=True)
            except Exception as ex:
                st.warning(f"Cannot load frame: {ex}")
        else:
            st.info("Live frame appears here once `main.py` is running.")

    with s_col:
        st.subheader("Live Status")
        st.metric("People In Scene", current_occ)
        cc1, cc2 = st.columns(2)
        cc1.metric("Entered Today", total_in)
        cc2.metric("Exited Today",  total_out)

        # Most recent alert (with its real timestamp — not a stale flag)
        if not alert_rows.empty:
            last = alert_rows.iloc[-1]
            st.markdown("**Most recent alert**")
            st.error(f"{last['alert_str']}  ·  {last['time'].strftime('%H:%M:%S')}")
        else:
            st.success("No alerts recorded")

        st.markdown("**Recent events**")
        recent = alert_rows.sort_values("time", ascending=False).head(10)
        if not recent.empty:
            for _, row in recent.iterrows():
                st.write(f"`{row['time'].strftime('%H:%M:%S')}`  {row['alert_str']}  "
                         f"({sev_of(row['alert_str'])})")
        else:
            st.caption("No events in the selected range.")

# ══ TAB 2 — PEOPLE FLOW ════════════════════════════════════════════════════════
with tab_flow:
    if occ_filt.empty:
        st.info("No data.")
    else:
        span_s = max(1, (end_ts - start_ts).total_seconds())
        freq, flabel = (
            ("1h",   "per hour")    if span_s > 86400 else
            ("5min", "per 5 min")   if span_s > 7200  else
            ("1min", "per minute")  if span_s > 1200  else
            ("15s",  "per 15 sec")  if span_s > 180   else
            ("5s",   "per 5 sec")
        )

        for cam in sel_cameras:
            cam_df = occ_filt[occ_filt["camera_id"] == cam].set_index("time").sort_index()
            if cam_df.empty:
                continue
            st.subheader(cam)

            # Footfall KPIs for this camera
            k1, k2, k3, k4 = st.columns(4)
            entered = int(max(0, cam_df["in_count"].max()  - cam_df["in_count"].min()))
            exited  = int(max(0, cam_df["out_count"].max() - cam_df["out_count"].min()))
            k1.metric("Entered",        entered)
            k2.metric("Exited",         exited)
            k3.metric("Peak Occupancy", int(cam_df["visible_count"].max()))
            k4.metric("Avg Occupancy",  round(cam_df["visible_count"].mean(), 1))

            # 1) Occupancy over time — peak per bucket (the security-relevant value)
            st.markdown(f"**Live Occupancy — peak people in scene ({flabel})**")
            occ_peak = cam_df["visible_count"].resample(freq).max().dropna()
            occ_avg  = cam_df["visible_count"].resample(freq).mean().dropna()
            if len(occ_peak) >= 2:
                st.line_chart(
                    pd.DataFrame({"Peak": occ_peak, "Average": occ_avg.round(1)}),
                    color=["#e63946", "#00b4d8"],
                )
            else:
                # Too few buckets to resample — show raw points
                st.line_chart(cam_df["visible_count"].rename("Occupancy"), color="#00b4d8")

            # 2) People currently inside = cumulative entered − exited (capacity view)
            st.markdown("**People Inside — live headcount (entered − exited)**")
            inside = (cam_df["in_count"] - cam_df["out_count"]).clip(lower=0)
            inside_r = inside.resample(freq).max().dropna()
            if len(inside_r) >= 2:
                st.area_chart(inside_r.rename("Inside"), color="#2a9d8f")
            else:
                st.line_chart(inside.rename("Inside"), color="#2a9d8f")

            # 3) Footfall RATE — entries/exits per bucket (difference of the cumulative
            #    counters). This is the real "flow"; the cumulative curve only ever rises.
            st.markdown(f"**Footfall Rate — entries vs exits ({flabel})**")
            cum  = cam_df[["in_count", "out_count"]].resample(freq).max().ffill()
            rate = cum.diff().clip(lower=0).dropna(how="all")
            if not rate.empty and rate.to_numpy().sum() > 0:
                rate = rate.rename(columns={"in_count": "Entered", "out_count": "Exited"})
                st.bar_chart(rate, color=["#2a9d8f", "#f4a261"])
            else:
                st.caption("Not enough movement yet to chart footfall rate.")

            # 4) Hourly profile — only hours with data
            st.markdown("**Busiest Hours**")
            hourly = cam_df.assign(h=cam_df.index.hour).groupby("h")["visible_count"].mean()
            if hourly.sum() > 0:
                peak_h = int(hourly.idxmax())
                fig, ax = plt.subplots(figsize=(max(5, len(hourly)*0.6+1), 2.8))
                ax.bar(range(len(hourly)), hourly.values,
                       color=["#e63946" if h == peak_h else "#457b9d" for h in hourly.index],
                       width=0.65, edgecolor="none")
                ax.set_xticks(range(len(hourly)))
                ax.set_xticklabels([f"{h:02d}:00" for h in hourly.index], rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Avg People")
                ax.set_title(f"Peak hour {peak_h:02d}:00 — staff up before this", fontsize=9)
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.divider()

# ══ TAB 3 — INSIGHTS ═══════════════════════════════════════════════════════════
with tab_insights:
    if occ_filt.empty:
        st.info("No data.")
    else:
        st.subheader("Business Intelligence")
        all_parts = [p for a in occ_filt["alert_str"] for p in a.split() if p]
        span_h = max(0.01, (end_ts - start_ts).total_seconds() / 3600)

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Visitors / Hour", round(total_in / span_h, 1))
        i2.metric("Alerts / Hour",   round(len(alert_rows) / span_h, 2))
        i3.metric("Peak Occupancy",  peak_occ)
        i4.metric("Avg Occupancy",   round(occ_filt["visible_count"].mean(), 1))

        st.markdown("---")
        h_occ = occ_filt.groupby("hour")["visible_count"].mean()
        # Quietest must come from hours that actually had people — otherwise it always
        # reports the closed-overnight hour (occ≈0), which is useless for staffing.
        active = h_occ[h_occ > 0]
        if not active.empty:
            ca, cb = st.columns(2)
            ca.success(f"**Busiest hour:** {int(h_occ.idxmax()):02d}:00 — {h_occ.max():.1f} avg")
            cb.info(   f"**Quietest open hour:** {int(active.idxmin()):02d}:00 — {active.min():.1f} avg")

        st.markdown("---")
        st.markdown("#### Alert Severity Mix")
        sev_counts = {"HIGH":0, "MEDIUM":0, "LOW":0}
        for p in all_parts:
            sev_counts[SEVERITY.get(p, "LOW")] += 1
        sa, sb, sc = st.columns(3)
        sa.metric("High",   sev_counts["HIGH"])
        sb.metric("Medium", sev_counts["MEDIUM"])
        sc.metric("Low",    sev_counts["LOW"])

        st.markdown("---")
        if all_parts:
            st.markdown("#### Alert Activity by Hour")
            a_h = alert_rows.groupby("hour").size()
            if not a_h.empty:
                fig, ax = plt.subplots(figsize=(max(5, len(a_h)*0.6+1), 2.5))
                ax.bar(range(len(a_h)), a_h.values, color="#e63946", width=0.6, edgecolor="none", alpha=0.85)
                ax.set_xticks(range(len(a_h)))
                ax.set_xticklabels([f"{h:02d}:00" for h in a_h.index], rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Alerts")
                ax.set_title("When do incidents happen?", fontsize=9)
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.markdown("#### Activity / Posture Mix")
        # posture is logged for every frame but was never visualised. Lying ↔ falls,
        # Sitting ↔ dwell/waiting — useful for healthcare & retail dwell analysis.
        pos = (occ_filt.loc[occ_filt["visible_count"] > 0, "posture"]
               .fillna("Unknown").replace("", "Unknown").value_counts())
        if not pos.empty:
            pc1, pc2 = st.columns([1.3, 1])
            with pc1:
                fig, ax = plt.subplots(figsize=(4.5, 2.6))
                pcolor = {"Lying": "#e63946", "Sitting": "#f4a261", "Walking": "#2a9d8f",
                          "Standing": "#457b9d", "Unknown": "#adb5bd"}
                ax.barh(pos.index[::-1], pos.values[::-1],
                        color=[pcolor.get(p, "#adb5bd") for p in pos.index[::-1]], edgecolor="none")
                ax.set_xlabel("Frames observed")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            with pc2:
                lying = int(pos.get("Lying", 0))
                if lying:
                    st.error(f"**{lying}** lying-posture frames — possible falls. Cross-check the Fall alerts & snapshots.")
                else:
                    st.success("No lying postures detected.")
                st.caption("Posture is inferred from bounding-box aspect ratio; treat as a hint, not a diagnosis.")

        st.markdown("---")
        st.markdown("#### Unusual Occupancy (statistical anomalies)")
        # Flag time buckets where occupancy spikes well above the norm (mean + 2σ).
        # Cheap, model-free anomaly detection — surfaces unexpected crowding for review.
        occ_series = occ_filt.set_index("time")["visible_count"].resample("5min").max().dropna()
        if len(occ_series) >= 6 and occ_series.std() > 0:
            thresh = occ_series.mean() + 2 * occ_series.std()
            spikes = occ_series[occ_series > thresh]
            if not spikes.empty:
                st.warning(f"{len(spikes)} interval(s) exceeded the normal range "
                           f"(> {thresh:.1f} people, baseline {occ_series.mean():.1f}).")
                st.dataframe(
                    spikes.rename("Peak People").reset_index().assign(
                        time=lambda d: d["time"].dt.strftime("%m/%d %H:%M")
                    ).rename(columns={"time": "Interval (5 min)"}),
                    use_container_width=True, hide_index=True)
            else:
                st.success("Occupancy stayed within the normal range — no spikes.")
        else:
            st.caption("Need a longer time span to flag occupancy anomalies.")

        st.markdown("---")
        daily = occ_filt.groupby("date").apply(
            lambda g: pd.Series({
                "visitors": window_entries(g, "in_count"),
                "peak_occ": int(g["visible_count"].max()),
                "avg_occ":  round(g["visible_count"].mean(), 1),
                "alerts":   int(g["alert_str"].ne("").sum()),
            }), include_groups=False,
        ).reset_index()
        if len(daily) > 1:
            st.markdown("#### Daily Trend")
            st.dataframe(daily.rename(columns={
                "date":"Date","visitors":"Entered","peak_occ":"Peak Occ",
                "avg_occ":"Avg Occ","alerts":"Alerts"}), use_container_width=True)
        else:
            st.caption("Daily trend appears once data spans multiple days.")

# ══ TAB 4 — ALERTS ═════════════════════════════════════════════════════════════
with tab_alerts:
    all_parts_a = [p for a in occ_filt["alert_str"] for p in a.split() if p]
    if not all_parts_a:
        st.info("No alerts in selected time range.")
    else:
        fc, tc = st.columns([1, 1.4])

        with fc:
            st.subheader("Alert Frequency")
            freq_s = pd.Series(all_parts_a).value_counts()
            freq_s = freq_s.reindex(
                [e for e in ALERT_TYPES if e in freq_s.index] +
                [e for e in freq_s.index if e not in ALERT_TYPES], fill_value=0)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            bcolors = [SEV_COLOR.get(SEVERITY.get(k,"LOW"),"#adb5bd") for k in freq_s.index[::-1]]
            ax.barh(freq_s.index[::-1], freq_s.values[::-1], color=bcolors, edgecolor="none")
            ax.set_xlabel("Count")
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(handles=[mpatches.Patch(color="#e63946",label="High"),
                               mpatches.Patch(color="#f4a261",label="Medium")],
                      fontsize=7, loc="lower right")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with tc:
            # Strip timeline: each alert = a dot at its time, row per type, colored by severity.
            # Far clearer than a line chart for sparse, bursty events.
            st.subheader("Alert Timeline")
            events = [
                {"time": row["time"], "type": p}
                for _, row in alert_rows.iterrows()
                for p in row["alert_str"].split() if p
            ]
            if events:
                ev = pd.DataFrame(events)
                types_present = [t for t in ALERT_TYPES if t in ev["type"].unique()]
                ypos = {t: i for i, t in enumerate(types_present)}
                fig, ax = plt.subplots(figsize=(7, max(2.2, 0.42*len(types_present)+1)))
                for t in types_present:
                    sub = ev[ev["type"] == t]
                    ax.scatter(sub["time"], [ypos[t]]*len(sub),
                               c=SEV_COLOR.get(SEVERITY.get(t,"LOW"),"#adb5bd"),
                               s=42, alpha=0.75, edgecolors="none")
                ax.set_yticks(range(len(types_present)))
                ax.set_yticklabels(types_present, fontsize=8)
                ax.set_ylim(-0.5, len(types_present)-0.5)
                ax.set_xlabel("Time")
                ax.grid(axis="x", linestyle=":", alpha=0.4)
                ax.spines[["top","right"]].set_visible(False)
                fig.autofmt_xdate()
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                st.caption("Each dot is one alert. Rows = alert type, colour = severity (red = high).")

        st.subheader("Alert Log" + (f" — {', '.join(sel_events)}" if sel_events else ""))
        log_df = (
            alert_log[["time","camera_id","alert_str","visible_count"]]
            .sort_values("time", ascending=False)
            .rename(columns={"alert_str":"alert","visible_count":"people_in_frame"})
            .head(300)
        ).copy()
        log_df.insert(0, "Severity", log_df["alert"].apply(sev_of))
        if not log_df.empty:
            st.dataframe(log_df, use_container_width=True)
            buf = io.StringIO(); log_df.to_csv(buf, index=False)
            st.download_button("Download Alert Log (CSV)", buf.getvalue(),
                               f"alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        else:
            st.info("No alerts match the current filter.")

# ══ TAB 5 — SNAPSHOTS ══════════════════════════════════════════════════════════
with tab_snaps:
    st.subheader("Alert Snapshots")
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR,"*.jpg")), key=os.path.getmtime, reverse=True)
    if not snaps:
        st.info("No snapshots yet — saved automatically when alerts fire.")
    else:
        ev_types = sorted({os.path.basename(f).split("_")[0] for f in snaps})
        sel_ev   = st.multiselect("Filter by event", ev_types, default=ev_types)
        shown    = [f for f in snaps if os.path.basename(f).split("_")[0] in sel_ev][:24]
        cols = st.columns(4)
        for i, sp in enumerate(shown):
            fname = os.path.basename(sp); etype = fname.split("_")[0]
            tpart = fname.replace(f"{etype}_","").replace(".jpg","")
            try:
                td = datetime.strptime(tpart,"%Y%m%d_%H%M%S").strftime("%m/%d %H:%M:%S")
            except Exception:
                td = tpart
            with cols[i%4]:
                try:
                    sev = SEVERITY.get(etype.capitalize(), "LOW")
                    st.image(Image.open(sp), caption=f"{etype.upper()} · {sev} · {td}",
                             use_container_width=True)
                except Exception:
                    st.warning(f"Cannot load {fname}")

# ══ TAB 6 — HEATMAP ════════════════════════════════════════════════════════════
with tab_heat:
    st.subheader("Motion Heatmap & Zone Activity")
    img_col, zone_col = st.columns([1.4, 1])

    with img_col:
        if os.path.exists(HEATMAP):
            mt = os.path.getmtime(HEATMAP)
            st.image(Image.open(HEATMAP),
                     caption=f"Generated {datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S')}",
                     use_container_width=True)
            st.caption("Warm colours (red/yellow) overlaid on the scene = where people spend the most time.")
        else:
            st.info("No heatmap yet — generated automatically while `main.py` runs.")

    with zone_col:
        st.markdown("**Activity by Zone (3×3 grid)**")
        if os.path.exists(ZONE_CSV):
            zdf = pd.read_csv(ZONE_CSV).sort_values("percent", ascending=False)
            if not zdf.empty and zdf["percent"].sum() > 0:
                top = zdf.iloc[0]
                st.success(f"Busiest zone: **{top['zone']}** ({top['percent']:.0f}% of activity)")
                fig, ax = plt.subplots(figsize=(4.5, 3))
                ax.barh(zdf["zone"][::-1], zdf["percent"][::-1], color="#e63946", edgecolor="none")
                ax.set_xlabel("% of total activity")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                with st.expander("How to use this"):
                    st.markdown(
                        "- **Retail:** put promotions / staff in the busiest zone, investigate dead zones.\n"
                        "- **Security:** confirm cameras cover high-traffic areas; spot blind spots.\n"
                        "- **Safety:** high-dwell zones near hazards (machinery, stairs) need attention.\n"
                        "- **Layout:** redesign flow if one zone is congested while others are empty."
                    )
            else:
                st.caption("No activity recorded yet.")
        else:
            st.caption("Zone breakdown appears after `main.py` saves a heatmap.")

# ── RAW LOGS ──────────────────────────────────────────────────────────────────
with st.expander("Raw Log Data"):
    raw = occ_filt.drop(columns=["alert_str","hour","date"], errors="ignore")\
                  .sort_values("time", ascending=False).head(500)
    st.dataframe(raw, use_container_width=True)
    buf2 = io.StringIO(); raw.to_csv(buf2, index=False)
    st.download_button("Download Raw Logs (CSV)", buf2.getvalue(),
                       f"raw_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
