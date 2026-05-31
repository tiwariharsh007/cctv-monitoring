"""
CCTV Surveillance and Monitoring Dashboard

Sidebar  : camera source | detection zones | activities to monitor | start/stop
Main     : live analysis (frame + real-time metrics)  OR  analytics tabs (idle)
"""
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import yaml
import sqlite3
import os, glob, io, time, json, subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import datetime

from db import init_db, insert_log
from analysis_engine import SurveillanceEngine

st.set_page_config(
    layout="wide",
    page_title="CCTV Surveillance and Monitoring",
    initial_sidebar_state="expanded",
)

# ── constants / config ─────────────────────────────────────────────────────────
DB_PATH   = "logs/analytics.db"
LIVE_PATH = "logs/live_feed.jpg"
HEATMAP   = "logs/heatmap_main.jpg"
ZONE_CSV  = "logs/heatmap_zones.csv"
ZONE_CFG  = "zones/zone_config.json"
SNAP_DIR  = "snapshots"
LIVE_THRESHOLD_SEC = 20

try:
    _CFG = yaml.safe_load(open("config.yaml")) or {}
except Exception:
    _CFG = {}

CAM_NAME         = _CFG.get("camera", {}).get("name", "Main CCTV")
DEFAULT_LIVE_URL = str(_CFG.get("camera", {}).get("live_source", "http://192.168.1.37:8080/video"))

ALERT_TYPES = [
    "Fall", "Tailgating", "Running", "Crowd",
    "Inactivity", "AbandonedObject",
    "Accident",
]
SEVERITY = {
    "Accident": "HIGH", "Fall": "HIGH", "Intrusion": "HIGH",
    "AbandonedObject": "HIGH",
    "Tailgating": "HIGH",
    "Crowd": "MEDIUM",
    "Running": "MEDIUM", "Inactivity": "MEDIUM",
}
SEV_COLOR = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#adb5bd"}

# ── session state defaults ─────────────────────────────────────────────────────
for _k, _v in {
    "an_active": False, "an_source": None,
    "an_label": "", "an_detect": True, "an_loop": True,
    "sel_activities": list(ALERT_TYPES),
    "_an_events": [],
}.items():
    st.session_state.setdefault(_k, _v)

# ── cached model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model...")
def get_detector():
    from detector import PersonDetector
    return PersonDetector()

# ── helpers ────────────────────────────────────────────────────────────────────
def _is_stream(src):
    return isinstance(src, str) and src.lower().startswith(("rtsp://", "http://", "https://"))

def open_source(src):
    cap = cv2.VideoCapture(src)
    if _is_stream(src):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap

def load_zone_cfg():
    if not os.path.exists(ZONE_CFG):
        return {}
    try:
        with open(ZONE_CFG) as f:
            return json.load(f)
    except Exception:
        return {}

def save_zone_cfg(data):
    os.makedirs("zones", exist_ok=True)
    with open(ZONE_CFG, "w") as f:
        json.dump(data, f, indent=4)

def sev_of(alert_str):
    sevs = [SEVERITY.get(p, "LOW") for p in str(alert_str).split()]
    return "HIGH" if "HIGH" in sevs else "MEDIUM" if "MEDIUM" in sevs else "LOW"

def _write_session_outputs(engine, frame=None):
    """Persist the motion heatmap + zone-activity CSV for the current run.

    Called periodically during analysis and once more on stop, so both the
    Heatmap image and the Zone Activity chart reflect THIS session's data.
    """
    if engine is None:
        return
    try:
        os.makedirs("logs", exist_ok=True)
        hm = engine.colored_heatmap(frame)
        if hm is not None:
            cv2.imwrite(HEATMAP, hm)
    except Exception:
        pass
    try:
        rows = engine.zone_activity() if hasattr(engine, "zone_activity") else []
        if rows:
            pd.DataFrame(rows).to_csv(ZONE_CSV, index=False)
    except Exception:
        pass

def _start_analysis(source, label, detect, loop):
    _stop_analysis()
    # Clear the session's traffic rows so analytics tabs show only this run's data
    try:
        import sqlite3 as _sq
        if os.path.exists(DB_PATH):
            _c = _sq.connect(DB_PATH)
            _c.execute("DELETE FROM traffic_logs")
            _c.commit()
            _c.close()
    except Exception:
        pass
    # Remove the previous run's heatmap / zone-activity so a new analysis never
    # shows stale visuals before its own data is written.
    for _stale in (HEATMAP, ZONE_CSV):
        try:
            os.remove(_stale)
        except OSError:
            pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state._an_events = []
    st.session_state.update(
        an_source=source, an_label=label,
        an_detect=detect, an_loop=loop, an_active=True,
    )

def _stop_analysis():
    # Flush the final heatmap + zone-activity for the run before tearing down.
    _write_session_outputs(st.session_state.get("_an_engine"))
    for k, m in [("_an_cap", "release"), ("_an_db", "close")]:
        o = st.session_state.get(k)
        if o:
            try:
                getattr(o, m)()
            except Exception:
                pass
    for k in ("_an_cap", "_an_engine", "_an_db"):
        st.session_state.pop(k, None)
    st.session_state.an_active = False
    try:
        st.cache_data.clear()   # refresh analytics tabs with the just-written session data
    except Exception:
        pass

def window_entries(frame: pd.DataFrame, col: str) -> int:
    if frame.empty:
        return 0
    per_cam = frame.groupby("camera_id")[col].agg(lambda s: max(0, int(s.max() - s.min())))
    return int(per_cam.sum())

# ── data loader (cached 1 s — short TTL so analytics update quickly during analysis)
@st.cache_data(ttl=1)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=[
            "time", "camera_id", "in_count", "out_count",
            "visible_count", "posture", "alert",
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

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — always visible
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## CCTV Surveillance")

    # ── Camera source ──────────────────────────────────────────────────────────
    st.markdown("### Camera Source")
    source_type = st.radio(
        "source_type", ["Recorded Video", "Live Camera", "Webcam"],
        label_visibility="collapsed",
    )

    source, label, loop = None, "", True
    if source_type == "Recorded Video":
        vids = sorted(glob.glob("data/test_videos/*.mp4"))
        if vids:
            pick   = st.selectbox("Video file", vids,
                                  format_func=os.path.basename,
                                  label_visibility="collapsed")
            source = pick
            label  = os.path.basename(pick)
        else:
            source = st.text_input("Video path", "data/test_videos/front.mp4",
                                   label_visibility="collapsed")
            label  = os.path.basename(source or "")
        loop = st.checkbox("Loop when finished", value=True)

    elif source_type == "Live Camera":
        source = st.text_input(
            "Stream URL", DEFAULT_LIVE_URL,
            help="IP Webcam app -> Start server -> http://PHONE_IP:8080/video",
        )
        label = "Live Camera"

    else:
        idx    = st.number_input("Webcam index", 0, 10, 0, 1)
        source = int(idx)
        label  = f"Webcam {int(idx)}"

    detect = st.checkbox("AI detection + alerts", value=True,
                         help="Run YOLO detection, counting, zone checks and alert firing")

    st.markdown("### Detection Zones")
    zone_data = load_zone_cfg()
    cam_zones = zone_data.get(CAM_NAME, [])

    if cam_zones:
        # "Use no zones" master toggle — when off, all zone detection is skipped
        _use_zones = st.checkbox(
            "Enable zone detection",
            value=st.session_state.get("_zones_enabled", True),
            key="_zones_enabled",
        )

        _active_ids = []
        if _use_zones:
            st.caption("Select individual zones to activate:")
            for z in cam_zones:
                acts = ", ".join(z.get("monitored_activities", [z.get("type", "?")]))
                _checked = st.checkbox(
                    f"**{z['id']}** — `{acts}`",
                    value=st.session_state.get(f"_zactive_{z['id']}", True),
                    key=f"_zactive_{z['id']}",
                )
                if _checked:
                    _active_ids.append(z["id"])
            if _active_ids:
                st.caption(f"{len(_active_ids)} of {len(cam_zones)} zone(s) active")
            else:
                st.warning("No zones selected — analysis runs globally.")
        else:
            st.caption("Zone detection disabled — all detections are global.")

        st.session_state.active_zone_ids = _active_ids  # empty list = no zones

        # Delete a saved zone
        to_del = st.selectbox(
            "Remove a saved zone",
            ["-- keep all --"] + [z["id"] for z in cam_zones],
            label_visibility="collapsed",
        )
        if to_del != "-- keep all --":
            if st.button("Remove selected zone", type="secondary",
                         use_container_width=True):
                zone_data[CAM_NAME] = [z for z in cam_zones if z["id"] != to_del]
                save_zone_cfg(zone_data)
                st.session_state.pop(f"_zactive_{to_del}", None)
                if st.session_state.get("_an_engine"):
                    st.session_state._an_engine.reload_zones()
                st.rerun()

        # Clear every zone for this camera permanently
        if st.button("Clear All Zones", type="secondary", use_container_width=True,
                     help="Permanently removes all zones from the config file"):
            zone_data[CAM_NAME] = []
            save_zone_cfg(zone_data)
            # Remove checkbox state keys
            for _k in [k for k in st.session_state if k.startswith("_zactive_")]:
                del st.session_state[_k]
            st.session_state.active_zone_ids = []
            st.session_state._az_applied     = None
            # Apply immediately to running engine
            _live_eng = st.session_state.get("_an_engine")
            if _live_eng is not None:
                try:
                    _live_eng.zones.zones[CAM_NAME] = []
                except Exception:
                    pass
            st.rerun()
    else:
        st.caption(f"No zones for **{CAM_NAME}** yet.")
        st.session_state.active_zone_ids = []

    zc1, zc2 = st.columns(2)
    with zc1:
        draw_zone_clicked = st.button("Draw Zone", use_container_width=True,
                                      help="Opens zone editor (new window)")
    with zc2:
        if st.button("Reload Zones", use_container_width=True,
                     help="Pick up zones after saving in the editor"):
            if st.session_state.get("_an_engine"):
                st.session_state._an_engine.reload_zones()
            st.rerun()

    if draw_zone_clicked:
        src_arg = str(source) if source else ""
        try:
            subprocess.Popen(
                ["python", "zone_draw_tool.py",
                 "--camera", CAM_NAME,
                 "--source", src_arg],
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            st.success("Zone editor launched — draw zones and press S to save. Zones auto-reload.")
        except Exception as exc:
            st.error(f"Could not launch zone editor: {exc}")

    st.markdown("### Activities to Monitor")
    if cam_zones:
        st.info("🎯 **Intrusion detection** is handled by zones. Select other activities to monitor.")
        sel_acts = st.multiselect(
            "activities",
            options=ALERT_TYPES,
            default=st.session_state.sel_activities,
            label_visibility="collapsed",
        )
    else:
        st.caption("Uncheck to silence an alert type globally")
        sel_acts = st.multiselect(
            "activities",
            options=ALERT_TYPES,
            default=st.session_state.sel_activities,
            label_visibility="collapsed",
        )
    st.session_state.sel_activities = sel_acts

    st.divider()

    # ── Start / Stop ───────────────────────────────────────────────────────────
    if st.session_state.an_active:
        st.success(f"Analysing: **{st.session_state.an_label}**")
        if st.button("Stop Analysis", type="secondary", use_container_width=True):
            _stop_analysis()
            st.rerun()
    else:
        btn_disabled = (source is None or source == "")
        if st.button("Start Analysis", type="primary",
                     use_container_width=True, disabled=btn_disabled):
            _start_analysis(source, label, detect, loop)
            st.rerun()
        if btn_disabled:
            st.caption("Select a source above to enable.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("CCTV Surveillance" + (" — Live Analysis" if st.session_state.an_active else " — Analytics"))

hdr_r = st.columns([6, 1])[1]
with hdr_r:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

# Load analytics data (cached 5 s)
df = load_data()

# ── Shared filter state (used by every analytics tab) ─────────────────────────
_all_cameras = sorted(df["camera_id"].dropna().unique().tolist()) if (df is not None and not df.empty) else []
_sel_cameras = _all_cameras
_start_ts    = df["time"].min() if (df is not None and not df.empty) else None
_end_ts      = df["time"].max() if (df is not None and not df.empty) else None
_sel_events: list = []

if df is not None and not df.empty:
    with st.expander("Analytics Filters", expanded=False):
        _cam_choice  = st.selectbox("Camera", ["All Cameras"] + _all_cameras)
        _sel_cameras = _all_cameras if _cam_choice == "All Cameras" else [_cam_choice]
        _sel_events  = st.multiselect("Alert type filter (empty = all)", options=ALERT_TYPES, default=[])
        _min_t, _max_t = df["time"].min(), df["time"].max()
        if _min_t != _max_t:
            _sv, _ev = st.slider(
                "Time range",
                min_value=_min_t.to_pydatetime(), max_value=_max_t.to_pydatetime(),
                value=(_min_t.to_pydatetime(), _max_t.to_pydatetime()),
                format="MM/DD HH:mm",
            )
            _start_ts, _end_ts = pd.Timestamp(_sv), pd.Timestamp(_ev)
        else:
            _start_ts, _end_ts = _min_t, _max_t

# Pre-compute filtered views once (shared across tabs)
if df is not None and not df.empty and _start_ts is not None:
    occ_filt = df[
        df["camera_id"].isin(_sel_cameras) &
        (df["time"] >= _start_ts) & (df["time"] <= _end_ts)
    ].copy()
    occ_filt["hour"]      = occ_filt["time"].dt.hour
    occ_filt["date"]      = occ_filt["time"].dt.date
    occ_filt["alert_str"] = occ_filt["alert"].fillna("").str.strip()

    def _alert_matches(val):
        if not _sel_events:
            return True
        s = str(val).strip() if pd.notna(val) else ""
        return bool(s) and any(p in _sel_events for p in s.split())

    alert_rows = occ_filt[occ_filt["alert_str"] != ""]
    alert_log  = occ_filt[occ_filt["alert"].apply(_alert_matches) & (occ_filt["alert_str"] != "")]
    total_in    = window_entries(occ_filt, "in_count")
    total_out   = window_entries(occ_filt, "out_count")
    peak_occ    = int(occ_filt["visible_count"].max()) if not occ_filt.empty else 0
    current_occ = (
        int(occ_filt.sort_values("time").groupby("camera_id")["visible_count"].last().sum())
        if not occ_filt.empty else 0
    )
else:
    occ_filt    = pd.DataFrame()
    alert_rows  = pd.DataFrame()
    alert_log   = pd.DataFrame()
    total_in = total_out = peak_occ = current_occ = 0

# ── Tabs ───────────────────────────────────────────────────────────────────────
_t0 = "Live Analysis" if st.session_state.an_active else "Overview"
tab0, tab_flow, tab_insights, tab_alerts, tab_snaps, tab_heat = st.tabs([
    _t0, "People Flow", "Insights", "Alerts", "Snapshots", "Heatmap",
])

# _needs_rerun is set inside the live analysis block then honoured at the
# very end of the script — after ALL tabs are fully rendered — so that
# analytics tabs remain accessible while analysis is running.
_needs_rerun = False

# ══ TAB 0 — LIVE ANALYSIS  /  OVERVIEW ════════════════════════════════════════
with tab0:
    if st.session_state.an_active:
        # ── Initialise capture + engine (once per session) ────────────────────
        src     = st.session_state.an_source
        _selset = set(st.session_state.sel_activities)

        if st.session_state.get("_an_cap") is None:
            _cap = open_source(src)
            if not _cap.isOpened():
                st.error(f"Cannot open: `{src}`")
                if _is_stream(src):
                    st.info("Check URL and confirm the stream is active.")
                _stop_analysis()
                st.rerun()

            _eng = SurveillanceEngine(
                _CFG, cam_name=CAM_NAME,
                detector=get_detector(), captions=True,
            )
            # Configure the brand-new engine with the CURRENT selection right away
            # so it never runs a single frame monitoring activities the user has
            # deselected.
            if hasattr(_eng, "set_monitored_activities"):
                _eng.set_monitored_activities(st.session_state.get("sel_activities", []))
            st.session_state._an_cap      = _cap
            st.session_state._an_engine   = _eng
            st.session_state._an_db       = init_db(DB_PATH)
            st.session_state._an_last     = (0, 0, 0)
            st.session_state._az_applied  = None   # force first-run zone apply

        _cap    = st.session_state._an_cap
        _engine = st.session_state._an_engine

        # Re-apply zone filter whenever the sidebar selection changes.
        # This makes the toggle and checkboxes take effect immediately on the
        # next frame without restarting analysis.
        _az      = st.session_state.get("active_zone_ids")   # list or None
        _az_prev = st.session_state.get("_az_applied")
        if _az != _az_prev:
            # Apply zone filter directly on the ZoneIntrusionDetector's
            # zones dict — works even when set_active_zones doesn't exist
            # in a stale cached module.
            _zid = getattr(_engine, "zones", None)
            if _zid is not None and hasattr(_zid, "zones") and isinstance(_zid.zones, dict):
                try:
                    # Reload full config from disk first
                    if hasattr(_zid, "reload"):
                        _zid.reload()
                    else:
                        with open(ZONE_CFG) as _zf:
                            _zid.zones = json.load(_zf)
                    # Now filter to selected IDs (empty list → clear all zones)
                    if _az is not None:
                        _cam_all = _zid.zones.get(CAM_NAME, [])
                        _zid.zones[CAM_NAME] = [
                            z for z in _cam_all if z.get("id") in _az
                        ]
                except Exception:
                    pass
            st.session_state._az_applied = list(_az) if _az is not None else None

        # Sync the engine's monitored-activity set to the CURRENT sidebar
        # selection on EVERY frame. This is a trivial set assignment, and doing
        # it unconditionally (instead of guarding on a cached "_activities_applied"
        # value that can drift across start/stop) guarantees that deselecting an
        # activity silences it immediately — it can never keep monitoring an
        # activity the user has turned off.
        _sel = st.session_state.get("sel_activities", [])
        if hasattr(_engine, "set_monitored_activities"):
            _engine.set_monitored_activities(_sel)

        ok, frame = _cap.read()
        if not ok:
            if _is_stream(src):
                st.warning("Stream interrupted.")
                _stop_analysis()
                st.rerun()
            if st.session_state.an_loop:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = _cap.read()
            if not ok:
                st.success("Video finished.")
                _stop_analysis()
                st.rerun()

        visible = cin = cout = 0
        posture = ""
        alerts  = []

        if st.session_state.an_detect:
            try:
                result = _engine.process(frame)
            except Exception as exc:
                st.error(f"Analysis error: {exc}")
                _stop_analysis()
                st.rerun()

            frame     = result["frame"]
            visible   = result["visible_count"]
            cin, cout = result["in_count"], result["out_count"]
            posture   = result.get("posture", "")
            # Intrusion alerts from zones always show; other alerts filter by selection
            alerts    = [a for a in result["alerts"] if a == "Intrusion" or a in _selset]
            alert_str = " ".join(alerts)

            if alerts:
                _ev = st.session_state._an_events
                _ev.append((datetime.now().strftime("%H:%M:%S"), alert_str))
                del _ev[:-300]

            fnum, l_in, l_out = st.session_state._an_last
            fnum += 1
            if alert_str or cin != l_in or cout != l_out or fnum % 8 == 0:
                insert_log(st.session_state._an_db, {
                    "time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_id":     CAM_NAME,
                    "in_count":      cin, "out_count": cout,
                    "visible_count": visible,
                    "posture":       posture,
                    "alert":         alert_str,
                })
                l_in, l_out = cin, cout
            st.session_state._an_last = (fnum, l_in, l_out)

            if fnum % 10 == 0:
                os.makedirs("logs", exist_ok=True)
                cv2.imwrite(LIVE_PATH, frame)
            if fnum % 30 == 0:
                # Refresh heatmap + zone-activity periodically so both update
                # live (and short clips get at least one write before they end).
                _write_session_outputs(_engine, frame)

        # Layout
        f_col, s_col = st.columns([2, 1])
        with f_col:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=(f"{st.session_state.an_label}  |  "
                         f"{datetime.now().strftime('%H:%M:%S')}  |  {CAM_NAME}"),
            )
        with s_col:
            st.subheader("📊 Live Metrics")
            col1, col2 = st.columns(2)
            col1.metric("👥 People", visible)
            col2.metric("📈 Entered", cin)
            col1.metric("📉 Exited", cout)
            if posture and posture != "Standing":
                col2.metric("⚠️ Posture", posture)

            st.divider()
            st.subheader("🚨 Alerts")
            _ev = st.session_state._an_events
            if _ev:
                _tl, _al = _ev[-1]
                _sev = sev_of(_al)
                _col = SEV_COLOR.get(_sev, "#e63946")
                st.markdown(
                    f'<div style="padding:8px 12px;border-left:5px solid {_col};'
                    f'background:#0f0f0f;border-radius:4px;margin-bottom:8px">'
                    f'<span style="color:{_col};font-weight:900">● {_sev}</span>'
                    f'&nbsp;&nbsp;{_al}'
                    f'<br><span style="color:#666;font-size:0.75em">{_tl}</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Recent events (last 10)")
                for _t, _a in reversed(_ev[-10:]):
                    _s = sev_of(_a); _c = SEV_COLOR.get(_s, "#888")
                    st.markdown(
                        f'<span style="color:#555;font-size:0.75em">{_t}</span> '
                        f'<span style="color:{_c};font-weight:600">{_a}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("✓ No alerts")

        _needs_rerun = True   # triggers st.rerun() at the very end of the page

    else:
        # ── Overview (idle) ───────────────────────────────────────────────────
        if df is None or df.empty:
            st.info("No session data yet — select a source and click **Start Analysis**.")
        else:
            if os.path.exists(LIVE_PATH):
                _fa = time.time() - os.path.getmtime(LIVE_PATH)
                if _fa < LIVE_THRESHOLD_SEC:
                    st.success(f"System online — last frame {int(_fa)}s ago")
                else:
                    st.error(f"System offline — last frame {int(_fa)}s ago. Start analysis.")
            else:
                st.info("No live frame yet — start analysis.")

            _lf, _ls = st.columns([2, 1])
            with _lf:
                st.subheader("Latest Frame")
                if os.path.exists(LIVE_PATH):
                    try:
                        _mt = os.path.getmtime(LIVE_PATH)
                        st.image(Image.open(LIVE_PATH),
                                 caption=f"Captured {datetime.fromtimestamp(_mt).strftime('%H:%M:%S')}",
                                 use_container_width=True)
                    except Exception as _ex:
                        st.warning(f"Cannot load frame: {_ex}")
                else:
                    st.info("Frame appears here after first analysis run.")

            with _ls:
                st.subheader("Current Status")
                st.metric("People In Scene", current_occ)
                _c1, _c2 = st.columns(2)
                _c1.metric("Entered", total_in)
                _c2.metric("Exited",  total_out)
                if not alert_rows.empty:
                    _last = alert_rows.iloc[-1]
                    st.markdown("**Most recent alert**")
                    st.error(f"{_last['alert_str']}  |  {_last['time'].strftime('%H:%M:%S')}")
                else:
                    st.success("No alerts recorded")
                st.markdown("**Recent events**")
                _recent = alert_rows.sort_values("time", ascending=False).head(10)
                if not _recent.empty:
                    for _, _row in _recent.iterrows():
                        st.write(f"`{_row['time'].strftime('%H:%M:%S')}`  "
                                 f"{_row['alert_str']}  ({sev_of(_row['alert_str'])})")
                else:
                    st.caption("No events in the selected range.")

# _charts_on: False while analysis runs — skips all matplotlib creation so the
# live-analysis rerun loop stays fast.  Full charts visible once stopped.
_charts_on = not st.session_state.an_active

# ══ TAB 2 — PEOPLE FLOW ════════════════════════════════════════════════════════
with tab_flow:
    if occ_filt.empty:
        if st.session_state.an_active:
            st.caption("Collecting data — charts appear within seconds...")
        else:
            st.info("No data — run an analysis first.")
    else:
        _span_s = max(1, (_end_ts - _start_ts).total_seconds())
        _freq, _flabel = (
            ("1h",   "per hour")    if _span_s > 86400 else
            ("5min", "per 5 min")   if _span_s > 7200  else
            ("1min", "per minute")  if _span_s > 1200  else
            ("15s",  "per 15 sec")  if _span_s > 180   else
            ("5s",   "per 5 sec")
        )
        for _cam in _sel_cameras:
            _cdf = occ_filt[occ_filt["camera_id"] == _cam].set_index("time").sort_index()
            if _cdf.empty:
                continue
            st.subheader(_cam)
            _k1, _k2, _k3, _k4 = st.columns(4)
            _ent = int(max(0, _cdf["in_count"].max()  - _cdf["in_count"].min()))
            _ext = int(max(0, _cdf["out_count"].max() - _cdf["out_count"].min()))
            _k1.metric("Entered",        _ent)
            _k2.metric("Exited",         _ext)
            _k3.metric("Peak Occupancy", int(_cdf["visible_count"].max()))
            _k4.metric("Avg Occupancy",  round(_cdf["visible_count"].mean(), 1))

            st.markdown(f"**Occupancy — peak {_flabel}**")
            _op = _cdf["visible_count"].resample(_freq).max().dropna()
            _oa = _cdf["visible_count"].resample(_freq).mean().dropna()
            if len(_op) >= 2:
                st.line_chart(pd.DataFrame({"Peak": _op, "Average": _oa.round(1)}),
                              color=["#e63946", "#00b4d8"])
            else:
                st.line_chart(_cdf["visible_count"].rename("Occupancy"), color="#00b4d8")

            _ins = (_cdf["in_count"] - _cdf["out_count"]).clip(lower=0)
            _ir  = _ins.resample(_freq).max().dropna()
            st.markdown("**People Inside (entered - exited)**")
            if len(_ir) >= 2:
                st.area_chart(_ir.rename("Inside"), color="#2a9d8f")
            else:
                st.line_chart(_ins.rename("Inside"), color="#2a9d8f")

            _cum  = _cdf[["in_count","out_count"]].resample(_freq).max().ffill()
            _rate = _cum.diff().clip(lower=0).dropna(how="all")
            st.markdown(f"**Footfall Rate ({_flabel})**")
            if not _rate.empty and _rate.to_numpy().sum() > 0:
                st.bar_chart(_rate.rename(columns={"in_count":"Entered","out_count":"Exited"}),
                             color=["#2a9d8f","#f4a261"])
            else:
                st.caption("Not enough movement yet to chart footfall rate.")

            if _charts_on:
                st.markdown("**Busiest Hours**")
                _hourly = _cdf.assign(h=_cdf.index.hour).groupby("h")["visible_count"].mean()
                if _hourly.sum() > 0:
                    _ph = int(_hourly.idxmax())
                    _fig, _ax = plt.subplots(figsize=(max(5, len(_hourly)*0.6+1), 2.8))
                    _ax.bar(range(len(_hourly)), _hourly.values,
                            color=["#e63946" if h == _ph else "#457b9d" for h in _hourly.index],
                            width=0.65, edgecolor="none")
                    _ax.set_xticks(range(len(_hourly)))
                    _ax.set_xticklabels([f"{h:02d}:00" for h in _hourly.index],
                                        rotation=45, ha="right", fontsize=8)
                    _ax.set_ylabel("Avg People")
                    _ax.set_title(f"Peak hour {_ph:02d}:00", fontsize=9)
                    _ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)
            st.divider()

# ══ TAB 3 — INSIGHTS ═══════════════════════════════════════════════════════════
with tab_insights:
    if occ_filt.empty:
        if st.session_state.an_active:
            st.caption("Collecting data...")
        else:
            st.info("No data.")
    else:
        st.subheader("Business Intelligence")
        _all_parts = [p for a in occ_filt["alert_str"] for p in a.split() if p]
        _span_h    = max(0.01, (_end_ts - _start_ts).total_seconds() / 3600)

        _i1, _i2, _i3, _i4 = st.columns(4)
        _i1.metric("Visitors / Hour", round(total_in / _span_h, 1))
        _i2.metric("Alerts / Hour",   round(len(alert_rows) / _span_h, 2))
        _i3.metric("Peak Occupancy",  peak_occ)
        _i4.metric("Avg Occupancy",   round(occ_filt["visible_count"].mean(), 1))

        _h_occ  = occ_filt.groupby("hour")["visible_count"].mean()
        _active_h = _h_occ[_h_occ > 0]
        if not _active_h.empty:
            st.markdown("---")
            _ca, _cb = st.columns(2)
            _ca.success(f"**Busiest hour:** {int(_h_occ.idxmax()):02d}:00 — {_h_occ.max():.1f} avg")
            _cb.info(   f"**Quietest open hour:** {int(_active_h.idxmin()):02d}:00 — {_active_h.min():.1f} avg")

        st.markdown("---")
        st.markdown("#### Alert Severity Mix")
        _sc = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for p in _all_parts:
            _sc[SEVERITY.get(p, "LOW")] += 1
        _sa, _sb, _scc = st.columns(3)
        _sa.metric("High", _sc["HIGH"]); _sb.metric("Medium", _sc["MEDIUM"]); _scc.metric("Low", _sc["LOW"])

        if _charts_on:
            if _all_parts:
                st.markdown("---")
                st.markdown("#### Alert Activity by Hour")
                _ah = alert_rows.groupby("hour").size()
                if not _ah.empty:
                    _fig, _ax = plt.subplots(figsize=(max(5, len(_ah)*0.6+1), 2.5))
                    _ax.bar(range(len(_ah)), _ah.values, color="#e63946",
                            width=0.6, edgecolor="none", alpha=0.85)
                    _ax.set_xticks(range(len(_ah)))
                    _ax.set_xticklabels([f"{h:02d}:00" for h in _ah.index],
                                        rotation=45, ha="right", fontsize=8)
                    _ax.set_ylabel("Alerts"); _ax.set_title("When do incidents happen?", fontsize=9)
                    _ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)

            st.markdown("---")
            st.markdown("#### Posture Mix")
            _pos = (occ_filt.loc[occ_filt["visible_count"] > 0, "posture"]
                    .fillna("Unknown").replace("", "Unknown").value_counts())
            if not _pos.empty:
                _pc1, _pc2 = st.columns([1.3, 1])
                with _pc1:
                    _fig, _ax = plt.subplots(figsize=(4.5, 2.6))
                    _pcol = {"Lying":"#e63946","Sitting":"#f4a261","Walking":"#2a9d8f",
                             "Standing":"#457b9d","Unknown":"#adb5bd"}
                    _ax.barh(_pos.index[::-1], _pos.values[::-1],
                             color=[_pcol.get(p,"#adb5bd") for p in _pos.index[::-1]], edgecolor="none")
                    _ax.set_xlabel("Frames"); _ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)
                with _pc2:
                    _lying = int(_pos.get("Lying", 0))
                    if _lying:
                        st.error(f"**{_lying}** lying-posture frames — possible falls.")
                    else:
                        st.success("No lying postures detected.")

            st.markdown("---")
            st.markdown("#### Occupancy Anomalies")
            _os = occ_filt.set_index("time")["visible_count"].resample("5min").max().dropna()
            if len(_os) >= 6 and _os.std() > 0:
                _thr = _os.mean() + 2 * _os.std()
                _spk = _os[_os > _thr]
                if not _spk.empty:
                    st.warning(f"{len(_spk)} interval(s) above normal (> {_thr:.1f} people).")
                    st.dataframe(
                        _spk.rename("Peak People").reset_index().assign(
                            time=lambda d: d["time"].dt.strftime("%m/%d %H:%M")
                        ).rename(columns={"time": "Interval (5 min)"}),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.success("Occupancy within normal range.")
            else:
                st.caption("Need a longer time span to detect anomalies.")

            st.markdown("---")
            _daily = occ_filt.groupby("date").apply(
                lambda g: pd.Series({
                    "visitors": window_entries(g, "in_count"),
                    "peak_occ": int(g["visible_count"].max()),
                    "avg_occ":  round(g["visible_count"].mean(), 1),
                    "alerts":   int(g["alert_str"].ne("").sum()),
                }), include_groups=False,
            ).reset_index()
            if len(_daily) > 1:
                st.markdown("#### Daily Trend")
                st.dataframe(_daily.rename(columns={
                    "date":"Date","visitors":"Entered","peak_occ":"Peak Occ",
                    "avg_occ":"Avg Occ","alerts":"Alerts",
                }), use_container_width=True)

# ══ TAB 4 — ALERTS ═════════════════════════════════════════════════════════════
with tab_alerts:
    _ap = [p for a in occ_filt.get("alert_str", pd.Series(dtype=str)) for p in a.split() if p] \
        if not occ_filt.empty else []
    if not _ap:
        if st.session_state.an_active:
            st.caption("No alerts yet in this session.")
        else:
            st.info("No alerts in selected range.")
    else:
        if _charts_on:
            _fc, _tc = st.columns([1, 1.4])
            with _fc:
                st.subheader("Alert Frequency")
                _fs = pd.Series(_ap).value_counts()
                _fs = _fs.reindex(
                    [e for e in ALERT_TYPES if e in _fs.index] +
                    [e for e in _fs.index if e not in ALERT_TYPES], fill_value=0)
                _fig, _ax = plt.subplots(figsize=(4.5, 3.5))
                _bc = [SEV_COLOR.get(SEVERITY.get(k,"LOW"),"#adb5bd") for k in _fs.index[::-1]]
                _ax.barh(_fs.index[::-1], _fs.values[::-1], color=_bc, edgecolor="none")
                _ax.set_xlabel("Count"); _ax.spines[["top","right"]].set_visible(False)
                _ax.legend(handles=[mpatches.Patch(color="#e63946",label="High"),
                                    mpatches.Patch(color="#f4a261",label="Medium")],
                           fontsize=7, loc="lower right")
                plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)
            with _tc:
                st.subheader("Alert Timeline")
                _evs = [{"time": row["time"], "type": p}
                        for _, row in alert_rows.iterrows()
                        for p in row["alert_str"].split() if p]
                if _evs:
                    _edf = pd.DataFrame(_evs)
                    _tp  = [t for t in ALERT_TYPES if t in _edf["type"].unique()]
                    _yp  = {t: i for i, t in enumerate(_tp)}
                    _fig, _ax = plt.subplots(figsize=(7, max(2.2, 0.42*len(_tp)+1)))
                    for t in _tp:
                        _sub = _edf[_edf["type"] == t]
                        _ax.scatter(_sub["time"], [_yp[t]]*len(_sub),
                                    c=SEV_COLOR.get(SEVERITY.get(t,"LOW"),"#adb5bd"),
                                    s=42, alpha=0.75, edgecolors="none")
                    _ax.set_yticks(range(len(_tp))); _ax.set_yticklabels(_tp, fontsize=8)
                    _ax.set_ylim(-0.5, len(_tp)-0.5); _ax.set_xlabel("Time")
                    _ax.grid(axis="x", linestyle=":", alpha=0.4)
                    _ax.spines[["top","right"]].set_visible(False)
                    _fig.autofmt_xdate(); plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)
                    st.caption("Each dot = one alert. Colour = severity (red = high).")

        st.subheader("Alert Log" + (f" — {', '.join(_sel_events)}" if _sel_events else ""))
        if not alert_log.empty:
            _ld = (alert_log[["time","camera_id","alert_str","visible_count"]]
                   .sort_values("time", ascending=False)
                   .rename(columns={"alert_str":"alert","visible_count":"people_in_frame"})
                   .head(300)).copy()
            _ld.insert(0, "Severity", _ld["alert"].apply(sev_of))
            st.dataframe(_ld, use_container_width=True)
            if _charts_on:
                _buf = io.StringIO(); _ld.to_csv(_buf, index=False)
                st.download_button("Download Alert Log (CSV)", _buf.getvalue(),
                                   f"alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        else:
            st.info("No alerts match the current filter.")

# ══ TAB 5 — SNAPSHOTS ══════════════════════════════════════════════════════════
with tab_snaps:
    st.subheader("Alert Snapshots")
    _snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "*.jpg")), key=os.path.getmtime, reverse=True)
    if not _snaps:
        st.info("No snapshots yet — saved automatically when alerts fire.")
    elif _charts_on:
        _et  = sorted({os.path.basename(f).split("_")[0] for f in _snaps})
        _sev = st.multiselect("Filter by event", _et, default=_et)
        _shown = [f for f in _snaps if os.path.basename(f).split("_")[0] in _sev][:24]
        _cols = st.columns(4)
        for _i, _sp in enumerate(_shown):
            _fn = os.path.basename(_sp); _etype = _fn.split("_")[0]
            _tp = _fn.replace(f"{_etype}_","").replace(".jpg","")
            try:
                _td = datetime.strptime(_tp, "%Y%m%d_%H%M%S").strftime("%m/%d %H:%M:%S")
            except Exception:
                _td = _tp
            with _cols[_i % 4]:
                try:
                    st.image(Image.open(_sp),
                             caption=f"{_etype.upper()} · {SEVERITY.get(_etype.capitalize(),'LOW')} · {_td}",
                             use_container_width=True)
                except Exception:
                    st.warning(f"Cannot load {_fn}")
    else:
        st.caption(f"{len(_snaps)} snapshot(s) saved. Stop analysis to browse.")

# ══ TAB 6 — HEATMAP ════════════════════════════════════════════════════════════
with tab_heat:
    st.subheader("Motion Heatmap & Zone Activity")
    _ic, _zc = st.columns([1.4, 1])
    with _ic:
        if os.path.exists(HEATMAP):
            _mt = os.path.getmtime(HEATMAP)
            st.image(Image.open(HEATMAP),
                     caption=f"Generated {datetime.fromtimestamp(_mt).strftime('%Y-%m-%d %H:%M:%S')}",
                     use_container_width=True)
            st.caption("Warm colours = where people spend the most time.")
        else:
            st.info("Heatmap generated automatically during analysis.")
    with _zc:
        st.markdown("**Activity by Zone**")
        if os.path.exists(ZONE_CSV) and _charts_on:
            _zdf = pd.read_csv(ZONE_CSV).sort_values("percent", ascending=False)
            if not _zdf.empty and _zdf["percent"].sum() > 0:
                _top = _zdf.iloc[0]
                st.success(f"Busiest: **{_top['zone']}** ({_top['percent']:.0f}%)")
                _fig, _ax = plt.subplots(figsize=(4.5, 3))
                _ax.barh(_zdf["zone"][::-1], _zdf["percent"][::-1], color="#e63946", edgecolor="none")
                _ax.set_xlabel("% of total activity"); _ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(_fig); plt.close(_fig)
            else:
                st.caption("No zone activity yet.")
        else:
            st.caption("Zone breakdown appears after analysis runs.")

        _zd = load_zone_cfg().get(CAM_NAME, [])
        if _zd:
            st.markdown(f"**Configured zones for {CAM_NAME}**")
            for _z in _zd:
                _acts = ", ".join(_z.get("monitored_activities", [_z.get("type","?")]))
                st.write(f"- **{_z['id']}** | {_acts}")

# ── Raw logs ───────────────────────────────────────────────────────────────────
with st.expander("Raw Log Data"):
    if not occ_filt.empty:
        _raw = (occ_filt.drop(columns=["alert_str","hour","date"], errors="ignore")
                .sort_values("time", ascending=False).head(500))
        st.dataframe(_raw, use_container_width=True)
        _b2 = io.StringIO(); _raw.to_csv(_b2, index=False)
        st.download_button("Download Raw Logs (CSV)", _b2.getvalue(),
                           f"raw_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.info("No data yet.")

# ══════════════════════════════════════════════════════════════════════════════
# Trigger rerun for live analysis — MUST be at the very end so that every tab
# above is fully rendered before the page refreshes.
# ══════════════════════════════════════════════════════════════════════════════
if _needs_rerun:
    time.sleep(0.05)
    st.rerun()

