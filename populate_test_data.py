"""Generate realistic business-day surveillance test data from 08:00 today."""
import sqlite3, random, math
from datetime import datetime, timedelta

DB_PATH = "logs/analytics.db"
random.seed(42)

conn = sqlite3.connect(DB_PATH)
c    = conn.cursor()
c.execute("DROP TABLE IF EXISTS traffic_logs")
c.execute("""
    CREATE TABLE traffic_logs (
        time TEXT, camera_id TEXT, in_count INTEGER, out_count INTEGER,
        visible_count INTEGER, posture TEXT, alert TEXT
    )
""")
conn.commit()

POSTURES = ["Standing"]*7 + ["Walking"]*4 + ["Sitting"] + ["Unknown"]


def occupancy_at(t: datetime) -> int:
    h = t.hour + t.minute / 60.0
    # Three peaks: morning (10), lunch (12:30), evening rush (17:30)
    morning = 4 * math.exp(-0.5 * ((h - 10.0) / 1.2) ** 2)
    lunch   = 9 * math.exp(-0.5 * ((h - 12.5) / 1.0) ** 2)
    evening = 11 * math.exp(-0.5 * ((h - 17.5) / 1.5) ** 2)
    base    = max(0, morning + lunch + evening + random.gauss(0, 0.6))
    # Taper off outside business hours
    if h < 8 or h > 21:
        base *= 0.15
    return max(0, min(20, int(round(base))))


def pick_alerts(count: int, elapsed_min: float, t: datetime) -> list:
    alerts = []
    hour   = t.hour
    if count > 8:
        alerts.append("Crowd")
    if 13 <= hour <= 16 and random.random() < 0.03:
        alerts.append("Loitering")
    if random.random() < 0.004:
        alerts.append("Running")
    if 180 <= elapsed_min <= 200 and random.random() < 0.12:
        alerts.append("Fall")
    if 240 <= elapsed_min <= 260 and random.random() < 0.07:
        alerts.append("AbandonedObject")
    if random.random() < 0.002:
        alerts.append("Intrusion")
    if random.random() < 0.005:
        alerts.append("Tailgating")
    if (hour < 8 or hour >= 22) and count > 0 and random.random() < 0.6:
        alerts.append("AfterHours")
    return alerts


# Generate data for the past 10 hours (covers a full business day regardless of timezone)
now_snap  = datetime.now()
today     = now_snap - timedelta(hours=10)
rows      = []
t         = today
in_count  = 0
out_count = 0
prev_occ  = 0

while t <= now_snap:
    elapsed = (t - today).total_seconds() / 60
    occ     = occupancy_at(t)
    posture = random.choice(POSTURES)

    if 180 <= elapsed <= 200 and random.random() < 0.10:
        posture = "Lying"

    delta = occ - prev_occ
    if delta > 0:
        in_count  += delta
    elif delta < 0:
        out_count += abs(delta)
    prev_occ = occ

    alerts = pick_alerts(occ, elapsed, t)

    rows.append((
        t.strftime("%Y-%m-%d %H:%M:%S"),
        "Main CCTV", in_count, out_count, occ,
        posture, " ".join(alerts),
    ))
    t += timedelta(seconds=5)

c.executemany("INSERT INTO traffic_logs VALUES (?,?,?,?,?,?,?)", rows)
conn.commit()
conn.close()

from collections import Counter
parts = [p for row in rows for p in row[6].split() if p]
print(f"✅  {len(rows)} records | 08:00 → now | camera: Main CCTV")
print(f"    IN: {rows[-1][2]}   OUT: {rows[-1][3]}   Peak: {max(r[4] for r in rows)}")
print(f"    Events: {dict(Counter(parts).most_common())}")
print("    Run: streamlit run dashboard_streamlit.py")
