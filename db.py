import sqlite3


def init_db(db_path="logs/analytics.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_logs (
            time          TEXT,
            camera_id     TEXT,
            in_count      INTEGER,
            out_count     INTEGER,
            visible_count INTEGER,
            posture       TEXT,
            alert         TEXT
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_time     ON traffic_logs(time)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_cam_time ON traffic_logs(camera_id, time)")
    conn.commit()
    return conn


def insert_log(conn, entry: dict):
    conn.execute(
        '''INSERT INTO traffic_logs
           (time, camera_id, in_count, out_count, visible_count, posture, alert)
           VALUES (?,?,?,?,?,?,?)''',
        (entry["time"], entry["camera_id"], entry["in_count"], entry["out_count"],
         entry["visible_count"], entry["posture"], entry["alert"])
    )
    conn.commit()
