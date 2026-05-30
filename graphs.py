"""Standalone PNG export of the core surveillance analytics.

Usage:  python3 graphs.py            # reads logs/analytics.db → logs/analytics_graphs.png

in_count / out_count are CUMULATIVE session counters, so per-interval footfall is
their difference and live occupancy is (in_count - out_count) — not the raw counters.
"""
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "logs/analytics.db"
OUT_PNG = "logs/analytics_graphs.png"


def load(db_path: str = DB_PATH) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM traffic_logs ORDER BY time", sqlite3.connect(db_path))
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).set_index("time").sort_index()


def plot_analytics(df: pd.DataFrame, out_png: str = OUT_PNG):
    if df.empty:
        print("No data to plot.")
        return

    inside = (df["in_count"] - df["out_count"]).clip(lower=0).resample("5min").max().dropna()
    cum    = df[["in_count", "out_count"]].resample("5min").max().ffill()
    rate   = cum.diff().clip(lower=0).dropna(how="all")
    occ    = df["visible_count"].resample("5min").max().dropna()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].fill_between(inside.index, inside.values, color="#2a9d8f", alpha=0.6)
    axes[0].set_ylabel("People inside")
    axes[0].set_title("Live headcount (entered − exited)")

    axes[1].bar(rate.index, rate["in_count"], width=0.002, color="#2a9d8f", label="Entered")
    axes[1].bar(rate.index, -rate["out_count"], width=0.002, color="#f4a261", label="Exited")
    axes[1].set_ylabel("Per 5 min")
    axes[1].set_title("Footfall rate")
    axes[1].legend()

    axes[2].plot(occ.index, occ.values, color="#e63946")
    axes[2].set_ylabel("Peak in scene")
    axes[2].set_title("Occupancy over time")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot_analytics(load())
