"""Step 1: synthetic ad-revenue data with injected anomalies (known ground truth)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

PUBLISHERS = ["NewsHub", "StreamBox", "GameZone", "DailyCart"]
PLATFORMS = ["CTV", "Web", "App"]
PARTNERS = ["AdX Prime", "BidFlow", "Nova SSP"]
ANOMALY_TYPES = ["ecpm_drop", "fill_drop", "traffic_drop"]

# Weekday multipliers (Mon..Sun): traffic peaks on weekends for CTV-heavy inventory.
WEEKLY = np.array([0.95, 0.95, 0.97, 1.0, 1.05, 1.12, 1.10])


def generate(days: int = 365, n_anomalies: int = 20, seed: int = 42):
    """Return (daily_df, events_df).

    daily_df: one row per date x publisher x platform x partner with
        ad_requests, fill_rate, ecpm, impressions, revenue.
    events_df: ground-truth injected anomalies (segment, type, start, end, magnitude).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-11-01", periods=days, freq="D")
    rows = []
    segments = list(itertools.product(PUBLISHERS, PLATFORMS, PARTNERS))

    base = {}
    for seg in segments:
        base[seg] = {
            "requests": rng.uniform(80_000, 400_000),
            "fill": rng.uniform(0.55, 0.90),
            "ecpm": rng.uniform(2.0, 14.0),  # revenue per 1000 impressions
        }

    # Choose anomaly events (avoid the first 45 days so detectors have history).
    events = []
    for _ in range(n_anomalies):
        seg = segments[rng.integers(len(segments))]
        start = int(rng.integers(45, days - 10))
        length = int(rng.integers(3, 8))
        events.append(
            {
                "publisher": seg[0],
                "platform": seg[1],
                "partner": seg[2],
                "type": ANOMALY_TYPES[rng.integers(len(ANOMALY_TYPES))],
                "start": dates[start],
                "end": dates[min(start + length - 1, days - 1)],
                "magnitude": float(rng.uniform(0.30, 0.55)),  # fractional drop
            }
        )
    events_df = pd.DataFrame(events)

    for seg in segments:
        b = base[seg]
        pub, plat, par = seg
        trend = np.linspace(1.0, 1.15, days)  # gentle growth
        for i, d in enumerate(dates):
            req = b["requests"] * trend[i] * WEEKLY[d.dayofweek] * rng.normal(1, 0.04)
            fill = float(np.clip(b["fill"] * rng.normal(1, 0.02), 0.05, 0.98))
            ecpm = b["ecpm"] * rng.normal(1, 0.04)
            rows.append([d, pub, plat, par, req, fill, ecpm])

    df = pd.DataFrame(
        rows,
        columns=["date", "publisher", "platform", "partner", "ad_requests", "fill_rate", "ecpm"],
    )

    # Apply injected anomalies.
    for ev in events:
        m = (
            (df.publisher == ev["publisher"])
            & (df.platform == ev["platform"])
            & (df.partner == ev["partner"])
            & (df.date >= ev["start"])
            & (df.date <= ev["end"])
        )
        col = {"ecpm_drop": "ecpm", "fill_drop": "fill_rate", "traffic_drop": "ad_requests"}[ev["type"]]
        df.loc[m, col] *= 1 - ev["magnitude"]

    df["impressions"] = df.ad_requests * df.fill_rate
    df["revenue"] = df.impressions * df.ecpm / 1000
    return df, events_df
