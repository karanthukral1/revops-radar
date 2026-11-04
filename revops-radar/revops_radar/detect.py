"""Step 2: anomaly detection (robust z-score) and Holt-Winters forecast band."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEGMENT = ["publisher", "platform", "partner"]


def detect_anomalies(df: pd.DataFrame, window: int = 28, z_thresh: float = -3.5) -> pd.DataFrame:
    """Flag days where revenue falls far below its recent weekday-adjusted baseline.

    Method (per segment): divide log revenue by its weekday effect, then compare each
    day to the median/MAD of the previous `window` days (current day excluded, so a
    long outage cannot hide itself). Drops only: revenue spikes are not alerts here.
    """
    out = []
    for keys, g in df.sort_values("date").groupby(SEGMENT):
        g = g.copy()
        y = np.log(g["revenue"].to_numpy())
        s = pd.Series(y, index=g.index)
        weekday_effect = s.groupby(g["date"].dt.dayofweek).transform("median") - s.median()
        adj = s - weekday_effect
        base = adj.shift(1).rolling(window, min_periods=window // 2)
        med = base.median()
        mad = base.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        g["z"] = (adj - med) / (1.4826 * mad.clip(lower=0.01))
        g["is_anomaly"] = g["z"] < z_thresh
        out.append(g)
    return pd.concat(out).sort_index()


def forecast_band(series: pd.Series, horizon: int = 30, z: float = 1.96) -> pd.DataFrame:
    """Holt-Winters (additive trend, weekly seasonality) forecast with a +/- z*sigma band."""
    series = series.asfreq("D")
    model = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated"
    ).fit()
    resid_sd = float(np.std(model.resid, ddof=1))
    fc = model.forecast(horizon)
    return pd.DataFrame({"forecast": fc, "lower": fc - z * resid_sd, "upper": fc + z * resid_sd})


def score_detection(flags: pd.DataFrame, events: pd.DataFrame, buffer_days: int = 2) -> dict:
    """Event-level recall and day-level precision against injected ground truth."""
    flagged = flags[flags.is_anomaly]
    detected = 0
    for _, ev in events.iterrows():
        m = (
            (flagged.publisher == ev.publisher)
            & (flagged.platform == ev.platform)
            & (flagged.partner == ev.partner)
            & (flagged.date >= ev.start)
            & (flagged.date <= ev.end + pd.Timedelta(days=buffer_days))
        )
        detected += bool(m.any())

    true_flags = 0
    for _, row in flagged.iterrows():
        e = events[
            (events.publisher == row.publisher)
            & (events.platform == row.platform)
            & (events.partner == row.partner)
            & (events.start <= row.date)
            & (events.end + pd.Timedelta(days=buffer_days) >= row.date)
        ]
        true_flags += len(e) > 0

    return {
        "events": len(events),
        "events_detected": detected,
        "recall": detected / len(events) if len(events) else float("nan"),
        "flagged_days": len(flagged),
        "precision": true_flags / len(flagged) if len(flagged) else float("nan"),
    }


def to_episodes(flags: pd.DataFrame, max_gap_days: int = 2) -> pd.DataFrame:
    """Merge consecutive flagged days in a segment into one episode (start, end)."""
    rows = []
    flagged = flags[flags.is_anomaly].sort_values("date")
    for keys, g in flagged.groupby(SEGMENT):
        start = prev = None
        for d in g.date:
            if start is None:
                start = prev = d
            elif (d - prev).days <= max_gap_days:
                prev = d
            else:
                rows.append((*keys, start, prev))
                start = prev = d
        if start is not None:
            rows.append((*keys, start, prev))
    return pd.DataFrame(rows, columns=SEGMENT + ["start", "end"])
