"""Step 3: root-cause decomposition of a revenue change.

Revenue = ad_requests x fill_rate x price_per_impression, so
ln(R1/R0) = ln(Q1/Q0) + ln(fill1/fill0) + ln(price1/price0)
The three terms add up exactly, which makes the split easy to explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIVERS = {"ad_requests": "traffic (ad requests)", "fill_rate": "fill rate", "price": "eCPM (price)"}


def _agg(g: pd.DataFrame) -> dict:
    req = g.ad_requests.sum()
    imps = g.impressions.sum()
    rev = g.revenue.sum()
    return {"ad_requests": req, "fill_rate": imps / req, "price": rev / imps, "revenue": rev}


def decompose(baseline: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Split the change in revenue between two windows into traffic / fill / price effects."""
    a, b = _agg(baseline), _agg(current)
    logs = {k: float(np.log(b[k] / a[k])) for k in DRIVERS}
    total_log = sum(logs.values())
    delta = b["revenue"] - a["revenue"]
    # Distribute the dollar change in proportion to each driver's log contribution.
    effects = {k: (delta * logs[k] / total_log if total_log else 0.0) for k in DRIVERS}
    main = max(effects, key=lambda k: abs(effects[k]))
    return {
        "baseline_revenue": a["revenue"],
        "current_revenue": b["revenue"],
        "change_pct": (b["revenue"] / a["revenue"] - 1) * 100,
        "effects": effects,
        "primary_driver": main,
        "primary_share_pct": (effects[main] / delta * 100) if delta else 0.0,
    }


def explain_segment(df, publisher, platform, partner, start, end, baseline_days: int = 28) -> dict:
    """Compare an anomaly window against the preceding `baseline_days` for one segment."""
    seg = df[(df.publisher == publisher) & (df.platform == platform) & (df.partner == partner)]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cur = seg[(seg.date >= start) & (seg.date <= end)]
    n = len(cur)
    base = seg[(seg.date < start) & (seg.date >= start - pd.Timedelta(days=baseline_days))]
    # Scale baseline to the same number of days so revenue totals are comparable.
    out = decompose(base.assign(**{c: base[c] * n / len(base) for c in ["ad_requests", "impressions", "revenue"]}), cur)
    out["segment"] = f"{publisher} / {platform} / {partner}"
    return out
