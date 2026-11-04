import pandas as pd

from revops_radar.data_gen import generate
from revops_radar.decompose import decompose
from revops_radar.detect import detect_anomalies, forecast_band, score_detection


def test_generate_shapes_and_revenue_identity():
    df, ev = generate(days=120, n_anomalies=5)
    assert len(ev) == 5
    assert (df.revenue - df.impressions * df.ecpm / 1000).abs().max() < 1e-6


def test_decomposition_adds_up():
    df, _ = generate(days=60, n_anomalies=0)
    a, b = df.iloc[:100], df.iloc[100:200]
    r = decompose(a, b)
    assert abs(sum(r["effects"].values()) - (r["current_revenue"] - r["baseline_revenue"])) < 1e-6


def test_detector_finds_most_injected_events():
    df, ev = generate()
    s = score_detection(detect_anomalies(df), ev)
    assert s["recall"] >= 0.8


def test_forecast_band_ordered():
    df, _ = generate(days=120, n_anomalies=0)
    s = df.groupby("date").revenue.sum()
    fc = forecast_band(s, horizon=14)
    assert len(fc) == 14 and (fc.lower < fc.upper).all()
