# RevOps Radar

Anomaly detection and root-cause analysis for ad revenue, built in Python.

Revenue teams in AdTech, CTV and publishing see daily revenue move for many reasons: less traffic, lower fill, or lower prices. RevOps Radar finds days when revenue drops abnormally and explains which driver caused the drop.

## What it does

1. **Generates realistic data.** Daily revenue for 4 publishers × 3 platforms (CTV, Web, App) × 3 demand partners over 365 days, with weekly seasonality, growth and noise. It injects 20 known anomalies (eCPM drops, fill-rate drops, traffic drops), so the detector can be scored against ground truth.
2. **Detects anomalies.** Per segment, it removes the weekday effect, then compares each day with the median and MAD of the previous 28 days (current day excluded). Days far below baseline are flagged.
3. **Forecasts.** A Holt-Winters model (additive trend, weekly seasonality) gives a 30-day forecast with a confidence band.
4. **Explains the cause.** Revenue = ad requests × fill rate × price per impression, so the log change in revenue is the sum of the log changes in the three drivers. The tool splits each drop into traffic, fill-rate and eCPM effects and names the main driver.

## Results (seed 42)

| Metric | Value |
| --- | --- |
| Injected anomalies | 20 |
| Anomalies detected (recall) | 20 / 20 (100%) |
| Flagged days that fall in a true anomaly window (precision) | 76% |

On the largest drops, the primary driver the tool names matches the injected cause.

Example output:

```
Detection vs injected ground truth: {'events': 20, 'events_detected': 20, 'recall': 1.0, 'flagged_days': 123, 'precision': 0.756...}

- DailyCart / App / AdX Prime (2026-08-11 to 2026-08-14): revenue -55% vs baseline, mainly eCPM (price)
- GameZone / CTV / BidFlow (2026-02-08 to 2026-02-12): revenue -52% vs baseline, mainly traffic (ad requests)
```

## Project structure

```
revops_radar/
  data_gen.py    synthetic data and injected anomalies
  detect.py      anomaly detection, forecast band, scoring, episode grouping
  decompose.py   traffic / fill / price root-cause split
main.py          end-to-end run
tests/           pytest tests
```

## Run it

```
pip install -r requirements.txt
python main.py
python -m pytest
```

## Tech

Python, pandas, NumPy, statsmodels, pytest.

## Roadmap

- LLM layer (Gemini API) that turns structured findings into plain-English alerts
- Streamlit dashboard
- Threshold tuning to raise precision
