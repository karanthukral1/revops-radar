"""Run the pipeline: generate -> detect -> score -> explain the biggest revenue drops."""
from revops_radar.data_gen import generate
from revops_radar.decompose import DRIVERS, explain_segment
from revops_radar.detect import detect_anomalies, score_detection, to_episodes


def main():
    df, events = generate()
    flags = detect_anomalies(df)
    print("Detection vs injected ground truth:", score_detection(flags, events))

    results = []
    for _, ep in to_episodes(flags).iterrows():
        r = explain_segment(df, ep.publisher, ep.platform, ep.partner, ep.start, ep.end)
        r["start"], r["end"] = ep.start.date(), ep.end.date()
        results.append(r)
    results = [r for r in results if r["change_pct"] < -10]  # material drops only
    results.sort(key=lambda r: r["change_pct"])

    print(f"\n{len(results)} material revenue drops. Worst 5:")
    for r in results[:5]:
        print(
            f"- {r['segment']} ({r['start']} to {r['end']}): revenue {r['change_pct']:.0f}% vs baseline, "
            f"mainly {DRIVERS[r['primary_driver']]}"
        )


if __name__ == "__main__":
    main()
