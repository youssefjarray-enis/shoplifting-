#!/usr/bin/env python3
"""Rapport de metriques sur les scenarios synthetiques (precision, rappel, F1, FP/h, latence)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visshop.evaluation import evaluate_all  # noqa: E402
from visshop.synthetic import SCENARIO_NAMES  # noqa: E402

TARGETS = dict(precision=0.92, recall=0.87, f1=0.89, false_positives_per_hour=2.0, mean_alert_latency_s=2.5)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "output/eval"
    os.makedirs(out, exist_ok=True)
    rep = evaluate_all(SCENARIO_NAMES, out)
    print(f"{'scenario':26s}{'evts':>5s}{'attendus':>9s}  suppressions")
    for k, v in rep["scenarios"].items():
        print(f"{k:26s}{v['events']:>5d}{v['expected']:>9d}  {v['suppressed']}")
    g = rep["global_metrics"]
    print("\nMetriques globales (cible)")
    print(f"  Precision  {g['precision']:.3f}  (>= {TARGETS['precision']})")
    print(f"  Rappel     {g['recall']:.3f}  (>= {TARGETS['recall']})")
    print(f"  F1         {g['f1']:.3f}  (>= {TARGETS['f1']})")
    print(f"  FP / heure {g['false_positives_per_hour']:.2f}  (< {TARGETS['false_positives_per_hour']})")
    print(f"  Latence    {g['mean_alert_latency_s']:.2f}s (< {TARGETS['mean_alert_latency_s']}s)")
    with open(os.path.join(out, "metrics_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapport: {os.path.join(out, 'metrics_report.json')}")
