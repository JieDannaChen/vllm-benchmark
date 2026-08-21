#!/usr/bin/env python3
"""
Regression Comparison Tool
===========================
Compares a new benchmark run against a saved baseline and flags
performance regressions that exceed configured thresholds.

Usage:
    python compare.py --baseline baseline.json --new results/run_latest.json
    python compare.py --baseline baseline.json --new results/run_latest.json --threshold 0.10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np


THRESHOLDS = {
    # Metric name : allowed degradation ratio (e.g. 0.10 = +10% is OK)
    "avg_ttft_s": 0.10,        # TTFT can degrade 10%
    "avg_tpot_ms": 0.10,       # TPOT can degrade 10%
    "avg_itl_ms": 0.10,        # ITL can degrade 10%
    "throughput_tok_s": -0.10,  # throughput should NOT drop more than 10% (negative = lower is worse)
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_metric(name: str, baseline_val: float, new_val: float,
                   threshold: float) -> Dict[str, Any]:
    """Compare a single metric. Returns status + delta info."""
    if baseline_val is None or new_val is None:
        return {"metric": name, "status": "skip", "reason": "missing value"}

    delta = new_val - baseline_val
    pct = delta / baseline_val if baseline_val != 0 else 0

    # For "higher is better" metrics (throughput), threshold is negative
    # For "lower is better" metrics (latency), threshold is positive
    if threshold < 0:
        # higher is better — regression means new < baseline * (1 + threshold)
        # e.g. threshold=-0.10 means drop of more than 10% is regression
        regression = pct < threshold
    else:
        # lower is better — regression means new > baseline * (1 + threshold)
        regression = pct > threshold

    direction = "↑" if delta > 0 else "↓" if delta < 0 else "="

    return {
        "metric": name,
        "baseline": baseline_val,
        "new": new_val,
        "delta": round(delta, 6),
        "delta_pct": round(pct * 100, 2),
        "direction": direction,
        "status": "regression" if regression else "ok",
    }


def compare_runs(baseline: dict, new_run: dict, custom_thresholds: Dict[str, float] = None):
    """Compare two full benchmark runs. Returns a structured report."""
    thresholds = {**THRESHOLDS, **(custom_thresholds or {})}

    base_summary = baseline["summary"]
    new_summary = new_run["summary"]

    # Overall-level comparisons
    overall_metrics = [
        ("throughput_tok_s", base_summary.get("throughput_tok_s"), new_summary.get("throughput_tok_s")),
    ]

    overall_results = []
    for name, bv, nv in overall_metrics:
        if name in thresholds:
            overall_results.append(compare_metric(name, bv, nv, thresholds[name]))

    # Per-prompt comparisons
    per_prompt_results = {}
    base_per_prompt = base_summary.get("per_prompt", {})
    new_per_prompt = new_summary.get("per_prompt", {})

    all_prompt_ids = set(base_per_prompt.keys()) | set(new_per_prompt.keys())
    for pid in sorted(all_prompt_ids):
        base_p = base_per_prompt.get(pid)
        new_p = new_per_prompt.get(pid)
        if not base_p or not new_p:
            per_prompt_results[pid] = {
                "name": (base_p or new_p).get("name", pid),
                "status": "skip",
                "reason": "prompt only in one run",
            }
            continue

        metrics = [
            ("avg_ttft_s", base_p.get("avg_ttft_s"), new_p.get("avg_ttft_s")),
            ("avg_tpot_ms", base_p.get("avg_tpot_ms"), new_p.get("avg_tpot_ms")),
            ("avg_itl_ms", base_p.get("avg_itl_ms"), new_p.get("avg_itl_ms")),
        ]
        prompt_results = []
        for name, bv, nv in metrics:
            if name in thresholds:
                prompt_results.append(compare_metric(name, bv, nv, thresholds[name]))

        regressions = [m for m in prompt_results if m["status"] == "regression"]
        per_prompt_results[pid] = {
            "name": new_p["name"],
            "metrics": prompt_results,
            "status": "regression" if regressions else "ok",
            "regression_count": len(regressions),
        }

    # Overall verdict
    all_ok = all(m["status"] != "regression" for m in overall_results) and \
             all(v["status"] != "regression" for v in per_prompt_results.values())

    return {
        "baseline_file": "baseline",
        "new_file": "new_run",
        "timestamp_new": new_run.get("timestamp"),
        "overall_metrics": overall_results,
        "per_prompt": per_prompt_results,
        "verdict": "PASS" if all_ok else "FAIL",
        "regression_count": sum(1 for v in per_prompt_results.values() if v["status"] == "regression")
                            + sum(1 for m in overall_results if m["status"] == "regression"),
    }


def print_report(report: dict):
    """Print a human-readable regression report."""
    width = 72
    print("\n" + "=" * width)
    verdict = report["verdict"]
    status_icon = "✅" if verdict == "PASS" else "❌"
    print(f"  {status_icon} REGRESSION REPORT — {verdict}")
    print("=" * width)

    # Overall metrics
    print(f"\n  Overall:")
    for m in report["overall_metrics"]:
        icon = "🔴" if m["status"] == "regression" else "🟢"
        print(f"    {icon} {m['metric']:<25s}  "
              f"base={m['baseline']:.2f}  "
              f"new={m['new']:.2f}  "
              f"delta={m['delta_pct']:+.1f}%")

    # Per-prompt
    print(f"\n  Per-prompt ({len(report['per_prompt'])} prompts):")
    for pid, info in sorted(report["per_prompt"].items()):
        if info["status"] == "skip":
            print(f"    ⚪ {pid:<20s}  skip ({info['reason']})")
            continue
        icon = "🔴" if info["status"] == "regression" else "🟢"
        print(f"    {icon} {pid:<20s}  ({info['name']})")
        if "metrics" in info:
            for m in info["metrics"]:
                micon = "↓" if m["status"] == "regression" else "·"
                print(f"         {micon} {m['metric']:<20s}  "
                      f"{m['delta_pct']:+.1f}%  "
                      f"({m['baseline']} → {m['new']})")

    print(f"\n  Total regressions: {report['regression_count']}")
    print("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark runs for regression")
    parser.add_argument("--baseline", required=True, help="Baseline JSON file")
    parser.add_argument("--new", dest="new_run", required=True, help="New run JSON file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Global degradation threshold (e.g. 0.10 = 10%%). "
                             "Applies to all latency metrics (lower-is-better).")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit with code 1 if any regression is found (for CI)")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    new_run = load_json(args.new_run)

    custom_thresholds = {}
    if args.threshold is not None:
        custom_thresholds["avg_ttft_s"] = args.threshold
        custom_thresholds["avg_tpot_ms"] = args.threshold
        custom_thresholds["avg_itl_ms"] = args.threshold
        custom_thresholds["throughput_tok_s"] = -args.threshold

    report = compare_runs(baseline, new_run, custom_thresholds)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if args.fail_on_regression and report["verdict"] == "FAIL":
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
