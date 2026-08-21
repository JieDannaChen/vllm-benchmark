#!/usr/bin/env python3
"""
Config Comparison Tool
=======================
Runs the same prompt set against multiple benchmark configurations
(different concurrency / batch size, max tokens, quantization profiles,
model variants, ...) and outputs a side-by-side comparison report.

Same workload, different serving setup — how do TTFT / TPOT / ITL /
throughput change?

Usage:
    python compare_configs.py --configs configs/default.yaml configs/concurrent.yaml
    python compare_configs.py --configs configs/a.yaml configs/b.yaml \
        --labels "Sequential" "Concurrent x8" --format html --output results/cmp.html
    python compare_configs.py --configs ... --json   # machine-readable output

Each config is benchmarked with its own settings; results are also saved
as individual run JSONs under results/ so they can be re-used later.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import benchmark  # reuse the core measurement logic


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one_config(config_path: str, label: str, clients_override: Optional[int]) -> dict:
    config = benchmark.load_config(config_path)
    if clients_override:
        config["benchmark"]["concurrent_clients"] = clients_override

    prompts = benchmark.load_prompts(config["prompts_file"])
    mode = config["benchmark"]["mode"]

    if mode == "concurrent":
        result = benchmark.run_concurrent(config, prompts)
    else:
        result = benchmark.run_sequential(config, prompts)

    # Annotate with the human-readable label
    result["label"] = label
    result["config_file"] = config_path

    # Save the individual run for later reuse / regression checks
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)
    safe_label = "".join(c if c.isalnum() else "_" for c in label)[:24] or "cfg"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"run_{safe_label}_{ts}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   💾 {label}: saved to {out}")
    return result


# ── Report generation ─────────────────────────────────────────────────────────

METRIC_ROWS = [
    ("avg_ttft_s", "TTFT (ms)", 1000.0),
    ("avg_tpot_ms", "TPOT (ms)", 1.0),
    ("avg_itl_ms", "ITL (ms)", 1.0),
]


def build_tables(runs: List[dict]) -> dict:
    """Extract per-prompt rows for each metric, columns = runs."""
    labels = [r["label"] for r in runs]
    all_pids = sorted({pid for r in runs for pid in r["summary"].get("per_prompt", {})})

    metric_tables = []
    for key, title, scale in METRIC_ROWS:
        rows = []
        for pid in all_pids:
            values = []
            for r in runs:
                info = r["summary"]["per_prompt"].get(pid, {})
                v = info.get(key)
                values.append(round(v * scale, 2) if v is not None else None)
            rows.append((pid, values))
        metric_tables.append({"title": title, "key": key, "rows": rows})

    overall = []
    for r in runs:
        s = r["summary"]
        overall.append({
            "label": r["label"],
            "config_file": r.get("config_file", ""),
            "mode": s.get("mode"),
            "clients": s.get("concurrent_clients"),
            "max_tokens": r.get("config", {}).get("benchmark", {}).get("max_tokens"),
            "requests": f"{s.get('successful_requests')}/{s.get('total_requests')}",
            "total_tokens": s.get("total_tokens"),
            "throughput": s.get("throughput_tok_s"),
            "wall_time_s": s.get("wall_time_s"),
        })

    return {"labels": labels, "all_pids": all_pids, "metric_tables": metric_tables, "overall": overall}


def _fmt(v):
    return "N/A" if v is None else f"{v:.2f}"


def generate_markdown(runs: List[dict], tables: dict) -> str:
    lines = []
    lines.append("# Benchmark Config Comparison")
    lines.append("")
    lines.append(f"**Timestamp:** {time.strftime('%Y-%m-%dT%H:%M:%S')}  ")
    lines.append(f"**Configs compared:** {len(runs)}  ")
    lines.append("")

    # Overall summary
    lines.append("## Overall")
    lines.append("")
    lines.append("| Config | Mode | Clients | Max tokens | Requests OK | Total tokens | Throughput (tok/s) | Wall time (s) |")
    lines.append("|--------|------|---------|------------|-------------|--------------|--------------------|---------------|")
    for o in tables["overall"]:
        lines.append(
            f"| {o['label']} | {o['mode']} | {o['clients'] or '-'} | {o['max_tokens']} | "
            f"{o['requests']} | {o['total_tokens']} | {o['throughput']} | {o['wall_time_s'] or '-'} |"
        )
    lines.append("")

    # Per-metric tables
    headers = "| Prompt | " + " | ".join(tables["labels"]) + " |"
    sep = "|--------|" + "".join("----------|" for _ in tables["labels"])
    for mt in tables["metric_tables"]:
        lines.append(f"## {mt['title']}")
        lines.append("")
        lines.append(headers)
        lines.append(sep)
        for pid, values in mt["rows"]:
            cells = " | ".join(_fmt(v) for v in values)
            lines.append(f"| {pid} | {cells} |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- All configs ran the same prompt set (same workload).")
    lines.append("- Lower is better for TTFT / TPOT / ITL; higher is better for throughput.")
    lines.append(f"- Individual run files: `results/run_*_{time.strftime('%Y%m%d_%H%M%S')}.json` (see output above).")
    lines.append("")
    return "\n".join(lines)


def generate_html(runs: List[dict], tables: dict) -> str:
    md = generate_markdown(runs, tables)
    try:
        import markdown as mdlib
        body = mdlib.markdown(md, extensions=["tables"])
    except ImportError:
        body = f"<pre>{md}</pre>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Benchmark Config Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1000px; margin: 2em auto; padding: 0 1em; line-height: 1.6; color: #24292f; }}
  h1 {{ border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
  h2 {{ margin-top: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: .2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; text-align: left; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: .2em .4em; border-radius: 3px; font-size: 85%; }}
  pre {{ background: #f6f8fa; padding: 1em; border-radius: 6px; overflow-x: auto; }}
  hr {{ border: 0; border-top: 1px solid #eaecef; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Compare multiple benchmark configs side by side")
    parser.add_argument("--configs", nargs="+", required=True, help="Config YAML files to compare")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Human-readable labels for each config (same count as --configs)")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override concurrent_clients for ALL configs")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown")
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    if args.labels:
        if len(args.labels) != len(args.configs):
            parser.error("--labels must have the same count as --configs")
        labels = args.labels
    else:
        labels = [Path(c).stem for c in args.configs]

    runs = []
    print(f"🔬 Config comparison — {len(args.configs)} configs")
    print("=" * 60)
    for cfg, label in zip(args.configs, labels):
        print(f"\n▶ Running [{label}] from {cfg}")
        runs.append(run_one_config(cfg, label, args.clients))
    print("=" * 60)

    tables = build_tables(runs)

    if args.format == "json":
        content = json.dumps(tables, indent=2, ensure_ascii=False, default=str)
    elif args.format == "html":
        content = generate_html(runs, tables)
    else:
        content = generate_markdown(runs, tables)

    if args.output:
        Path(args.output).write_text(content)
        print(f"\n📄 Comparison report written to: {args.output}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
