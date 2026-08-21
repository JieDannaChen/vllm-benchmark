#!/usr/bin/env python3
"""
vLLM Inference Benchmark Tool
================================
Measures end-to-end inference performance against any OpenAI-compatible
API (vLLM, Ollama, llama-cpp-python, etc.) with streaming support for
accurate per-token timing.

Metrics collected per request:
  - TTFT  (Time To First Token)   : latency until the first token arrives
  - TPOT  (Time Per Output Token) : average latency per generated token
  - ITL   (Inter-Token Latency)   : latency between consecutive tokens
  - Throughput                    : tokens per second (for the full run)

Usage:
    python benchmark.py --config configs/default.yaml
    python benchmark.py --config configs/default.yaml --mode concurrent --clients 16
    python benchmark.py --baseline   # also save as baseline
"""

import argparse
import http.client
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import yaml


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def _stream_sse_lines(url: str, headers: dict, payload: dict, timeout: int = 300):
    """
    Stream SSE `data:` lines from an OpenAI-compatible endpoint with minimal
    client-side buffering.

    Why not requests.iter_lines()? requests wraps the socket in a buffered
    reader that pre-reads large chunks, so tokens that the server emits at a
    steady pace arrive at the client in bursts. That skews inter-token timing
    (ITL percentiles collapse to ~0 within a burst). Reading from the raw
    SocketIO lets each token timestamp reflect the server's actual pacing.
    """
    u = urlparse(url)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)

    conn.request("POST", u.path or "/", body=json.dumps(payload), headers=headers)
    resp = conn.getresponse()
    if resp.status != 200:
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")

    fp = resp.fp.raw  # _io.SocketIO — unbuffered, near-real-time line reads
    try:
        while True:
            line = fp.readline()
            if not line:  # EOF
                break
            yield line.decode("utf-8", errors="replace").rstrip("\r\n")
    finally:
        conn.close()


# ── Core measurement ─────────────────────────────────────────────────────────

def measure_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
    stream: bool = True,
) -> Dict[str, Any]:
    """
    Send one chat completion request and measure timing.
    Returns a dict with TTFT, TPOT, ITL stats, total tokens, etc.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if not stream:
        # Non-streaming fallback — only ITL is meaningful, not TTFT/TPOT
        t0 = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        t_end = time.perf_counter()
        content = data["choices"][0]["message"]["content"]
        total_tokens = len(content)  # rough; better with tokenizer but keep dep-free
        return {
            "itl_s": (t_end - t0) / max(total_tokens, 1),
            "total_tokens": total_tokens,
            "ttft_s": None,
            "tpot_s": None,
            "itl_percentiles": None,
            "content_length": len(content),
            "status": "ok",
        }

    # ── Streaming path ──
    token_timestamps: List[float] = []
    full_content_parts: List[str] = []
    t0 = time.perf_counter()

    for line in _stream_sse_lines(url, headers, payload, timeout=300):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            token_timestamps.append(time.perf_counter())
            full_content_parts.append(content)

    t_end = time.perf_counter()
    full_content = "".join(full_content_parts)
    n_tokens = len(token_timestamps)

    if n_tokens == 0:
        return {
            "ttft_s": None,
            "tpot_s": None,
            "itl_s": None,
            "total_tokens": 0,
            "itl_percentiles": None,
            "content_length": 0,
            "status": "error_no_tokens",
        }

    # TTFT: time from request start to first token
    ttft = token_timestamps[0] - t0

    # ITL: inter-token latencies (between consecutive tokens)
    itls = [token_timestamps[i] - token_timestamps[i - 1] for i in range(1, n_tokens)]
    itl_avg = float(np.mean(itls)) if itls else ttft / max(n_tokens, 1)
    itl_p50 = float(np.percentile(itls, 50)) if itls else itl_avg
    itl_p95 = float(np.percentile(itls, 95)) if itls else itl_avg
    itl_p99 = float(np.percentile(itls, 99)) if itls else itl_avg

    # TPOT: total time / total output tokens (end-to-end, including TTFT)
    total_time = t_end - t0
    tpot = total_time / n_tokens

    return {
        "ttft_s": round(ttft, 4),
        "tpot_s": round(tpot, 5),
        "itl_s": round(itl_avg, 5),
        "itl_p50_s": round(itl_p50, 5),
        "itl_p95_s": round(itl_p95, 5),
        "itl_p99_s": round(itl_p99, 5),
        "total_tokens": n_tokens,
        "total_time_s": round(total_time, 4),
        "content_length": len(full_content),
        "status": "ok",
    }


# ── Sequential benchmark ────────────────────────────────────────────────────

def run_sequential(config: dict, prompts: List[dict]) -> Dict[str, Any]:
    target = config["target"]
    bench = config["benchmark"]
    repeats = bench.get("repeats_per_prompt", 1)
    warmup = bench.get("warmup_requests", 0)

    print(f"\n🚀 Sequential benchmark — {len(prompts)} prompts × {repeats} repeats")
    print(f"   Target: {target['base_url']}  model={target['model']}")
    print(f"   Warmup requests: {warmup}")

    # Warm-up
    if warmup > 0:
        print(f"   Warming up ({warmup} requests)...")
        for i in range(warmup):
            wp = prompts[i % len(prompts)]
            msgs = [
                {"role": "system", "content": wp["system"]},
                {"role": "user", "content": wp["prompt"]},
            ]
            measure_request(
                base_url=target["base_url"],
                api_key=target["api_key"],
                model=target["model"],
                messages=msgs,
                max_tokens=bench["max_tokens"],
                temperature=bench["temperature"],
                stream=bench["stream"],
            )
        print("   Warmup complete.")

    all_results = []
    prompt_results: Dict[str, List[dict]] = {}

    for prompt in prompts:
        prompt_id = prompt["id"]
        prompt_results[prompt_id] = []
        msgs = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["prompt"]},
        ]
        print(f"\n   ▸ [{prompt_id}] {prompt['name']}")
        for r in range(repeats):
            result = measure_request(
                base_url=target["base_url"],
                api_key=target["api_key"],
                model=target["model"],
                messages=msgs,
                max_tokens=bench["max_tokens"],
                temperature=bench["temperature"],
                stream=bench["stream"],
            )
            result["prompt_id"] = prompt_id
            result["prompt_name"] = prompt["name"]
            result["repeat"] = r + 1
            all_results.append(result)
            prompt_results[prompt_id].append(result)
            if result["status"] == "ok":
                print(f"     run {r+1}: TTFT={result['ttft_s']:.3f}s  "
                      f"TPOT={result['tpot_s']*1000:.2f}ms  "
                      f"tokens={result['total_tokens']}")
            else:
                print(f"     run {r+1}: {result['status']}")

    # Aggregate per-prompt stats
    per_prompt_summary = {}
    for prompt_id, runs in prompt_results.items():
        ok_runs = [r for r in runs if r["status"] == "ok"]
        if not ok_runs:
            continue
        ttfts = [r["ttft_s"] for r in ok_runs if r["ttft_s"] is not None]
        tpots = [r["tpot_s"] for r in ok_runs if r["tpot_s"] is not None]
        itls = [r["itl_s"] for r in ok_runs if r["itl_s"] is not None]
        itl_p50s = [r["itl_p50_s"] for r in ok_runs if r.get("itl_p50_s") is not None]
        itl_p99s = [r["itl_p99_s"] for r in ok_runs if r.get("itl_p99_s") is not None]
        tokens = [r["total_tokens"] for r in ok_runs]
        per_prompt_summary[prompt_id] = {
            "name": ok_runs[0]["prompt_name"],
            "runs": len(ok_runs),
            "avg_ttft_s": round(float(np.mean(ttfts)), 4) if ttfts else None,
            "p50_ttft_s": round(float(np.percentile(ttfts, 50)), 4) if ttfts else None,
            "avg_tpot_ms": round(float(np.mean(tpots)) * 1000, 2) if tpots else None,
            "p50_tpot_ms": round(float(np.percentile(tpots, 50)) * 1000, 2) if tpots else None,
            "p95_tpot_ms": round(float(np.percentile(tpots, 95)) * 1000, 2) if tpots else None,
            "avg_itl_ms": round(float(np.mean(itls)) * 1000, 2) if itls else None,
            "itl_p50_ms": round(float(np.mean(itl_p50s)) * 1000, 2) if itl_p50s else None,
            "p95_itl_ms": round(float(np.percentile(itls, 95)) * 1000, 2) if itls else None,
            "itl_p99_ms": round(float(np.mean(itl_p99s)) * 1000, 2) if itl_p99s else None,
            "avg_tokens": round(float(np.mean(tokens)), 1),
        }

    # Overall throughput: total tokens / total wall-clock time
    ok_runs_all = [r for r in all_results if r["status"] == "ok"]
    total_tokens = sum(r["total_tokens"] for r in ok_runs_all)
    total_wall_time = sum(r["total_time_s"] for r in ok_runs_all) if ok_runs_all else 1
    throughput = total_tokens / total_wall_time if total_wall_time > 0 else 0

    summary = {
        "mode": "sequential",
        "model": target["model"],
        "total_requests": len(all_results),
        "successful_requests": len(ok_runs_all),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 2),
        "per_prompt": per_prompt_summary,
    }

    return {
        "summary": summary,
        "raw_results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
    }


# ── Concurrent benchmark (simplified, thread-based) ─────────────────────────

def run_concurrent(config: dict, prompts: List[dict]) -> Dict[str, Any]:
    """Run N concurrent clients, each sending requests in a loop."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target = config["target"]
    bench = config["benchmark"]
    n_clients = bench.get("concurrent_clients", 8)
    # Each client runs through each prompt once for a clean apples-to-apples comparison
    repeats = bench.get("repeats_per_prompt", 1)

    print(f"\n🚀 Concurrent benchmark — {n_clients} clients × {len(prompts)} prompts × {repeats} runs")
    print(f"   Target: {target['base_url']}  model={target['model']}")

    all_results: List[dict] = []
    lock = threading.Lock()

    def client_worker(client_id: int) -> List[dict]:
        results = []
        for prompt in prompts:
            for r in range(repeats):
                msgs = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["prompt"]},
                ]
                result = measure_request(
                    base_url=target["base_url"],
                    api_key=target["api_key"],
                    model=target["model"],
                    messages=msgs,
                    max_tokens=bench["max_tokens"],
                    temperature=bench["temperature"],
                    stream=bench["stream"],
                )
                result["prompt_id"] = prompt["id"]
                result["prompt_name"] = prompt["name"]
                result["client_id"] = client_id
                result["repeat"] = r + 1
                results.append(result)
        return results

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [pool.submit(client_worker, i) for i in range(n_clients)]
        for f in as_completed(futures):
            with lock:
                all_results.extend(f.result())
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    ok_runs = [r for r in all_results if r["status"] == "ok"]
    total_tokens = sum(r["total_tokens"] for r in ok_runs)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    # Per-prompt summary
    from collections import defaultdict
    by_prompt: Dict[str, List[dict]] = defaultdict(list)
    for r in ok_runs:
        by_prompt[r["prompt_id"]].append(r)

    per_prompt = {}
    for pid, runs in by_prompt.items():
        ttfts = [r["ttft_s"] for r in runs if r["ttft_s"] is not None]
        tpots = [r["tpot_s"] for r in runs if r["tpot_s"] is not None]
        itls = [r["itl_s"] for r in runs if r["itl_s"] is not None]
        itl_p50s = [r["itl_p50_s"] for r in runs if r.get("itl_p50_s") is not None]
        itl_p99s = [r["itl_p99_s"] for r in runs if r.get("itl_p99_s") is not None]
        per_prompt[pid] = {
            "name": runs[0]["prompt_name"],
            "runs": len(runs),
            "avg_ttft_s": round(float(np.mean(ttfts)), 4) if ttfts else None,
            "p50_ttft_s": round(float(np.percentile(ttfts, 50)), 4) if ttfts else None,
            "avg_tpot_ms": round(float(np.mean(tpots)) * 1000, 2) if tpots else None,
            "p50_tpot_ms": round(float(np.percentile(tpots, 50)) * 1000, 2) if tpots else None,
            "p95_tpot_ms": round(float(np.percentile(tpots, 95)) * 1000, 2) if tpots else None,
            "avg_itl_ms": round(float(np.mean(itls)) * 1000, 2) if itls else None,
            "itl_p50_ms": round(float(np.mean(itl_p50s)) * 1000, 2) if itl_p50s else None,
            "p95_itl_ms": round(float(np.percentile(itls, 95)) * 1000, 2) if itls else None,
            "itl_p99_ms": round(float(np.mean(itl_p99s)) * 1000, 2) if itl_p99s else None,
            "avg_tokens": round(float(np.mean([r["total_tokens"] for r in runs])), 1),
        }

    summary = {
        "mode": "concurrent",
        "model": target["model"],
        "concurrent_clients": n_clients,
        "total_requests": len(all_results),
        "successful_requests": len(ok_runs),
        "total_tokens": total_tokens,
        "wall_time_s": round(wall_time, 2),
        "throughput_tok_s": round(throughput, 2),
        "per_prompt": per_prompt,
    }

    return {
        "summary": summary,
        "raw_results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM inference benchmark tool")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["sequential", "concurrent"], default=None,
                        help="Override benchmark mode from config")
    parser.add_argument("--clients", type=int, default=None, help="Override concurrent_clients")
    parser.add_argument("--baseline", action="store_true",
                        help="Save this run as the regression baseline")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/run_<timestamp>.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config["benchmark"]["mode"] = args.mode
    if args.clients:
        config["benchmark"]["concurrent_clients"] = args.clients

    prompts = load_prompts(config["prompts_file"])

    # Run benchmark
    mode = config["benchmark"]["mode"]
    if mode == "concurrent":
        result = run_concurrent(config, prompts)
    else:
        result = run_sequential(config, prompts)

    # Print summary table
    print("\n" + "=" * 72)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 72)
    s = result["summary"]
    print(f"   Mode:              {s['mode']}")
    print(f"   Model:             {s['model']}")
    print(f"   Requests:          {s['successful_requests']} / {s['total_requests']} OK")
    print(f"   Total tokens:      {s['total_tokens']}")
    print(f"   Throughput:        {s['throughput_tok_s']} tokens/s")
    if s["mode"] == "concurrent":
        print(f"   Concurrent clients:{s['concurrent_clients']}")
        print(f"   Wall time:         {s['wall_time_s']}s")

    print(f"\n   Per-prompt breakdown:")
    print(f"   {'Prompt ID':<20} {'TTFT(s)':>8} {'TPOT(ms)':>9} {'ITL(ms)':>9} {'Tokens':>8}")
    print(f"   {'-'*20} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    for pid, info in s["per_prompt"].items():
        ttft = f"{info['avg_ttft_s']:.3f}" if info.get("avg_ttft_s") else "N/A"
        tpot = f"{info['avg_tpot_ms']:.2f}" if info.get("avg_tpot_ms") else "N/A"
        itl = f"{info['avg_itl_ms']:.2f}" if info.get("avg_itl_ms") else "N/A"
        print(f"   {pid:<20} {ttft:>8} {tpot:>9} {itl:>9} {info['avg_tokens']:>8.0f}")
    print("=" * 72 + "\n")

    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"run_{s['mode']}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   💾 Results saved to: {out_path}")

    # Save baseline if requested
    if args.baseline:
        baseline_path = config.get("output", {}).get("baseline_file", "baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"   🎯 Baseline saved to: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
