# vLLM Inference Benchmark Toolkit

A lightweight toolkit for measuring, baseline-tracking, and regression-testing
LLM inference performance. Built with a quality-engineering mindset:
reproducible results, automated regression detection, CI integration,
and clear reports.

> The architecture follows the same patterns used in enterprise
> performance engineering (baseline management, regression gates, CI
> integration) and is designed to scale from a local CPU setup to a
> production GPU vLLM deployment without changing the workflow.

## Features

- **Four core metrics** — TTFT, TPOT, ITL (with percentiles), and Throughput
- **Streaming-native** — accurate per-token timing via SSE streaming
- **Two modes** — sequential (latency-focused) and concurrent (throughput-focused)
- **Regression detection** — compare new runs against a saved baseline with configurable thresholds
- **Markdown + HTML reports** — publishable results for PRs or dashboards
- **CI-ready** — GitHub Actions workflow with PR comment reporting
- **Backend-agnostic** — works with any OpenAI-compatible API (vLLM, Ollama, llama-cpp-python, etc.)

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Start an inference backend

**Option 0 — Mock server (zero dependency, for CI / local testing)**
```bash
python mock_server.py --port 8000 --tokens-per-sec 40 --ttft-ms 80
python benchmark.py --config configs/mock.yaml --baseline
```

**Option A — vLLM (GPU recommended)**
```bash
pip install vllm
vllm serve qwen2.5-0.5b-instruct --host 0.0.0.0 --port 8000
```

**Option B — Ollama (easy CPU/Mac setup)**
```bash
ollama serve &
ollama pull qwen2.5:0.5b
# Update config: base_url → http://localhost:11434/v1, model → qwen2.5:0.5b
```

**Option C — Any OpenAI-compatible API**
Just set `base_url` and `model` in the config.

### 3. Run a benchmark

```bash
# Sequential latency benchmark
python benchmark.py --config configs/default.yaml

# Concurrent throughput benchmark
python benchmark.py --config configs/default.yaml --mode concurrent --clients 8
```

### 4. Save a baseline

```bash
python benchmark.py --config configs/default.yaml --baseline
```

### 5. Run regression check

```bash
# After making a change (new config, model version, etc.)
python benchmark.py --config configs/default.yaml --output results/new_run.json
python compare.py --baseline baseline.json --new results/new_run.json --fail-on-regression
```

### 6. Generate a report

```bash
python report.py results/run_sequential_xxx.json --format markdown
python report.py results/run_sequential_xxx.json --format html --output report.html
```

## Metrics Explained

| Metric | Full Name | What It Measures | Why It Matters |
|--------|-----------|-----------------|----------------|
| **TTFT** | Time To First Token | Time from request start to the first output token | Perceived responsiveness / user experience |
| **TPOT** | Time Per Output Token | Average total time per generated token (end-to-end) | Overall generation speed |
| **ITL** | Inter-Token Latency | Time between consecutive tokens (post-first-token) | Smoothness of streaming output; jitter detection |
| **Throughput** | Tokens per second | Total output tokens / total wall time | System-level capacity and cost efficiency |

## Project Structure

```
vllm-benchmark/
├── benchmark.py          # Core benchmark runner (sequential + concurrent)
├── compare.py            # Regression comparison tool
├── compare_configs.py    # Side-by-side config comparison (A/B testing)
├── report.py             # Markdown / HTML report generator
├── mock_server.py        # Mock OpenAI-compatible backend (no GPU needed)
├── baseline.json         # Saved baseline (after --baseline run)
├── configs/
│   ├── default.yaml      # Test configuration (model, prompts, thresholds)
│   ├── mock.yaml         # Config for the mock backend
│   └── concurrent.yaml   # Concurrent-mode variant
├── prompts/
│   └── test_prompts.json # Test prompts with categories
├── results/              # Generated run results
├── requirements.txt
├── .github/workflows/
│   └── benchmark.yml     # GitHub Actions CI
└── README.md
```

## Config Comparison (A/B Testing)

Run the same prompts under different configs and compare side by side:

```bash
python compare_configs.py configs/mock.yaml configs/concurrent.yaml
```

## CI Integration

The GitHub Actions workflow (`.github/workflows/benchmark.yml`) runs benchmarks
on every PR, compares against the baseline, and posts the results as a PR comment.

For real GPU benchmarks, you'll need a self-hosted runner. The workflow file
includes commented-out instructions.

## Motivation

LLM inference performance regresses silently unless someone measures it.
This toolkit applies standard performance engineering practices — baseline
management, regression gates, CI integration, clear reporting — to LLM
serving. It is not a replacement for production load-test tools like vLLM's
built-in `benchmark_serving`; it focuses on the regression-workflow side of
performance engineering.

## Design Notes

- **Metric selection**: TTFT reflects perceived responsiveness, TPOT/ITL reflect generation speed and streaming smoothness, throughput reflects system capacity — each covers a distinct failure mode
- **Client-side buffering pitfall**: `requests.iter_lines()` pre-buffers the socket, causing tokens to arrive in bursts and ITL percentiles to collapse to ~0. The benchmark reads from the unbuffered `http.client` socket directly so per-token timestamps reflect the server's actual pacing
- **Regression gate**: baseline + threshold + `--fail-on-regression` exit code, wired into CI so PRs are blocked on performance degradation
- **Reproducibility**: fixed prompt set, temperature 0, committed baselines, and a mock backend so the full pipeline runs identically on any machine without a GPU

## License

MIT
