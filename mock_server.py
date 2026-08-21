#!/usr/bin/env python3
"""
Mock OpenAI-Compatible Inference Server
========================================
A dependency-free mock of an OpenAI-compatible chat completions endpoint
(like the one vLLM exposes). Useful for:

  * End-to-end pipeline testing when no GPU / real model is available
  * CI runs on standard GitHub runners (no GPU required)
  * Validating the benchmark pipeline (TTFT/TPOT/ITL/Throughput) before
    pointing it at a real vLLM server

It simulates per-token generation latency so TTFT, TPOT, ITL and
throughput are all measurable and behave like a slow-but-real model.

Usage:
    python mock_server.py --port 8000
    python mock_server.py --port 8000 --tokens-per-sec 30 --ttft-ms 80

Point configs/default.yaml at http://localhost:8000/v1 and run
    python benchmark.py --config configs/default.yaml
"""

import argparse
import json
import random
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ── Content generation ────────────────────────────────────────────────────────
# A small pool of words so generated responses look token-ish (one word ≈ one token).
WORDS = [
    "performance", "inference", "latency", "throughput", "token", "streaming",
    "baseline", "regression", "benchmark", "optimization", "parallelism",
    "quantization", "attention", "context", "model", "deployment", "monitoring",
    "autoscale", "reliability", "scalability", "throughput", "efficiency",
    "quality", "service", "request", "response", "metric", "percentile",
    "pipeline", "scheduler", "batching", "prefill", "decode", "cache", "prompt",
    "guardrail", "policy", "observability", "telemetry", "dashboard", "alert",
]

SENTENCE_PATTERNS = [
    "The {a} improved by {p}% after {b} was applied.",
    "Measuring {a} requires accurate {b} and stable {c}.",
    "We observed a {p}ms {a} under {b} concurrency.",
    "A robust {a} pipeline tracks {b} and {c} over time.",
    "Comparing {a} across {b} helps detect {c} early.",
    "The {a} shows {p} tokens per second in this {b}.",
]

ADJECTIVES = ["measured", "estimated", "observed", "reported", "expected", "typical"]


def _pick_word(rng: random.Random) -> str:
    return rng.choice(WORDS)


def generate_response_text(rng: random.Random, n_words: int) -> str:
    """Generate a deterministic-ish pseudo-token response of ~n_words words."""
    words = []
    while len(words) < n_words:
        pattern = rng.choice(SENTENCE_PATTERNS)
        a = _pick_word(rng)
        b = _pick_word(rng)
        c = _pick_word(rng)
        p = rng.randint(5, 60)
        sentence = pattern.format(a=a, b=b, c=c, p=p)
        words.extend(sentence.split())
    return " ".join(words[:n_words])


def estimate_target_tokens(prompt_text: str, max_tokens: int, rng: random.Random) -> int:
    """
    Decide how many tokens the mock "model" will generate.
    Longer prompts get proportionally more output, capped by max_tokens,
    plus run-to-run jitter so benchmarks show realistic variance.
    """
    # Rough input-token proxy: ~2 chars per token for ASCII
    input_len = max(len(prompt_text) // 2, 1)
    ratio = min(0.95, max(0.15, (20 + input_len) / 150.0))
    jitter = rng.uniform(0.8, 1.2)
    n = int(min(max_tokens, max(8, max_tokens * ratio * jitter)))
    return max(8, min(max_tokens, n))


# ── HTTP handler ──────────────────────────────────────────────────────────────

class MockHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"

    # Injected from CLI args (set in main)
    tokens_per_sec = 40.0
    ttft_ms = 100
    seed = 42

    # ── helpers ──
    def _rng(self) -> random.Random:
        return random.Random(self.seed + time.time_ns() % 1_000_000_000)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_headers(self, content_type: str = "application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _send_error_json(self, status: int, message: str):
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error"}})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, fmt, *args):  # quieter logging
        return

    # ── routes ──
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send_headers()
            body = {
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}],
            }
            self.wfile.write(json.dumps(body).encode())
        else:
            self._send_error_json(404, f"Not found: {path}")

    def do_POST(self):
        path = urlparse(self.path).path
        if path.endswith("/chat/completions"):
            self._handle_chat_completions()
        else:
            self._send_error_json(404, f"Not found: {path}")

    def _handle_chat_completions(self):
        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_error_json(400, f"Bad JSON body: {e}")
            return

        model = body.get("model", "mock-model")
        stream = body.get("stream", False)
        max_tokens = int(body.get("max_tokens", 256))

        # Concatenate all message text to size the response
        prompt_text = " ".join(
            m.get("content", "") for m in body.get("messages", []) if isinstance(m, dict)
        )

        rng = self._rng()
        n_tokens = estimate_target_tokens(prompt_text, max_tokens, rng)
        content = generate_response_text(rng, n_tokens)

        if stream:
            self._handle_stream(model, content, n_tokens, rng)
        else:
            self._handle_non_stream(model, content, prompt_text, n_tokens, rng)

    def _handle_stream(self, model: str, content: str, n_tokens: int, rng: random.Random):
        self._send_headers(content_type="text/event-stream")

        words = content.split()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        def chunk(delta_content, finish_reason=None):
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta_content} if delta_content else {},
                    "finish_reason": finish_reason,
                }],
            }
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        # First token: simulate prefill / TTFT
        time.sleep(self.ttft_ms / 1000.0 + rng.uniform(0, 0.05))
        chunk(words[0] + " ")

        # Remaining tokens: simulate decode speed
        inter_token = 1.0 / self.tokens_per_sec
        for w in words[1:]:
            time.sleep(inter_token * rng.uniform(0.6, 1.4))
            chunk(w + " ")

        chunk(None, finish_reason="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_non_stream(self, model: str, content: str, prompt_text: str,
                           n_tokens: int, rng: random.Random):
        # Simulate the whole generation, then return one big JSON response.
        total_sleep = self.ttft_ms / 1000.0 + n_tokens / self.tokens_per_sec
        time.sleep(total_sleep)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        resp = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": max(len(prompt_text) // 4, 1),
                "completion_tokens": n_tokens,
                "total_tokens": max(len(prompt_text) // 4, 1) + n_tokens,
            },
        }
        self._send_headers()
        self.wfile.write(json.dumps(resp).encode())


def main():
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible inference server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--tokens-per-sec", type=float, default=40.0,
                        help="Simulated decode speed in tokens/second (default: 40)")
    parser.add_argument("--ttft-ms", type=int, default=100,
                        help="Simulated base time-to-first-token in ms (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    MockHandler.tokens_per_sec = args.tokens_per_sec
    MockHandler.ttft_ms = args.ttft_ms
    MockHandler.seed = args.seed

    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    print(f"✅ Mock OpenAI-compatible server listening on http://{args.host}:{args.port}/v1")
    print(f"   model=mock-model  tokens/s={args.tokens_per_sec}  ttft={args.ttft_ms}ms")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
