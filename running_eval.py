#!/usr/bin/env python3
"""Small, auditable running-data regression eval for compatible chat endpoints.

This complements benchmark.py. It checks structured analysis against synthetic
fixtures and measures non-streaming end-to-end latency, not medical validity.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parent
GRADER_VERSION = "running-contract-v1"
EXPECTED_FIELDS = {
    "observed_distance_km", "activity_count", "goal_seconds", "plan_status",
    "evidence_ids", "missing_fields", "needs_clarification",
    "can_assess_full_week", "next_action",
}
SYSTEM_PROMPT = """你是跑步数据分析助手。本任务只检查数据理解和用户指令遵循。
输入中的数据全部是合成测试数据。仅分析输入，不作专业健康判断，不开训练处方，
不推断未查询到的健康指标，不承诺成绩或恢复时间。
按 input 中的明确任务、统计范围、来源及规则分析；同ID内容完全相同的活动
只计一次，有冲突且无优先级的来源不得擅自选择。未确认的目标变更不是生效目标。
用户明确暂停时，直接尊重暂停，不要求用户再次确认。不要仅凭背景记录改变目标。
只输出一个 JSON 对象，不要 Markdown 或额外字段，必须包含：
observed_distance_km: 已观察记录的公里数和；距离缺失或冲突时为null；
activity_count: 范围内去重后的活动数；查询失败不能确定时为null；
goal_seconds: 最近已确认生效目标的秒数；未知为null；
plan_status: active、paused 或 unknown；
evidence_ids: 本次判断实际采用的数据/指令source_id，唯一字符串数组；
missing_fields: 待补充的信息路径，唯一字符串数组；
needs_clarification: 布尔值；can_assess_full_week: 布尔值；
can_assess_full_week 表示完整周距离统计可评估：查询成功、覆盖范围完整、
窗口已结束，且活动距离无缺失或冲突。尚未发生的剩余周无需向用户澄清。
next_action: summarize、clarify、pause_plan 或 review_goal；
explanation: 简短中文说明，说明证据和不确定性，不能与结构化字段矛盾。
input 中给出的字段路径用于 missing_fields，source_id 用于 evidence_ids。
"""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def reject_constant(value):
    raise ValueError("Non-finite JSON number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(text):
    return json.loads(text, parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


def validate_answer(answer, explanation=True):
    if not isinstance(answer, dict):
        raise ValueError("Response must be a JSON object")
    required = EXPECTED_FIELDS | ({"explanation"} if explanation else set())
    if set(answer) != required:
        raise ValueError("Response fields do not match the output contract")
    for field in ("observed_distance_km", "activity_count", "goal_seconds"):
        value = answer[field]
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid numeric field: " + field)
            if field != "observed_distance_km" and type(value) is not int:
                raise ValueError("Expected an integer: " + field)
    for field in ("needs_clarification", "can_assess_full_week"):
        if type(answer[field]) is not bool:
            raise ValueError("Expected a boolean: " + field)
    for field in ("evidence_ids", "missing_fields"):
        value = answer[field]
        if not isinstance(value, list) or any(type(x) is not str for x in value):
            raise ValueError("Expected a string array: " + field)
        if len(value) != len(set(value)):
            raise ValueError("Duplicate array values: " + field)
    if answer["plan_status"] not in ("active", "paused", "unknown"):
        raise ValueError("Invalid plan_status")
    if answer["next_action"] not in ("summarize", "clarify", "pause_plan", "review_goal"):
        raise ValueError("Invalid next_action")
    if explanation and (not isinstance(answer["explanation"], str)
                        or not answer["explanation"].strip()):
        raise ValueError("Expected a nonempty explanation")


def load_dataset(path):
    dataset = strict_json(Path(path).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Dataset must contain unique, nonempty case IDs")
    for case in cases:
        validate_answer(case["expected"], explanation=False)
        if not set(case["critical_fields"]) <= EXPECTED_FIELDS:
            raise ValueError("Unknown critical field")
    return dataset


def grade(case, raw_response, request_error=None):
    """Exact independent expectations; arrays compare as sets, km within 0.01."""
    base = {"case_id": case["id"], "category": case["category"],
            "raw_response": raw_response, "checks": [], "critical_failures": []}
    if request_error:
        return dict(base, status="error", error=request_error, parsed_response=None)
    try:
        answer = strict_json(raw_response)
        validate_answer(answer)
    except (ValueError, TypeError) as exc:
        return dict(base, status="error", error=str(exc), parsed_response=None)
    checks = []
    for field, expected in case["expected"].items():
        actual = answer[field]
        if isinstance(expected, list):
            passed = set(actual) == set(expected)
        elif field == "observed_distance_km" and expected is not None and actual is not None:
            passed = abs(actual - expected) <= 0.01
        else:
            passed = type(actual) is type(expected) and actual == expected
        checks.append({"field": field, "passed": passed, "expected": expected,
                       "actual": actual, "critical": field in case["critical_fields"]})
    failed = [c for c in checks if not c["passed"]]
    return dict(base, status="fail" if failed else "pass", error=None,
                checks=checks, parsed_response=answer,
                critical_failures=[c["field"] for c in failed if c["critical"]])


def summarize(results):
    counts = {s: sum(r["status"] == s for r in results) for s in ("pass", "fail", "error")}
    failed = [r for r in results if r["status"] == "fail"]
    gate = "BLOCK" if failed else "UNKNOWN" if counts["error"] or not results else "PASS"
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {"planned_trials": len(results), **counts,
            "pass_rate": counts["pass"] / len(results) if results else None,
            "critical_failed_trials": sum(bool(r["critical_failures"]) for r in results),
            "automated_contract_gate": gate,
            "human_explanation_review": "pending",
            "mean_response_latency_s": statistics.mean(latencies) if latencies else None}


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    target = config["target"]
    url = urlsplit(target["base_url"])
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an http(s) base URL without embedded credentials/query")
    settings = config.get("evaluation", {})
    values = {"repeats_per_case": settings.get("repeats_per_case", 3),
              "max_tokens": settings.get("max_tokens", 1024),
              "temperature": settings.get("temperature", 0.0),
              "timeout_seconds": settings.get("timeout_seconds", 60)}
    for field in ("repeats_per_case", "max_tokens", "timeout_seconds"):
        if type(values[field]) is not int or values[field] < 1:
            raise ValueError("Expected a positive integer: " + field)
    temp = values["temperature"]
    if type(temp) not in (float, int) or not math.isfinite(temp) or temp < 0:
        raise ValueError("Invalid temperature")
    config["evaluation"] = values
    return config


def request_completion(target, settings, messages):
    """Only E2E response latency. No character/chunk counts masquerade as tokens."""
    key_env = target.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise ValueError("Configured API key environment variable is not set")
    else:
        key = target.get("api_key", "EMPTY")
    payload = {"model": target["model"], "messages": messages,
               "max_tokens": settings["max_tokens"],
               "temperature": settings["temperature"], "stream": False}
    req = urllib.request.Request(target["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=settings["timeout_seconds"]) as response:
            data = strict_json(response.read().decode("utf-8"))
        latency = time.perf_counter() - start
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Expected text content")
        if choice.get("finish_reason") not in (None, "stop"):
            return content, latency, "Response did not finish normally"
        return content, latency, None
    except urllib.error.HTTPError as exc:
        # Never persist an upstream error body, headers, API key or full config.
        exc.close()
        return "", time.perf_counter() - start, "HTTP " + str(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return "", time.perf_counter() - start, "Request/response error: " + type(exc).__name__


def execute(config, dataset, *, fixture=False):
    settings, target = config["evaluation"], config["target"]
    results = []
    for case in dataset["cases"]:
        user_message = json.dumps({"case_id": case["id"], "input": case["input"]}, ensure_ascii=False)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}]
        for repeat in range(1, settings["repeats_per_case"] + 1):
            raw, latency, error = request_completion(target, settings, messages)
            result = grade(case, raw, error)
            result.update(repeat=repeat, latency_s=latency)
            results.append(result)
    endpoint = urlsplit(target["base_url"])
    # Store origin only: even a path can contain an access token on some gateways.
    origin = endpoint.scheme + "://" + endpoint.netloc
    return {
        "schema": "running-analysis-run/v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_kind": "fixture" if fixture else "model_endpoint",
        "label": config.get("label", target["model"]),
        "target": {"endpoint_origin": origin, "endpoint_path_sha256": canonical_hash(endpoint.path),
                   "model": target["model"]},
        "contract": {"dataset_version": dataset["version"], "dataset_sha256": canonical_hash(dataset),
                     "case_ids": [case["id"] for case in dataset["cases"]],
                     "system_prompt_sha256": canonical_hash(SYSTEM_PROMPT),
                     "grader_version": GRADER_VERSION,
                     "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "settings": settings},
        "limitations": ["Synthetic regression set; not a medical or training-effectiveness evaluation.",
                        "Free-text explanations require human review even when structured checks pass.",
                        "Small repeated sample; no claim of statistical significance or production readiness."],
        "summary": summarize(results), "results": results,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare_runs(baseline, candidate):
    for result in (baseline, candidate):
        if result.get("schema") != "running-analysis-run/v1":
            raise ValueError("Not a running-analysis result")
    if baseline["execution_kind"] != candidate["execution_kind"]:
        raise ValueError("Cannot compare fixture output with real endpoint output")
    if baseline["contract"] != candidate["contract"]:
        raise ValueError("Dataset, prompt, evaluator or run settings differ; comparison refused")
    def index(run):
        indexed = {(r["case_id"], r["repeat"]): r for r in run["results"]}
        if len(indexed) != len(run["results"]):
            raise ValueError("Duplicate trial IDs")
        contract = run["contract"]
        case_ids = contract.get("case_ids", [])
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("Invalid case manifest")
        expected = {(case_id, repeat) for case_id in case_ids
                    for repeat in range(1, contract["settings"]["repeats_per_case"] + 1)}
        if set(indexed) != expected:
            raise ValueError("Trials do not cover the complete case/repeat manifest")
        for result in indexed.values():
            if result["status"] not in ("pass", "fail", "error"):
                raise ValueError("Unknown trial status")
            if result["status"] != "error":
                checks = result["checks"]
                if len(checks) != len(EXPECTED_FIELDS) or {c["field"] for c in checks} != EXPECTED_FIELDS:
                    raise ValueError("Incomplete trial checks")
                if any(type(c["passed"]) is not bool for c in checks):
                    raise ValueError("Invalid check status")
                if (result["status"] == "pass") != all(c["passed"] for c in checks):
                    raise ValueError("Trial status contradicts its checks")
        return indexed
    old, new = index(baseline), index(candidate)
    if not old or set(old) != set(new):
        raise ValueError("Trial coverage differs or is empty; comparison refused")
    old_summary, new_summary = summarize(list(old.values())), summarize(list(new.values()))
    regression = [key for key in old if old[key]["status"] == "pass" and new[key]["status"] != "pass"]
    gate = new_summary["automated_contract_gate"]
    if gate == "PASS" and old_summary["error"]:
        gate = "UNKNOWN"
    return {"schema": "running-analysis-comparison/v1", "execution_kind": baseline["execution_kind"],
            "baseline_label": baseline["label"], "candidate_label": candidate["label"],
            "baseline_target": baseline.get("target", {}), "candidate_target": candidate.get("target", {}),
            "baseline_summary": old_summary, "candidate_summary": new_summary,
            "automated_contract_gate": gate, "human_explanation_review": "pending",
            "regressed_trials": [{"case_id": k[0], "repeat": k[1]} for k in regression],
            "pairs": [{"baseline": old[k], "candidate": new[k]} for k in old],
            "contract": baseline["contract"]}


def render_report(comparison):
    fixture = comparison["execution_kind"] == "fixture"
    lines = ["# 跑步数据分析：模型升级回归检查", "",
             "**模拟演示：响应由预设测试桩生成，没有调用真实模型，不能据此推断模型优劣。**" if fixture
             else "**真实端点调用；模型标签由配置提供，需另行记录实际部署版本。**",
             "", "自动项只检查结构、算术、来源和已编码的用户指令约束。说明文字待人工审核；不评判医学正确性或长期训练效果。", "",
             "自动契约门禁：**" + comparison["automated_contract_gate"] + "**；人工说明审核：**待完成**。", "",
             "| 版本 | 通过/计划运行 | 失败 | 请求或格式错误 | 关键字段失败次数 | 平均完整响应耗时(ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for label, summary in ((comparison["baseline_label"], comparison["baseline_summary"]),
                           (comparison["candidate_label"], comparison["candidate_summary"])):
        latency = summary["mean_response_latency_s"]
        latency = f"{latency * 1000:.1f}" if latency is not None else "N/A"
        lines.append(f"| {cell(label)} | {summary['pass']}/{summary['planned_trials']} | {summary['fail']} | {summary['error']} | {summary['critical_failed_trials']} | {latency} |")
    lines += ["", "耗时为非流式请求的端到端响应时间；模拟模式只演示取舍，不是 vLLM/GPU 性能结果。", "",
              "| 场景 | 重复 | 基线 | 候选 | 候选未通过字段/错误 |", "|---|---:|---|---|---|"]
    for pair in comparison["pairs"]:
        old, new = pair["baseline"], pair["candidate"]
        reasons = [c["field"] + (" [关键]" if c["critical"] else "") for c in new["checks"] if not c["passed"]]
        reasons = ", ".join(reasons) or new.get("error") or "—"
        lines.append(f"| {cell(new['case_id'])} | {new['repeat']} | {old['status']} | {new['status']} | {cell(reasons)} |")
    lines += ["", "## 可复现信息", "", "```json",
              json.dumps({"baseline_target": comparison.get("baseline_target", {}),
                          "candidate_target": comparison.get("candidate_target", {})}, ensure_ascii=False, indent=2),
              "```", "", "```json",
              json.dumps(comparison["contract"], ensure_ascii=False, indent=2), "```", "",
              "逐次原始回答、解析结果、期望值和实际值保存在同目录的运行 JSON 中。", "",
              "PASS 仅表示这些合成样本的自动契约检查通过，仍需审阅自由文本及扩大独立样本；小样本分差不代表统计显著。"]
    return "\n".join(lines) + "\n"


def demo(output_dir):
    dataset = load_dataset(ROOT / "scenarios/running_analysis.json")
    cases = {c["id"]: c for c in dataset["cases"]}
    # These deliberately wrong outputs test that the grader can detect failures.
    mutations = {"02_missing_distance": {"observed_distance_km": 8.0},
                 "05_unconfirmed_goal": {"goal_seconds": 16200},
                 "06_explicit_pause": {"plan_status": "active", "next_action": "summarize"}}
    if not set(mutations) <= set(cases):
        raise ValueError("Demo mutation case IDs do not match dataset")

    class FixtureHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = request.get("model")
            if model not in ("fixture-baseline", "fixture-regressed"):
                self.send_error(400)
                return
            case_id = json.loads(request["messages"][-1]["content"])["case_id"]
            answer = copy.deepcopy(cases[case_id]["expected"])
            answer["explanation"] = "预设测试桩回答，仅用于验证评测链路。"
            if model == "fixture-regressed":
                answer.update(mutations.get(case_id, {}))
            time.sleep(0.03 if model == "fixture-baseline" else 0.005)
            data = json.dumps({"model": model, "choices": [{"message": {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                                                                "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_dir = Path(output_dir)
    try:
        runs = []
        for model in ("fixture-baseline", "fixture-regressed"):
            config = {"label": model, "target": {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": model},
                      "evaluation": {"repeats_per_case": 2, "max_tokens": 1024,
                                     "temperature": 0.0, "timeout_seconds": 10}}
            run = execute(config, dataset, fixture=True)
            write_json(output_dir / (model + ".json"), run)
            runs.append(run)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    comparison = compare_runs(*runs)
    write_json(output_dir / "comparison.json", comparison)
    (output_dir / "comparison.md").write_text(render_report(comparison), encoding="utf-8")
    if runs[0]["summary"]["automated_contract_gate"] != "PASS" or comparison["automated_contract_gate"] != "BLOCK":
        raise RuntimeError("Fixture demonstration failed its expected control checks")
    print("模拟链路验证成功：预设候选的退化已被阻断；没有测量真实模型能力。")
    print(output_dir / "comparison.md")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo_parser = commands.add_parser("demo", help="Run two explicit fixture endpoints locally")
    demo_parser.add_argument("--output-dir", default="results/running-demo")
    run_parser = commands.add_parser("run", help="Evaluate a real compatible chat endpoint")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True)
    compare_parser = commands.add_parser("compare", help="Compare matching evaluation runs")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--report", required=True)
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            demo(args.output_dir)
            return 0  # Expected BLOCK is a successful test of the control.
        if args.command == "run":
            config = read_config(args.config)
            path = Path(config.get("scenario_file", "scenarios/running_analysis.json"))
            dataset_path = path if path.is_absolute() else ROOT / path
            if Path(args.output).resolve() in (Path(args.config).resolve(), dataset_path.resolve()):
                raise ValueError("Output must not overwrite the config or dataset")
            dataset = load_dataset(dataset_path)
            result = execute(config, dataset)
            write_json(args.output, result)
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            return {"PASS": 0, "BLOCK": 1, "UNKNOWN": 2}[result["summary"]["automated_contract_gate"]]
        report = Path(args.report)
        if report.suffix.lower() != ".md":
            raise ValueError("Comparison report must have a .md suffix")
        outputs = {report.resolve(), report.with_suffix(".json").resolve()}
        inputs = {Path(args.baseline).resolve(), Path(args.candidate).resolve()}
        if len(outputs) != 2 or outputs & inputs:
            raise ValueError("Comparison output must not overwrite input results")
        comparison = compare_runs(strict_json(Path(args.baseline).read_text()),
                                  strict_json(Path(args.candidate).read_text()))
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(comparison), encoding="utf-8")
        write_json(report.with_suffix(".json"), comparison)
        print("自动契约门禁：" + comparison["automated_contract_gate"])
        return int(args.fail_on_regression and comparison["automated_contract_gate"] != "PASS")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Evaluation failed: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
