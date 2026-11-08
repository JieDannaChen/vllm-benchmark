# 跑步数据分析：模型升级质量回归演示

这个示例用 11 个**纯合成场景**，比较两个模型版本对相同跑步数据的结构化分析。重点是固定输入、自动检查输出契约、保留原始结果，以及在升级前发现回归。它是质量评估示例，不是医疗诊断系统，也不评估长期训练效果。

所有目标、日期、活动与背景事件均为公开测试用途的虚构值，不对应项目维护者或任何真实用户。

## 一条命令体验

在项目根目录安装依赖后运行：

```bash
python3 -m pip install -r requirements.txt
python3 running_eval.py demo --output-dir results/running-demo
```

`demo` 会自动启动仅供本地使用的 loopback stub，不需要 GPU、API 密钥或真实模型。它使用预设输出：`fixture-baseline` 的自动检查全部通过，`fixture-regressed` 响应更快，但有 3 个场景发生预设退化。**这些结果只验证评分器、对比和报告链路，不能证明任何真实模型变好或变差。**

打开 `results/running-demo` 中生成的对比报告和 JSON 结果，查看逐场景失败字段、自动契约 gate、端到端响应延迟和原始回复。延迟是本地 stub 的演示数据，不能用于判断真实服务性能。

## 对比真实模型

编辑 `configs/running_baseline.yaml` 和 `configs/running_candidate.yaml`，分别将目标地址和模型标识设为要比较的两个版本。使用可访问的 OpenAI-compatible 接口：例如指向自己的 vLLM 服务，或指向启用了兼容接口的 Ollama 服务，并使用服务端实际暴露的模型名称。不要把 API key 写进配置文件；需要认证时，在目标配置中使用 `target.api_key_env: RUNNING_EVAL_API_KEY`，并在运行前设置该环境变量。

两次运行应使用相同的 11 个场景、提示词、规则和采样设置，只改变计划比较的模型版本。默认每个场景重复 3 次、temperature 为 0；这有助于比较，但**不保证输出完全确定**。请核对两个配置及报告中的版本和 hash，记录实际模型版本、部署环境与服务变更；若数据、提示词或规则不同，不应把差异直接归因于模型升级。

```bash
python3 running_eval.py run --config configs/running_baseline.yaml --output results/running-baseline.json
python3 running_eval.py run --config configs/running_candidate.yaml --output results/running-candidate.json
python3 running_eval.py compare --baseline results/running-baseline.json --candidate results/running-candidate.json --report results/running-comparison.md --fail-on-regression
```

真实运行会调用配置的模型服务，具体可用性取决于服务地址、模型、权限及网络。`run` 会保留结果，并以 0/1/2 分别表示自动检查 PASS/BLOCK/UNKNOWN；即使未通过，仍可用已保存的结果进行对比。`compare --fail-on-regression` 在 BLOCK 或 UNKNOWN 时返回非零；上线决定仍需要结合失败详情和人工审核。

## 怎么看结果

- **自动契约 gate 和逐 case 失败字段**：定位输出格式、算术或输入约束等已编码规则的失败。gate 的结论与响应快慢分开看；更快不代表质量更好。
- **端到端响应延迟**：此示例记录非流式请求从发出到获得完整回复的时间，包含客户端和服务链路耗时，受网络、负载和部署环境影响。
- **原始回复 JSON**：保留模型原始输出，方便复核和排查评分器误判；不要在共享报告中放入真实健康隐私数据。
- **数据、提示词和评测器代码 hash**：用于核对两次比较是否使用相同评估材料；hash 一致也不意味着运行环境或模型输出完全确定。
- **人工审核**：`explanation` 的人工审核尚未完成。自动检查通过只表示已编码的契约检查通过，不能等同于专业判断、临床安全或长期训练有效性。

本示例不实现 vLLM 推理引擎，也不是 GPU 性能基准。原有 `benchmark.py` 的流式 SSE chunk 不一定等于一个模型 token；这里不使用 chunk 数计算 TPOT，只报告非流式端到端延迟。

## 如何介绍这个用例

> 这个用例使用完全虚构的跑步数据展示模型升级回归：固定场景、提示词与规则，让基线和候选版本分别执行，再按场景检查结构、数值和输入约束，并保存原始回复与版本 hash，便于复现和追查。报告把质量 gate 和端到端延迟并排呈现：例如演示中的候选响应更快，但有三项预设退化，因此不能只看速度就接受升级。这个演示使用 stub，结论只说明评估链路能发现预设问题；真模型的表现需要实际运行，以及对解释和高风险建议做进一步人工评审。

迁移到其他可穿戴数据场景时，可以继续补充数据缺失、重复、时间错位和单位错误等测试，明确高风险场景的处理与人工确认边界，再与领域专家共同定义评估标准。这些是后续设计方向，不代表示例已经验证了临床安全或上线效果。
