# HABEAS

**Graph-powered financial-crime investigations, built on FalkorDB.**  
Track: **Agents That Act on Connected Data**

HABEAS investigates synthetic financial-transfer alerts by tracing connected accounts, collecting graph-backed evidence, testing competing explanations, and checking whether a deterministic policy permits a simulated account freeze. An independent verifier then checks the recorded evidence and outcome.

> **Demo safety:** Synthetic data only. HABEAS does not connect to real bank accounts or payment systems. Any freeze is a mutation in a disposable demo graph. Findings are risk indicators for human review, not legal conclusions.

## Demo video

[Watch the HABEAS demo](https://github.com/user-attachments/assets/b12e6278-e677-4516-8ff6-01dbf8e1b9d8)

## Why a graph?

Fraud-like patterns can span multiple transfers and accounts: circular flows, many-to-one concentration, rapid pass-through activity, or shared identifiers. FalkorDB lets HABEAS traverse those relationships instead of treating every transaction in isolation.

## How it works

1. **Investigate:** A bounded LangGraph agent selects approved MCP tools to inspect alerts, account relationships, transfer paths and cycles.
2. **Collect evidence:** Results retain query IDs, parameters, hashes and truncation information.
3. **Check the policy:** Deterministic Policy v2 requires independent, target-relevant evidence; LLM confidence alone cannot authorize an action.
4. **Simulate and act:** HABEAS calculates the potential routing impact, then may apply a guarded freeze to the disposable graph.
5. **Verify:** A separate non-LLM verifier checks recorded evidence and the post-action graph state.

The agent accesses FalkorDB through domain-specific MCP tools—not arbitrary model-generated Cypher.

## Quick start

Requirements: Docker and Python 3.11+.

```bash
git clone https://github.com/thxaman/HABEAS.git
cd HABEAS
docker compose up -d
python -m venv .venv
```

Activate the environment, then install and prepare the demo:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux / Git Bash: use this instead of the Windows activation command
# source .venv/bin/activate

pip install -r requirements.txt
python -m src.seed.generate --seed 42 --reset-demo-graph
```

Run the UI:

```bash
streamlit run app.py
```

Or run the deterministic three-case CLI demo (FakeLLM; no API key required):

```bash
python -m src.demo.run_demo
```

For a live-provider demo, copy `.env.example` to `.env` and configure the appropriate provider key. Never commit `.env` or API keys.

## Checks

```bash
python -m pytest -q
python -m src.probes
python -m pip check
```

Build or verify a report:

```bash
python -m src.report build --case CASE-ALR-0021 --reports-dir reports/demo
python -m src.report verify --case CASE-ALR-0021
```

## Evaluation snapshot

Measured on the synthetic seed-42 benchmark using FakeLLM over the real MCP stdio transport:

| Metric | Result |
|---|---:|
| Alerts evaluated | 21 |
| Strict recall | 4/10 (40%) |
| False-positive rate on benign alerts | 0/11 (0%) |
| Planted pattern-family coverage | 2/4 (50%) |
| Investigation completion | 21/21 (100%) |
| Evidence claims passing verifier | 129/158 (81.6%) |
| Verifier mismatches | 0 |

These results are from a small synthetic benchmark and do not establish real-world detection performance. The high abstention rate and unverifiable claims are documented limitations. HABEAS does not claim that case memory reduces tool calls; the measured comparison did not show savings.

Full evaluation results: [`eval/results.md`](eval/results.md).

## Project details

- [Architecture](docs/architecture.md)
- [Graph model](docs/graph-model.md)
- [FalkorDB capability probes](docs/falkordb-capabilities.md)
- [Evaluation results](eval/results.md)

## AI-assisted development

This project was developed with an AI coding assistant under human direction. Tests, capability probes and evaluation outputs are recorded in the repository; performance claims above are limited to the measured synthetic benchmark.
