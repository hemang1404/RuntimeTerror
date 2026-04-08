---
title: RuntimeTerror IncidentEnv
emoji: "🚨"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
    - openenv
    - reinforcement-learning
    - agent
    - incident-response
    - benchmark
license: mit
pinned: false
---

# IncidentEnv — AI On-Call Incident Response Environment

> **An AI agent receives a production alert, investigates using diagnostic tools, identifies the root cause, and submits a code fix — validated by actual test execution.**

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the Meta × PyTorch × Hugging Face Hackathon.

---

## Why Incident Response?

Incident response is the **highest-stakes task in software engineering**. When a system goes down at 3 AM, an on-call engineer must:

1. **Read alerts and logs** — triage severity, identify affected services
2. **Run diagnostics** — query metrics, inspect code, correlate signals
3. **Identify root cause** — connect multiple data points into a hypothesis
4. **Apply a fix** — write and validate a code change that resolves the issue

This environment simulates that full loop, testing multi-step reasoning, code comprehension, and debugging — capabilities no existing benchmark covers.

---

## Architecture

```
reset(task_id)
│
├─ PHASE 1: INVESTIGATION (up to 10 steps)
│   ├─ query_logs(service, keyword)      → filtered log entries
│   ├─ query_metrics(metric, time_range) → time-series data
│   ├─ inspect_code(file)                → source file with line numbers
│   ├─ run_diagnostic(command)           → diagnostic output
│   └─ submit_root_cause(root_cause)     → transitions to Phase 2
│
└─ PHASE 2: REMEDIATION (up to 5 steps)
    ├─ suggest_fix(file, patch_code)     → pytest runs → pass/fail
    └─ submit_resolution()               → episode ends → grader score
```

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `query_logs`, `query_metrics`, `inspect_code`, `run_diagnostic`, `submit_root_cause`, `suggest_fix`, `submit_resolution` |
| `service` | `str` | Target service name (for `query_logs`) |
| `keyword` | `str` | Log search filter (optional) |
| `metric` | `str` | Metric name (for `query_metrics`) |
| `time_range` | `str` | `1m`, `5m`, `15m`, or `1h` |
| `file` | `str` | File path (for `inspect_code` or `suggest_fix`) |
| `command` | `str` | Diagnostic command (for `run_diagnostic`) |
| `root_cause` | `str` | Root cause explanation (for `submit_root_cause`) |
| `patch_code` | `str` | Complete fixed file content (for `suggest_fix`) |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether episode has ended |
| `reward` | `float` | Immediate reward from last action |
| `alert_title` | `str` | Incident alert title |
| `alert_description` | `str` | Incident description |
| `severity_level` | `str` | P1, P2, or P3 |
| `affected_service` | `str` | Primary affected service |
| `output` | `str` | Result of last diagnostic action |
| `output_type` | `str` | `logs`, `metrics`, `code`, `diagnostic`, `test_result`, `info` |
| `available_services` | `list[str]` | Services the agent can query |
| `available_files` | `list[str]` | Source files the agent can inspect |
| `available_metrics` | `list[str]` | Metrics the agent can query |
| `available_commands` | `list[str]` | Diagnostic commands available |
| `phase` | `str` | `investigation` or `remediation` |
| `step_number` | `int` | Current step |
| `max_steps` | `int` | Maximum steps (15) |
| `test_output` | `str` | pytest output (Phase 2 only) |
| `tests_passed` | `bool\|None` | Whether fix passed tests |

---

## Tasks

| Task ID | Difficulty | Scenarios | Description | Pass Threshold |
|---------|-----------|-----------|-------------|---------------|
| `easy_triage` | Easy | 5 | Single clear root cause with obvious log signal | ≥ 0.7 |
| `medium_triage` | Medium | 5 | Multi-signal correlation across services | ≥ 0.5 |
| `hard_triage` | Hard | 5 | Subtle bugs: race conditions, encoding, clock skew | ≥ 0.3 |

### Scenario Examples

- **Easy**: DB connection pool leak, null pointer crash, unbounded SQL query
- **Medium**: Memory leak in batch processor, cascading timeout mismatch, off-by-one rate limiter
- **Hard**: Race condition in job queue, UTF-8/Latin-1 encoding corruption, JWT clock skew, silent event buffer drops, cache stampede

---

## Reward Function

### Investigation Phase

| Action | Condition | Reward |
|--------|-----------|--------|
| `query_logs` | Returns root-cause-relevant data | **+0.05** |
| `query_logs` | Unknown service | **-0.05** |
| `query_metrics` | Valid metric | **+0.05** |
| `inspect_code` | Inspects the buggy file | **+0.10** |
| `run_diagnostic` | Valid command | **+0.05** |
| `submit_root_cause` | ≥ 50% keyword match | **+0.40** |
| `submit_root_cause` | ≥ 30% keyword match | **+0.20** |
| `submit_root_cause` | Poor match | **-0.20** |
| Any | Duplicate query | **-0.10** |

### Remediation Phase

| Action | Condition | Reward |
|--------|-----------|--------|
| `suggest_fix` | All tests pass ✅ | **+0.50** |
| `suggest_fix` | Some tests pass | **+0.20** |
| `suggest_fix` | All tests fail | **-0.10** |
| `submit_resolution` | After successful fix | **+0.10** |
| `submit_resolution` | Without attempting fix | **-0.20** |
| Any | Max steps exceeded | **-0.30** |

---

## Grading

Deterministic formula (0.0 – 1.0):

```
score = 0.30 × root_cause_accuracy
      + 0.15 × investigation_quality  (inspected buggy file?)
      + 0.35 × fix_quality            (tests passed?)
      + 0.10 × efficiency             (fewer steps = better)
      + 0.10 × decision_quality       (submitted root cause?)
```

---

## Setup

### Local Development

```bash
# Install dependencies
pip install pydantic fastapi uvicorn requests pytest

# Run tests
python -m pytest tests/ -v

# Start server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t incident-env:latest -f server/Dockerfile .
docker run -p 8000:8000 incident-env:latest
```

### HuggingFace Space

```bash
openenv push --repo-id yourname/incident-env
```

---

## Usage

```python
# Connect to running server
from client import IncidentEnv

env = IncidentEnv(base_url="http://localhost:8000")

# Start an episode
obs = env.reset(task_id="easy_triage")
print(obs["alert_title"])  # 🚨 HIGH ERROR RATE: user-api

# Investigate
result = env.step({"action_type": "query_logs", "service": "user-api", "keyword": "error"})
print(result["observation"]["output"])  # Log entries...

result = env.step({"action_type": "inspect_code", "file": "db/pool.py"})
print(result["observation"]["output"])  # Source code...

# Submit root cause
result = env.step({
    "action_type": "submit_root_cause",
    "root_cause": "DB connections leak in get_user() — never released"
})

# Fix the bug
result = env.step({
    "action_type": "suggest_fix",
    "file": "db/pool.py",
    "patch_code": "...fixed code with try/finally..."
})
print(result["observation"]["tests_passed"])  # True ✅

# Finalize
result = env.step({"action_type": "submit_resolution"})
state = env.state()
print(f"Grader score: {state['grader_score']}")  # 0.87
```

---

## Inference Script

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

python inference.py                          # all tasks
python inference.py --task easy_triage       # just easy
python inference.py --episodes 3             # 3 episodes per task
```

---

## Baseline Scores

Latest measured runs (HF Space deployment):

| Task | Model | Avg Score | Episodes | Notes |
|------|-------|-----------|----------|-------|
| `easy_triage` | `Qwen/Qwen2.5-7B-Instruct` | `0.4256` | `3` | Episode 1 reached `0.7767`; later episodes degraded due to exhausted provider credits |
| `easy_triage` | `Qwen/Qwen2.5-7B-Instruct` | `0.1500` | `1` | Earlier baseline before policy improvements |

Reproduce with:

```bash
export ENV_URL="https://hemang1404-runtimeterror.hf.space"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"

python inference.py --task easy_triage --episodes 3
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | API key |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:8000`) |

---

## License

MIT
