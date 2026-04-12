---
title: NitpickAI
emoji: "🔍"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
    - openenv
    - reinforcement-learning
    - agent
    - code-debugging
    - benchmark
license: mit
pinned: false
---

# 🔍 NitpickAI — Interactive Debugging Benchmark for AI Agents

> **An AI agent receives buggy Python code, investigates using execution tools, identifies the bug, and submits a fix — validated by actual test execution.**

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the Meta × PyTorch × Hugging Face Hackathon.

---

## Why Interactive Debugging?

Debugging is the **most cognitively demanding task in software engineering**. Unlike static code review, real debugging requires:

1. **Reading code** — understand the intended behavior
2. **Running code** — observe actual vs expected behavior
3. **Forming hypotheses** — identify what's wrong
4. **Patching code** — write a targeted fix
5. **Validating** — run tests to confirm the fix works

NitpickAI simulates this full loop, testing multi-step reasoning, code comprehension, and execution-based debugging — capabilities no existing benchmark adequately covers.

---

## Architecture

```
reset(task_id)
│
├─ step(run_code, code="print(calculate_total([Item(10)]))")
│   → "0"  (bug visible! should be 10)
│
├─ step(run_tests)
│   → "FAILED: test_total_single_item - expected 10, got 0"
│
├─ step(create_issue, issue_description="Off-by-one: range(1,...)")
│   → "+0.30 reward"
│
├─ step(suggest_fix, patch_code="def calculate_total(...)")
│   → "4/4 tests passed ✅"  (+0.50 reward)
│
└─ step(request_changes, message="Fixed off-by-one in range()")
    → episode ends (+1.00 reward) → grader score: 0.85
```

---

## Action Space

| Action | Parameters | Description | Reward |
|--------|-----------|-------------|--------|
| `run_code` | `code` | Execute a snippet in sandbox | +0.05 (useful) |
| `run_tests` | — | Run visible test suite | +0.05 (reveals failures) |
| `create_issue` | `issue_description` | Describe the bug | +0.3 (accurate) / -0.2 (poor) |
| `suggest_fix` | `patch_code` | Submit fixed source | +0.5 (all pass) / -0.3 (fail) |
| `request_changes` | `message` | Finalize session | +1.0 (after fix) / -0.3 (no fix) |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether episode has ended |
| `reward` | `float` | Immediate reward from last action |
| `code` | `str` | Current source code (with line numbers) |
| `visible_tests` | `list[str]` | Test code the agent can see |
| `execution_output` | `str` | stdout/stderr from last run_code |
| `test_results` | `str` | pytest output from last run_tests/suggest_fix |
| `tests_passed` | `bool\|None` | Whether all tests passed |
| `step_number` | `int` | Current step |
| `max_steps` | `int` | Maximum steps for this task |
| `action_feedback` | `str` | Human-readable feedback |

---

## Tasks

| Task ID | Difficulty | Scenarios | Bug Types | Max Steps |
|---------|-----------|-----------|-----------|-----------|
| `easy_debug` | Easy | 5 | Off-by-one, missing return, wrong operator, empty string, division by zero | 15 |
| `medium_debug` | Medium | 5 | Mutable default, type coercion, pagination, KeyError, boolean logic | 20 |
| `hard_debug` | Hard | 5 | Closure capture, generator exhaustion, cache corruption, float precision, class variable leak | 25 |

### Scenario Examples

- **Easy**: `range(1, len(items))` skips first element; function missing `return` statement
- **Medium**: Shallow dict copy causes config bleed; string prices compared lexicographically
- **Hard**: Lambda in loop captures loop variable by reference; generator exhausted after first pass

---

## Reward Design

Rewards are **dense and incremental** — not binary end-of-episode scores:

| Phase | Reward Range | Signal |
|-------|-------------|--------|
| Exploration (run_code, run_tests) | -0.1 to +0.05 | Encourages useful investigation |
| Diagnosis (create_issue) | -0.2 to +0.3 | Rewards accurate bug identification |
| Remediation (suggest_fix) | -0.3 to +0.5 | Penalizes bad fixes, rewards working ones |
| Finalization (request_changes) | -0.3 to +1.0 | Big bonus for completing with a working fix |

---

## Grading Formula

Deterministic score in [0.0, 1.0]:

```
score = 0.25 × issue_accuracy      (keyword overlap with ground truth)
      + 0.15 × code_execution_quality  (did agent run code to investigate?)
      + 0.40 × fix_quality            (fraction of tests passed)
      + 0.10 × efficiency             (fewer steps = better)
      + 0.10 × decision_quality       (submitted issue + attempted fix?)
```

---

## Setup

### Local Development

```bash
# Install dependencies
pip install pydantic fastapi uvicorn requests pytest

# Run tests
python -m pytest tests/ -v

# Start server (API only, no UI)
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Start server with Gradio UI
pip install gradio
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t nitpick-ai:latest .
docker run -p 7860:7860 nitpick-ai:latest
```

---

## Usage

### Python Client

```python
from client import NitpickEnv

env = NitpickEnv(base_url="http://localhost:7860")

# Start a debugging session
obs = env.reset(task_id="easy_debug")
print(obs["code"])  # Buggy source code

# Run the code to observe behavior
result = env.step({"action_type": "run_code", "code": "from code import *\nprint(calculate_total([Item(10)]))"})
print(result["observation"]["execution_output"])  # "0" (should be 10!)

# Run tests
result = env.step({"action_type": "run_tests"})
print(result["observation"]["test_results"])  # FAILED: expected 10, got 0

# Identify the bug
result = env.step({
    "action_type": "create_issue",
    "issue_description": "Off-by-one: range(1, len(items)) skips the first item"
})

# Fix it
result = env.step({
    "action_type": "suggest_fix",
    "patch_code": "class Item:\n  def __init__(self, price):\n    self.price = price\n\ndef calculate_total(items):\n  total = 0\n  for i in range(len(items)):\n    total += items[i].price\n  return total\n"
})
print(result["observation"]["tests_passed"])  # True ✅

# Finalize
result = env.step({"action_type": "request_changes", "message": "Fixed range"})
state = env.state()
print(f"Grader score: {state['grader_score']}")  # 0.85+
```

---

## Baseline Agent

A rule-based agent (no LLM required) that follows a fixed strategy:

```bash
# Run locally
python -m agent.baseline

# Evaluate across all tasks
python baseline_eval.py --episodes 3
python baseline_eval.py --task easy_debug --episodes 5 --seed 42
```

---

## Baseline Scores

| Task | Avg Score | Success Rate | Notes |
|------|-----------|-------------|-------|
| `easy_debug` | — | — | Pattern-matching heuristics |
| `medium_debug` | — | — | Harder: requires execution insight |
| `hard_debug` | — | — | Most bugs resist naive fixes |

Run `python baseline_eval.py --episodes 5` to reproduce.

---

## Gradio UI

The UI has two modes:

### 🎮 Interactive Mode
- View buggy code and tests
- Run code snippets in the sandbox
- Execute test suite
- Submit bug descriptions
- Submit fixes manually
- Track rewards in real-time

### 🤖 Agent Demo Mode
- Watch the baseline agent debug step-by-step
- See action choices with rationale
- Track reward progression
- View final grader score

---

## OpenEnv Compliance

- **REST API**: `POST /reset`, `POST /step/{id}`, `GET /state/{id}`
- **WebSocket**: `/ws`
- **Schema**: `GET /schema`
- **Health**: `GET /health`
- **Config**: `openenv.yaml`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUCCESS_SCORE_THRESHOLD` | No | Score threshold for success (default: 0.3) |

---

## License

MIT
