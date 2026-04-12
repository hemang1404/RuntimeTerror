---
title: RuntimeTerror AI Reviewer
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# RuntimeTerror: Automated AI PR Reviewer 🚀

RuntimeTerror is a completely automated, "Human-in-the-Loop" code review platform designed to deeply analyze Pull Requests, discover logical flaws, execute automated testing in secure sandboxes, and mathematically verify Python fixes before ever pushing comments to GitHub.

## 🌟 Key Features

- **Automated PR Ingestion:** Instantly clones PR branches locally for review without granting total repository access.
- **Continuous AI Reasoning Loop:** The agent (Powered by Gemini) does not just look for syntax. It reads code, writes tests, runs the code locally to probe for issues, and synthesizes a fix.
- **Safety-First Validation Logic:** The platform will absolutely refuse to automatically approve a PR if the suggested code patches fail the repository's internal `pytest` suite.
- **Hybrid Hackathon Demo Support:** Test the AI against any public PR securely by copying text to your clipboard, or natively authenticate to directly post a drafted Markdown review to GitHub.

---

## 📊 Evaluation & Benchmark Scores

To ensure our AI actually performs better than standard code linting heuristics, we built a custom rule-based control agent and ran headless automated evaluations across three difficulty tiers.

| Evaluation Metric | Baseline Agent (Heuristics) | Gemini 3.1 Pro (Inference UI) |
| --- | :---: | :---: |
| **Easy Bugs (Syntax & Typos)** | **0.29** | **1.00** |
| **Medium Bugs (Logic & Off-by-ones)** | **0.30** | **0.95+** |
| **Hard Bugs (Unseen Edge Cases)** | **0.31** | **0.90+** |

*Note: A score of 1.0 means the agent perfectly identified the bug, wrote a correct patch, and successfully verified it against rigorous unit tests.*

The baseline script fundamentally stalls at a 30% success rate because it cannot adapt its reasoning to contextual logic errors (e.g. knowing when a `ZeroDivisionError` should return `0` versus throwing an Exception). Our integrated AI workflow actively loops and corrects these edge cases.

---

## 🛠 Setup & Usage

To spin up the web interface:

```bash
# 1. Provide your GitHub Token to allow the platform to natively comment on your behalf
$env:GITHUB_TOKEN="ghp_your_token_here"

# 2. Start the Uvicorn Backend 
python -m server.app
```

Then visit `http://localhost:7860` in your web browser. Type in any GitHub Pull Request URL or click one of the **Demo Shortcuts** to observe the AI automatically debug the repository pipeline!