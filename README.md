---
title: EntropyEnv
emoji: 🌀
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🌀 EntropyEnv — Multi-Agent Dev Tools Environment.

> A multi-domain RL environment for training and evaluating AI agents on **real-world developer and clinical tasks**.
> Built for the **Scaler × Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026**.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1-blue)](https://huggingface.co/docs/openenv)
[![Tasks](https://img.shields.io/badge/Tasks-9-green)](https://huggingface.co/spaces/immortalindeed/EntropyEnv)
[![Domains](https://img.shields.io/badge/Domains-3-purple)]()
[![Cases](https://img.shields.io/badge/Ground--Truth%20Cases-39-orange)]()

---

## 💡 Why This Environment?

Most RL benchmarks test agents on **static, single-turn tasks** — classify this image, answer this question. But real developer workflows are **multi-turn, iterative, and require revision**:

- A security reviewer doesn't just find a bug — they **identify → propose a fix → revise after feedback**
- A DevOps engineer doesn't just flag outdated packages — they **resolve version conflicts across an entire dependency graph**
- A clinical coordinator doesn't just spot missing steps — they **prioritize by urgency and plan a dependency-safe recovery**

**No existing RL environment tests agents on this full identify → act → revise cycle.** EntropyEnv fills that gap with 9 tasks across 3 real-world domains, progressive difficulty, rich partial-credit scoring, and iterative multi-turn episodes.

---

## 🎯 What Is This?


EntropyEnv is a **training gym for AI agents** — not the agent itself.
Think of it like a driving test course: we build the course, and different AI "drivers" take the test.

An AI agent connects via API, receives a **task** (e.g., "find the vulnerability in this code"), sends back an **action** (its answer), and gets a **reward score** based on how good the answer is.

```
                    POST /reset
AI Agent  ────────────────────────►  EntropyEnv
                                     │
                                     ├── Picks a task case from the dataset
                                     ├── Returns: observation (the problem)
          ◄────────────────────────  │
                                     │
                    POST /step       │
          ────────────────────────►  │
                                     ├── Validates the action (3 stages)
                                     ├── Grades it (domain-specific grader)
          ◄────────────────────────  ├── Returns: reward + done + next observation
                                     │
             (repeat until done)     │
```

---

## 🏗️ Three Domains, Nine Tasks

### 🔒 Domain 1: MCP Security Auditing

Agents identify vulnerabilities in code snippets, propose secure fixes, and iteratively revise based on adversarial reviewer feedback.

| Task | Difficulty | What the Agent Does |
|------|-----------|---------------------|
| `sec_easy` | 🟢 Easy | Classify a single vulnerability (type, CVSS, severity) |
| `sec_medium` | 🟡 Medium | Identify → propose a code fix |
| `sec_hard` | 🔴 Hard | Identify → fix → revise with adversarial reviewer feedback |

**Coverage:** SQL injection, XSS, IDOR, hardcoded secrets, missing auth, JWT misuse, path traversal, SSRF, XXE

### 📦 Domain 2: PyTorch Migration Time-Machine

Agents detect deprecated APIs, resolve version conflicts using compatibility matrices, and fix `torch.compile` graph-break patterns in dependency order.

| Task | Difficulty | What the Agent Does |
|------|-----------|---------------------|
| `dep_easy` | 🟢 Easy | Flag outdated packages and deprecated API usage |
| `dep_medium` | 🟡 Medium | Resolve version conflicts across package constraints |
| `dep_hard` | 🔴 Hard | Fix torch.compile graph-breaks in correct dependency order |

**Coverage:** Variable, cuda(), DataParallel, ONNX export, torch.compile, vmap, torch.export

### 🏥 Domain 3: Clinical Workflow Chaos Simulator

Agents detect missing steps in hospital workflows, rank them by clinical priority, and plan dependency-ordered recovery sequences.

| Task | Difficulty | What the Agent Does |
|------|-----------|---------------------|
| `cli_easy` | 🟢 Easy | Detect missing workflow steps and assess risk |
| `cli_medium` | 🟡 Medium | Detect gaps → rank by clinical priority |
| `cli_hard` | 🔴 Hard | Detect → rank → plan dependency-safe recovery |

**Coverage:** Surgery prep, ER triage, chemotherapy, cardiac emergency, blood transfusion, organ transplant, stroke code

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Partial-Credit Scoring** | F1, NDCG, weighted multi-component grading — not binary pass/fail |
| 🔄 **Multi-Turn Episodes** | Agents iterate through identify → act → revise workflows |
| 🛡️ **3-Stage Validation** | Schema → Domain → Consistency checks with helpful error hints |
| 📊 **Score Breakdown** | Per-component feedback in every step so agents learn *what* to improve |
| 🏎️ **Fatal Error Handling** | Automatic 402/401/403 detection stops wasted API calls immediately |
| 🌐 **Universal LLM Support** | Works with any OpenAI-compatible model (Qwen, Llama, DeepSeek, Gemini, etc.) |
| 🐳 **Docker-Ready** | One-command deploy to Hugging Face Spaces |
| 📈 **GRPO-Compatible** | Smooth reward gradients designed for policy optimization training |

---

## 📡 API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Health check | Returns status and available tasks |
| `POST /reset` | Start episode | `{"task_id": "sec_easy"}` → `{episode_id, observation}` |
| `POST /step` | Submit action | `{episode_id, action_type, ...}` → `{reward, done, observation}` |
| `GET /state` | Query state | `?episode_id=xxx` → current episode info |
| `GET /debug` | Debug panel | Interactive HTML benchmark runner |
| `GET /web` | Gradio UI | Full task browser with run history |

### Quick Example

```python
import requests

# 1. Start an episode
resp = requests.post("http://localhost:7860/reset", json={"task_id": "sec_easy"})
data = resp.json()
episode_id = data["episode_id"]
observation = data["observation"]

print(observation["task_description"])
# → "Identify the SQL injection vulnerability in this code snippet."

# 2. Send an action
action = {
    "episode_id": episode_id,
    "action_type": "identify_vulnerability",
    "vuln_type": "sql_injection",
    "cvss_score": 9.1,
    "severity": "critical",
    "affected_line": 3
}
result = requests.post("http://localhost:7860/step", json=action).json()

print(f"Reward: {result['reward']}, Done: {result['done']}")
# → Reward: 0.85, Done: true
```

---

## 🚀 Getting Started

### Run Locally

```bash
# Install dependencies
pip install fastapi uvicorn openai requests packaging gradio python-dotenv

# Start the environment
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t entropyenv .
docker run -p 7860:7860 entropyenv
```

### Run the Baseline Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --repo-id <username>/EntropyEnv
```

---

## 🏛️ Project Structure

```
entropyenv/
├── inference.py                # Baseline agent with smart prompt engineering
├── openenv.yaml                # OpenEnv manifest (9 tasks)
├── pyproject.toml              # Package configuration
├── Dockerfile                  # Multi-stage Docker build
├── server/
│   ├── app.py                  # FastAPI server with session management
│   ├── router.py               # Task dispatcher with Counter-based sequence checking
│   ├── session.py              # Episode state management
│   ├── web_ui.py               # Gradio UI with performance dashboard
│   ├── demo_agent.py           # Rule-based demo agent
│   ├── benchmark_store.py      # Persistent results storage
│   ├── debug_panel.html        # Interactive debug interface
│   ├── validation/
│   │   └── validator.py        # 3-stage validation with type-casting
│   ├── graders/
│   │   ├── base_grader.py      # Universal reward pipeline
│   │   ├── security_grader.py  # Security domain grader
│   │   ├── dependency_grader.py # Dependency domain grader
│   │   └── clinical_grader.py  # Clinical domain grader
│   └── datasets/
│       ├── security_cases.py   # 13 ground-truth security cases
│       ├── dependency_cases.py # 13 ground-truth dependency cases
│       └── clinical_cases.py   # 13 ground-truth clinical cases
└── results/
    └── run_history.json        # Benchmark history (auto-created)
```

---

## 📈 Baseline Performance

> **Note:** Scores below are from the latest grading revision (v3: weighted 0.60×max + 0.40×mean scoring, difficulty_multiplier removed, dep_hard done-condition fixed). Re-benchmarking across 14+ models in progress.

| Model | Provider | sec_easy | sec_med | sec_hard | dep_easy | dep_med | dep_hard | cli_easy | cli_med | cli_hard | **Avg** |
|-------|----------|:--------:|:-------:|:--------:|:--------:|:-------:|:--------:|:--------:|:-------:|:--------:|:-------:|
| *Benchmarking in progress...* | | | | | | | | | | | |

**Scoring formula:** `score = 0.60 × max(step_rewards) + 0.40 × mean(step_rewards)`, clamped to `[0.01, 0.99]`

**Design principles:**
- 🎯 **No artificial difficulty caps** — scores reflect actual grader correctness
- 📊 **Weighted blend** — rewards consistently good episodes over single-lucky-step flukes
- 🔬 **Spec-compliant** — `[END]` lines perfectly match the 3-line format mandatory rules
- 🧠 **14+ model families tested** for universal compatibility

---

## 📝 Inference Log Format

The baseline `inference.py` emits structured logs matching the OpenEnv spec:

```
[START] task=sec_easy env=EntropyEnv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=identify_vulnerability reward=0.85 done=false error=null
[STEP] step=2 action=propose_fix reward=0.92 done=true error=null
[END] success=true steps=2 score=0.89 rewards=0.85,0.92
```

---

## 🤝 Built With

- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance async API framework
- **[Gradio](https://gradio.app/)** — Interactive web UI for testing and visualization
- **[PyTorch](https://pytorch.org/)** — Domain expertise for migration tasks
- **[OpenEnv](https://huggingface.co/docs/openenv)** — Standardized RL environment specification

---

<p align="center">
  <b>Built with ❤️ for the Scaler × Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026</b>
</p>
