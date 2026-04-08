# 🛠️ EntropyEnv: Multi-Agent Dev Tools Environment

> A multi-domain RL environment for training and evaluating AI agents on **real-world developer and clinical tasks**.
> Built for the **Scaler × Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026**.

---

## 💡 Why This Environment?

Most existing RL benchmarks test agents on **static, single-turn tasks** — classify this image, answer this question. But real developer workflows are **multi-turn, iterative, and require revision**:

- A security reviewer doesn't just find a bug — they **identify → propose a fix → revise after feedback**
- A DevOps engineer doesn't just flag outdated packages — they **resolve version conflicts across an entire dependency graph**
- A clinical coordinator doesn't just spot missing steps — they **prioritize by urgency and plan a dependency-safe recovery**

**No existing RL environment tests agents on this full identify → act → revise cycle.** This environment fills that gap by providing 9 tasks across 3 real-world domains with progressive difficulty, rich partial-credit scoring, and iterative multi-turn episodes.

**Who would use this?** Teams training AI coding assistants (code review bots), dependency management agents (Dependabot-like systems), and clinical decision support systems.

---

## 🎯 What Is This?

This is a **training gym for AI agents** — not the agent itself.
Think of it like a driving test course: you build the course, and different AI "drivers" take the test.

An AI agent connects to this environment via API, receives a **task** (e.g., "find the vulnerability in this code"), sends back an **action** (its answer), and gets a **reward score** (0.0 – 1.0) based on how good the answer is.

```
                    POST /reset
AI Agent  ────────────────────────►  This Environment
                                     │
                                     ├── Picks a task case
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

Agents must identify vulnerabilities in code snippets, propose fixes, and iteratively revise based on reviewer feedback.

| Task | Difficulty | Subtype | Max Steps | Threshold | Actions |
|------|-----------|---------|-----------|-----------|---------|
| `sec_easy` | Easy | `single` | 4 | 0.80 | `identify_vulnerability` |
| `sec_medium` | Medium | `multi` | 6 | 0.75 | `identify` → `propose_fix` → `revise_fix` |
| `sec_hard` | Hard | `adversarial` | 8 | 0.70 | `identify` → `propose_fix` → `revise_fix` (reviewer) |

**Dataset:** 10 ground-truth cases covering SQL injection, XSS, IDOR, hardcoded secrets, missing auth, JWT misuse, path traversal, SSRF.

### 📦 Domain 2: PyTorch Migration Time-Machine

Agents must detect deprecated APIs, resolve version conflicts, and fix `torch.compile` graph-break patterns.

| Task | Difficulty | Subtype | Max Steps | Threshold | Actions |
|------|-----------|---------|-----------|-----------|---------|
| `dep_easy` | Easy | `flag` | 4 | 0.80 | `flag_outdated` |
| `dep_medium` | Medium | `resolve` | 6 | 0.75 | `resolve_conflict` |
| `dep_hard` | Hard | `migrate` | 8 | 0.70 | `migrate_api` / `validate_tree` |

**Dataset:** 10 ground-truth cases covering Variable, cuda(), DataParallel, ONNX export, torch.compile graph-breaks.

### 🏥 Domain 3: Clinical Workflow Chaos Simulator

Agents must detect missing steps in hospital workflows, rank them by priority, and plan dependency-ordered recovery sequences.

| Task | Difficulty | Max Steps | Threshold | Actions |
|------|-----------|-----------|-----------|---------|
| `cli_easy` | Easy | 4 | 0.80 | `detect_gap` |
| `cli_medium` | Medium | 6 | 0.75 | `detect_gap` → `rank_issues` |
| `cli_hard` | Hard | 6 | 0.70 | `detect_gap` → `rank_issues` → `order_steps` |

**Dataset:** 10 ground-truth cases covering surgery prep, ER triage, chemotherapy, cardiac emergency, blood transfusion.

---

## 📊 Observation & Action Spaces

### Observation Space

Every observation includes these core fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `str` | Domain: `security`, `dependency`, or `clinical` |
| `task_id` | `str` | Task identifier (e.g., `sec_easy`) |
| `task_subtype` | `str` | Variant: `single`, `multi`, `flag`, `resolve`, `migrate` |
| `task_description` | `str` | Human-readable problem description |
| `available_actions` | `list[dict]` | Valid actions with parameter specs |
| `turn` | `int` | Current step number |
| `done` | `bool` | Whether episode has ended |

Domain-specific fields are added (e.g., `code_snippet` for security, `compatibility_matrix` for dependency, `events` and `dependency_graph` for clinical).

### Action Space

Actions are JSON objects with `action_type` and domain-specific parameters:

```json
{"action_type": "identify_vulnerability", "vuln_type": "sql_injection", "cvss_score": 8.5, "severity": "critical", "affected_line": 3}
{"action_type": "propose_fix", "fix_code": "db.execute(query, (param,))", "explanation": "Use parameterized queries"}
{"action_type": "flag_outdated", "packages": {"torch": "1.9.0"}, "deprecated_api": "torch.autograd.Variable", "replacement": "plain tensor"}
{"action_type": "detect_gap", "missing_steps": ["pre_op_consent"], "risk_level": "critical"}
```

---

## 📊 Scoring System

### Two-Layer Grading Architecture

**Layer 1: `base_grader.py`** — Universal reward pipeline applied to ALL domains:

```
reward = safe_score(correctness + repetition_penalty + harmful_penalty + efficiency_bonus)
```

| Component | Formula | Range |
|-----------|---------|-------|
| `compute_correctness()` | Domain-specific (see below) | 0.0 – 1.0 |
| `repetition_penalty` | −0.15 × count(same action in last 3 turns) | −0.45 – 0.0 |
| `harmful_output_penalty` | −0.30 if forbidden pattern detected | −0.30 – 0.0 |
| `efficiency_bonus` | +0.10 if `correctness >= 0.8` and early finish | 0.0 – 0.10 |
| `safe_score()` | `clamp(score, 0.0, 1.0)` | 0.0 – 1.0 |

**Layer 2: Domain-specific graders:**

#### Security Grader
| Action | Component | Weight |
|--------|-----------|--------|
| `identify_vulnerability` | vuln_type match | ×0.45 |
| `identify_vulnerability` | CVSS in range (partial: ±3.0) | ×0.30 |
| `identify_vulnerability` | severity match (adjacent: ×0.40) | ×0.25 |
| `propose_fix` | token coverage + identifier preserved (floor: 0.25) | up to 1.15 |
| `revise_fix` | feedback keyword coverage − regression (floor: 0.20) | 0.0 – 1.0 |

#### Dependency Grader
| Action | Formula |
|--------|---------|
| `flag_outdated` | F1 × 0.55 + deprecated_api_match × 0.45 |
| `resolve_conflict` | valid_pkgs / conflict_count + tree_bonus(0.15) − downgrade(0.10) |
| `migrate_api` | order_score × 0.30 + completeness × 0.40 + fix_quality × 0.30 |

#### Clinical Grader
| Action | Formula |
|--------|---------|
| `detect_gap` | F1(predicted, expected) × 0.65 + risk_match × 0.35 |
| `rank_issues` | completeness × 0.40 + NDCG@k × 0.60 |
| `order_steps` | order_violations × 0.40 + completeness × 0.40 + efficiency × 0.20 |

### GRPO Training Signal Quality

This environment is specifically designed for **Group Relative Policy Optimization**:

- **Smooth reward ramp** — Scores transition smoothly from 0.0 → 1.0, never binary
- **Partial credit everywhere** — F1 scoring, NDCG ranking, adjacent-severity credit
- **Progressive penalty learning** — Schema penalty (−0.20), repetition (−0.15), harmful (−0.30)
- **Efficiency bonus** — Agents learn to solve faster by finishing early
- **Floor scores** — Valid workflow attempts always get minimum credit (0.20–0.25)

---

## 🔐 Validation (3 Stages)

Every action goes through 3-stage validation before reaching the grader:

1. **Schema** — Required fields present? Correct types? (Auto-casts `"8.5"` → `8.5`)
2. **Domain** — Is `vuln_type` in the valid set? Is `cvss_score` in [0, 10]?
3. **Consistency** — Is `revise_fix` called after `propose_fix`? No identical repeats?

If validation fails, the agent gets a **rich feedback observation** (not just 0.0):
```json
{
  "validation_failed": true,
  "error_type": "domain_error",
  "message": "cvss_score 12.5 out of range",
  "hint": "cvss_score must be a float between 0.0 and 10.0",
  "available_actions": ["identify_vulnerability", "propose_fix", "revise_fix"]
}
```

---

## 🏛️ Architecture

```
project-root/
├── inference.py                # Baseline agent (OpenAI-compatible, spec-compliant logs)
├── openenv.yaml                # OpenEnv manifest (9 tasks declared)
├── pyproject.toml              # Python package config with openenv-core dependency
├── Dockerfile                  # Docker build for HF Spaces (port 7860)
├── server/
│   ├── app.py                  # FastAPI endpoints: /, /reset, /step, /state, /debug
│   ├── router.py               # Central dispatcher: observations, done conditions, score_details
│   ├── session.py              # In-memory session state management
│   ├── benchmark_store.py      # Persistent JSON results store (survives restarts)
│   ├── demo_agent.py           # Rule-based demo agent for Gradio UI
│   ├── web_ui.py               # Gradio UI with task runner and history
│   ├── debug_panel.html        # Interactive HTML debug panel
│   ├── validation/
│   │   └── validator.py        # 3-stage validation: Schema → Domain → Consistency
│   ├── graders/
│   │   ├── base_grader.py      # safe_score, grade_dynamic, penalties, bonuses
│   │   ├── security_grader.py  # Vuln detection, fix quality, feedback coverage
│   │   ├── dependency_grader.py # F1 scoring, version checking, graph ordering
│   │   └── clinical_grader.py  # F1, NDCG ranking, dependency-violation counting
│   └── datasets/
│       ├── security_cases.py   # 10 cases: SQL injection, XSS, IDOR, SSRF, etc.
│       ├── dependency_cases.py # 10 cases: Variable, cuda(), DataParallel, graph-breaks
│       └── clinical_cases.py   # 10 cases: surgery prep, ER triage, chemo, cardiac
└── results/
    └── run_history.json        # Persistent benchmark results (auto-created)
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Health check | Returns status, task list, spec version |
| `POST /reset` | Start episode | `{"task_id": "sec_easy"}` → `{episode_id, observation}` |
| `POST /step` | Submit action | `{episode_id, action_type, ...}` → `{reward, done, observation}` |
| `GET /state` | Query state | `?episode_id=xxx` → `{step_count, done, reward_acc}` |
| `GET /debug` | Debug panel | Interactive HTML benchmark runner |
| `GET /web` | Gradio UI | Full task browser with run history |

### Step Response Format

```json
{
  "episode_id": "uuid-string",
  "step_count": 2,
  "reward": 0.75,
  "done": false,
  "observation": {
    "task_type": "security",
    "task_id": "sec_easy",
    "task_subtype": "single",
    "task_description": "Identify the SQL injection vulnerability...",
    "turn": 1,
    "done": false,
    "available_actions": [...]
  },
  "score_details": {
    "vuln_type_match": 1.0,
    "cvss_in_range": 1.0,
    "severity_match": 0.0
  }
}
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- `pip install fastapi uvicorn openai requests packaging gradio python-dotenv`

### Running Locally

```bash
# 1. Start the environment server
cd multi-agent-dev-tools-env
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. Run baseline inference (in another terminal)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t multi-agent-dev-tools-env .
docker run -p 7860:7860 multi-agent-dev-tools-env
```

### Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --repo-id <username>/multi-agent-dev-tools-env
```

---

## 📝 Mandatory Log Format

The `inference.py` emits structured stdout logs matching the spec exactly:

```
[START] task=sec_easy env=multi-agent-dev-tools-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=identify_vulnerability reward=0.85 done=false error=null
[STEP] step=2 action=propose_fix reward=1.00 done=true error=null
[END] success=true steps=2 score=1.00 rewards=0.85,1.00
```

### Environment Variables (Required)

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | API key / HF token | `hf_xxxxx` or `gsk_xxxxx` |
| `ENV_URL` | Environment URL | `http://localhost:7860` |

---

## 📈 Baseline Scores

Tested with multiple model families for universal compatibility:

| Model | Family | Parameters | Average Score |
|-------|--------|------------|---------------|
| Llama 3.3 70B | Meta | 70B | **0.97** |
| Qwen3-32B | Alibaba | 32B | **0.99** |
| DeepSeek V3.2 | DeepSeek | MoE | **0.96** |

The environment provides smooth reward gradients that enable GRPO training of smaller models (8B+).

---

## 🔧 Key Design Decisions

1. **Data-driven done conditions** — `completion_threshold` and `required_sequence` stored per case
2. **Universal model compatibility** — Strips `<think>`, `<reasoning>`, `<antThinking>` etc.
3. **Type-casting validator** — Auto-converts `"8.5"` → `8.5` before rejecting
4. **Floor scores** — Valid workflow attempts always get minimum credit
5. **Deterministic case selection** — `hash(episode_id) % len(cases)` for reproducibility
6. **Compatibility matrix separation** — Prevents context truncation for large observations
7. **Patch-level version fuzzy** — `2.1.1` matches `2.1.0` by major.minor
8. **Hallucination filter** — `_score_rank` filters step IDs not in `available_steps`
9. **Persistent results** — `benchmark_store.py` writes to disk, survives restarts
10. **Robust dependency fallback** — Works without `packaging` module via manual version parsing

---

## ☑️ Compliance Checklist

### Phase 1: Automated Validation (Pass/Fail)
- [x] HF Space deploys and responds to `GET /`
- [x] `openenv.yaml` present with all 9 task IDs
- [x] `POST /reset` returns `episode_id` + `observation` for all 9 tasks
- [x] `POST /step` returns `reward` (float, 0.0–1.0) + `done` (bool) + `observation`
- [x] `GET /state` returns episode state
- [x] All endpoints return HTTP 200 (never 500)
- [x] `Dockerfile` at project root, builds cleanly
- [x] `inference.py` at project root, runs under 20 min
- [x] `openenv validate` passes

### Phase 2: Agentic Evaluation (Scored)
- [x] Observations include `task_type`, `task_subtype`, `task_description`, `available_actions`
- [x] Partial credit graders (F1, NDCG, weighted sub-scores) — not binary
- [x] Score variance across 9 tasks (varied difficulty = varied scores)
- [x] `score_details` in step response for grading transparency
- [x] `safe_score()` clamps all rewards to [0.0, 1.0]

### Phase 3: Human Review
- [x] 3 real-world domains (security, dependency, clinical)
- [x] Multi-turn iterative workflows (identify → fix → revise)
- [x] Rich validation hints for agent learning
- [x] Debug panel with benchmark runner UI
- [x] GRPO-compatible reward shaping
