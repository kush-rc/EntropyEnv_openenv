# inference.py  <-- MUST be at project root
# Mandatory baseline inference script for OpenEnv hackathon.
# Uses OpenAI-compatible client for HuggingFace Inference API.
#
# STDOUT FORMAT (mandatory — any deviation causes scoring failure):
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
#
# Universal model compatibility:
#   Strips <think>, <thinking>, <reasoning>, <reflection>, <thought>, <antThinking>
#   Handles unclosed thinking tags, markdown fences, prose before/after JSON
#   Type coercion for string→float, string→list, etc.

import os
import re
import json
import textwrap
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Mandatory environment variables (spec-required names) ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 8
TEMPERATURE = 0.1
MAX_TOKENS  = 400
BENCHMARK   = "EntropyEnv"

TASKS = [
    "sec_easy", "sec_medium", "sec_hard",
    "dep_easy", "dep_medium", "dep_hard",
    "cli_easy", "cli_medium", "cli_hard",
]

# ── Generic System Prompt (works for ALL LLMs) ──
SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous multi-domain analyst agent inside an RL environment.

YOUR JOB:
1. Read the observation — it contains task_type, task_subtype, task_description,
   available_actions (with parameter specs), and domain-specific data.
2. Choose the correct action from available_actions.
3. Respond with ONLY a valid JSON object. No markdown fences. No prose. No thinking tags.

DOMAIN RULES:
- security: Workflow is ALWAYS: identify_vulnerability → propose_fix → revise_fix (if feedback)
  vuln_type MUST be one of: sql_injection|xss|idor|hardcoded_secret|missing_auth|jwt_misuse|path_traversal|ssrf|rate_limit_missing|xxe
  severity MUST be: critical|high|medium|low. cvss_score: 0.0-10.0 (float).
  NEVER call identify_vulnerability twice. After identify, ALWAYS call propose_fix next.

- dependency:
  task_subtype=flag    → flag_outdated   (find deprecated packages/APIs)
  task_subtype=resolve → resolve_conflict (pick compatible versions from compatibility_matrix)
  task_subtype=migrate → migrate_api     (fix ALL graph-break IDs, include code_changes for each)

- clinical: ALWAYS follow this order: detect_gap → rank_issues → order_steps
  Use ONLY step IDs from observation.available_steps.
  risk_level MUST be: critical|high|medium|low
  If dependency_graph is present, ensure prerequisites come BEFORE dependent steps.

EXACT FORMAT EXAMPLES — copy field names exactly:
{"action_type": "identify_vulnerability", "vuln_type": "sql_injection", "cvss_score": 8.5, "severity": "critical", "affected_line": 3}
{"action_type": "propose_fix", "fix_code": "db.execute(query, (param,))", "explanation": "Use parameterized query to prevent SQL injection"}
{"action_type": "revise_fix", "fix_code": "cursor.execute(sql, values)", "addressed_feedback": "Used parameterized queries and added input validation"}
{"action_type": "flag_outdated", "packages": {"torch": "1.9.0"}, "deprecated_api": "torch.autograd.Variable", "replacement": "plain tensor"}
{"action_type": "resolve_conflict", "packages": {"torch": "2.1.0", "numpy": "1.24.0"}, "reasoning": "torch 2.1 requires numpy >=1.24"}
{"action_type": "migrate_api", "completed_items": ["break_001", "break_002", "break_003"], "code_changes": {"break_001": "use torch.where", "break_002": "use tensor.shape[0]", "break_003": "use .detach().numpy()"}}
{"action_type": "detect_gap", "missing_steps": ["pre_op_consent"], "risk_level": "critical"}
{"action_type": "rank_issues", "priority_order": ["resolve_insurance", "pre_op_consent", "book_specialist"]}
{"action_type": "order_steps", "recovery_steps": ["resolve_insurance", "complete_pre_op", "book_specialist", "schedule_surgery"]}

CRITICAL: Output ONLY the JSON object. Nothing before or after it.
""")


def build_user_prompt(step_num: int, obs: dict, history: list) -> str:
    """Build a focused user prompt from observation and history.
    Works with ALL models — keeps context compact to avoid truncation.
    """
    task_type = obs.get("task_type", "unknown")
    task_id   = obs.get("task_id", "unknown")
    task_sub  = obs.get("task_subtype", "")

    parts = [f"Step {step_num} | task_type={task_type} | task_id={task_id} | subtype={task_sub}"]

    # History summary — short to avoid confusing models
    if history:
        used = [h["action_type"] for h in history]
        last = history[-1]
        parts.append(f"Actions used so far: {used}")
        parts.append(f"Last reward: {last['reward']:.2f}")
        if last["reward"] == 0.0:
            parts.append("WARNING: Last action scored 0.0 — it was wrong or invalid. Do NOT repeat it.")
        elif last["reward"] < 0.4:
            parts.append(f"WARNING: Low score ({last['reward']:.2f}). Try a better approach.")

    # Validation failure — show prominently
    if obs.get("validation_failed"):
        parts.append(f"\nACTION VALIDATION FAILED!")
        parts.append(f"Error: {obs.get('message', 'unknown error')}")
        hint = obs.get("hint", obs.get("available_actions", ""))
        parts.append(f"Hint: {hint}")
        parts.append("Fix your JSON and try again with a VALID action.")

    # Reviewer feedback for security tasks
    if obs.get("reviewer_feedback"):
        parts.append(f"\nREVIEWER FEEDBACK (address this in your revise_fix):")
        parts.append(obs["reviewer_feedback"])

    # Full observation — separate compat matrix to avoid truncation
    obs_copy = dict(obs)
    compat = obs_copy.pop("compatibility_matrix", None)
    obs_text = json.dumps(obs_copy, default=str)
    if len(obs_text) > 1800:
        obs_text = obs_text[:1800] + "..."
    parts.append(f"\nObservation:\n{obs_text}")

    if compat:
        parts.append(f"\nCompatibility Matrix (use this to choose correct versions):\n{json.dumps(compat, indent=2)}")

    # Next action hint — helps ALL models stay on track
    if task_type == "security":
        used_types = [h["action_type"] for h in history]
        if not history or "identify_vulnerability" not in used_types:
            parts.append("\nNEXT ACTION: identify_vulnerability")
        elif "propose_fix" not in used_types:
            parts.append("\nNEXT ACTION: propose_fix")
        else:
            parts.append("\nNEXT ACTION: revise_fix (address the reviewer_feedback)")
    elif task_type == "clinical":
        used_types = [h["action_type"] for h in history]
        if "detect_gap" not in used_types:
            parts.append("\nNEXT ACTION: detect_gap")
        elif "rank_issues" not in used_types:
            parts.append("\nNEXT ACTION: rank_issues (use the step IDs from available_steps)")
        elif "order_steps" not in used_types:
            parts.append("\nNEXT ACTION: order_steps (respect dependency_graph ordering)")

    parts.append("\nOutput ONLY a single JSON object:")
    return "\n".join(parts)


def parse_action(raw_text: str) -> dict:
    """Parse LLM response into action dict.

    Universal compatibility — handles ALL known model output patterns:
    - Qwen3/DeepSeek R1: <think>...</think>{json}
    - QwQ: <reasoning>...</reasoning>{json}
    - Gemini: <thought>...</thought>{json}
    - Claude: <antThinking>...</antThinking>{json}
    - Mistral/Mixtral: plain prose before JSON
    - All models: ```json fences, unclosed tags, nested JSON
    """
    text = raw_text.strip()

    # Strip ALL known reasoning/thinking blocks (closed and unclosed)
    for tag in ["think", "thinking", "reasoning", "reflection", "thought", "antThinking"]:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if open_tag in text:
            if close_tag in text:
                # Normal case: strip everything between tags
                text = text.split(close_tag)[-1].strip()
            else:
                # Unclosed tag: take everything after the open tag and find JSON
                text = text.split(open_tag)[-1].strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Find first JSON object if text has prose before/after
    if not text.startswith("{"):
        start = text.find("{")
        if start >= 0:
            end = text.rfind("}")
            if end > start:
                text = text[start:end + 1]

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback: find outermost JSON object (handles nested braces)
    match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return {"action_type": "error", "raw": text[:100]}


def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task through the environment. Returns score in [0, 1]."""

    # Reset environment
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    data = resp.json()

    if "error" in data and not data.get("episode_id"):
        # ── MANDATORY: [START] line even on error ──
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return 0.0

    episode_id = data.get("episode_id", "unknown")
    obs = data.get("observation", data)

    # ── MANDATORY [START] — exact spec format ──
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    history = []
    step_num = 0
    last_error = None

    for step_num in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(step_num, obs, history)

        error_msg = None
        try:
            reply = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = (reply.choices[0].message.content or "").strip()
        except Exception as e:
            error_msg = str(e)
            response_text = '{"action_type": "error"}'

        action = parse_action(response_text)
        action_type = action.get("action_type", "unknown")
        action["episode_id"] = episode_id

        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_data = step_resp.json()
        except Exception as e:
            error_msg = str(e)
            # ── MANDATORY [STEP] line on connection error ──
            print(f"[STEP] step={step_num} action={action_type} reward=0.00 done=true error={error_msg}", flush=True)
            rewards.append(0.0)
            break

        reward = float(step_data.get("reward", 0.0))
        done   = bool(step_data.get("done", False))
        obs    = step_data.get("observation", step_data)
        step_error = step_data.get("error") or error_msg
        last_error = step_error

        rewards.append(reward)
        history.append({"step": step_num, "action_type": action_type, "reward": reward, "done": done})

        # Show 'invalid' for validation failures
        display_action = action_type
        if obs.get("validation_failed"):
            display_action = "invalid"

        # ── MANDATORY [STEP] — exact spec format ──
        error_val = step_error if step_error else "null"
        print(f"[STEP] step={step_num} action={display_action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

        if done:
            break

    # Score = max(rewards) — agent's best single-step performance, clamped to [0, 1]
    score = round(min(max(max(rewards) if rewards else 0.0, 0.0), 1.0), 2)
    success = score > 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── MANDATORY [END] — exact spec format ──
    print(f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}", flush=True)

    return score


def main() -> None:
    """Run all 9 tasks and report final scores."""
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Health check
    try:
        health = requests.get(f"{ENV_URL}/", timeout=10, headers={"Accept": "application/json"})
        health_data = health.json()
        print(f"Environment: {health_data.get('env', 'unknown')} | Tasks: {health_data.get('tasks', 0)}", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {ENV_URL}: {e}", flush=True)
        return

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(client, task_id)
        except Exception as e:
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            scores[task_id] = 0.0

    avg = round(sum(scores.values()) / max(len(scores), 1), 2)
    print(f"\n✅ All tasks complete! Average: {avg:.2f}", flush=True)

    # Final scores JSON — evaluator may parse this
    print(json.dumps({"final_scores": scores}), flush=True)

    # Persist results to disk
    try:
        from server.benchmark_store import append_result
        append_result(MODEL_NAME, MODEL_NAME, scores)
        print(f"💾 Results saved (avg: {avg:.4f})", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to save results to disk: {e}", flush=True)


if __name__ == "__main__":
    main()
