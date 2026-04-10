# inference.py  <-- MUST be at project root
# Mandatory baseline inference script for OpenEnv hackathon.
# Uses OpenAI-compatible client for HuggingFace Inference API.
#
# STDOUT FORMAT (mandatory — any deviation causes scoring failure):
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

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

# ── Mandatory environment variables ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 8
TEMPERATURE = 0.1
MAX_TOKENS  = 400
BENCHMARK   = "EntropyEnv"

# ── FATAL error codes: stop the entire task immediately, don't loop ──
# 402 = payment required, 401 = unauthorized, 403 = forbidden
# 429 = rate limit (stop task, not whole run), 503 = model unavailable
FATAL_HTTP_CODES = {402, 401, 403}
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
MAX_CONSECUTIVE_ERRORS = 3  # stop task after 3 consecutive API errors

TASKS = [
    "sec_easy", "sec_medium", "sec_hard",
    "dep_easy", "dep_medium", "dep_hard",
    "cli_easy", "cli_medium", "cli_hard",
]

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


def _extract_http_code(error_str: str) -> int:
    """Extract HTTP status code from error message string. Returns 0 if not found."""
    # Matches patterns like "Error code: 402" or "status_code=402" or "HTTP 402"
    match = re.search(r'(?:Error code:|status_code=|HTTP )\s*(\d{3})', str(error_str))
    if match:
        return int(match.group(1))
    # Also check for bare 4xx/5xx at start of error
    match = re.search(r'\b(4\d{2}|5\d{2})\b', str(error_str))
    if match:
        return int(match.group(1))
    return 0


def _is_fatal_error(error_str: str) -> bool:
    """Return True if this error means we should stop ALL tasks (not just this one)."""
    code = _extract_http_code(error_str)
    if code in FATAL_HTTP_CODES:
        return True
    # Also catch keyword patterns
    fatal_keywords = ['insufficient credits', 'unauthorized', 'invalid api key',
                      'authentication failed', 'no api key', 'forbidden']
    err_lower = str(error_str).lower()
    return any(kw in err_lower for kw in fatal_keywords)


def _is_task_fatal_error(error_str: str) -> bool:
    """Return True if this error means we should stop THIS task but try others."""
    code = _extract_http_code(error_str)
    if code in RETRYABLE_HTTP_CODES:
        return True
    task_fatal_keywords = ['model not found', 'model unavailable', 'context length',
                           'maximum context', 'rate limit']
    err_lower = str(error_str).lower()
    return any(kw in err_lower for kw in task_fatal_keywords)


def build_user_prompt(step_num: int, obs: dict, history: list) -> str:
    task_type = obs.get("task_type", "unknown")
    task_id   = obs.get("task_id", "unknown")
    task_sub  = obs.get("task_subtype", "")

    parts = [f"Step {step_num} | task_type={task_type} | task_id={task_id} | subtype={task_sub}"]

    if history:
        used = [h["action_type"] for h in history]
        last = history[-1]
        parts.append(f"Actions used: {used}")
        parts.append(f"Last reward: {last['reward']:.2f}")
        if last["reward"] < 0.4:
            parts.append(f"⚠️ Low score. Try different approach.")

    if obs.get("validation_failed"):
        parts.append(f"\n❌ VALIDATION FAILED!")
        parts.append(f"Error: {obs.get('message', 'unknown')}")
        parts.append(f"Fix: {obs.get('hint', '')}")

    if obs.get("reviewer_feedback"):
        parts.append(f"\n📝 REVIEWER FEEDBACK:")
        parts.append(obs["reviewer_feedback"])

    obs_copy = dict(obs)
    compat_matrix = obs_copy.pop("compatibility_matrix", None)
    dep_graph = obs_copy.pop("dependency_graph", None)

    core_text = json.dumps(obs_copy, default=str, indent=2)
    parts.append(f"\nObservation:\n{core_text}")

    if compat_matrix:
        parts.append(f"\nCompatibility Matrix (use this to resolve conflicts):")
        for pkg, versions in compat_matrix.items():
            parts.append(f"  {pkg}:")
            for ver, deps in versions.items():
                if deps:
                    parts.append(f"    {ver} → requires {deps}")
                else:
                    parts.append(f"    {ver} → (no deps)")

    if dep_graph:
        parts.append(f"\nDependency Graph (prerequisites must come first):")
        for step, prereqs in dep_graph.items():
            if prereqs:
                parts.append(f"  {step} requires: {prereqs}")
            else:
                parts.append(f"  {step} → (no prereqs)")

    if task_type == "security":
        used_types = [h["action_type"] for h in history]
        if not used_types or "identify_vulnerability" not in used_types:
            parts.append("\n➡️ NEXT: identify_vulnerability")
        elif "propose_fix" not in used_types:
            parts.append("\n➡️ NEXT: propose_fix")
        else:
            parts.append("\n➡️ NEXT: revise_fix (address reviewer_feedback)")

    elif task_type == "clinical":
        used_types = [h["action_type"] for h in history]
        if "detect_gap" not in used_types:
            parts.append("\n➡️ NEXT: detect_gap")
        elif "rank_issues" not in used_types:
            parts.append("\n➡️ NEXT: rank_issues")
        elif "order_steps" not in used_types:
            parts.append("\n➡️ NEXT: order_steps (respect dependency_graph)")

    parts.append("\n📤 Output ONLY a single JSON object:")
    return "\n".join(parts)


def parse_action(raw_text: str) -> dict:
    """Parse LLM response into action dict. Universal model compatibility."""
    text = raw_text.strip()

    for tag in ["think", "thinking", "reasoning", "reflection", "thought", "antThinking"]:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if open_tag in text:
            if close_tag in text:
                text = text.split(close_tag)[-1].strip()
            else:
                text = text.split(open_tag)[-1].strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

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

    match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return {"action_type": "error", "raw": text[:100]}


def run_task(client: OpenAI, task_id: str) -> tuple:
    """Run a single task. Returns (score, is_fatal_api_error).

    is_fatal_api_error=True means the caller should stop ALL remaining tasks.
    """
    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
        return 0.01, False

    if "error" in data and not data.get("episode_id"):
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
        return 0.01, False

    episode_id = data.get("episode_id", "unknown")
    obs = data.get("observation", data)

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    history = []
    step_num = 0
    consecutive_errors = 0

    for step_num in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(step_num, obs, history)

        error_msg = None
        fatal_error = False
        task_fatal = False

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
            consecutive_errors = 0  # reset on success

        except Exception as e:
            error_msg = str(e)
            response_text = '{"action_type": "error"}'
            consecutive_errors += 1

            # Check if this is a fatal error (auth/payment) — stop everything
            if _is_fatal_error(error_msg):
                fatal_error = True
                short_err = error_msg[:120].replace('\n', ' ')
                print(f"[STEP] step={step_num} action=invalid reward=0.01 done=true error=FATAL:{short_err}", flush=True)
                rewards.append(0.01)
                step_num_final = step_num
                break

            # Check if this is a task-level fatal (rate limit, model unavailable)
            if _is_task_fatal_error(error_msg) or consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                task_fatal = True
                short_err = error_msg[:120].replace('\n', ' ')
                print(f"[STEP] step={step_num} action=invalid reward=0.01 done=true error=TASK_STOP:{short_err}", flush=True)
                rewards.append(0.01)
                step_num_final = step_num
                break

        action = parse_action(response_text)
        action_type = action.get("action_type", "unknown")
        action["episode_id"] = episode_id

        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_data = step_resp.json()
        except Exception as e:
            short_err = str(e)[:100]
            print(f"[STEP] step={step_num} action={action_type} reward=0.01 done=true error={short_err}", flush=True)
            rewards.append(0.01)
            step_num_final = step_num
            fatal_error = False
            break

        reward = float(step_data.get("reward", 0.0))
        done   = bool(step_data.get("done", False))
        obs    = step_data.get("observation", step_data)
        step_error = step_data.get("error") or error_msg

        rewards.append(reward)
        history.append({"step": step_num, "action_type": action_type, "reward": reward, "done": done})

        display_action = action_type
        if obs.get("validation_failed"):
            display_action = "invalid"

        error_val = step_error if step_error else "null"
        # Truncate long error messages in output
        if error_val and error_val != "null" and len(str(error_val)) > 150:
            error_val = str(error_val)[:150] + "..."

        print(f"[STEP] step={step_num} action={display_action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

        step_num_final = step_num

        if done:
            fatal_error = False
            break
    else:
        step_num_final = step_num
        fatal_error = False

    avg_reward = sum(rewards) / max(len(rewards), 1) if rewards else 0.01
    score = round(min(max(avg_reward, 0.01), 0.99), 4)
    success = score > 0.01
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={step_num_final} score={score:.2f} rewards={rewards_str}", flush=True)

    return score, fatal_error


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
            score, is_fatal = run_task(client, task_id)
            scores[task_id] = score

            # If we hit a fatal API error (402/401/403), stop ALL remaining tasks
            if is_fatal:
                print(f"\n🚫 Fatal API error on {task_id}. Stopping all remaining tasks.", flush=True)
                print(f"   Likely cause: invalid token, no credits, or unauthorized access.", flush=True)
                # Fill remaining tasks with 0.01
                for remaining in TASKS:
                    if remaining not in scores:
                        scores[remaining] = 0.01
                        print(f"[START] task={remaining} env={BENCHMARK} model={MODEL_NAME}", flush=True)
                        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
                break

        except Exception as e:
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
            scores[task_id] = 0.01

    avg = round(sum(scores.values()) / max(len(scores), 1), 2)
    print(f"\n✅ All tasks complete! Average: {avg:.2f}", flush=True)
    print(json.dumps({"final_scores": scores}), flush=True)

    try:
        from server.benchmark_store import append_result
        append_result(MODEL_NAME, MODEL_NAME, scores)
        print(f"💾 Results saved (avg: {avg:.4f})", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to save results to disk: {e}", flush=True)


if __name__ == "__main__":
    main()
