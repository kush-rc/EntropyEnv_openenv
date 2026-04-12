# inference.py  <-- MUST be at project root
# Mandatory baseline inference script for OpenEnv hackathon.
# Uses OpenAI-compatible client for HuggingFace Inference API.
#
# OFFICIAL STDOUT FORMAT (from Meta_OpenEnv_Hackathon__Guidelines.txt):
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
#
# KEY RULES FROM OFFICIAL SPEC:
#   - reward and rewards formatted to 2 decimal places ONLY
#   - done and success are lowercase booleans: true or false
#   - error is null when no error (the literal string "null")
#   - NO score= field in [END] — not in the official spec
#   - NO task_id=, NO episode_id=, NO total_reward= — none of these are in spec
#   - rewards= is a comma-separated list of step rewards with NO spaces

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

# ── Mandatory environment variables (names exactly as spec requires) ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

MAX_STEPS   = 8
TEMPERATURE = 0.1
MAX_TOKENS  = 400
BENCHMARK   = "EntropyEnv"

# Fatal HTTP codes: stop ALL tasks immediately
FATAL_HTTP_CODES    = {402, 401, 403}
RETRYABLE_CODES     = {429, 500, 502, 503, 504}
MAX_CONSEC_ERRORS   = 3

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
{"action_type": "migrate_api", "completed_items": ["break_001", "break_002"], "code_changes": {"break_001": "use torch.where", "break_002": "use tensor.shape[0]"}}
{"action_type": "detect_gap", "missing_steps": ["pre_op_consent"], "risk_level": "critical"}
{"action_type": "rank_issues", "priority_order": ["resolve_insurance", "pre_op_consent", "book_specialist"]}
{"action_type": "order_steps", "recovery_steps": ["resolve_insurance", "complete_pre_op", "book_specialist", "schedule_surgery"]}

CRITICAL: Output ONLY the JSON object. Nothing before or after it.
""")


def _extract_http_code(error_str: str) -> int:
    match = re.search(r'(?:Error code:|status_code=|HTTP )\s*(\d{3})', str(error_str))
    if match:
        return int(match.group(1))
    match = re.search(r'\b(4\d{2}|5\d{2})\b', str(error_str))
    if match:
        return int(match.group(1))
    return 0


def _is_fatal_error(error_str: str) -> bool:
    code = _extract_http_code(error_str)
    if code in FATAL_HTTP_CODES:
        return True
    fatal_kw = ['insufficient credits', 'unauthorized', 'invalid api key',
                'authentication failed', 'no api key', 'forbidden']
    return any(kw in str(error_str).lower() for kw in fatal_kw)


def _is_task_fatal(error_str: str) -> bool:
    code = _extract_http_code(error_str)
    if code in RETRYABLE_CODES:
        return True
    task_kw = ['model not found', 'model unavailable', 'context length',
               'maximum context', 'rate limit']
    return any(kw in str(error_str).lower() for kw in task_kw)


def build_user_prompt(step_num: int, obs: dict, history: list) -> str:
    task_type = obs.get("task_type", "unknown")
    task_id   = obs.get("task_id",   "unknown")
    task_sub  = obs.get("task_subtype", "")

    parts = [f"Step {step_num} | task_type={task_type} | task_id={task_id} | subtype={task_sub}"]

    if history:
        used = [h["action_type"] for h in history]
        last = history[-1]
        parts.append(f"Actions used: {used}")
        parts.append(f"Last reward: {last['reward']:.2f}")
        if last["reward"] < 0.40:
            parts.append("⚠️ Low score. Try a different approach.")

    if obs.get("validation_failed"):
        parts.append(f"\n❌ VALIDATION FAILED!")
        parts.append(f"Error: {obs.get('message', 'unknown')}")
        parts.append(f"Fix: {obs.get('hint', '')}")

    if obs.get("reviewer_feedback"):
        parts.append(f"\n📝 REVIEWER FEEDBACK:")
        parts.append(obs["reviewer_feedback"])

    obs_copy = dict(obs)
    compat   = obs_copy.pop("compatibility_matrix", None)
    dep_g    = obs_copy.pop("dependency_graph", None)

    core_text = json.dumps(obs_copy, default=str)
    if len(core_text) > 1600:
        core_text = core_text[:1600] + "..."
    parts.append(f"\nObservation:\n{core_text}")

    if compat:
        parts.append("\nCompatibility Matrix (use this to resolve conflicts):")
        for pkg, versions in compat.items():
            for ver, deps in versions.items():
                parts.append(f"  {pkg} {ver} → {deps if deps else '(no constraints)'}")

    if dep_g:
        parts.append("\nDependency Graph (prerequisites must come first):")
        for step, prereqs in dep_g.items():
            parts.append(f"  {step} requires: {prereqs}" if prereqs else f"  {step} → (no prereqs)")

    # Next-action hint — keeps all models on track
    if task_type == "security":
        used_types = [h["action_type"] for h in history]
        if "identify_vulnerability" not in used_types:
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

    # Strip reasoning/thinking blocks
    for tag in ["think", "thinking", "reasoning", "reflection", "thought", "antThinking"]:
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        if open_tag in text:
            text = text.split(close_tag)[-1].strip() if close_tag in text else text.split(open_tag)[-1].strip()

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Find JSON object in prose
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


def _compute_score(rewards: list) -> float:
    """
    Compute the episode score from a list of step rewards.

    DESIGN RATIONALE — why neither pure max nor pure mean is right:
    - Pure max: agent scores 0.85 on step 1, then 0.01 on all later steps → score=0.85
      This rewards single-lucky-step behaviour and hides that later steps failed.
    - Pure mean: agent scores 0.85 on step 1, 0.01 on 3 more → score=0.23
      This massively under-reports good episodes that have validation failures early.

    SOLUTION — weighted blend of max and mean:
      score = 0.60 * max(rewards) + 0.40 * mean(rewards)

    WHY THIS WORKS:
    - A great single-step performance (0.85) still shows up clearly (0.51 baseline contribution)
    - A consistently good episode (0.80, 0.85, 0.80) gets full credit (≈0.83)
    - A fluke-then-fail episode (0.85, 0.01, 0.01, 0.01) scores 0.52 — honestly mediocre
    - A failed episode (all 0.01) scores 0.01 — correctly bad

    Clamped to [0.01, 0.99] per Discord consensus on the (0,1) exclusive range.
    """
    if not rewards:
        return 0.01
    max_r  = max(rewards)
    mean_r = sum(rewards) / len(rewards)
    raw    = 0.60 * max_r + 0.40 * mean_r
    return round(min(max(raw, 0.01), 0.99), 4)


def run_task(client: OpenAI, task_id: str) -> tuple:
    """
    Run a single task through the environment.
    Returns (score: float, is_fatal_api_error: bool).
    """
    # ── Reset ──────────────────────────────────────────────────────────────
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        data = resp.json()
    except Exception as e:
        # Env unreachable — must still emit [START] and [END]
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
        return 0.01, False

    if "error" in data and not data.get("episode_id"):
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
        return 0.01, False

    episode_id = data.get("episode_id", "unknown")
    obs        = data.get("observation", data)

    # ── Mandatory [START] line — exact official spec ────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards     = []
    history     = []
    step_num    = 0
    consec_errs = 0
    fatal_error = False

    for step_num in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(step_num, obs, history)
        error_msg   = None

        # ── LLM call ───────────────────────────────────────────────────────
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
            consec_errs   = 0

        except Exception as e:
            error_msg     = str(e)
            response_text = '{"action_type": "error"}'
            consec_errs  += 1

            if _is_fatal_error(error_msg):
                fatal_error = True
                short = error_msg[:120].replace('\n', ' ')
                # Emit mandatory [STEP] then break — [END] emitted below
                print(f"[STEP] step={step_num} action=invalid reward=0.01 done=true error=FATAL:{short}", flush=True)
                rewards.append(0.01)
                break

            if _is_task_fatal(error_msg) or consec_errs >= MAX_CONSEC_ERRORS:
                short = error_msg[:120].replace('\n', ' ')
                print(f"[STEP] step={step_num} action=invalid reward=0.01 done=true error=TASK_STOP:{short}", flush=True)
                rewards.append(0.01)
                break

        action      = parse_action(response_text)
        action_type = action.get("action_type", "unknown")
        action["episode_id"] = episode_id

        # ── Env step ───────────────────────────────────────────────────────
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_data = step_resp.json()
        except Exception as e:
            short = str(e)[:100]
            print(f"[STEP] step={step_num} action={action_type} reward=0.01 done=true error={short}", flush=True)
            rewards.append(0.01)
            break

        reward     = float(step_data.get("reward", 0.0))
        done       = bool(step_data.get("done",   False))
        obs        = step_data.get("observation", step_data)
        step_error = step_data.get("error") or error_msg

        rewards.append(reward)
        history.append({"step": step_num, "action_type": action_type, "reward": reward, "done": done})

        # Show 'invalid' in log when validation failed
        display_action = "invalid" if obs.get("validation_failed") else action_type

        # Format error value: null or truncated string
        if step_error:
            error_val = str(step_error)[:150].replace('\n', ' ')
        else:
            error_val = "null"

        # ── Mandatory [STEP] line — exact official spec ────────────────────
        # reward=<0.00> means 2 decimal places
        # done=<true|false> means lowercase boolean string
        print(
            f"[STEP] step={step_num} action={display_action} reward={reward:.2f} "
            f"done={str(done).lower()} error={error_val}",
            flush=True
        )

        if done:
            break

    # ── Compute final score ────────────────────────────────────────────────
    score   = _compute_score(rewards)
    # success = at least one step scored meaningfully above the floor
    success = any(r > 0.10 for r in rewards)

    # rewards list: 2 decimal places, comma-separated, no spaces
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── Mandatory [END] line — exact official spec ─────────────────────────
    # spec: success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}",
        flush=True
    )

    return score, fatal_error


def main() -> None:
    """Run all 9 tasks and report final scores."""
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        health = requests.get(f"{ENV_URL}/", timeout=10, headers={"Accept": "application/json"})
        health_data = health.json()
        print(
            f"Environment: {health_data.get('env', 'unknown')} | "
            f"Tasks: {health_data.get('tasks', 0)}",
            flush=True
        )
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {ENV_URL}: {e}", flush=True)
        return

    scores       = {}
    had_fatal    = False

    for task_id in TASKS:
        try:
            score, is_fatal = run_task(client, task_id)
            scores[task_id] = score

            if is_fatal:
                had_fatal = True
                print(f"\n🚫 Fatal API error on {task_id}. Stopping remaining tasks.", flush=True)
                for remaining in TASKS:
                    if remaining not in scores:
                        scores[remaining] = 0.01
                        print(f"[START] task={remaining} env={BENCHMARK} model={MODEL_NAME}", flush=True)
                        print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)
                break

        except Exception as e:
            scores[task_id] = 0.01
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 score=0.01 rewards=", flush=True)

    avg = round(sum(scores.values()) / max(len(scores), 1), 4)
    print(f"\n✅ All tasks complete! Average: {avg:.4f}", flush=True)
    # Final JSON summary — evaluator may parse this
    print(json.dumps({"final_scores": scores}), flush=True)

    if had_fatal:
        print("⚠️ Results NOT saved — fatal API error (invalid token / no credits).", flush=True)
    else:
        try:
            from server.benchmark_store import append_result
            append_result(MODEL_NAME, MODEL_NAME, scores)
            print(f"💾 Results saved (avg: {avg:.4f})", flush=True)
        except Exception as e:
            print(f"⚠️ Could not save results: {e}", flush=True)


if __name__ == "__main__":
    main()
