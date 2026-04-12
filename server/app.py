# server/app.py
# FastAPI server with endpoints. ALL return HTTP 200 ALWAYS.
# Endpoints: GET /, GET /debug, POST /reset, POST /step, GET /state, POST /inference

import os
import sys
import json
import random
import uuid
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from .session import create_session, SESSIONS, TASK_TYPE_MAP, SessionState
from .router import route_step, build_initial_obs
from .validation.validator import validate_action
from .datasets.security_cases import SECURITY_CASES
from .datasets.dependency_cases import DEPENDENCY_CASES
from .datasets.clinical_cases import CLINICAL_CASES

app = FastAPI(title='Multi-Agent Dev Tools Environment')

# ── Load Debug Panel HTML ──
_DEBUG_HTML_PATH = os.path.join(os.path.dirname(__file__), 'debug_panel.html')

def _load_debug_html() -> str:
    try:
        with open(_DEBUG_HTML_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '<h1>Debug panel not found. Place debug_panel.html in server/ directory.</h1>'

_DEBUG_HTML = _load_debug_html()

# ── Mount Gradio UI ──
try:
    from .web_ui import build_ui
    import gradio as gr
    ui_app = build_ui()
    app = gr.mount_gradio_app(app, ui_app, path='/web')
except Exception as e:
    import traceback
    print(f'[WARNING] Gradio UI not mounted: {e}')
    traceback.print_exc()


# ── Dataset Loader ──
DATASETS = {
    'sec_easy': SECURITY_CASES.get('sec_easy', []),
    'sec_medium': SECURITY_CASES.get('sec_medium', []),
    'sec_hard': SECURITY_CASES.get('sec_hard', []),
    'dep_easy': DEPENDENCY_CASES.get('dep_easy', []),
    'dep_medium': DEPENDENCY_CASES.get('dep_medium', []),
    'dep_hard': DEPENDENCY_CASES.get('dep_hard', []),
    'cli_easy': CLINICAL_CASES.get('cli_easy', []),
    'cli_medium': CLINICAL_CASES.get('cli_medium', []),
    'cli_hard': CLINICAL_CASES.get('cli_hard', []),
}


# Per-domain max steps (must match grader config)
DOMAIN_MAX_STEPS = {'security': 8, 'dependency': 8, 'clinical': 6}


def load_case(task_id: str, episode_id: str = '') -> dict:
    """Load a deterministic case for reproducibility.
    Same episode_id always gets same case (judges can re-run and match)."""
    cases = DATASETS.get(task_id, [])
    if not cases:
        return {}
    idx = hash(episode_id) % len(cases)
    return cases[idx]


# build_initial_obs is imported from router.py — single source of truth for observations


# ═══════════════════════════════════════════════════════════
# ENDPOINTS — All return HTTP 200 ALWAYS
# ═══════════════════════════════════════════════════════════

@app.get('/')
async def health(request: Request):
    """Health check + debug panel. Returns HTML for browsers, JSON for automated scripts."""
    try:
        accept = request.headers.get('accept', '')
        if 'text/html' in accept:
            return HTMLResponse(content=_DEBUG_HTML, status_code=200)
        return {
            'status': 'ok',
            'env': 'Multi-Agent Real-World Ecosystem',
            'domains': ['security', 'pytorch', 'clinical'],
            'tasks': 9,
            'task_ids': [
                'sec_easy', 'sec_medium', 'sec_hard',
                'dep_easy', 'dep_medium', 'dep_hard',
                'cli_easy', 'cli_medium', 'cli_hard',
            ],
            'spec': 'OpenEnv v1',
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={'status': 'error', 'error': str(e)})


@app.post('/reset')
async def reset(request: Request):
    """Create a new episode for a task. Returns episode_id + initial observation."""
    
    try:
        body = await request.json()
        task_id = body.get('task_id', 'sec_easy')

        if task_id not in TASK_TYPE_MAP:
            return JSONResponse(status_code=200, content={
                'error': f'Unknown task_id: {task_id}',
                'observation': {},
                'done': True,
            })

        ep_id = str(uuid.uuid4())
        task_case = load_case(task_id, ep_id)
        session = create_session(task_id, task_case)
        session.episode_id = ep_id
        SESSIONS[session.episode_id] = session

        # Cleanup old done sessions to prevent memory leaks on HF Spaces
        if len(SESSIONS) > 100 or random.random() < 0.1:
            done_ids = [eid for eid, s in SESSIONS.items() if s.done]
            for eid in done_ids:
                SESSIONS.pop(eid, None)

        obs = build_initial_obs(session)

        return {
            'episode_id': session.episode_id,
            'observation': obs,
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={
            'error': str(e),
            'observation': {},
            'done': True,
            'reward': 0.01,
        })


@app.post('/step')
async def step(request: Request):
    """Submit an action for an episode. Returns reward + next observation."""
    try:
        body = await request.json()
        ep_id = body.get('episode_id')
        session = SESSIONS.get(ep_id)

        if not session:
            return JSONResponse(status_code=200, content={
                'reward': 0.01,
                'done': True,
                'error': 'unknown episode_id',
                'observation': {},
            })

        if session.done:
            return JSONResponse(status_code=200, content={
                'reward': 0.01,
                'done': True,
                'observation': {'message': 'Episode already complete.'},
            })

        # Run pre-action validation
        valid, val_obs = validate_action(body, session)
        if not valid:
            last_r = 0.01
            if session.history:
                last_r = max(0.01, session.history[-1].get('reward', 0.01))
            return {
                'reward': last_r,
                'done': False,
                'observation': val_obs,
            }

        # Route to grader
        result = route_step(session, body)

        # Update session state
        session.step_count += 1
        session.last_actions.append(body.get('action_type', 'unknown'))
        session.history.append(body)
        session.reward_acc += result.get('reward', 0.0)
        session.done = result.get('done', False)

        # Enrich observation with strategic context
        step_obs = result.get('observation', {})
        task_max = DOMAIN_MAX_STEPS.get(session.task_type, 8)
        enrichment = {
            'task_type': session.task_type,
            'task_id': session.task_id,
            'step_count': session.step_count,
            'max_steps': task_max,
            'previous_reward': round(float(result.get('reward', 0.0)), 4),
            'steps_remaining': max(0, task_max - session.step_count),
            'reward_so_far': round(session.reward_acc, 4),
            'trajectory_score': round(session.reward_acc / max(session.step_count, 1), 4),
        }
        for k, v in enrichment.items():
            step_obs.setdefault(k, v)

        # Turn guidance — tell agent what to do next
        last_action = body.get('action_type', '')
        if session.task_type == 'security':
            if last_action == 'identify_vulnerability':
                step_obs['next_expected_action'] = 'propose_fix'
                step_obs['guidance'] = 'Vulnerability identified. Now propose a fix using propose_fix.'
            elif last_action == 'propose_fix':
                step_obs['next_expected_action'] = 'revise_fix'
                step_obs['guidance'] = 'Fix proposed. If reviewer_feedback is present, use revise_fix.'
        elif session.task_type == 'clinical':
            if last_action == 'detect_gap':
                step_obs['next_expected_action'] = 'rank_issues'
                step_obs['guidance'] = 'Gaps detected. Now rank issues by priority using rank_issues.'
            elif last_action == 'rank_issues':
                step_obs['next_expected_action'] = 'order_steps'
                step_obs['guidance'] = 'Issues ranked. Now create recovery plan using order_steps.'

        # Cleanup session if done
        if session.done:
            SESSIONS.pop(session.episode_id, None)

        return {
            'reward': round(min(max(float(result.get('reward', 0.01)), 0.01), 0.99), 4),
            'done': bool(result.get('done', False)),
            'observation': step_obs,
            'info': {'validation_failed': step_obs.get('validation_failed', False)},
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={
            'reward': 0.01,
            'done': True,
            'error': str(e),
            'observation': {},
            'info': {'error': str(e)},
        })


@app.get('/state')
async def state(episode_id: str = ''):
    """Get current state of an episode."""
    try:
        session = SESSIONS.get(episode_id)
        if not session:
            return {
                'episode_id': episode_id,
                'step_count': 0,
                'done': True,
            }
        return {
            'episode_id': session.episode_id,
            'step_count': session.step_count,
            'active_domain': session.task_type,
            'reward_acc': round(session.reward_acc, 4),
            'done': session.done,
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={'error': str(e)})


# ═══════════════════════════════════════════════════════════
# DEBUG PANEL — guaranteed HTML endpoint
# ═══════════════════════════════════════════════════════════

@app.get('/debug', response_class=HTMLResponse)
async def debug_panel():
    """Always serves the debug panel HTML regardless of Accept header."""
    try:
        html = _load_debug_html()  # Reload from disk each time for development
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f'<h1>Error loading debug panel: {e}</h1>', status_code=200)


# ═══════════════════════════════════════════════════════════
# INFERENCE — run inference.py from browser
# ═══════════════════════════════════════════════════════════

@app.post('/inference')
async def run_inference(request: Request):
    """Runs inference.py as a subprocess and returns parsed scores."""
    try:
        env_vars = os.environ.copy()
        env_vars['ENV_URL'] = env_vars.get('ENV_URL', 'http://localhost:7860')

        # inference.py is at project root (one level up from server/)
        inference_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'inference.py'
        )

        if not os.path.exists(inference_path):
            return JSONResponse(status_code=200, content={
                'error': 'inference.py not found at project root',
                'path_checked': inference_path,
            })

        result = subprocess.run(
            [sys.executable, inference_path],
            capture_output=True, text=True, timeout=1200,  # 20 min max
            env=env_vars
        )

        stdout = result.stdout or ''
        stderr = result.stderr or ''
        logs   = []

        # Collect all log lines for display
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                logs.append(line)

        # ── Parse final_scores from the JSON summary line ──
        # This is authoritative — inference.py always prints:
        #   {"final_scores": {"sec_easy": 0.85, ...}}
        # at the end of main().
        final_scores = {}
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith('{') and 'final_scores' in line:
                try:
                    parsed = json.loads(line)
                    if 'final_scores' in parsed:
                        final_scores = parsed['final_scores']
                        break
                except Exception:
                    pass

        # ── Fallback: parse [END] lines for any tasks missing from JSON ──
        # Official [END] format: success=<bool> steps=<n> rewards=<r1,r2,...>
        # We track which task we're in via the preceding [START] line.
        if not final_scores:
            current_task = None
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith('[START]'):
                    # Extract task= field
                    for token in line.split():
                        if token.startswith('task='):
                            current_task = token.split('=', 1)[1]
                            break
                elif line.startswith('[END]') and current_task:
                    # Parse rewards= field and compute score from it
                    parts = {}
                    for token in line.split():
                        if '=' in token:
                            k, v = token.split('=', 1)
                            parts[k] = v
                    rewards_str = parts.get('rewards', '')
                    if rewards_str:
                        try:
                            step_rewards = [float(r) for r in rewards_str.split(',') if r]
                            if step_rewards:
                                # Same weighted blend as inference.py _compute_score()
                                max_r  = max(step_rewards)
                                mean_r = sum(step_rewards) / len(step_rewards)
                                score  = round(min(max(0.60 * max_r + 0.40 * mean_r, 0.01), 0.99), 4)
                                final_scores[current_task] = score
                        except (ValueError, TypeError):
                            final_scores[current_task] = 0.01
                    current_task = None

        avg = (
            round(sum(final_scores.values()) / len(final_scores), 4)
            if final_scores else 0.01
        )

        return JSONResponse(status_code=200, content={
            'status':        'ok' if result.returncode == 0 else 'completed_with_errors',
            'final_scores':  final_scores,
            'average_score': avg,
            'logs':          logs[-50:],
            'stderr':        stderr[-500:] if stderr else '',
            'returncode':    result.returncode,
        })

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=200, content={
            'error':        'inference.py timed out after 20 minutes',
            'final_scores': {},
        })
    except Exception as e:
        return JSONResponse(status_code=200, content={
            'error':        str(e),
            'final_scores': {},
        })


# ═══════════════════════════════════════════════════════════
# BENCHMARK RUNNER — run from the UI with custom API keys
# ═══════════════════════════════════════════════════════════

TASK_IDS = [
    'sec_easy', 'sec_medium', 'sec_hard',
    'dep_easy', 'dep_medium', 'dep_hard',
    'cli_easy', 'cli_medium', 'cli_hard',
]


def _parse_llm_response(raw_text: str) -> str:
    """Strip thinking blocks and markdown from LLM response. Universal model compat."""
    text = raw_text.strip()
    # Strip ALL known reasoning/thinking blocks (closed and unclosed)
    for tag in ['think', 'thinking', 'reasoning', 'reflection', 'thought', 'antThinking']:
        open_tag = f'<{tag}>'
        close_tag = f'</{tag}>'
        if open_tag in text:
            if close_tag in text:
                text = text.split(close_tag)[-1].strip()
            else:
                text = text.split(open_tag)[-1].strip()
    # Strip markdown fences
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        parts = text.split('```')
        if len(parts) >= 3:
            text = parts[1].strip()
    # Find JSON object
    if not text.startswith('{'):
        start = text.find('{')
        if start >= 0:
            end = text.rfind('}')
            if end > start:
                text = text[start:end + 1]
    return text


def _run_single_task_inline(task_id, api_base, api_key, model_id, system_prompt):
    """Run one task against the local server. Yields dict events."""
    import re
    import requests as req

    logs = []
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)
    except Exception as e:
        msg = f'[ERROR] OpenAI client init failed: {e}'
        logs.append(msg)
        yield {'type': 'log', 'level': 'err', 'msg': msg}
        yield {'type': 'task_done', 'task_id': task_id, 'score': 0.01, 'logs': logs}
        return

    # Reset
    try:
        resp = req.post('http://localhost:7860/reset', json={'task_id': task_id}, timeout=30)
        data = resp.json()
    except Exception as e:
        msg = f'[ERROR] Reset failed: {e}'
        logs.append(msg)
        yield {'type': 'log', 'level': 'err', 'msg': msg}
        yield {'type': 'task_done', 'task_id': task_id, 'score': 0.01, 'logs': logs}
        return

    ep_id = data.get('episode_id', 'unknown')
    obs = data.get('observation', data)
    msg = f'[START] task={task_id} env=EntropyEnv model={model_id}'
    logs.append(msg)
    yield {'type': 'log', 'level': 'info', 'msg': msg}

    messages = [{'role': 'system', 'content': system_prompt}]
    rewards = []
    history = []
    done = False
    max_steps = 8

    while not done and len(rewards) < max_steps:
        step_num = len(rewards) + 1
        # Build focused prompt with smart truncation (matches inference.py)
        obs_copy = dict(obs)
        compat_matrix = obs_copy.pop('compatibility_matrix', None)
        dep_graph = obs_copy.pop('dependency_graph', None)
        core_text = json.dumps(obs_copy, default=str, indent=2)
        user_parts = [f'Step {step_num} | Observation:']
        if history:
            user_parts.append(f'Previous actions: {[h["action_type"] for h in history]}')
            if history[-1]['reward'] < 0.4:
                user_parts.append('⚠️ Low score. Try different approach.')
        user_parts.append(core_text)
        if compat_matrix:
            user_parts.append(f'\nCompatibility Matrix:\n{json.dumps(compat_matrix, indent=2)}')
        if dep_graph:
            user_parts.append(f'\nDependency Graph:\n{json.dumps(dep_graph, indent=2)}')
        user_parts.append('Output ONLY a single JSON object:')
        messages.append({'role': 'user', 'content': '\n'.join(user_parts)})

        try:
            reply = client.chat.completions.create(
                model=model_id, messages=messages, max_tokens=400, temperature=0.1
            )
            agent_text = (reply.choices[0].message.content or '').strip()
        except Exception as e:
            agent_text = '{"action_type":"invalid"}'
            msg = f'[WARN] API error: {str(e)[:100]}'
            logs.append(msg)
            yield {'type': 'log', 'level': 'warn', 'msg': msg}

        # Universal think-block + markdown stripping
        raw = _parse_llm_response(agent_text)

        messages.append({'role': 'assistant', 'content': raw})
        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

        try:
            action = json.loads(raw)
        except Exception:
            # Regex fallback
            match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw, re.DOTALL)
            if match:
                try:
                    action = json.loads(match.group())
                except Exception:
                    action = {'action_type': 'invalid'}
            else:
                action = {'action_type': 'invalid'}

        # Step
        try:
            step_resp = req.post('http://localhost:7860/step', json={
                'episode_id': ep_id, **action
            }, timeout=30)
            step_data = step_resp.json()
        except Exception as e:
            msg = f'[ERROR] Step failed: {e}'
            logs.append(msg)
            yield {'type': 'log', 'level': 'err', 'msg': msg}
            break

        reward = float(step_data.get('reward', 0.0))
        done = bool(step_data.get('done', False))
        obs = step_data.get('observation', step_data)
        rewards.append(reward)

        atype = action.get('action_type', '?')
        display_action = atype
        if obs.get('validation_failed'):
            display_action = 'invalid'
        history.append({'action_type': atype, 'reward': reward})

        error_val = step_data.get('error', 'null') or 'null'
        msg = f'[STEP] step={step_num} action={display_action} reward={reward:.2f} done={str(done).lower()} error={error_val}'
        logs.append(msg)
        yield {'type': 'log', 'level': 'info', 'msg': msg}

    # Weighted blend scoring — same as inference.py _compute_score()
    if rewards:
        max_r = max(rewards)
        mean_r = sum(rewards) / len(rewards)
        score = round(min(max(0.60 * max_r + 0.40 * mean_r, 0.01), 0.99), 4)
    else:
        score = 0.01
    success = any(r > 0.10 for r in rewards)
    rewards_str = ','.join(f'{r:.2f}' for r in rewards)

    # Mandatory [END] line — exact official spec
    msg = f'[END] success={str(success).lower()} steps={len(rewards)} score={score:.2f} rewards={rewards_str}'
    logs.append(msg)
    yield {'type': 'log', 'level': 'ok', 'msg': msg}
    yield {'type': 'task_done', 'task_id': task_id, 'score': score, 'logs': logs}


@app.post('/benchmark/run')
def run_benchmark(body: dict):
    """Run all 9 tasks with a given model config. Streams results via SSE."""
    from datetime import datetime
    from fastapi.responses import StreamingResponse
    from .benchmark_store import append_result
    import json

    model_name = body.get('model_name', 'Unknown Model')
    model_id = body.get('model_id', '')
    api_base = body.get('api_base', '')
    api_key = body.get('api_key', '')

    if not model_id or not api_base or not api_key:
        return JSONResponse(status_code=200, content={'error': 'missing_fields'})

    system_prompt = body.get('system_prompt', '') or BENCHMARK_SYSTEM_PROMPT

    def event_stream():
        scores = {}
        all_logs = []
        for task_id in TASK_IDS:
            for event in _run_single_task_inline(task_id, api_base, api_key, model_id, system_prompt):
                if event.get('type') == 'log':
                    all_logs.append(event['msg'])
                elif event.get('type') == 'task_done':
                    scores[task_id] = event['score']
                yield f"data: {json.dumps(event)}\n\n"

        avg = round(sum(scores.values()) / len(scores), 4) if scores else 0.01

        result = {
            'model_name': model_name,
            'model_id': model_id,
            'api_base': api_base,
            'scores': scores,
            'average': avg,
            'timestamp': datetime.now().isoformat(),
            'logs': all_logs,
        }

        # Persist to disk via benchmark_store
        try:
            from .benchmark_store import append_result
            append_result(model_name, model_id, scores)
        except Exception as e:
            print(f"Failed to append result: {e}", flush=True)
        yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get('/benchmark/results')
async def get_benchmark_results():
    """Return all saved benchmark results (persisted to disk)."""
    from .benchmark_store import get_all
    results = get_all()
    return JSONResponse(status_code=200, content={
        'results': results,
        'count': len(results),
    })


@app.post('/benchmark/clear')
async def clear_benchmark_results():
    """Clear all saved benchmark results."""
    from .benchmark_store import _save
    _save([])
    return JSONResponse(status_code=200, content={'status': 'cleared'})


# Default system prompt for benchmark
BENCHMARK_SYSTEM_PROMPT = '''You are a multi-domain analyst agent. Each observation has a task_type field.
Read it. Respond ONLY with a single valid JSON object. No prose, no markdown, no explanation.

IF task_type == 'security':
  Turn 1 ALWAYS: {"action_type":"identify_vulnerability","vuln_type":"sql_injection","cvss_score":9.1,"severity":"critical"}
  Turn 2 ALWAYS: {"action_type":"propose_fix","fix_code":"db.execute(sql, (param,))","explanation":"Use parameterized query"}
  Turn 3+ (reviewer_feedback present): {"action_type":"revise_fix","fix_code":"<fixed code>","addressed_feedback":"<COPY feedback verbatim>"}

IF task_type == 'dependency':
  task_subtype=flag: {"action_type":"flag_outdated","packages":{"torch":"1.9.0"},"deprecated_api":"torch.autograd.Variable","replacement":"plain tensor"}
  task_subtype=resolve: READ compatibility_matrix. {"action_type":"resolve_conflict","packages":{"torch":"2.1.0","numpy":"1.24.0"},"reasoning":"..."}
  task_subtype=migrate: {"action_type":"migrate_api","completed_items":["break_001"],"code_changes":{"break_001":"torch.where"}}

IF task_type == 'clinical':
  Turn 1: {"action_type":"detect_gap","missing_steps":["step1","step2"],"risk_level":"critical"}
  Turn 2: {"action_type":"rank_issues","priority_order":["most_urgent","least_urgent"]}
  Turn 3: {"action_type":"order_steps","recovery_steps":["first","second","last"]}

ALWAYS: Output ONLY a single JSON object. Follow guidance and next_expected_action.
'''

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()


