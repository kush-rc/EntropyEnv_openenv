# server/web_ui.py
# Gradio UI with task descriptions, how-it-works, model performance tracking.

import os
import gradio as gr
import requests
import json
import time
from datetime import datetime

ENV_URL = 'http://localhost:7860'
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'run_history.json')
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# ── Task info for the UI ──
TASK_INFO = {
    'sec_easy': {
        'name': '🔒 Security — Easy',
        'desc': 'Identify a single vulnerability in a code snippet.\nThe agent must classify the vulnerability type (e.g., SQL injection, XSS), estimate the CVSS score, and determine severity.',
        'domain': 'Security (MCP Sandbox)',
        'example': '{"action_type":"identify_vulnerability","vuln_type":"sql_injection","cvss_score":9.1,"severity":"critical","affected_line":3}',
    },
    'sec_medium': {
        'name': '🔒 Security — Medium',
        'desc': 'Identify a vulnerability AND propose a secure code fix.\nThe agent performs vulnerability identification on turn 1, then proposes a fix on turn 2.',
        'domain': 'Security (MCP Sandbox)',
        'example': 'Turn 1: identify_vulnerability → Turn 2: propose_fix with fix_code',
    },
    'sec_hard': {
        'name': '🔒 Security — Hard',
        'desc': 'Identify → Fix → Revise based on reviewer feedback.\nMulti-turn: the agent must iteratively improve its fix when a reviewer provides feedback.',
        'domain': 'Security (MCP Sandbox)',
        'example': 'Turn 1: identify → Turn 2: propose_fix → Turn 3+: revise_fix (with reviewer feedback)',
    },
    'dep_easy': {
        'name': '📦 Dependency — Easy',
        'desc': 'Flag outdated packages and deprecated API usage.\nThe agent scans code for old package versions and deprecated function calls.',
        'domain': 'PyTorch Migration',
        'example': '{"action_type":"flag_outdated","packages":{"torch":"1.7.0"},"deprecated_api":"torch.no_grad","replacement":"torch.inference_mode"}',
    },
    'dep_medium': {
        'name': '📦 Dependency — Medium',
        'desc': 'Resolve version conflicts using a compatibility matrix.\nThe agent must propose compatible versions that satisfy cross-package constraints.',
        'domain': 'PyTorch Migration',
        'example': '{"action_type":"resolve_conflict","packages":{"torch":"2.1.0","numpy":"1.24.0"},"reasoning":"torch 2.1 requires numpy >= 1.24"}',
    },
    'dep_hard': {
        'name': '📦 Dependency — Hard',
        'desc': 'Fix torch.compile graph-break patterns in dependency order.\nThe agent must fix multiple graph-break issues in the correct order based on their dependencies.',
        'domain': 'PyTorch Migration',
        'example': '{"action_type":"migrate_api","completed_items":["break_1"],"code_changes":{"break_1":"replaced torch.no_grad with inference_mode"}}',
    },
    'cli_easy': {
        'name': '🏥 Clinical — Easy',
        'desc': 'Detect missing steps in a clinical workflow and assess risk.\nThe agent identifies which required steps are missing from a patient workflow.',
        'domain': 'Clinical Workflow Recovery',
        'example': '{"action_type":"detect_gap","missing_steps":["insurance_auth","pre_op_consent"],"risk_level":"critical"}',
    },
    'cli_medium': {
        'name': '🏥 Clinical — Medium',
        'desc': 'Detect gaps AND rank them by clinical priority.\nThe agent must both find missing steps and rank them by importance.',
        'domain': 'Clinical Workflow Recovery',
        'example': 'Turn 1: detect_gap → Turn 2: rank_issues with priority_order list',
    },
    'cli_hard': {
        'name': '🏥 Clinical — Hard',
        'desc': 'Plan a dependency-ordered recovery sequence.\nThe agent must respect the dependency graph when ordering recovery steps.',
        'domain': 'Clinical Workflow Recovery',
        'example': 'insurance_auth → pre_op_consent → specialist → surgery (respecting dependencies)',
    },
}


def _load_history():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_run(run_data):
    history = _load_history()
    history.append(run_data)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_task_info(task_id):
    """Return description for selected task."""
    info = TASK_INFO.get(task_id, {})
    return (
        f"### {info.get('name', task_id)}\n\n"
        f"**Domain:** {info.get('domain', '?')}\n\n"
        f"{info.get('desc', '')}\n\n"
        f"**Example action:**\n```json\n{info.get('example', '')}\n```"
    )


def run_single_task(task_id: str):
    """Run a single task with the demo agent."""
    from .demo_agent import demo_action

    logs = []
    rewards = []

    r = requests.post(f'{ENV_URL}/reset', json={'task_id': task_id}, timeout=30).json()
    ep_id = r.get('episode_id', '')
    obs = r.get('observation', r)
    logs.append(f'[START] task={task_id} episode={ep_id[:12]}...')

    done = False
    step = 0
    while not done and step < 8:
        action = demo_action(obs)
        action['episode_id'] = ep_id
        sr = requests.post(f'{ENV_URL}/step', json=action, timeout=30).json()
        reward = sr.get('reward', 0.0)
        done = sr.get('done', False)
        obs = sr.get('observation', sr)
        rewards.append(round(reward, 4))
        atype = action.get('action_type', '?')
        logs.append(f'  Step {step + 1}: action={atype}  reward={reward:.4f}  done={done}')
        step += 1

    total = round(sum(rewards) / max(len(rewards), 1), 4)
    logs.append(f'[END] avg_reward={total}  steps={step}')
    return '\n'.join(logs), rewards, total


def run_task_ui(task_id: str, model_name: str):
    """Run a single task and return display outputs."""
    if not model_name.strip():
        model_name = 'Demo Agent (rule-based)'

    log_str, rewards, total = run_single_task(task_id)

    reward_lines = ['Reward per step:']
    for i, r in enumerate(rewards):
        bar = '█' * int(r * 20)
        reward_lines.append(f'  Step {i + 1}: {bar}  {r:.4f}')
    reward_str = '\n'.join(reward_lines)

    info = TASK_INFO.get(task_id, {})
    domain = info.get('domain', 'Unknown')
    difficulty = task_id.split('_')[1].upper()
    score = min(max(total / max(len(rewards), 1), 0.01), 0.99)

    score_md = f'''### ✅ Results
| Field | Value |
|-------|-------|
| **Model** | `{model_name}` |
| **Task** | `{task_id}` |
| **Domain** | {domain} |
| **Difficulty** | {difficulty} |
| **Score** | **{score:.4f}** |
| **Total Reward** | {total:.4f} |
| **Steps** | {len(rewards)} |
'''

    _save_run({
        'model': model_name, 'task_id': task_id, 'domain': domain,
        'total_reward': total, 'score': round(score, 4),
        'steps': len(rewards), 'timestamp': datetime.now().isoformat(),
    })

    return log_str, reward_str, score_md


def run_all_tasks_ui(model_name: str):
    """Run all 9 tasks and return a performance dashboard."""
    if not model_name.strip():
        model_name = 'Demo Agent (rule-based)'

    tasks = list(TASK_INFO.keys())
    all_logs = []
    all_scores = {}

    for task_id in tasks:
        log_str, rewards, total = run_single_task(task_id)
        all_logs.append(log_str)
        score = min(max(total / max(len(rewards), 1), 0.01), 0.99)
        all_scores[task_id] = round(score, 4)

    full_log = '\n\n'.join(all_logs)

    sec = [all_scores[t] for t in tasks if t.startswith('sec')]
    dep = [all_scores[t] for t in tasks if t.startswith('dep')]
    cli = [all_scores[t] for t in tasks if t.startswith('cli')]

    rows = []
    for task_id, score in all_scores.items():
        info = TASK_INFO.get(task_id, {})
        bar = '█' * int(min(score, 1.0) * 15)
        rows.append(f'| `{task_id}` | {info.get("domain", "?")} | {bar} | **{score:.4f}** |')

    avg = sum(all_scores.values()) / 9
    sec_avg = sum(sec) / 3
    dep_avg = sum(dep) / 3
    cli_avg = sum(cli) / 3

    dashboard = f'''## 📊 Model Performance Dashboard

**Model:** `{model_name}`  
**Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Per-Task Scores
| Task | Domain | Performance | Score |
|------|--------|-------------|-------|
{chr(10).join(rows)}

### Domain Averages
| Domain | Avg Score | Rating |
|--------|-----------|--------|
| 🔒 Security | {sec_avg:.4f} | {"🟢 Excellent" if sec_avg > 0.7 else "🟡 Good" if sec_avg > 0.4 else "🔴 Needs Work"} |
| 📦 PyTorch Migration | {dep_avg:.4f} | {"🟢 Excellent" if dep_avg > 0.7 else "🟡 Good" if dep_avg > 0.4 else "🔴 Needs Work"} |
| 🏥 Clinical Workflow | {cli_avg:.4f} | {"🟢 Excellent" if cli_avg > 0.7 else "🟡 Good" if cli_avg > 0.4 else "🔴 Needs Work"} |

### Overall: **{avg:.4f}**
'''

    _save_run({
        'model': model_name, 'type': 'full_run', 'scores': all_scores,
        'avg': round(avg, 4), 'timestamp': datetime.now().isoformat(),
    })

    return full_log, dashboard


def show_history():
    history = _load_history()
    if not history:
        return 'No runs yet. Run a task first!'
    lines = ['## 📜 Run History\n']
    for i, run in enumerate(reversed(history[-10:])):
        ts = run.get('timestamp', '?')[:19]
        model = run.get('model', '?')
        if run.get('type') == 'full_run':
            avg = run.get('avg', 0)
            lines.append(f'**#{len(history) - i}** | `{ts}` | `{model}` | All 9 tasks | Avg: **{avg:.4f}**')
        else:
            task = run.get('task_id', '?')
            score = run.get('score', 0)
            lines.append(f'**#{len(history) - i}** | `{ts}` | `{model}` | `{task}` | Score: **{score:.4f}**')
    return '\n\n'.join(lines)


def build_ui():
    with gr.Blocks(title='Multi-Agent Dev Tools Env', theme=gr.themes.Soft()) as demo:
        gr.Markdown('''# 🛠️ Multi-Agent Dev Tools Environment
**A multi-domain RL environment for training AI agents on real-world tasks.**

This environment tests AI agents across **3 domains** with **9 tasks** of increasing difficulty.
Agents receive observations (problems), send actions (answers), and get reward scores (0.01 – 0.99).
''')

        with gr.Tab('🎯 Single Task'):
            with gr.Row():
                task_dd = gr.Dropdown(
                    choices=list(TASK_INFO.keys()),
                    value='sec_easy',
                    label='🎯 Select Task',
                )
                model_input = gr.Textbox(
                    label='🤖 Model Name',
                    value='Demo Agent (rule-based)',
                    placeholder='e.g. Qwen/Qwen2.5-72B-Instruct',
                )
                run_btn = gr.Button('▶️ Run Task', variant='primary', scale=1)

            task_info_md = gr.Markdown(get_task_info('sec_easy'))
            task_dd.change(fn=get_task_info, inputs=[task_dd], outputs=[task_info_md])

            with gr.Row():
                logs_box = gr.Textbox(label='📋 Episode Log', lines=10)
                rewards_box = gr.Textbox(label='📊 Reward History', lines=10)

            score_md = gr.Markdown('*Results will appear after running a task...*')

            run_btn.click(
                fn=run_task_ui,
                inputs=[task_dd, model_input],
                outputs=[logs_box, rewards_box, score_md],
            )

        with gr.Tab('🏆 Run All 9 Tasks'):
            gr.Markdown('Run all 9 tasks at once and see a full performance dashboard with domain averages.')
            with gr.Row():
                model_all = gr.Textbox(
                    label='🤖 Model Name',
                    value='Demo Agent (rule-based)',
                )
                run_all_btn = gr.Button('🚀 Run All 9 Tasks', variant='primary')

            all_logs = gr.Textbox(label='📋 Full Run Log', lines=12)
            dashboard_md = gr.Markdown('*Dashboard will appear after running all tasks...*')

            run_all_btn.click(
                fn=run_all_tasks_ui,
                inputs=[model_all],
                outputs=[all_logs, dashboard_md],
            )

        with gr.Tab('📜 Run History'):
            history_md = gr.Markdown('Click refresh to see past runs.')
            refresh_btn = gr.Button('🔄 Refresh History')
            refresh_btn.click(fn=show_history, outputs=[history_md])

        with gr.Tab('📖 How It Works'):
            gr.Markdown('''## How This Environment Works

### Overview
This is a **training gym for AI agents**. You build an agent, connect it to this environment
via the API, and it gets scored on how well it solves real-world tasks.

### The Flow
```
1. Agent calls POST /reset with a task_id → Gets an observation (the problem)
2. Agent analyzes the observation and sends POST /step with its action
3. Environment validates the action and grades it
4. Returns a reward score (0.01 – 0.99) and the next observation
5. Repeat until the episode ends (done=true) or max steps reached
```

### Three Domains
| Domain | Tasks | What Agents Do |
|--------|-------|---------------|
| 🔒 **Security** | sec_easy, sec_medium, sec_hard | Identify vulnerabilities, propose fixes, revise based on feedback |
| 📦 **Dependency** | dep_easy, dep_medium, dep_hard | Flag outdated packages, resolve conflicts, fix graph-breaks |
| 🏥 **Clinical** | cli_easy, cli_medium, cli_hard | Detect workflow gaps, rank by priority, plan recovery |

### Reward Signals
- Scores range from **0.01** (completely wrong) to **0.99** (near-perfect)
- Partial credit is awarded for partially correct answers
- Invalid or malformed actions receive lower scores
- The environment provides feedback on validation failures to help agents improve

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Health check | Returns status and task count |
| `POST /reset` | Start episode | `{"task_id":"sec_easy"}` → observation |
| `POST /step` | Submit action | `{action_type, ...}` → reward + next observation |
| `GET /state` | Get state | Query current episode state |

### Getting Started
```python
import requests

# Start an episode
resp = requests.post("http://localhost:7860/reset", json={"task_id": "sec_easy"})
data = resp.json()
episode_id = data["episode_id"]
observation = data["observation"]

# Send an action
action = {"episode_id": episode_id, "action_type": "identify_vulnerability", ...}
result = requests.post("http://localhost:7860/step", json=action)
print(result.json())  # {"reward": 0.85, "done": true, "observation": {...}}
```
''')

    return demo
