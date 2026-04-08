# server/demo_agent.py
# Simple rule-based demo agent for the Gradio UI.
# Uses hardcoded heuristics to show the environment works without calling a real LLM.


def demo_action(obs):
    """Generate a simple action based on observation. Used by the UI demo."""
    task_type = obs.get('task_type', '')
    task_id = obs.get('task_id', '')
    turn = obs.get('turn', 0)

    if task_type == 'security':
        return _security_action(obs, task_id, turn)
    elif task_type == 'dependency':
        return _dependency_action(obs, task_id, turn)
    elif task_type == 'clinical':
        return _clinical_action(obs, task_id, turn)
    else:
        return {'action_type': 'invalid'}


def _security_action(obs, task_id, turn):
    if turn == 0:
        tool_call = obs.get('tool_call', '')
        # Simple heuristic to detect common vulnerability types
        vuln_type = 'sql_injection'
        severity = 'critical'
        cvss = 8.5
        if 'script' in tool_call.lower() or 'xss' in tool_call.lower():
            vuln_type = 'xss'
            severity = 'medium'
            cvss = 5.0
        elif 'password' in tool_call.lower() or 'secret' in tool_call.lower():
            vuln_type = 'hardcoded_secret'
            severity = 'high'
            cvss = 6.5
        elif 'jwt' in tool_call.lower() or 'token' in tool_call.lower():
            vuln_type = 'jwt_misuse'
            severity = 'critical'
            cvss = 8.0
        elif 'path' in tool_call.lower() or '..' in tool_call:
            vuln_type = 'path_traversal'
            severity = 'high'
            cvss = 7.0
        elif 'auth' in tool_call.lower() and 'no' in tool_call.lower():
            vuln_type = 'missing_auth'
            severity = 'critical'
            cvss = 8.5

        return {
            'action_type': 'identify_vulnerability',
            'vuln_type': vuln_type,
            'cvss_score': cvss,
            'severity': severity,
            'affected_line': 1,
        }

    elif 'reviewer_feedback' in obs:
        return {
            'action_type': 'revise_fix',
            'fix_code': 'sanitize_input(parameterized_query)',
            'addressed_feedback': obs.get('reviewer_feedback', 'fixed the issue'),
        }
    else:
        return {
            'action_type': 'propose_fix',
            'fix_code': 'use parameterized query with ? placeholder',
            'explanation': 'Replace string concatenation with parameterized queries',
        }


def _dependency_action(obs, task_id, turn):
    task_subtype = obs.get('task_subtype', 'flag')

    if task_subtype == 'flag':
        return {
            'action_type': 'flag_outdated',
            'packages': {'torch': '1.9.0'},
            'deprecated_api': 'torch.autograd.Variable',
            'replacement': 'plain tensor',
        }
    elif task_subtype == 'resolve':
        return {
            'action_type': 'resolve_conflict',
            'packages': {'torch': '2.1.0', 'numpy': '1.24.0'},
            'reasoning': 'PyTorch 2.1 requires NumPy 1.24+',
        }
    else:  # migrate
        return {
            'action_type': 'migrate_api',
            'completed_items': ['break_001', 'break_002'],
            'code_changes': {
                'break_001': 'torch.where(condition, x*2, x)',
                'break_002': 'x.shape[0]',
            },
        }


def _clinical_action(obs, task_id, turn):
    available_steps = obs.get('available_steps', [])

    if turn == 0:
        return {
            'action_type': 'detect_gap',
            'missing_steps': available_steps[:2] if available_steps else ['unknown_step'],
            'risk_level': 'critical',
        }
    elif turn == 1:
        return {
            'action_type': 'rank_issues',
            'priority_order': available_steps[:3] if available_steps else ['unknown_step'],
        }
    else:
        dep_graph = obs.get('dependency_graph', {})
        # Simple topological sort attempt
        ordered = _simple_topo_sort(available_steps, dep_graph)
        return {
            'action_type': 'order_steps',
            'recovery_steps': ordered,
        }


def _simple_topo_sort(steps, dep_graph):
    """Simple topological sort for dependency ordering."""
    if not dep_graph:
        return steps
    result = []
    remaining = set(steps)
    for _ in range(len(steps) + 1):
        if not remaining:
            break
        for step in list(remaining):
            prereqs = dep_graph.get(step, [])
            if all(p in result for p in prereqs):
                result.append(step)
                remaining.remove(step)
                break
    # Add any unresolved steps
    result.extend(remaining)
    return result
