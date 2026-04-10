# server/router.py
# Central dispatcher. Routes validated actions to the correct domain grader.
#
# KEY FIX: The _check_done() mastery condition was firing after just 2 steps
# if avg_reward >= 0.90. This caused:
#   - sec_easy: identify_vulnerability scores 0.99 → avg = 0.99 → done=True immediately
#   - dep_easy, cli_easy: same problem — 1-step episodes ending with 0.99
#
# The mastery condition is now DISABLED. Done is determined by:
#   1. max_steps reached (hard limit)
#   2. required_sequence fully completed (all actions in sequence done)
#   3. completion_threshold met AND min_actions satisfied
#
# This forces multi-step tasks to actually run all required steps,
# and prevents easy tasks from short-circuiting at step 1.

from typing import Dict
from .session import SessionState
from .graders import security_grader, dependency_grader, clinical_grader

GRADERS = {
    'security': security_grader,
    'dependency': dependency_grader,
    'clinical': clinical_grader,
}


def route_step(session: SessionState, action: Dict) -> Dict:
    """Route a validated action to the correct grader and return enriched result."""
    grader = GRADERS.get(session.task_type)
    if not grader:
        return {
            'reward': 0.01,
            'done': True,
            'observation': {'error': f'Unknown task_type: {session.task_type}'},
        }

    reward = grader.grade(action, session)

    case = session.task_case
    max_steps = case.get('max_steps', 8)
    done = _check_done(session, action, reward, max_steps)

    obs = _build_step_obs(session, action, reward, done)
    score_details = _compute_score_details(action, session)
    obs['score_breakdown'] = score_details

    return {
        'episode_id': session.episode_id,
        'step_count': session.step_count + 1,
        'reward': round(float(reward), 4),
        'done': bool(done),
        'observation': obs,
        'score_details': score_details,
    }


def _check_done(session: SessionState, action: Dict, reward: float, max_steps: int) -> bool:
    """
    Determine if the episode should end.

    Rules (in priority order):
    1. Hard limit: max_steps reached → always done
    2. min_actions not yet reached → never done early
    3. Required sequence: each action in required_sequence must appear
       at least as many times as it appears in the list → done
       (e.g. ['migrate_api', 'migrate_api'] requires 2 migrate_api calls)
    4. Single-step tasks (min_actions=1, no required_sequence): threshold met → done
    5. Otherwise: not done

    BUG FIX: Previously used `all(a in all_actions ...)` which treated
    ['migrate_api', 'migrate_api'] as satisfied after just 1 migrate_api call
    because Python `in` checks set membership, not count.
    Now uses Counter to check that each action appears enough times.
    """
    next_step = session.step_count + 1
    case = session.task_case
    done_conditions = case.get('done_conditions', {})
    min_actions = done_conditions.get('min_actions', 1)
    required_seq = done_conditions.get('required_sequence', [])

    # Rule 1: Hard limit — always terminates
    if next_step >= max_steps:
        return True

    # Build the full action history including the current action
    all_actions = session.last_actions + [action.get('action_type', '')]

    # Rule 2: min_actions guard — episode cannot end before this many steps
    if next_step < min_actions:
        return False

    # Rule 3: Required sequence check using COUNTS not set membership
    # This correctly handles repeated actions like ['migrate_api', 'migrate_api']
    if required_seq:
        from collections import Counter
        required_counts = Counter(required_seq)
        actual_counts   = Counter(all_actions)
        # Every required action must appear at least as many times as required
        seq_complete = all(
            actual_counts[act] >= count
            for act, count in required_counts.items()
        )
        if seq_complete:
            return True
        return False  # required_seq defined but not complete → keep going

    # Rule 4: Single-step tasks with no required sequence — threshold met
    if min_actions == 1:
        threshold = case.get('completion_threshold', 0.85)
        if reward >= threshold:
            return True

    return False


def build_initial_obs(session: SessionState) -> dict:
    """Build the initial observation returned by /reset."""
    case = session.task_case
    task_type = session.task_type
    task_id = session.task_id

    obs = {
        'task_type': task_type,
        'task_id': task_id,
        'task_subtype': case.get('task_subtype', 'standard'),
        'task_description': case.get('task_description', ''),
        'turn': 0,
        'done': False,
    }

    if task_type == 'security':
        obs['code_snippet'] = case.get('tool_call', '')
        obs['reviewer_feedback'] = None
        obs['available_actions'] = [
            {'name': 'identify_vulnerability',
             'params': ['vuln_type:str', 'cvss_score:float', 'severity:str', 'affected_line:int']},
            {'name': 'propose_fix',
             'params': ['fix_code:str', 'explanation:str']},
            {'name': 'revise_fix',
             'params': ['fix_code:str', 'addressed_feedback:str']},
        ]

    elif task_type == 'dependency':
        obs['code_snippet'] = case.get('code_snippet', '')
        subtype = case.get('task_subtype', '')
        if subtype == 'flag':
            obs['requirements'] = case.get('requirements', {})
            obs['available_actions'] = [
                {'name': 'flag_outdated',
                 'params': ['packages:dict', 'deprecated_api:str|null', 'replacement:str|null']},
            ]
        elif subtype == 'resolve':
            obs['conflict_packages'] = case.get('conflict_packages', [])
            obs['compatibility_matrix'] = case.get('compatibility_matrix', {})
            obs['current_requirements'] = case.get('requirements', {})
            obs['available_actions'] = [
                {'name': 'resolve_conflict',
                 'params': ['packages:dict', 'reasoning:str']},
            ]
        elif subtype == 'migrate':
            obs['graph_break_report'] = case.get('graph_break_report', case.get('break_descriptions', []))
            obs['available_actions'] = [
                {'name': 'migrate_api',
                 'params': ['completed_items:list', 'code_changes:dict']},
                {'name': 'validate_tree',
                 'params': ['completed_items:list']},
            ]

    elif task_type == 'clinical':
        obs['patient_id'] = case.get('patient_id', '')
        obs['events'] = case.get('events', case.get('patient_events', []))
        obs['available_steps'] = case.get('available_steps', [])
        if task_id in ('cli_medium', 'cli_hard'):
            obs['dependency_graph'] = case.get('dependency_graph', {})
        obs['available_actions'] = [
            {'name': 'detect_gap',
             'params': ['missing_steps:list', 'risk_level:str']},
            {'name': 'rank_issues',
             'params': ['priority_order:list']},
            {'name': 'order_steps',
             'params': ['recovery_steps:list']},
        ]

    return obs


def _build_step_obs(session: SessionState, action: Dict, reward: float, done: bool) -> Dict:
    """Build observation returned after each step()."""
    case = session.task_case
    task_type = session.task_type

    obs = {
        'task_type': task_type,
        'task_id': session.task_id,
        'task_subtype': case.get('task_subtype', 'standard'),
        'turn': session.step_count + 1,
        'done': done,
        'last_reward': round(reward, 4),
    }

    if done:
        obs['message'] = 'Episode complete.'
        return obs

    if task_type == 'security':
        obs['task_description'] = case.get('task_description', '')
        obs['code_snippet'] = case.get('tool_call', '')
        atype = action.get('action_type', '')
        if atype == 'propose_fix':
            fb = case.get('reviewer_feedback', '')
            if fb:
                obs['reviewer_feedback'] = fb
        elif atype == 'revise_fix':
            fb_seq = case.get('reviewer_feedback_sequence', [])
            if fb_seq:
                fb_idx = min(len(session.history), len(fb_seq) - 1)
                if fb_idx >= 0:
                    obs['reviewer_feedback'] = fb_seq[fb_idx]
        obs['available_actions'] = [
            {'name': 'identify_vulnerability',
             'params': ['vuln_type:str', 'cvss_score:float', 'severity:str', 'affected_line:int']},
            {'name': 'propose_fix',
             'params': ['fix_code:str', 'explanation:str']},
            {'name': 'revise_fix',
             'params': ['fix_code:str', 'addressed_feedback:str']},
        ]

    elif task_type == 'dependency':
        obs['task_description'] = case.get('task_description', '')
        obs['code_snippet'] = case.get('code_snippet', '')
        subtype = case.get('task_subtype', '')
        if subtype == 'migrate':
            obs['graph_break_report'] = case.get('graph_break_report', case.get('break_descriptions', []))
            obs['available_actions'] = [
                {'name': 'migrate_api', 'params': ['completed_items:list', 'code_changes:dict']},
                {'name': 'validate_tree', 'params': ['completed_items:list']},
            ]
        elif subtype == 'resolve':
            obs['conflict_packages'] = case.get('conflict_packages', [])
            obs['compatibility_matrix'] = case.get('compatibility_matrix', {})
            obs['available_actions'] = [
                {'name': 'resolve_conflict', 'params': ['packages:dict', 'reasoning:str']},
            ]
        else:
            obs['available_actions'] = [
                {'name': 'flag_outdated',
                 'params': ['packages:dict', 'deprecated_api:str|null', 'replacement:str|null']},
            ]

    elif task_type == 'clinical':
        obs['task_description'] = case.get('task_description', '')
        obs['patient_id'] = case.get('patient_id', '')
        obs['events'] = case.get('events', case.get('patient_events', []))
        obs['available_steps'] = case.get('available_steps', [])
        if session.task_id in ('cli_medium', 'cli_hard'):
            obs['dependency_graph'] = case.get('dependency_graph', {})
        obs['available_actions'] = [
            {'name': 'detect_gap', 'params': ['missing_steps:list', 'risk_level:str']},
            {'name': 'rank_issues', 'params': ['priority_order:list']},
            {'name': 'order_steps', 'params': ['recovery_steps:list']},
        ]

    return obs


def _compute_score_details(action: Dict, session: SessionState) -> Dict[str, float]:
    """Compute per-component score breakdown for UI display."""
    atype = action.get('action_type', '')
    case = session.task_case
    details = {}

    if session.task_type == 'security':
        if atype == 'identify_vulnerability':
            details['vuln_type_match'] = 1.0 if action.get('vuln_type') == case.get('expected_vuln_type') else 0.0
            lo, hi = case.get('cvss_range', [0, 10])
            try:
                v = float(action.get('cvss_score', -1))
                details['cvss_in_range'] = 1.0 if lo <= v <= hi else (0.4 if abs(v - (lo + hi) / 2) <= 1.5 else 0.0)
            except (TypeError, ValueError):
                details['cvss_in_range'] = 0.0
            details['severity_match'] = 1.0 if action.get('severity') == case.get('expected_severity') else 0.0
        elif atype == 'propose_fix':
            tokens = case.get('required_fix_tokens', [])
            if isinstance(tokens, dict):
                tokens = tokens.get(case.get('expected_vuln_type', ''), [])
            tokens = [t for t in tokens if isinstance(t, str)]
            fix = action.get('fix_code', '')
            details['token_coverage'] = sum(1 for t in tokens if t.lower() in fix.lower()) / max(len(tokens), 1) if fix else 0.0
            key_id = case.get('must_preserve_identifier', '')
            details['id_preserved'] = 1.0 if key_id and key_id in fix else 0.0
        elif atype == 'revise_fix':
            kws = case.get('current_feedback_keywords', [])
            addressed = action.get('addressed_feedback', '')
            details['feedback_addressed'] = sum(1 for kw in kws if kw.lower() in addressed.lower()) / max(len(kws), 1) if addressed else 0.0

    elif session.task_type == 'dependency':
        if atype == 'flag_outdated':
            expected = set(case.get('expected_outdated_packages', []))
            provided = set(action.get('packages', {}).keys())
            if expected:
                tp = len(expected & provided)
                p = tp / max(len(provided), 1)
                r = tp / max(len(expected), 1)
                details['pkg_f1'] = round(2 * p * r / max(p + r, 0.001), 4)
            details['api_match'] = 1.0 if action.get('deprecated_api') == case.get('expected_deprecated_api') else 0.0
        elif atype == 'resolve_conflict':
            proposed = action.get('packages', {})
            conflict = case.get('conflict_packages', [])
            details['packages_proposed'] = len(proposed)
            details['conflict_count'] = len(conflict)
        elif atype in ('migrate_api', 'validate_tree'):
            checklist = case.get('graph_breaks', [])
            completed = action.get('completed_items', [])
            details['items_completed'] = len(completed)
            details['total_items'] = len(checklist)

    elif session.task_type == 'clinical':
        if atype == 'detect_gap':
            expected = set(case.get('expected_missing_steps', []))
            provided = set(action.get('missing_steps', []))
            if expected:
                tp = len(expected & provided)
                p = tp / max(len(provided), 1)
                r = tp / max(len(expected), 1)
                details['step_f1'] = round(2 * p * r / max(p + r, 0.001), 4)
            details['risk_match'] = 1.0 if action.get('risk_level') == case.get('expected_risk') else 0.0
        elif atype == 'rank_issues':
            expected = case.get('priority_order', [])
            provided = action.get('priority_order', [])
            details['ranking_overlap'] = len(set(expected) & set(provided)) / max(len(expected), 1) if expected else 0.0
        elif atype == 'order_steps':
            expected = case.get('required_steps', case.get('expected_missing_steps', []))
            provided = action.get('recovery_steps', [])
            details['steps_overlap'] = len(set(expected) & set(provided)) / max(len(expected), 1) if expected else 0.0

    return details
