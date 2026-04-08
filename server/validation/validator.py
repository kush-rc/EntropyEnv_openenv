# server/validation/validator.py
# 3-stage pre-action validation: Schema → Domain → Consistency.
# IMPORTANT: Validator should HELP agents, not trap them.
# - Auto-coerce types where possible (string "8.5" → float 8.5)
# - Only hard-reject truly unrecoverable actions (wrong domain)
# - Silently truncate oversized fields instead of rejecting
# - Rich hints so agent can self-correct on next step

from typing import Dict, Tuple

VALID_VULN_TYPES = {
    'sql_injection', 'xss', 'idor', 'hardcoded_secret', 'missing_auth',
    'jwt_misuse', 'path_traversal', 'ssrf', 'rate_limit_missing', 'xxe'
}
VALID_SEVERITIES = {'critical', 'high', 'medium', 'low'}
VALID_RISK_LEVELS = {'critical', 'high', 'medium', 'low'}

# Which actions belong to which domain
DOMAIN_ACTIONS = {
    'security':   {'identify_vulnerability', 'propose_fix', 'revise_fix'},
    'dependency': {'flag_outdated', 'resolve_conflict', 'migrate_api', 'validate_tree'},
    'clinical':   {'detect_gap', 'rank_issues', 'order_steps'},
}

# Required fields and their types for each action
ACTION_SCHEMAS = {
    'identify_vulnerability': {
        'vuln_type': str,
        'cvss_score': (int, float),
        'severity': str,
    },
    'propose_fix': {
        'fix_code': str,
        'explanation': str,
    },
    'revise_fix': {
        'fix_code': str,
        'addressed_feedback': str,
    },
    'flag_outdated': {
        'packages': dict,
        # deprecated_api and replacement are optional — handled below
    },
    'resolve_conflict': {
        'packages': dict,
        'reasoning': str,
    },
    'migrate_api': {
        'completed_items': list,
        'code_changes': dict,
    },
    'validate_tree': {
        'completed_items': list,
    },
    'detect_gap': {
        'missing_steps': list,
        'risk_level': str,
    },
    'rank_issues': {
        'priority_order': list,
    },
    'order_steps': {
        'recovery_steps': list,
    },
}

# Fields that are optional (won't cause hard rejection if missing)
OPTIONAL_FIELDS = {
    'flag_outdated': {'deprecated_api', 'replacement'},
    'identify_vulnerability': {'affected_line'},
}


def _coerce(action: Dict, schema: Dict) -> Dict:
    """Try to coerce field types before validating. Modifies action in-place.
    
    This is critical for model compatibility — different LLMs output
    numbers as strings, lists as comma-separated strings, etc.
    """
    for field, expected_type in schema.items():
        if field not in action:
            continue
        val = action[field]
        # Already correct type
        if isinstance(val, expected_type):
            continue
        # Try coercions
        try:
            target = expected_type[0] if isinstance(expected_type, tuple) else expected_type
            if target == float:
                action[field] = float(val)
            elif target == int:
                action[field] = int(val)
            elif target == str and not isinstance(val, str):
                action[field] = str(val)
            elif target == list and isinstance(val, str):
                # Try JSON parse first, then comma split
                try:
                    import json as _j
                    parsed = _j.loads(val)
                    if isinstance(parsed, list):
                        action[field] = parsed
                except Exception:
                    action[field] = [x.strip(' "\'') for x in val.strip('[]').split(',') if x.strip()]
            elif target == dict and isinstance(val, str):
                import json as _j
                action[field] = _j.loads(val)
        except Exception:
            pass  # Leave as-is; domain check will catch real problems
    return action


def validate_action(action: Dict, session) -> Tuple[bool, Dict]:
    """3-stage validation. Returns (is_valid, feedback_observation).

    Philosophy: be lenient on format (coerce types), strict on cross-domain actions.
    An action in the wrong domain = hard reject.
    An action with slightly wrong types = coerce and pass through.
    """
    atype = action.get('action_type', '')

    # ── Stage 1: Is this a known action type? ──
    all_valid = set(ACTION_SCHEMAS.keys())
    if atype not in all_valid:
        return False, _fb(
            'invalid_action_type',
            f'Unknown action_type: {repr(atype)}',
            session,
            hint=f'Valid actions for {session.task_type}: {sorted(DOMAIN_ACTIONS.get(session.task_type, []))}',
        )

    # ── Cross-domain check FIRST (before coercion) ──
    domain_valid = DOMAIN_ACTIONS.get(session.task_type, set())
    if atype not in domain_valid:
        return False, _fb(
            'wrong_domain_action',
            f'{repr(atype)} is not valid for task_type={repr(session.task_type)}',
            session,
            hint=f'Valid actions: {sorted(domain_valid)}',
        )

    # ── Coerce types before schema check (be helpful to all models) ──
    schema = ACTION_SCHEMAS.get(atype, {})
    action = _coerce(action, schema)

    # ── Stage 2: Check required fields are present ──
    optional = OPTIONAL_FIELDS.get(atype, set())
    required_fields = [f for f in schema if f not in optional]
    missing = [f for f in required_fields if f not in action]
    if missing:
        return False, _fb(
            'missing_fields',
            f'Missing required fields: {missing}',
            session,
            hint=f'Required for {atype}: {required_fields}',
        )

    # ── Stage 3: Domain value validation ──
    errs = _domain_check(action, atype)
    if errs:
        return False, _fb(
            'domain_error',
            f'Invalid field values: {errs}',
            session,
            hint=_domain_hint(atype, errs),
        )

    # ── Stage 4: Consistency check ──
    cons = _consistency_check(action, atype, session)
    if cons:
        return False, _fb('consistency_error', cons['message'], session, hint=cons['hint'])

    return True, {}


def _domain_check(action: Dict, atype: str) -> list:
    """Check values are within allowed ranges/enums. Returns list of error dicts."""
    errors = []

    if atype == 'identify_vulnerability':
        vt = action.get('vuln_type', '')
        if vt not in VALID_VULN_TYPES:
            errors.append({'field': 'vuln_type', 'value': vt, 'allowed': sorted(VALID_VULN_TYPES)})
        try:
            cvss = float(action.get('cvss_score', -1))
            if not (0.0 <= cvss <= 10.0):
                errors.append({'field': 'cvss_score', 'value': cvss, 'allowed': '0.0 to 10.0'})
        except (TypeError, ValueError):
            errors.append({'field': 'cvss_score', 'value': action.get('cvss_score'), 'allowed': '0.0 to 10.0'})
        sev = action.get('severity', '')
        if sev not in VALID_SEVERITIES:
            errors.append({'field': 'severity', 'value': sev, 'allowed': sorted(VALID_SEVERITIES)})

    elif atype in ('propose_fix', 'revise_fix'):
        fix = action.get('fix_code', '')
        if len(fix) > 2000:
            # Silently truncate instead of rejecting — don't penalize verbose agents
            action['fix_code'] = fix[:2000]

    elif atype == 'detect_gap':
        rl = action.get('risk_level', '')
        if rl not in VALID_RISK_LEVELS:
            errors.append({'field': 'risk_level', 'value': rl, 'allowed': sorted(VALID_RISK_LEVELS)})

    elif atype == 'resolve_conflict':
        pkgs = action.get('packages', {})
        if not isinstance(pkgs, dict) or len(pkgs) == 0:
            errors.append({'field': 'packages', 'issue': 'must be a non-empty dict of {package: version}'})

    elif atype == 'migrate_api':
        items = action.get('completed_items', [])
        changes = action.get('code_changes', {})
        if not isinstance(items, list) or len(items) == 0:
            errors.append({'field': 'completed_items', 'issue': 'must be a non-empty list of break IDs'})
        if not isinstance(changes, dict):
            errors.append({'field': 'code_changes', 'issue': 'must be a dict of {break_id: fix_description}'})

    return errors


def _domain_hint(atype: str, errors: list) -> str:
    """Generate a helpful hint for domain errors."""
    fields = [e.get('field', '') for e in errors]
    if 'vuln_type' in fields:
        return "vuln_type must be one of: sql_injection, xss, idor, hardcoded_secret, missing_auth, jwt_misuse, path_traversal, ssrf, rate_limit_missing, xxe"
    if 'severity' in fields:
        return "severity must be one of: critical, high, medium, low"
    if 'risk_level' in fields:
        return "risk_level must be one of: critical, high, medium, low"
    if 'cvss_score' in fields:
        return "cvss_score must be a float between 0.0 and 10.0"
    return f"Check field values for: {fields}"


def _consistency_check(action: Dict, atype: str, session) -> dict:
    """Check that action makes sense given session history."""
    hist_types = [h.get('action_type') for h in session.history]

    if atype == 'revise_fix' and 'propose_fix' not in hist_types:
        return {
            'message': 'Cannot call revise_fix before propose_fix',
            'hint': 'Call propose_fix first, then revise_fix if you get reviewer feedback'
        }

    if atype == 'rank_issues' and 'detect_gap' not in hist_types:
        return {
            'message': 'Cannot call rank_issues before detect_gap',
            'hint': 'Call detect_gap first, then rank_issues'
        }

    if atype == 'order_steps' and 'detect_gap' not in hist_types:
        return {
            'message': 'Cannot call order_steps before detect_gap',
            'hint': 'Call detect_gap first, then rank_issues, then order_steps'
        }

    # Reject identical resolve_conflict proposals (infinite loop prevention)
    if atype == 'resolve_conflict':
        for prev in session.history:
            if (prev.get('action_type') == 'resolve_conflict' and
                    prev.get('packages') == action.get('packages', {})):
                return {
                    'message': 'Identical version proposal already submitted — this combination was rejected',
                    'hint': 'Try different package versions. Check the compatibility_matrix in the observation.'
                }

    return {}


def _fb(error_type: str, message: str, session, **kwargs) -> Dict:
    """Build a feedback observation for validation failures."""
    obs = {
        'validation_failed': True,
        'error_type': error_type,
        'message': message,
        'turn': session.step_count,
        'task_type': session.task_type,
        'task_id': getattr(session, 'task_id', ''),
        'available_actions': sorted(DOMAIN_ACTIONS.get(session.task_type, [])),
    }
    obs.update(kwargs)
    return obs
