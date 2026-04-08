# server/graders/security_grader.py
# Grader for MCP Security Sandbox tasks (sec_easy, sec_medium, sec_hard).
# Bug 4 FIXED: _score_identify does NOT early-return on wrong vuln_type.

from typing import Dict
from .base_grader import grade_dynamic, safe_score

VALID_ACTIONS = ['identify_vulnerability', 'propose_fix', 'revise_fix']
FORBIDDEN = ['os.remove', 'shutil.rmtree', 'drop table', 'delete from']
SEV_ORDER = ['low', 'medium', 'high', 'critical']


def _adj_sev(predicted, target):
    """Check if severity is off by exactly one level (partial credit)."""
    try:
        return abs(SEV_ORDER.index(predicted) - SEV_ORDER.index(target)) == 1
    except ValueError:
        return False


def _score_identify(action: Dict, case: Dict) -> float:
    """Score vulnerability identification. Bug 4 FIX: always score all 3 components."""
    # Detection: correct vuln_type? (45% weight)
    det = 1.0 if action.get('vuln_type') == case.get('expected_vuln_type', '') else 0.0

    # BUG 4 FIX: do NOT early-return here. Always score CVSS and severity.
    # This gives the agent partial credit even when vuln_type is wrong.

    # CVSS: within expected range? (30% weight)
    lo, hi = case.get('cvss_range', [0.0, 10.0])
    v = float(action.get('cvss_score', -1))
    cvss = 1.0 if lo <= v <= hi else (0.5 if abs(v - (lo + hi) / 2) <= 3.0 else 0.0)

    # Severity: exact match or adjacent? (25% weight)
    s, es = action.get('severity', ''), case.get('expected_severity', '')
    sev = 1.0 if s == es else (0.4 if _adj_sev(s, es) else 0.0)

    return det * 0.45 + cvss * 0.30 + sev * 0.25


def _score_propose(action: Dict, case: Dict) -> float:
    """Score proposed fix. Checks token coverage and identifier preservation."""
    tokens = case.get('required_fix_tokens', [])
    if isinstance(tokens, dict):
        tokens = tokens.get(case.get('expected_vuln_type', ''), [])
    # Safety: flatten to list of strings only
    tokens = [t for t in tokens if isinstance(t, str)]

    fix = action.get('fix_code', '')
    if not fix:
        return 0.0

    # Token coverage: allow missing 1 token to still get full score
    if not tokens:
        coverage = 0.5
    else:
        divisor = max(1, len(tokens) - 1)
        coverage = min(1.0, sum(1 for t in tokens if t.lower() in fix.lower()) / divisor)

    # Identifier preservation: did the fix keep the key function name?
    key_id = case.get('must_preserve_identifier', '')
    preservation = 0.15 if key_id and key_id in fix else 0.0

    # Floor: any non-empty fix_code gets at least 0.25 (agent showed correct workflow)
    return max(0.25, safe_score(coverage + preservation))


def _score_revise(action: Dict, case: Dict) -> float:
    """Score revised fix after reviewer feedback. Checks coverage and regression."""
    kw = case.get('current_feedback_keywords', [])
    addressed = action.get('addressed_feedback', '')
    fix = action.get('fix_code', '')

    # Feedback keyword coverage: allow missing 1 keyword
    divisor = max(1, len(kw) - 1)
    cov = min(1.0, sum(1 for k in kw if k.lower() in addressed.lower()) / divisor)

    # Regression check: does the fix_code still contain the original vulnerability? (-20%)
    reg = 0.20 if case.get('original_vuln_pattern', '') in fix else 0.0

    # Floor: any non-empty addressed_feedback gets at least 0.20
    return max(0.20, safe_score(cov - reg))


def compute_correctness(action: Dict, case: Dict) -> float:
    """Route to correct scoring function based on action_type."""
    atype = action.get('action_type')
    if atype == 'identify_vulnerability':
        return _score_identify(action, case)
    if atype == 'propose_fix':
        return _score_propose(action, case)
    if atype == 'revise_fix':
        return _score_revise(action, case)
    return None  # safe_score(None) = 0.0


def grade(action: Dict, session) -> float:
    """Entry point called by router. Runs full reward pipeline."""
    return grade_dynamic(action, session, compute_correctness, VALID_ACTIONS, FORBIDDEN, max_steps=8)
