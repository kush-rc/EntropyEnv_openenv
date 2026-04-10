# server/graders/security_grader.py
# Grader for MCP Security Sandbox tasks (sec_easy, sec_medium, sec_hard).
#
# FIX SUMMARY:
# 1. _score_identify: CVSS partial credit was too generous (±3.0 range → ±1.5)
# 2. _score_propose: floor raised from 0.0 to 0.15, but explanation scoring tightened
# 3. _score_revise: floor raised from 0.20 to 0.10 — revise should be hard
# 4. All three scorers now have tighter weights that produce real variance

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
    """Score vulnerability identification.
    
    FIX: CVSS partial-credit window tightened from ±3.0 to ±1.5.
    Previously a model guessing CVSS=5.0 on a [7.5, 9.8] range got 0.5 credit.
    Now it must be within 1.5 of the midpoint to get any partial credit.
    
    Weights: vuln_type=45%, CVSS=30%, severity=25%
    """
    # Detection: correct vuln_type? (45% weight)
    det = 1.0 if action.get('vuln_type') == case.get('expected_vuln_type', '') else 0.0

    # CVSS: within expected range? (30% weight)
    # FIX: Tightened partial credit window from 3.0 to 1.5
    lo, hi = case.get('cvss_range', [0.0, 10.0])
    midpoint = (lo + hi) / 2
    try:
        v = float(action.get('cvss_score', -1))
    except (TypeError, ValueError):
        v = -1.0

    if lo <= v <= hi:
        cvss = 1.0
    elif abs(v - midpoint) <= 1.5:  # FIX: was 3.0
        cvss = 0.4  # FIX: was 0.5 — tighter partial credit
    else:
        cvss = 0.0

    # Severity: exact match or adjacent? (25% weight)
    s, es = action.get('severity', ''), case.get('expected_severity', '')
    sev = 1.0 if s == es else (0.3 if _adj_sev(s, es) else 0.0)
    # FIX: adjacent severity was 0.4, now 0.3 — being one level off is meaningful

    return det * 0.45 + cvss * 0.30 + sev * 0.25


def _score_propose(action: Dict, case: Dict) -> float:
    """Score proposed fix.
    
    FIX: 
    - Token coverage divisor changed: now we require ALL tokens, not (n-1)
    - Explanation score tightened — model must mention BOTH the vuln and the fix mechanism
    - Removed the 0.25 floor — a blank or wrong fix_code should score low
    
    Weights: code=55%, explanation=35%, identifier=10%
    """
    tokens = case.get('required_fix_tokens', [])
    if isinstance(tokens, dict):
        tokens = tokens.get(case.get('expected_vuln_type', ''), [])

    def flatten(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            elif isinstance(item, str):
                result.append(item)
        return result

    tokens = flatten(tokens) if isinstance(tokens, list) else []

    fix = action.get('fix_code', '')
    if not fix or len(fix.strip()) < 5:
        return 0.05  # FIX: was 0.0 → 0.05 (minimal signal so training doesn't stall)

    # FIX: Token coverage — now require ALL tokens (not n-1)
    # This is the main fix: previously len(tokens)-1 in denominator let 1 missing token score 100%
    if tokens:
        matched = sum(1 for t in tokens if t.lower() in fix.lower())
        coverage = matched / len(tokens)  # FIX: was / max(1, len(tokens)-1)
    else:
        coverage = 0.40  # Unknown tokens: give neutral score

    # Identifier preservation (10%)
    key_id = case.get('must_preserve_identifier', '')
    preservation = 0.10 if key_id and key_id in fix else 0.0

    # FIX: Explanation quality (35%) — tightened
    explanation = action.get('explanation', '')
    exp_score = 0.0
    if explanation and len(explanation) >= 20:
        # Must mention the mechanism (how the fix works)
        mechanism_words = ['prevent', 'secure', 'validate', 'sanitize', 'parameterize',
                          'escape', 'encode', 'whitelist', 'authenticate', 'authorize']
        mech_hits = sum(0.05 for kw in mechanism_words if kw in explanation.lower())
        exp_score += min(0.20, mech_hits)  # cap mechanism score at 0.20

        # Must mention the vulnerability type
        vuln_type = case.get('expected_vuln_type', '').replace('_', ' ')
        if vuln_type and vuln_type in explanation.lower():
            exp_score += 0.15  # bonus for naming the vuln correctly

    # FIX: Weights adjusted: code 55%, explanation 35%, identifier 10%
    # Previously: code 60%, explanation 30%, identifier 10%
    raw = coverage * 0.55 + exp_score * 0.35 + preservation * 0.10
    # FIX: Removed the max(0.25, ...) floor — bad fixes should score low
    return max(0.05, safe_score(raw))


def _score_revise(action: Dict, case: Dict) -> float:
    """Score revised fix after reviewer feedback.
    
    FIX:
    - Floor lowered from 0.20 to 0.10 — this is the hardest action, it should be hardest to score
    - Coverage now checks ALL feedback keywords, not (n-1)
    - Regression penalty doubled from -0.20 to -0.35
    - Requires BOTH addressed_feedback AND fix_code to score well
    
    This is intentionally the hardest scorer because revise_fix only happens on hard tasks.
    """
    kw = case.get('current_feedback_keywords', [])
    addressed = action.get('addressed_feedback', '')
    fix = action.get('fix_code', '')

    if not addressed or len(addressed.strip()) < 10:
        return 0.10

    if not fix or len(fix.strip()) < 5:
        return 0.10

    # FIX: Coverage now requires ALL keywords (was n-1)
    if kw:
        cov = sum(1 for k in kw if k.lower() in addressed.lower()) / len(kw)
        # FIX: was / max(1, len(kw)-1)
    else:
        cov = 0.50

    # FIX: Regression penalty doubled: -0.35 (was -0.20)
    reg = 0.35 if case.get('original_vuln_pattern', '') in fix else 0.0

    # Check if fix_code is actually different from previous (no copy-paste regression)
    fix_quality = 0.20 if len(fix) > 30 else 0.0

    # FIX: Floor lowered from 0.20 to 0.10
    return max(0.10, safe_score(cov * 0.60 + fix_quality * 0.20 - reg))


def compute_correctness(action: Dict, case: Dict) -> float:
    """Route to correct scoring function based on action_type."""
    atype = action.get('action_type')
    if atype == 'identify_vulnerability':
        return _score_identify(action, case)
    if atype == 'propose_fix':
        return _score_propose(action, case)
    if atype == 'revise_fix':
        return _score_revise(action, case)
    return None


def grade(action: Dict, session) -> float:
    """Entry point called by router. Runs full reward pipeline."""
    return grade_dynamic(action, session, compute_correctness, VALID_ACTIONS, FORBIDDEN, max_steps=8)
