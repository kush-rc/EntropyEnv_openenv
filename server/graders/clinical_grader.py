# server/graders/clinical_grader.py
# Grader for Clinical Workflow Chaos Simulator tasks (cli_easy, cli_medium, cli_hard).
# Bug 2 FIXED: propose_recovery is NOT in VALID_ACTIONS.
# Uses NDCG ranking and dependency violation counting.

import math
from typing import Dict, List
from .base_grader import grade_dynamic, safe_score

# Bug 2 FIX: propose_recovery is NOT here — it has no grader branch
VALID_ACTIONS = ['detect_gap', 'rank_issues', 'order_steps']
FORBIDDEN = []
RISK_ORDER = ['low', 'medium', 'high', 'critical']


def _adj_risk(predicted, target):
    """Check if risk level is off by exactly one level (partial credit)."""
    try:
        return abs(RISK_ORDER.index(predicted) - RISK_ORDER.index(target)) == 1
    except ValueError:
        return False


def _f1(predicted: List, expected: List) -> float:
    """Compute F1 score between predicted and expected lists."""
    if not expected:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    p_s = set(str(x).strip() for x in predicted)
    e_s = set(str(x).strip() for x in expected)
    tp = len(p_s & e_s)
    prec = tp / len(p_s) if p_s else 0.0
    rec = tp / len(e_s) if e_s else 0.0
    return round(2 * prec * rec / max(prec + rec, 0.001), 4)


def _ndcg(predicted: List, ideal: List, k: int = None) -> float:
    """NDCG@k: rewards getting highest-priority items ranked first.

    If ideal = ['insurance_auth', 'pre_op_consent', 'book_specialist']:
      - Getting 'insurance_auth' first is worth more than getting it last.
      - Each position is worth less than the previous (logarithmic discount).
      - NDCG=1.0 means perfect ranking. NDCG=0.0 means completely reversed.
    """
    if not ideal:
        return 1.0
    if k is None:
        k = len(ideal)

    def dcg(order):
        score = 0.0
        for i, item in enumerate(order[:k]):
            if item in ideal:
                relevance = len(ideal) - ideal.index(item)
                score += relevance / math.log2(i + 2)
        return score

    ideal_dcg = dcg(ideal)
    return round(dcg(predicted) / ideal_dcg, 4) if ideal_dcg > 0 else 0.0


def _count_violations(proposed: List, dep_graph: Dict) -> int:
    """Count steps where a prerequisite appears AFTER the step needing it."""
    violations = 0
    for i, step in enumerate(proposed):
        for prereq in dep_graph.get(step, []):
            if prereq not in proposed[:i]:
                violations += 1
    return violations


def _score_detect(action: Dict, case: Dict) -> float:
    """Score gap detection (cli_easy). F1 on missing steps + risk level match."""
    exp = case.get('expected_missing_steps', [])
    pred = action.get('missing_steps', [])

    # Normalize to lists
    if isinstance(exp, str):
        exp = [exp]
    if isinstance(pred, str):
        pred = [pred]

    # F1 on missing step detection (65% weight)
    step_score = _f1(pred, exp)

    # Risk level match: exact or adjacent (35% weight)
    er = case.get('expected_risk', '')
    pr = action.get('risk_level', '')
    risk_score = 1.0 if pr == er else (0.5 if _adj_risk(pr, er) else 0.0)

    return 0.65 * step_score + 0.35 * risk_score


def _score_rank(action: Dict, case: Dict) -> float:
    """Score priority ranking (cli_medium). Completeness + NDCG ordering."""
    ideal = case.get('priority_order', [])
    predicted = action.get('priority_order', [])

    if not ideal:
        return 0.5

    # Filter predicted to only include valid step IDs (prevents hallucinated IDs from scoring)
    valid_ids = set(case.get('available_steps', []))
    if valid_ids:
        predicted = [p for p in predicted if p in valid_ids]

    # Completeness: are all items present? (40% weight)
    completeness = _f1(predicted, ideal)

    # Ranking quality: NDCG (60% weight)
    ranking = _ndcg(predicted, ideal)

    return 0.40 * completeness + 0.60 * ranking


def _score_order(action: Dict, case: Dict) -> float:
    """Score dependency-ordered recovery (cli_hard). Order + completeness + efficiency."""
    dep_graph = case.get('dependency_graph', {})
    required = case.get('required_steps', [])
    proposed = action.get('recovery_steps', [])

    if not proposed:
        return 0.0

    # Dependency violations: -0.25 each (40% weight)
    viol = _count_violations(proposed, dep_graph)
    order = max(0.0, 1.0 - viol * 0.25)

    # Completeness: F1 against required steps (40% weight)
    completeness = _f1(proposed, required)

    # Efficiency: penalize extra unnecessary steps (20% weight)
    extra = max(0, len(proposed) - len(required))
    efficiency = max(0.0, 1.0 - extra * 0.10)

    return safe_score(order * 0.40 + completeness * 0.40 + efficiency * 0.20)


def compute_correctness(action: Dict, case: Dict) -> float:
    """Route to correct scoring function based on action_type."""
    atype = action.get('action_type')
    if atype == 'detect_gap':
        return _score_detect(action, case)
    if atype == 'rank_issues':
        return _score_rank(action, case)
    if atype == 'order_steps':
        return _score_order(action, case)
    return None


def grade(action: Dict, session) -> float:
    """Entry point called by router. Runs full reward pipeline."""
    return grade_dynamic(action, session, compute_correctness, VALID_ACTIONS, FORBIDDEN, max_steps=6)
