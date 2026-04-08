# server/graders/base_grader.py
# Core grading utilities used by ALL domain graders.
# Contains: safe_score (Bug 1 fix), penalty functions, grade_dynamic entry point.

from typing import Dict, Any, List, Callable


def safe_score(raw) -> float:
    """Always clamp to [0.0, 1.0]. Never crash. Handles None, str, out-of-range."""
    if raw is None:
        return 0.0                    # BUG 1 FIX — must be first line
    try:
        return round(max(0.0, min(1.0, float(raw))), 4)
    except (TypeError, ValueError):
        return 0.0


def repetition_penalty(action_type: str, last_actions: List[str], window: int = 3) -> float:
    """Penalise repeating the same action type in the last N steps."""
    count = last_actions[-window:].count(action_type)
    return -0.15 * count


def invalid_action_penalty(action_type: str, valid_actions: List[str]) -> float:
    """Penalise actions not in the valid set for this domain."""
    return -0.20 if action_type not in valid_actions else 0.0


def harmful_output_penalty(action: Dict, forbidden_patterns: List[str]) -> float:
    """Penalise destructive patterns like 'os.remove' or 'drop table'."""
    action_str = str(action).lower()
    for p in forbidden_patterns:
        if p.lower() in action_str:
            return -0.30
    return 0.0


def efficiency_bonus(step_count: int, max_steps: int, done: bool) -> float:
    """Reward finishing early (before half the max steps)."""
    return 0.10 if done and step_count < max_steps // 2 else 0.0


def grade_dynamic(
    action: Dict[str, Any],
    session,
    compute_correctness_fn: Callable,
    valid_actions: List[str],
    forbidden_patterns: List[str] = None,
    max_steps: int = 8
) -> float:
    """Full reward pipeline. Entry point for all domain graders.

    Pipeline: invalid check → repetition → correctness → harmful → efficiency → clamp
    """
    if forbidden_patterns is None:
        forbidden_patterns = []

    action_type = action.get('action_type', 'unknown')

    # Penalties
    inv  = invalid_action_penalty(action_type, valid_actions)
    rep  = repetition_penalty(action_type, session.last_actions)
    harm = harmful_output_penalty(action, forbidden_patterns)

    # If action type is invalid, skip the grader entirely
    if inv < 0:
        return safe_score(inv + rep)

    # Core correctness score from domain-specific grader
    correctness = compute_correctness_fn(action, session.task_case)

    # Efficiency bonus — session.done is always False at this point (set by router
    # AFTER grade() returns), so use correctness >= 0.8 as proxy for "solved well"
    eff = efficiency_bonus(session.step_count + 1, max_steps, correctness is not None and correctness >= 0.8)

    # Combine and clamp
    raw = correctness + rep + harm + eff
    return safe_score(raw)

