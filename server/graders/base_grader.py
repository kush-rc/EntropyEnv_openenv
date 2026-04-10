# server/graders/base_grader.py
# Core grading utilities used by ALL domain graders.
# FIX: safe_score now uses [0.01, 0.99] range but with REAL variance in between.
# The key issue was that graders were returning values too close to 1.0 for partial answers.

from typing import Dict, Any, List, Callable


def safe_score(raw) -> float:
    """Clamp to [0.01, 0.99]. Never crash. Returns float with 4 decimal precision."""
    if raw is None:
        return 0.01
    try:
        val = float(raw)
        # FIX: Don't round aggressively — keep 4 decimal places so variance is visible
        return round(max(0.01, min(0.99, val)), 4)
    except (TypeError, ValueError):
        return 0.01


def repetition_penalty(action_type: str, last_actions: List[str], window: int = 3) -> float:
    """Penalise repeating the same action type in the last N steps."""
    count = last_actions[-window:].count(action_type)
    # FIX: Increased penalty from -0.15 to -0.20 per repeat so it actually stings
    return -0.20 * count


def invalid_action_penalty(action_type: str, valid_actions: List[str]) -> float:
    """Penalise actions not in the valid set for this domain."""
    # FIX: Increased from -0.20 to -0.40 — wrong domain is a serious mistake
    return -0.40 if action_type not in valid_actions else 0.0


def harmful_output_penalty(action: Dict, forbidden_patterns: List[str]) -> float:
    """Penalise destructive patterns like 'os.remove' or 'drop table'."""
    action_str = str(action).lower()
    for p in forbidden_patterns:
        if p.lower() in action_str:
            return -0.50
    return 0.0


def efficiency_bonus(step_count: int, max_steps: int, done: bool) -> float:
    """Small bonus for finishing early. FIX: reduced from 0.10 to 0.05 so it doesn't
    inflate scores — the correctness score should be the main signal."""
    return 0.05 if done and step_count < max_steps // 2 else 0.0


def difficulty_multiplier(task_id: str) -> float:
    """
    FIX: NEW FUNCTION — Scale raw correctness by task difficulty so easy tasks 
    genuinely can't score as high as hard tasks even with correct answers.
    
    - easy tasks: correctness score is NOT boosted (agents should get high scores)
    - medium tasks: a perfect answer gets 0.90 max (10% cap)  
    - hard tasks: a perfect answer gets 0.80 max (20% cap) — they're SUPPOSED to be hard
    
    This ensures there's real spread between easy/medium/hard scores.
    """
    if 'hard' in task_id:
        return 0.80
    elif 'medium' in task_id:
        return 0.90
    else:
        return 0.99  # easy — allow near-perfect


def grade_dynamic(
    action: Dict[str, Any],
    session,
    compute_correctness_fn: Callable,
    valid_actions: List[str],
    forbidden_patterns: List[str] = None,
    max_steps: int = 8
) -> float:
    """Full reward pipeline. Entry point for all domain graders.

    Pipeline: invalid check → repetition → correctness → harmful → efficiency → difficulty cap → clamp
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

    if correctness is None:
        correctness = 0.0

    # FIX: Apply difficulty cap BEFORE efficiency bonus
    task_id = getattr(session, 'task_id', '')
    max_allowed = difficulty_multiplier(task_id)
    correctness = min(correctness, max_allowed)

    # Efficiency bonus — small
    eff = efficiency_bonus(session.step_count + 1, max_steps, correctness >= 0.75)

    # Combine and clamp
    raw = correctness + rep + harm + eff
    return safe_score(raw)
