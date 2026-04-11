# server/graders/base_grader.py
# Core grading utilities used by ALL domain graders.
#
# CHANGES FROM PREVIOUS VERSION:
# 1. difficulty_multiplier() — REMOVED ENTIRELY.
#    The cap (hard→0.80, medium→0.90) made every hard task score identically
#    at 0.80 and every medium task at 0.90, regardless of agent quality.
#    This is exactly the wrong behaviour for an RL training environment:
#    GRPO needs variance WITHIN difficulty levels, not a uniform ceiling.
#    Task difficulty now comes from the grader logic and case design alone.
#
# 2. safe_score range: [0.01, 0.99]
#    The official spec says "strictly between 0 and 1".
#    Discord consensus from many participants confirmed 0.01/0.99 as the
#    correct interpretation. Do not change this back to [0.0, 1.0].
#
# 3. Penalty values kept as-is (increased in last revision):
#    - repetition_penalty:    -0.20 per repeat (was -0.15)
#    - invalid_action_penalty: -0.40 for wrong domain action (was -0.20)
#    - harmful_output_penalty: -0.50 for destructive patterns
#    These are intentionally higher to create real signal.
#
# 4. efficiency_bonus reduced to 0.05 (was 0.10).
#    Small enough that it doesn't inflate scores, but still rewards
#    agents that solve tasks efficiently.

from typing import Dict, Any, List, Callable


def safe_score(raw) -> float:
    """
    Clamp score to [0.01, 0.99]. Never crash. Returns float.

    WHY [0.01, 0.99] NOT [0.0, 1.0]:
    - Official spec says scores must be strictly between 0 and 1
    - Discord confirmed 0.01/0.99 as the correct practical interpretation
    - A score of exactly 0.0 from a broken run looks like a crash
    - A score of exactly 1.0 means the grader is trivially solved

    WHY 4 DECIMAL PLACES:
    - Keeps variance visible (0.4500 vs 0.4750 are meaningfully different)
    - round() handles float precision artifacts
    """
    if raw is None:
        return 0.01
    try:
        val = float(raw)
        return round(max(0.01, min(0.99, val)), 4)
    except (TypeError, ValueError):
        return 0.01


def repetition_penalty(action_type: str, last_actions: List[str], window: int = 3) -> float:
    """
    Penalise repeating the same action type in the last N steps.

    WHY: Without this, GRPO agents discover they can emit the same
    high-scoring action repeatedly within an episode. The penalty
    forces genuine strategy exploration each turn.

    -0.20 per repeat (capped by window=3, so max penalty is -0.60).
    """
    count = last_actions[-window:].count(action_type)
    return -0.20 * count


def invalid_action_penalty(action_type: str, valid_actions: List[str]) -> float:
    """
    Penalise actions not in the valid set for this domain.

    -0.40 because calling a dependency action on a security task is a
    fundamental routing error — it should hurt significantly.
    """
    return -0.40 if action_type not in valid_actions else 0.0


def harmful_output_penalty(action: Dict, forbidden_patterns: List[str]) -> float:
    """
    Penalise destructive patterns like 'os.remove', 'drop table'.

    -0.50 because these patterns represent the agent trying to "cheat"
    by deleting things rather than fixing them.
    """
    action_str = str(action).lower()
    for p in forbidden_patterns:
        if p.lower() in action_str:
            return -0.50
    return 0.0


def efficiency_bonus(step_count: int, max_steps: int, done: bool) -> float:
    """
    Small bonus for finishing early — rewards decisive, confident agents.

    WHY ONLY 0.05: The correctness score must be the dominant signal.
    The efficiency bonus should never flip a mediocre answer into a good score.
    """
    return 0.05 if done and step_count < max_steps // 2 else 0.0


def grade_dynamic(
    action:                Dict[str, Any],
    session,
    compute_correctness_fn: Callable,
    valid_actions:          List[str],
    forbidden_patterns:     List[str] = None,
    max_steps:              int       = 8,
) -> float:
    """
    Full reward pipeline. Entry point for all domain graders.

    Pipeline:
    1. Invalid action check — if wrong domain action, return penalised score immediately
    2. Repetition penalty — subtract for repeated action types
    3. compute_correctness_fn — domain-specific grader (security/dep/clinical)
    4. Harmful output penalty — subtract for destructive patterns
    5. Efficiency bonus — add small bonus for early completion
    6. safe_score — clamp to [0.01, 0.99]

    NOTE: difficulty_multiplier has been REMOVED.
    The task difficulty is expressed through:
    - Tighter CVSS ranges in hard cases (harder to guess)
    - More required_fix_tokens in hard cases
    - Adversarial reviewer_feedback in hard cases
    - Dependency graphs in hard clinical cases
    - Multiple checklist items with ordering in hard dep cases
    The grader itself should produce lower scores for harder tasks naturally.
    """
    if forbidden_patterns is None:
        forbidden_patterns = []

    action_type = action.get('action_type', 'unknown')

    # Step 1: Invalid action → skip grader entirely, return penalised score
    inv = invalid_action_penalty(action_type, valid_actions)
    rep = repetition_penalty(action_type, session.last_actions)
    if inv < 0:
        return safe_score(inv + rep)

    # Step 2: Domain-specific correctness
    correctness = compute_correctness_fn(action, session.task_case)
    if correctness is None:
        correctness = 0.01

    # Step 3: Harmful output check
    harm = harmful_output_penalty(action, forbidden_patterns)

    # Step 4: Efficiency bonus
    eff = efficiency_bonus(session.step_count + 1, max_steps, correctness >= 0.75)

    # Step 5: Combine and clamp
    raw = correctness + rep + harm + eff
    return safe_score(raw)
