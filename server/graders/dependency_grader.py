# server/graders/dependency_grader.py
# Grader for PyTorch Migration Time-Machine tasks (dep_easy, dep_medium, dep_hard).
# Covers: deprecated API detection, version conflict resolution, graph-break fixing.

from typing import Dict
from .base_grader import grade_dynamic, safe_score

try:
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    _HAS_PACKAGING = True
except ImportError:
    _HAS_PACKAGING = False

VALID_ACTIONS = ['flag_outdated', 'resolve_conflict', 'migrate_api', 'validate_tree']
FORBIDDEN = []


def _normalize_ver(v: str) -> str:
    """Normalize version: '2.1' → '2.1.0', '1' → '1.0.0'."""
    parts = str(v).strip().split('.')
    while len(parts) < 3:
        parts.append('0')
    return '.'.join(parts[:3])


def _parse_version_tuple(v: str) -> tuple:
    """Parse '2.1.0' into (2, 1, 0). Robust fallback when packaging is unavailable."""
    try:
        parts = _normalize_ver(v).split('.')
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _simple_version_check(ver_str: str, constraint: str) -> bool:
    """Check if ver_str satisfies a constraint like '>=1.24,<2.0' WITHOUT packaging.
    Handles: >=, <=, >, <, ==, != and comma-separated constraints.
    """
    ver = _parse_version_tuple(ver_str)
    parts = [c.strip() for c in constraint.split(',') if c.strip()]
    for part in parts:
        if part.startswith('>='):
            if ver < _parse_version_tuple(part[2:]):
                return False
        elif part.startswith('<='):
            if ver > _parse_version_tuple(part[2:]):
                return False
        elif part.startswith('!='):
            if ver == _parse_version_tuple(part[2:]):
                return False
        elif part.startswith('>'):
            if ver <= _parse_version_tuple(part[1:]):
                return False
        elif part.startswith('<'):
            if ver >= _parse_version_tuple(part[1:]):
                return False
        elif part.startswith('=='):
            if ver != _parse_version_tuple(part[2:]):
                return False
        else:
            # Bare version string — treat as ==
            if ver != _parse_version_tuple(part):
                return False
    return True


def _f1(predicted, expected):
    """Compute F1 score between predicted and expected sets."""
    if not expected:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    pred_s = set(str(p).strip() for p in predicted)
    exp_s = set(str(e).strip() for e in expected)
    tp = len(pred_s & exp_s)
    p = tp / len(pred_s) if pred_s else 0.0
    r = tp / len(exp_s) if exp_s else 0.0
    return round(2 * p * r / max(p + r, 0.001), 4)


def _downgrades(proposed: Dict, case: Dict) -> int:
    """Count unnecessary version downgrades (dep_medium penalty)."""
    reqs = case.get('requirements', {})
    count = 0
    for pkg, ver in proposed.items():
        if pkg in reqs:
            try:
                if _HAS_PACKAGING:
                    if Version(_normalize_ver(ver)) < Version(_normalize_ver(reqs[pkg])):
                        count += 1
                else:
                    if _parse_version_tuple(ver) < _parse_version_tuple(reqs[pkg]):
                        count += 1
            except Exception:
                pass
    return count


def _score_flag(action: Dict, case: Dict) -> float:
    """Score deprecated API detection (dep_easy)."""
    exp = set(case.get('expected_outdated_packages', []))
    flagged = set(action.get('packages', {}).keys())

    # F1 on package detection (55% weight)
    p = len(flagged & exp) / max(len(flagged), 1)
    r = len(flagged & exp) / max(len(exp), 1)
    f1 = 2 * p * r / max(p + r, 0.001)

    # Deprecated API match (45% weight) — fuzzy for model variations
    expected_api = case.get('expected_deprecated_api', '')
    actual_api = action.get('deprecated_api', '') or ''
    if actual_api == expected_api:
        dep_ok = 1.0
    elif expected_api and expected_api.split('.')[-1] in actual_api:
        dep_ok = 0.7  # last segment match e.g. "Variable" in "autograd.Variable"
    elif expected_api and any(p in actual_api for p in expected_api.split('.')):
        dep_ok = 0.4  # partial segment match
    else:
        dep_ok = 0.0

    return f1 * 0.55 + dep_ok * 0.45


def _score_resolve(action: Dict, case: Dict) -> float:
    """Score version conflict resolution (dep_medium). Cross-checks compatibility matrix constraints."""
    compat = case.get('compatibility_matrix', {})
    proposed = action.get('packages', {})
    conflict_pkgs = case.get('conflict_packages', [])

    # Count valid proposed versions WITH cross-constraint checking
    valid = 0
    for pkg, ver in proposed.items():
        if pkg not in compat:
            continue
        norm_ver = _normalize_ver(ver)
        # Try exact match first, then normalized
        pkg_versions = compat[pkg]
        matched_ver = None
        if ver in pkg_versions:
            matched_ver = ver
        elif norm_ver in pkg_versions:
            matched_ver = norm_ver
        else:
            for k in pkg_versions:
                if _normalize_ver(k) == norm_ver:
                    matched_ver = k
                    break
        # Patch-level fuzzy: match major.minor only (e.g. "2.1.1" → "2.1.0")
        if not matched_ver:
            norm_major_minor = '.'.join(norm_ver.split('.')[:2])
            for k in pkg_versions:
                if '.'.join(_normalize_ver(k).split('.')[:2]) == norm_major_minor:
                    matched_ver = k
                    break
        if not matched_ver:
            continue

        # Check cross-dependency constraints using packaging or fallback
        deps = pkg_versions[matched_ver]
        cross_ok = True
        if isinstance(deps, dict):
            for dep_pkg, constraint in deps.items():
                if dep_pkg in proposed:
                    dep_ver = _normalize_ver(proposed[dep_pkg])
                    try:
                        if _HAS_PACKAGING:
                            if Version(dep_ver) not in SpecifierSet(constraint):
                                cross_ok = False
                                break
                        else:
                            if not _simple_version_check(dep_ver, constraint):
                                cross_ok = False
                                break
                    except Exception:
                        pass
        if cross_ok:
            valid += 1

    base = valid / max(len(conflict_pkgs), 1)
    bonus = 0.15 if valid == len(conflict_pkgs) else 0.0
    down = _downgrades(proposed, case) * 0.10

    return safe_score(base + bonus - down)


def _score_migrate(action: Dict, case: Dict) -> float:
    """Score graph-break migration (dep_hard). Checks coverage, order, fix quality."""
    checklist = case.get('graph_breaks', [])       # list of break IDs
    dep_graph = case.get('checklist_dependency_graph', {})
    completed = action.get('completed_items', [])
    fix_map = case.get('correct_fix_map', {})      # break_id -> required_token

    if not checklist:
        return 0.5

    # Early exit: if agent submitted nothing, score is 0
    if not completed:
        return 0.0

    # Dependency order violations
    viol = sum(
        1 for item in completed
        for pre in dep_graph.get(item, [])
        if pre not in completed
    )
    order_score = max(0.0, 1.0 - viol * 0.20)

    # Checklist coverage
    covered = [b for b in checklist if b in completed]
    completeness = len(covered) / max(len(checklist), 1)

    # Fix quality: does each fix contain the required token?
    fix_qs = []
    for b in covered:
        if b not in fix_map:
            continue
        expected_token = fix_map[b].lower()
        actual_fix = str(action.get('code_changes', {}).get(b, '')).lower()
        if expected_token in actual_fix or actual_fix in expected_token:
            fix_qs.append(1.0)
        else:
            fix_qs.append(0.6)  # Generous partial credit
    fix_quality = sum(fix_qs) / max(len(fix_qs), 1) if fix_qs else 0.0

    return safe_score(order_score * 0.30 + completeness * 0.40 + fix_quality * 0.30)


def compute_correctness(action: Dict, case: Dict) -> float:
    """Route to correct scoring function based on action_type."""
    atype = action.get('action_type')
    if atype == 'flag_outdated':
        return _score_flag(action, case)
    if atype == 'resolve_conflict':
        return _score_resolve(action, case)
    if atype in ('migrate_api', 'validate_tree'):
        return _score_migrate(action, case)
    return None


def grade(action: Dict, session) -> float:
    """Entry point called by router. Runs full reward pipeline."""
    return grade_dynamic(action, session, compute_correctness, VALID_ACTIONS, FORBIDDEN, max_steps=8)
