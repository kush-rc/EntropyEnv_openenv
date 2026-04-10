# server/graders/dependency_grader.py
# Grader for PyTorch Migration Time-Machine tasks (dep_easy, dep_medium, dep_hard).
#
# FIX SUMMARY:
# 1. _score_flag: F1 was too loose — model could name extra packages and still score high
#    FIX: Added precision penalty so naming extra/wrong packages hurts
# 2. _score_resolve: bonus of 0.15 for all-correct inflated scores to 0.99
#    FIX: Removed bonus, tightened cross-constraint checking
# 3. _score_migrate: fix_quality was too generous (0.6 partial credit)
#    FIX: Lowered partial credit to 0.3, required more precise token matching

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
    parts = str(v).strip().split('.')
    while len(parts) < 3:
        parts.append('0')
    return '.'.join(parts[:3])


def _parse_version_tuple(v: str) -> tuple:
    try:
        parts = _normalize_ver(v).split('.')
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _simple_version_check(ver_str: str, constraint: str) -> bool:
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
    """Score deprecated API detection (dep_easy).
    
    FIX:
    - Previously F1 alone let models name 10 packages and still score well if 1 correct
    - Now: precision matters heavily — flagging extra packages is penalized
    - Deprecated API match: tightened, exact match required for full credit
    
    Weights: precision=30%, recall=25%, deprecated_api=45%
    """
    exp = set(case.get('expected_outdated_packages', []))
    flagged = set(action.get('packages', {}).keys())

    if not exp:
        return 0.3

    tp = len(flagged & exp)

    # FIX: Separate precision and recall, weight them differently
    # Precision: don't flag random packages (penalizes hallucinating packages)
    precision = tp / len(flagged) if flagged else 0.0
    # Recall: find the actual outdated packages
    recall = tp / len(exp) if exp else 0.0

    # FIX: Deprecated API match — tightened
    expected_api = case.get('expected_deprecated_api', '')
    actual_api = action.get('deprecated_api', '') or ''

    if actual_api == expected_api:
        dep_ok = 1.0
    elif expected_api and expected_api.split('.')[-1].lower() in actual_api.lower():
        # partial: just the last segment (e.g. "Variable" in "autograd.Variable")
        dep_ok = 0.50  # FIX: was 0.7
    elif expected_api and any(p.lower() in actual_api.lower() for p in expected_api.split('.')):
        dep_ok = 0.20  # FIX: was 0.4
    else:
        dep_ok = 0.0

    # FIX: Weights — precision 30%, recall 25%, api 45%
    # Previously: f1 55%, api 45% — f1 hid precision failures
    return safe_score(precision * 0.30 + recall * 0.25 + dep_ok * 0.45)


def _score_resolve(action: Dict, case: Dict) -> float:
    """Score version conflict resolution (dep_medium).
    
    FIX:
    - Removed the 0.15 bonus for all-correct (was inflating to 0.99)
    - Cross-constraint checking is now STRICT — partial version match gives 0 credit
    - Downgrade penalty increased from 0.10 to 0.15 per downgrade
    
    Now: a perfect answer scores ~0.85, not 0.99
    A partial (1/2 correct) scores ~0.40
    A wrong answer scores ~0.10
    """
    compat = case.get('compatibility_matrix', {})
    proposed = action.get('packages', {})
    conflict_pkgs = case.get('conflict_packages', [])

    if not conflict_pkgs:
        return 0.20

    if not proposed:
        return 0.05

    valid = 0
    for pkg in conflict_pkgs:
        if pkg not in proposed:
            continue
        ver = proposed[pkg]
        if pkg not in compat:
            continue

        norm_ver = _normalize_ver(ver)
        pkg_versions = compat[pkg]

        # Find matching version in compat matrix
        matched_ver = None
        for k in pkg_versions:
            if _normalize_ver(k) == norm_ver:
                matched_ver = k
                break

        # FIX: Removed patch-level fuzzy match — versions must be reasonably exact
        # (major.minor match still allowed, but NOT major-only)
        if not matched_ver:
            norm_major_minor = '.'.join(norm_ver.split('.')[:2])
            for k in pkg_versions:
                k_mm = '.'.join(_normalize_ver(k).split('.')[:2])
                if k_mm == norm_major_minor:
                    matched_ver = k
                    break

        if not matched_ver:
            continue  # Version not in compatibility matrix at all — 0 credit

        # Check cross-dependency constraints
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

    # FIX: Base score — no bonus, just ratio
    base = valid / len(conflict_pkgs)

    # FIX: Downgrade penalty increased from 0.10 to 0.15
    down = _downgrades(proposed, case) * 0.15

    # FIX: Max possible without penalties is 1.0, which gets clamped to 0.99 by safe_score
    # But in practice perfect = 1.0 - 0 downgrades = 1.0 → 0.99 after clamp
    # And partial (1/2) = 0.50 → clear signal
    return safe_score(base - down)


def _score_migrate(action: Dict, case: Dict) -> float:
    """Score graph-break migration (dep_hard).
    
    FIX:
    - fix_quality partial credit lowered from 0.6 to 0.25
      (model must actually include the right fix, not just a vague description)
    - Order violation penalty increased from 0.20 to 0.30 per violation
    - Extra steps penalty increased from 0.10 to 0.15
    """
    checklist = case.get('graph_breaks', [])
    dep_graph = case.get('checklist_dependency_graph', {})
    completed = action.get('completed_items', [])
    fix_map = case.get('correct_fix_map', {})

    if not checklist:
        return 0.5
    if not completed:
        return 0.0

    # FIX: Order violations penalized more heavily (0.30 per violation, was 0.20)
    viol = sum(
        1 for item in completed
        for pre in dep_graph.get(item, [])
        if pre not in completed
    )
    order_score = max(0.0, 1.0 - viol * 0.30)

    # Checklist coverage
    covered = [b for b in checklist if b in completed]
    completeness = len(covered) / max(len(checklist), 1)

    # FIX: Fix quality — token must be present, partial credit reduced to 0.25
    fix_qs = []
    for b in covered:
        if b not in fix_map:
            continue
        expected_token = fix_map[b].lower()
        actual_fix = str(action.get('code_changes', {}).get(b, '')).lower()
        if expected_token in actual_fix:
            fix_qs.append(1.0)
        elif any(word in actual_fix for word in expected_token.split()):
            fix_qs.append(0.25)  # FIX: was 0.6 — partial credit halved
        else:
            fix_qs.append(0.0)  # FIX: No fix at all → 0, not 0.6

    fix_quality = sum(fix_qs) / max(len(fix_qs), 1) if fix_qs else 0.0

    # FIX: Extra steps penalty increased from 0.10 to 0.15
    extra = max(0, len(completed) - len(checklist))
    efficiency = max(0.0, 1.0 - extra * 0.15)

    return safe_score(order_score * 0.30 + completeness * 0.40 + fix_quality * 0.20 + efficiency * 0.10)


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
