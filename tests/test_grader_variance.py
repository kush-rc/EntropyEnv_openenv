# tests/test_grader_variance.py
# Phase 2 of judging runs a variance check. If all graders return the same score
# for different quality answers, the submission is DISQUALIFIED.
# Run: python -m pytest tests/test_grader_variance.py -v

import sys
sys.path.insert(0, '.')

from server.graders.base_grader import safe_score
from server.graders.security_grader import compute_correctness as sec_cc
from server.graders.dependency_grader import compute_correctness as dep_cc
from server.graders.clinical_grader import compute_correctness as cli_cc


# ── Security Case for Testing ──
SEC_CASE = {
    'expected_vuln_type': 'sql_injection',
    'cvss_range': [7.5, 9.8],
    'expected_severity': 'critical',
    'required_fix_tokens': ['?', 'parameterized'],
    'current_feedback_keywords': ['sql', 'injection'],
    'original_vuln_pattern': 'query+',
}


def test_sec_identify_variance():
    """Security grader must return 3+ different scores for different quality answers."""
    perfect = {
        'action_type': 'identify_vulnerability',
        'vuln_type': 'sql_injection',
        'cvss_score': 8.5,
        'severity': 'critical',
        'affected_line': 1,
    }
    partial = {
        'action_type': 'identify_vulnerability',
        'vuln_type': 'xss',           # wrong vuln_type
        'cvss_score': 8.5,            # but correct CVSS
        'severity': 'critical',       # and correct severity
        'affected_line': 1,
    }
    wrong = {
        'action_type': 'identify_vulnerability',
        'vuln_type': 'xss',           # wrong everything
        'cvss_score': 2.0,
        'severity': 'low',
        'affected_line': 1,
    }

    s1 = safe_score(sec_cc(perfect, SEC_CASE))
    s2 = safe_score(sec_cc(partial, SEC_CASE))
    s3 = safe_score(sec_cc(wrong, SEC_CASE))

    assert len({round(s, 2) for s in [s1, s2, s3]}) >= 3, f'No variance: {s1},{s2},{s3}'
    assert s1 > s2 > s3, f'Wrong ordering: {s1},{s2},{s3}'
    print(f'  Security identify variance: {s1:.4f} > {s2:.4f} > {s3:.4f} PASS')


def test_dep_resolve_variance():
    """Dependency grader must return different scores for different quality answers."""
    case = {
        'conflict_packages': ['torch', 'numpy'],
        'compatibility_matrix': {
            'torch': {'2.1.0': {'numpy': '>=1.24'}, '1.9.0': {}},
            'numpy': {'1.24.0': {}, '1.16.0': {}},
        },
        'requirements': {'torch': '1.9.0', 'numpy': '1.16.0'},
    }

    full = {'action_type': 'resolve_conflict', 'packages': {'torch': '2.1.0', 'numpy': '1.24.0'}, 'reasoning': 'ok'}
    part = {'action_type': 'resolve_conflict', 'packages': {'torch': '2.1.0', 'numpy': '1.16.0'}, 'reasoning': 'ok'}
    empty = {'action_type': 'resolve_conflict', 'packages': {}, 'reasoning': 'ok'}

    s1 = safe_score(dep_cc(full, case))
    s2 = safe_score(dep_cc(part, case))
    s3 = safe_score(dep_cc(empty, case))

    assert s1 > s2 >= s3, f'No variance: {s1},{s2},{s3}'
    print(f'  Dependency resolve variance: {s1:.4f} > {s2:.4f} >= {s3:.4f} PASS')


def test_cli_order_variance():
    """Clinical grader must return different scores for correct vs violated dependency order."""
    case = {
        'dependency_graph': {
            'schedule_surgery': ['resolve_insurance', 'complete_pre_op'],
            'complete_pre_op': ['resolve_insurance'],
            'resolve_insurance': [],
        },
        'required_steps': ['resolve_insurance', 'complete_pre_op', 'schedule_surgery'],
    }

    correct = {
        'action_type': 'order_steps',
        'recovery_steps': ['resolve_insurance', 'complete_pre_op', 'schedule_surgery'],
    }
    violated = {
        'action_type': 'order_steps',
        'recovery_steps': ['schedule_surgery', 'complete_pre_op', 'resolve_insurance'],
    }
    partial = {
        'action_type': 'order_steps',
        'recovery_steps': ['resolve_insurance', 'complete_pre_op'],
    }

    s1 = safe_score(cli_cc(correct, case))
    s2 = safe_score(cli_cc(violated, case))
    s3 = safe_score(cli_cc(partial, case))

    assert s1 > s2, f'Violation not penalised: correct={s1}, violated={s2}'
    assert s1 > s3, f'Completeness not rewarded: correct={s1}, partial={s3}'
    print(f'  Clinical order variance: {s1:.4f} > violated:{s2:.4f}, partial:{s3:.4f} PASS')


def test_safe_score_clamp():
    """
    safe_score clamps to [0.01, 0.99] — strictly between 0 and 1.

    WHY 0.01 not 0.0:  The official spec says scores must be strictly > 0.
                        A score of 0.0 from a crashed run looks indistinguishable
                        from a broken environment. 0.01 signals "ran but failed".

    WHY 0.99 not 1.0:  A score of exactly 1.0 means the grader is trivially solved
                        or broken. 0.99 signals "excellent but not perfect".
    """
    # Floor: None, negative, bad types → 0.01
    assert safe_score(None)   == 0.01, f"Expected 0.01, got {safe_score(None)}"
    assert safe_score(-0.5)   == 0.01, f"Expected 0.01, got {safe_score(-0.5)}"
    assert safe_score(-999)   == 0.01, f"Expected 0.01, got {safe_score(-999)}"
    assert safe_score('bad')  == 0.01, f"Expected 0.01, got {safe_score('bad')}"
    assert safe_score([])     == 0.01, f"Expected 0.01, got {safe_score([])}"

    # Ceiling: values > 1 → 0.99
    assert safe_score(1.5)    == 0.99, f"Expected 0.99, got {safe_score(1.5)}"
    assert safe_score(2.0)    == 0.99, f"Expected 0.99, got {safe_score(2.0)}"
    assert safe_score(100)    == 0.99, f"Expected 0.99, got {safe_score(100)}"

    # Exact boundary values
    assert safe_score(0.01)   == 0.01, f"Expected 0.01, got {safe_score(0.01)}"
    assert safe_score(0.99)   == 0.99, f"Expected 0.99, got {safe_score(0.99)}"

    # Pass-through: normal values in range stay unchanged
    assert safe_score(0.5)    == 0.5,  f"Expected 0.5, got {safe_score(0.5)}"
    assert safe_score(0.85)   == 0.85, f"Expected 0.85, got {safe_score(0.85)}"
    assert safe_score(0.0001) == 0.01, f"Expected 0.01 (below floor), got {safe_score(0.0001)}"
    assert safe_score(0.9999) == 0.99, f"Expected 0.99 (above ceiling), got {safe_score(0.9999)}"

    print('  safe_score clamp [0.01, 0.99]: PASS')


def test_clinical_valid_actions():
    """Bug 2 fix: propose_recovery must NOT be in clinical VALID_ACTIONS."""
    from server.graders.clinical_grader import VALID_ACTIONS
    assert 'propose_recovery' not in VALID_ACTIONS, 'Bug 2 still present!'
    assert set(VALID_ACTIONS) == {'detect_gap', 'rank_issues', 'order_steps'}
    print('  Clinical VALID_ACTIONS (Bug 2): PASS')


if __name__ == '__main__':
    test_safe_score_clamp()
    test_clinical_valid_actions()
    test_sec_identify_variance()
    test_dep_resolve_variance()
    test_cli_order_variance()
    print('\nALL VARIANCE TESTS PASSED ✅')
