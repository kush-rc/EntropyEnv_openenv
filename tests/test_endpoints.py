# tests/test_endpoints.py
# Basic endpoint tests for the environment.
# Run: python -m pytest tests/ -v

import requests
import pytest

BASE_URL = 'http://localhost:7860'


def test_health_check():
    """GET / should return 200 with status ok."""
    r = requests.get(f'{BASE_URL}/')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert data['tasks'] == 9


def test_reset_valid_task():
    """POST /reset with valid task_id should return episode_id and observation."""
    r = requests.post(f'{BASE_URL}/reset', json={'task_id': 'sec_easy'})
    assert r.status_code == 200
    data = r.json()
    assert 'episode_id' in data
    assert 'observation' in data
    assert data['observation']['task_type'] == 'security'


def test_reset_all_tasks():
    """POST /reset should work for all 9 task IDs."""
    tasks = [
        'sec_easy', 'sec_medium', 'sec_hard',
        'dep_easy', 'dep_medium', 'dep_hard',
        'cli_easy', 'cli_medium', 'cli_hard',
    ]
    for task_id in tasks:
        r = requests.post(f'{BASE_URL}/reset', json={'task_id': task_id})
        assert r.status_code == 200
        data = r.json()
        assert 'episode_id' in data, f'No episode_id for {task_id}'
        assert 'observation' in data, f'No observation for {task_id}'


def test_reset_invalid_task():
    """POST /reset with invalid task_id should still return 200."""
    r = requests.post(f'{BASE_URL}/reset', json={'task_id': 'nonexistent'})
    assert r.status_code == 200


def test_step_valid_action():
    """POST /step with valid action should return reward and observation."""
    # Reset first
    r = requests.post(f'{BASE_URL}/reset', json={'task_id': 'sec_easy'})
    ep_id = r.json()['episode_id']

    # Step
    action = {
        'episode_id': ep_id,
        'action_type': 'identify_vulnerability',
        'vuln_type': 'sql_injection',
        'cvss_score': 9.1,
        'severity': 'critical',
        'affected_line': 1,
    }
    r = requests.post(f'{BASE_URL}/step', json=action)
    assert r.status_code == 200
    data = r.json()
    assert 'reward' in data
    assert 'done' in data
    assert 'observation' in data
    assert 0.0 <= data['reward'] <= 1.0


def test_step_invalid_episode():
    """POST /step with invalid episode_id should return 200 with done=True."""
    r = requests.post(f'{BASE_URL}/step', json={
        'episode_id': 'nonexistent',
        'action_type': 'identify_vulnerability',
    })
    assert r.status_code == 200
    data = r.json()
    assert data['done'] is True


def test_state_endpoint():
    """GET /state should return episode info."""
    r = requests.post(f'{BASE_URL}/reset', json={'task_id': 'sec_easy'})
    ep_id = r.json()['episode_id']

    r = requests.get(f'{BASE_URL}/state', params={'episode_id': ep_id})
    assert r.status_code == 200
    data = r.json()
    assert data['episode_id'] == ep_id
    assert data['done'] is False


def test_reward_range():
    """Rewards should always be in [0.0, 1.0]."""
    tasks = ['sec_easy', 'dep_easy', 'cli_easy']
    for task_id in tasks:
        r = requests.post(f'{BASE_URL}/reset', json={'task_id': task_id})
        ep_id = r.json()['episode_id']

        # Send an invalid action
        r = requests.post(f'{BASE_URL}/step', json={
            'episode_id': ep_id,
            'action_type': 'invalid_action_type',
        })
        data = r.json()
        assert 0.0 <= data['reward'] <= 1.0, f'Reward out of range for {task_id}'


def test_step_enriched_observation():
    """Step observations should include task context fields."""
    r = requests.post(f'{BASE_URL}/reset', json={'task_id': 'sec_easy'})
    ep_id = r.json()['episode_id']

    action = {
        'episode_id': ep_id,
        'action_type': 'identify_vulnerability',
        'vuln_type': 'sql_injection',
        'cvss_score': 9.1,
        'severity': 'critical',
        'affected_line': 1,
    }
    r = requests.post(f'{BASE_URL}/step', json=action)
    obs = r.json()['observation']
    assert 'task_type' in obs
    assert 'max_steps' in obs
    assert 'steps_remaining' in obs
