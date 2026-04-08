# server/benchmark_store.py
# Persists benchmark results to disk so they survive server restarts.
# Used by both inference.py (CLI) and web_ui.py (frontend).

import json
import os
from datetime import datetime
from typing import List, Dict

_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'run_history.json'
)
os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)


def _load() -> List[Dict]:
    """Load all benchmark results from disk."""
    if not os.path.exists(_STORE_PATH):
        return []
    try:
        with open(_STORE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save(results: List[Dict]) -> None:
    """Save all benchmark results to disk."""
    try:
        with open(_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
    except IOError as e:
        print(f"[benchmark_store] WARNING: Could not save results: {e}")


def append_result(model: str, model_id: str, scores: Dict[str, float]) -> Dict:
    """Add a new benchmark result and persist to disk. Returns the saved entry."""
    avg = round(sum(scores.values()) / max(len(scores), 1), 4)
    entry = {
        'model_name': model,
        'model_id': model_id,
        'scores': scores,
        'average': avg,
        'type': 'full_run',
        'timestamp': datetime.utcnow().isoformat(),
    }
    results = _load()
    results.append(entry)
    _save(results)
    return entry


def get_all() -> List[Dict]:
    """Return all benchmark results, newest first."""
    results = _load()
    for r in results:
        if 'average' not in r and 'avg' in r:
            r['average'] = r['avg']
        if 'model_name' not in r and 'model' in r:
            r['model_name'] = r['model']
    return sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)


def get_leaderboard() -> List[Dict]:
    """Return deduplicated leaderboard: best score per model_id."""
    results = _load()
    best: Dict[str, Dict] = {}
    for r in results:
        mid = r.get('model_id', r.get('model_name', r.get('model', 'unknown')))
        val = r.get('average', r.get('avg', 0))
        best_val = best[mid].get('average', best[mid].get('avg', 0)) if mid in best else -1
        if mid not in best or val > best_val:
            best[mid] = r
    return sorted(best.values(), key=lambda x: x.get('average', x.get('avg', 0)), reverse=True)
