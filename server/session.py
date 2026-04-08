# server/session.py
# Foundation module — everything depends on this.
# Manages episode state, task-to-domain mapping, and in-memory session storage.

from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid


@dataclass
class SessionState:
    """Holds all data for a single episode (one run of one task)."""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ''        # 'security' | 'dependency' | 'clinical'
    task_id: str = ''          # e.g. 'sec_easy'
    task_case: Dict[str, Any] = field(default_factory=dict)   # ground truth — NEVER shared with agent
    history: List[Dict] = field(default_factory=list)          # all past actions
    last_actions: List[str] = field(default_factory=list)      # action_type strings for repetition penalty
    step_count: int = 0
    reward_acc: float = 0.0
    done: bool = False


# Maps each of the 9 task IDs to its domain
TASK_TYPE_MAP = {
    'sec_easy': 'security',   'sec_medium': 'security',   'sec_hard': 'security',
    'dep_easy': 'dependency', 'dep_medium': 'dependency', 'dep_hard': 'dependency',
    'cli_easy': 'clinical',   'cli_medium': 'clinical',   'cli_hard': 'clinical',
}

# In-memory store for all active sessions
SESSIONS: Dict[str, SessionState] = {}


def create_session(task_id: str, task_case: Dict) -> SessionState:
    """Create a new session for a given task. Returns the SessionState object."""
    s = SessionState()
    s.task_id = task_id
    s.task_type = TASK_TYPE_MAP.get(task_id, 'unknown')
    s.task_case = task_case
    return s
