# server/models/dependency_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class FlagOutdated(BaseModel):
    action_type: str = 'flag_outdated'
    packages: Dict[str, str] = Field(..., description='package_name: current_version')
    deprecated_api: Optional[str] = None
    replacement: Optional[str] = None


class ResolveConflict(BaseModel):
    action_type: str = 'resolve_conflict'
    packages: Dict[str, str] = Field(..., description='package_name: proposed_version')
    reasoning: str = Field(..., max_length=100)


class MigrateApi(BaseModel):
    action_type: str = 'migrate_api'
    completed_items: List[str] = Field(..., description='list of break_ids fixed')
    code_changes: Dict[str, str] = Field(..., description='break_id: fix summary')
