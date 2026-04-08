# server/models/clinical_models.py
from pydantic import BaseModel, Field
from typing import List


class DetectGap(BaseModel):
    action_type: str = 'detect_gap'
    missing_steps: List[str] = Field(..., description='IDs of missing workflow steps')
    risk_level: str = Field(..., description='critical|high|medium|low')


class RankIssues(BaseModel):
    action_type: str = 'rank_issues'
    priority_order: List[str] = Field(..., description='step IDs from highest to lowest priority')


class OrderSteps(BaseModel):
    action_type: str = 'order_steps'
    recovery_steps: List[str] = Field(..., description='step IDs in dependency-safe execution order')
