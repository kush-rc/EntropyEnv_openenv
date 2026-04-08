# server/models/security_models.py
from pydantic import BaseModel, Field
from typing import Optional


class IdentifyVulnerability(BaseModel):
    action_type: str = 'identify_vulnerability'
    vuln_type: str = Field(..., description='Type of vulnerability detected')
    cvss_score: float = Field(..., ge=0.0, le=10.0)
    severity: str = Field(..., description='critical|high|medium|low')
    affected_line: int = Field(..., ge=1)


class ProposeFix(BaseModel):
    action_type: str = 'propose_fix'
    fix_code: str = Field(..., max_length=500)
    explanation: str = Field(..., max_length=200)


class ReviseFix(BaseModel):
    action_type: str = 'revise_fix'
    fix_code: str = Field(..., max_length=500)
    addressed_feedback: str = Field(..., max_length=200)
