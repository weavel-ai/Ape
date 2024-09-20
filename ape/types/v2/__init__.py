from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class InstructionSuggestion(BaseModel):
    reason: str
    suggestion: str
    
class ValueAnalysisResult(BaseModel):
    think: str
    is_success: bool
    score: int

class SplitAnalysisResult(BaseModel):
    think: str
    analysis: Dict[str, Any]

class ParaphraseAnalysisResult(BaseModel):
    think: str
    success_analysis: Optional[str]
    failure_analysis: Optional[str]