from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class InstructionSuggestion(BaseModel):
    reason: str
    suggestion: str
    
class ValueAnalysisResult(BaseModel):
    think: str
    is_worthy: bool

class SplitAnalysisResult(BaseModel):
    think: str
    analysis: Dict[str, Any]

class ParaphraseAnalysisResult(BaseModel):
    think: str
    best_index: int