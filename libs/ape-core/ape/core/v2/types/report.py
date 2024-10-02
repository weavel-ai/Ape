from typing import Any, Dict, List
from pydantic import BaseModel

class BaseReport(BaseModel):
    pass

class TextGradientTrainerReport(BaseReport):
    scores: List[Dict[str, Any]]
    text_gradients: List[Dict[str, Any]]
    
class ExpelTrainerReport(BaseReport):
    scores: List[Dict[str, Any]]
    feedbacks: List[Dict[str, Any]]

class OptunaTrainerReport(BaseReport):
    scores: List[Dict[str, Any]]
    trial_logs: Dict[str, Any]
    best_score: float

class FewShotTrainerReport(BaseReport):
    scores: List[Dict[str, Any]]
    choices: List[Dict[str, Any]]
    best_params: Dict[str, Any]
