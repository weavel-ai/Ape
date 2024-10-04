from typing import Any, Dict, List
from pydantic import BaseModel


class BaseReport(BaseModel):
    scores: List[Dict[str, Any]]


class TextGradientTrainerReport(BaseReport):
    text_gradients: List[Dict[str, Any]]


class ExpelTrainerReport(BaseReport):
    feedbacks: List[Dict[str, Any]]


class OptunaTrainerReport(BaseReport):
    trial_logs: Dict[str, Any]
    best_score: float


class FewShotTrainerReport(BaseReport):
    choices: List[Dict[str, Any]]
    best_params: Dict[str, Any]
