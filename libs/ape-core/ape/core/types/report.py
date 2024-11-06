from typing import Any, Dict, List
from pydantic import BaseModel


class BaseReport(BaseModel):
    scores: List[Dict[str, Any]]
    best_score: float = 0.0


class TextGradientTrainerReport(BaseReport):
    text_gradients: List[Dict[str, Any]]

class ExpelTrainerReport(BaseReport):
    feedbacks: List[Dict[str, Any]]


class OptunaTrainerReport(BaseReport):
    trial_logs: Dict[str, Any]


class FewShotTrainerReport(BaseReport):
    choices: List[Dict[str, Any]]
    best_params: Dict[str, Any]


class EvoPromptReport(BaseReport):
    pass

class TextGradEvoTrainerReport(BaseReport):
    evolution_steps: List[Dict[str, Any]]
