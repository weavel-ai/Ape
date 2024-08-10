from pydantic import BaseModel
from typing import Any, Awaitable, Callable, Dict, Optional


class MIPROBase(BaseModel):
    prompt_model: Optional[str] = "gpt-4o"
    task_model: Optional[str] = "gpt-4o-mini"
    teacher_settings: Dict = {}
    num_candidates: int = 10
    metric: Optional[Callable[..., Awaitable[Any]]] = None
    init_temperature: float = 1.0
    verbose: bool = False
    track_stats: bool = True
    log_dir: Optional[str] = None
    view_data_batch_size: int = 10
    minibatch_size: int = 25
    minibatch_full_eval_steps: int = 10

    class Config:
        arbitrary_types_allowed = True
