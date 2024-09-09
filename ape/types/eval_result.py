from pydantic import BaseModel
from typing import Any, Dict, Optional, Union


class EvaluationResult(BaseModel):
    example: Union[dict, str]
    prediction: Union[dict, str]
    score: float
    trace: Optional[Dict[str, Any]] = None
