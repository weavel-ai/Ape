from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

class EvaluationResult(BaseModel):
    example: Union[dict, str]
    prediction: Union[dict, str]
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None
    
class MetricResult(BaseModel):
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None
    
class GlobalMetricResult(BaseModel):
    score: float
    metadata: Optional[Dict[str, Any]] = None