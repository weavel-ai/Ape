from pydantic import BaseModel
from typing import Any, Dict, Optional

class MetricResult(BaseModel):
    score: float
    metadata: Optional[Dict[str, Any]] = None

class GlobalMetricResult(BaseModel):
    score: float
    metadata: Optional[Dict[str, Any]] = None
