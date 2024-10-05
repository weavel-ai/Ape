from pydantic import BaseModel
from typing import Any, Dict, Optional


class MetricResult(BaseModel):
    score: float
    trace: Optional[Dict[str, Any]] = None


class GlobalMetricResult(BaseModel):
    score: float
    trace: Optional[Dict[str, Any]] = None
