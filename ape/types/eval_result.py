from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

from ape.types.dataset_item import DataItem


class EvaluationResult(BaseModel):
    example: DataItem
    prediction: DataItem
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None


class MetricResult(BaseModel):
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None


class GlobalMetricResult(BaseModel):
    score: float
    metadata: Optional[Dict[str, Any]] = None
