from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

from .dataset_item import DatasetItem


class EvaluationResult(BaseModel):
    example: DatasetItem
    prediction: Union[str, Dict[str, Any]]
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None


class MetricResult(BaseModel):
    score: float
    intermediate_values: Optional[Dict[str, Any]] = None


class GlobalMetricResult(BaseModel):
    score: float
    metadata: Optional[Dict[str, Any]] = None
