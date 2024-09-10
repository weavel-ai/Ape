from typing import Any, Awaitable, Callable, Dict, List, Tuple, Union
from .dataset_item import DatasetItem
from .response_format import (
    ResponseFormat, 
    JsonSchema,
)
from .eval_result import EvaluationResult, MetricResult, GlobalMetricResult
DataItem = Union[Dict[str, Any], DatasetItem]

Dataset = Union[List[Dict[str, Any]], List[DatasetItem]]

Evaluator = Callable[
    ..., Awaitable[Tuple[int, DataItem, Union[str, Dict[str, Any], float]]]
]


__all__ = [
    "Dataset",
    "DataItem",
    "Evaluator",
    "EvaluationResult",
    "MetricResult",
    "ResponseFormat",
    "JsonSchema",
    "GlobalMetricResult",
]
