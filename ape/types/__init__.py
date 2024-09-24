from typing import Any, Awaitable, Callable, Dict, Tuple, Union
from .dataset_item import DatasetItem, DataItem, Dataset
from .response_format import (
    ResponseFormat,
    JsonSchema,
)
from .eval_result import EvaluationResult, MetricResult, GlobalMetricResult


Evaluator = Callable[
    ..., Awaitable[Tuple[int, DataItem, Union[str, Dict[str, Any], float]]]
]


__all__ = [
    "DatasetItem",
    "Dataset",
    "DataItem",
    "Evaluator",
    "EvaluationResult",
    "MetricResult",
    "ResponseFormat",
    "JsonSchema",
    "GlobalMetricResult",
]
