from typing import Any, Awaitable, Callable, Dict, List, Tuple, Union
from .dataset_item import DatasetItem

DataItem = Union[Dict[str, Any], DatasetItem]

Dataset = Union[List[Dict[str, Any]], List[DatasetItem]]

Evaluator = Callable[
    ..., Awaitable[Tuple[int, DataItem, Union[str, Dict[str, Any], float]]]
]
