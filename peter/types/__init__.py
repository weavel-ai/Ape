import weavel.types
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Union

DataItem = Union[Dict[str, Any], weavel.types.DatasetItem]

Dataset = Union[List[Dict[str, Any]], List[weavel.types.DatasetItem]]

Evaluator = Callable[
    ..., Awaitable[Tuple[int, DataItem, Union[str, Dict[str, Any], float]]]
]
