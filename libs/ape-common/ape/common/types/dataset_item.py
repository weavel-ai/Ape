from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict


class DatasetItem(TypedDict):
    inputs: Dict[str, Any]
    outputs: Optional[Union[Dict[str, Any], str]] = None
