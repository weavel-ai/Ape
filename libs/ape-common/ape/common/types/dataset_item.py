from typing import Any, Dict, List, Optional, TypedDict, Union


class DatasetItem(TypedDict):
    inputs: Dict[str, Any]
    outputs: Optional[Union[Dict[str, Any], str]] = None
