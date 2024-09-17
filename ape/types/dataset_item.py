from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class DatasetItem(BaseModel):
    inputs: Dict[str, Any]
    outputs: Optional[Union[Dict[str, Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


DataItem = Union[Dict[str, Any], DatasetItem]

Dataset = Union[List[Dict[str, Any]], List[DatasetItem]]
