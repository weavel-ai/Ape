from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class DatasetItem(BaseModel):
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")
