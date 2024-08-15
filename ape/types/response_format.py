from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel


class ResponseFormatType(str, Enum):
    JSON = "json_object"
    JSON_SCHEMA = "json_schema"
    XML = "xml"


class ResponseFormat(BaseModel):
    type: ResponseFormatType
    json_schema: Optional[Dict[str, Any]] = None
