from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel


class ResponseFormatType(str, Enum):
    JSON = "json_object"
    JSON_SCHEMA = "json_schema"
    XML = "xml"


class JsonSchema(BaseModel):
    name: str
    schema: Dict[str, Any]
    strict: bool = True


class ResponseFormat(BaseModel):
    type: ResponseFormatType
    json_schema: Optional[JsonSchema] = None
