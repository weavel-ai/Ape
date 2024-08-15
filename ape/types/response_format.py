from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel


class ResponseFormatType(str, Enum):
    JSON = "json_object"
    JSON_SCHEMA = "json_schema"
    XML = "xml"


class ResponseFormatXML(BaseModel):
    type: Literal[ResponseFormatType.XML] = ResponseFormatType.XML


class ResponseFormatJSON(BaseModel):
    type: Literal[ResponseFormatType.JSON] = ResponseFormatType.JSON


class ResponseFormatJSONSchema(BaseModel):
    type: Literal[ResponseFormatType.JSON_SCHEMA] = ResponseFormatType.JSON_SCHEMA
    json_schema: Dict[str, Any]


ResponseFormat = Union[ResponseFormatXML, ResponseFormatJSON, ResponseFormatJSONSchema]
