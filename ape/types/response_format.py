from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


class JsonSchema(BaseModel):
    name: str
    schema: Dict[str, Any]
    strict: bool = True


class ResponseFormat(BaseModel):
    type: Literal["json_object", "json_schema", "xml"]
    json_schema: Optional[JsonSchema] = None
