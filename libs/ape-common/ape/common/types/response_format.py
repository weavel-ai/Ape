from typing import Type
from pydantic import BaseModel
from openai.types.chat.completion_create_params import ResponseFormat as OpenAIResponseFormat

ResponseFormat = OpenAIResponseFormat | Type[BaseModel] | None

__all__ = ["ResponseFormat", "OpenAIResponseFormat"]
