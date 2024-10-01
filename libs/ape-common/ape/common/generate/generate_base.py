from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Union
import asyncio

from pydantic import BaseModel
from openai.types.chat.completion_create_params import ChatCompletionMessageParam, ResponseFormat

from ape.common.prompt.prompt_base import Prompt
from ape.common.types import MetricResult


class BaseGenerate(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any] = {},
        lm_config: Dict[str, Any] = {},
    ) -> Union[str, Dict[str, Any], Awaitable[Union[str, Dict[str, Any]]]]:
        pass

    async def __call__(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any] = {},
        lm_config: Dict[str, Any] = {},
    ) -> Union[str, Dict[str, Any], Awaitable[Union[str, Dict[str, Any]]]]:
        """
        Unified method to compute the metric, handling both sync and async implementations.

        Args:
            messages (List[Message]): The messages.
            model (str): The model.
            response_format (Optional[ResponseFormat]): The response format.

        Returns:
            MetricResult: An object containing the score and intermediate values.
        """
        result = self.generate(prompt, inputs)
        if asyncio.iscoroutine(result):
            return await result
        return result
