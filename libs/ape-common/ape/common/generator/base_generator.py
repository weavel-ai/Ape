from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Union
import asyncio

from pydantic import BaseModel
from openai.types.chat.completion_create_params import ChatCompletionMessageParam, ResponseFormat

from ape.common.prompt.prompt_base import Prompt
from ape.common.types import MetricResult
from ape.common.cache.generator_cache import GeneratorCache
from ape.common.utils import logger


class BaseGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any] = {},
    ) -> Union[str, Dict[str, Any], Awaitable[Union[str, Dict[str, Any]]]]:
        """
        Generate a response from the model. This method can be implemented as either synchronous or asynchronous.

        Args:
            prompt (Prompt): The prompt to generate a response for.
            inputs (Dict[str, Any], optional): The inputs to the prompt. Defaults to {}.

        Returns:
            Union[str, Dict[str, Any], Awaitable[Union[str, Dict[str, Any]]]]: The generated response.
        """
        pass

    async def __call__(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any] = {},
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
        cache = GeneratorCache.get_instance()
        if cache:
            cached_result = cache.get(prompt.model_dump(), inputs)
            if cached_result:
                # logger.debug("Cache hit on Generator")
                return cached_result

        result = self.generate(prompt, inputs)
        if asyncio.iscoroutine(result):
            result = await result

        if cache:
            cache.set(prompt.model_dump(), inputs, result)
            # logger.debug("Cache set on Generator")

        return result
