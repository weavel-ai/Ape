import asyncio
import json
import time
from typing import Any, Dict, Type, Union

from pydantic import BaseModel
from ape.common.generator.base_generator import BaseGenerator
from ape.common.prompt.prompt_base import Prompt
from ape.common.utils.logging import logger
from litellm import acompletion


class Generator(BaseGenerator):
    def __init__(self, timeout: float = 25.0, stream_timeout: float = 5.0, frequency_penalty: float = 0.1, retry_count: int = 3):
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.frequency_penalty = frequency_penalty
        self.retry_count = retry_count
        
    async def generate(
        self, prompt: Prompt, inputs: Dict[str, Any] = {}
    ) -> Union[str, Dict[str, Any], Type[BaseModel]]:
        retry_count = 0
        messages = prompt.format(**inputs).messages
        model = prompt.model
        response_format = prompt.response_format
        
        while retry_count < self.retry_count:
            try:
                start_time = time.time()
                stream_response = await asyncio.wait_for(
                    acompletion(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=prompt.temperature,
                        stream=True,
                        stream_options={"include_usage": True},
                        frequency_penalty=self.frequency_penalty
                    ),
                    timeout=self.stream_timeout
                )
                full_response = ""
                async for chunk in stream_response:
                    if time.time() - start_time > self.timeout:
                        logger.error(f"SLOW, {full_response}")
                        raise Exception("TimeoutError")
                    if len(chunk.choices) == 0:
                        continue
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                
                if response_format is not None:
                    return json.loads(full_response)
                else:
                    return full_response

            except asyncio.TimeoutError:
                logger.error("TimeoutError")
                retry_count += 1
                if retry_count == self.retry_count:
                    if response_format is not None:
                        return {}
                    else:
                        return ""
            except Exception as e:
                logger.error(f"Other Error, {e}")
                retry_count += 1
                if retry_count == self.retry_count:
                    if response_format is not None:
                        return {}
                    else:
                        return ""

        if response_format is not None:
            return {}
        else:
            return ""
