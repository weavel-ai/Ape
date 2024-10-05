from typing import Any, Dict, Type, Union

from pydantic import BaseModel
from ape.common.generator.base_generator import BaseGenerator
from ape.common.prompt.prompt_base import Prompt


class Generator(BaseGenerator):
    async def generate(
        self, prompt: Prompt, inputs: Dict[str, Any] = {}
    ) -> Union[str, Dict[str, Any], Type[BaseModel]]:
        return await prompt(**inputs)
