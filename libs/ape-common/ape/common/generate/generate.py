from typing import Any, Dict, Type, Union

from pydantic import BaseModel
from ape.common.generate.generate_base import BaseGenerate
from ape.common.prompt.prompt_base import Prompt


class Generate(BaseGenerate):
    async def generate(
        self, prompt: Prompt, inputs: Dict[str, Any] = {}, lm_config: Dict[str, Any] = {}
    ) -> Union[str, Dict[str, Any], Type[BaseModel]]:
        return await prompt(**inputs, lm_config=lm_config)
