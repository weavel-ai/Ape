import logging
from typing import Any, Dict, List, Literal, Optional, Self, Union
from litellm import acompletion
from ape.prompt.utils import format_fewshot_xml
from ape.utils import parse_xml_outputs, logger
from pydantic import BaseModel, ConfigDict
import promptfile as pf


class Prompt(pf.PromptConfig):
    _optimized = False

    model_config = ConfigDict(extra="allow")

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_metadata()

    def _ensure_metadata(self):
        self.metadata.setdefault("fewshot", [])
        self.metadata.setdefault("inputs", {})
        self.metadata.setdefault("outputs", {})

    @property
    def fewshot(self) -> Optional[List[Dict[str, Any]]]:
        return self.metadata["fewshot"]

    @fewshot.setter
    def fewshot(self, value: Optional[List[Dict[str, Any]]]):
        self.metadata["fewshot"] = value

    @property
    def inputs_desc(self) -> Optional[Dict[str, str]]:
        return self.metadata["inputs"]

    @inputs_desc.setter
    def inputs_desc(self, value: Optional[Dict[str, str]]):
        self.metadata["inputs"] = value

    @property
    def outputs_desc(self) -> Optional[Dict[str, str]]:
        return self.metadata["outputs"]

    @outputs_desc.setter
    def outputs_desc(self, value: Optional[Dict[str, str]]):
        self.metadata["outputs"] = value

    async def __call__(
        self, lm_config: Dict[str, Any] = {}, **kwargs
    ) -> Union[str, Dict[str, Any]]:
        if self.inputs_desc:
            inputs = {k: v for k, v in kwargs.items() if k in self.inputs_desc}
            if len(inputs) != len(self.inputs_desc):
                logger.error(
                    f"Error: Expected inputs {self.inputs_desc.keys()}, got {inputs.keys()}"
                )
                return None

        messages = self.format(**kwargs).messages

        res = await acompletion(
            model=self.model,
            messages=messages,
            **lm_config,
        )

        res_text = res.choices[0].message.content
        if not res_text:
            logger.error(res)

        if not self.outputs_desc:
            logger.info(f"Generated: {res_text}")
            return res_text

        try:
            logger.debug(res_text)
            parsed_outputs = parse_xml_outputs(res_text)
            logger.debug("Parsed outputs")
            logger.debug(parsed_outputs)
            return parsed_outputs
        except Exception as e:
            logger.error(f"Error parsing outputs: {e}")
            logger.error(res_text)
            return res_text

    @classmethod
    def load(cls, content: str) -> "Prompt":
        config = super().load(content)
        return cls(**config.model_dump())

    @classmethod
    def from_filename(cls, name: str) -> "Prompt":
        config = super().from_filename(name)
        return cls(**config.model_dump())

    @classmethod
    def load_file(cls, file_path: str) -> "Prompt":
        content = super().load_file(file_path)
        return cls(**content.model_dump())

    def format(self, **kwargs) -> Self:
        if self.fewshot:
            kwargs["_FEWSHOT_"] = format_fewshot_xml(self.fewshot or [])
        return super().format(**kwargs)

    def reset_copy(self):
        new = self.deepcopy()
        new.fewshot = []
        return new
