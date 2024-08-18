import json
import os
from typing import Any, Dict, List, Optional, Self, Union
from litellm import acompletion
from ape.prompt.utils import format_fewshot
from ape.types.dataset_item import DatasetItem
from ape.types.response_format import (
    ResponseFormat,
)
from ape.utils import parse_xml_outputs, logger
from pydantic import ConfigDict
import promptfile as pf


# Get the directory of the current file (prompt_base.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach the root 'ape' directory
ape_root = os.path.dirname(current_dir)

# Construct the path to the 'prompts' directory
base_path = os.path.join(ape_root, "ape_prompts")

# Initialize promptfile with the constructed base_path
if os.path.exists(base_path):
    pf.init(base_path=base_path)
else:
    raise FileNotFoundError(f"Prompts directory not found at {base_path}")


class Prompt(pf.PromptConfig):
    # response_format: Optional[ResponseFormat] = None
    temperature: float = 0.0
    _optimized = False

    model_config = ConfigDict(extra="allow")

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_metadata()
        self._restructure_models()

    def _ensure_metadata(self):
        self.metadata.setdefault("response_format", None)
        self.metadata.setdefault("fewshot", [])
        self.metadata.setdefault("inputs", {})
        self.metadata.setdefault("outputs", {})

    def _restructure_models(self):
        if self.response_format:
            self.response_format = ResponseFormat(**self.response_format)
        if self.fewshot:
            self.fewshot = [
                DatasetItem(**x) if isinstance(x, dict) else x for x in self.fewshot
            ]

    @property
    def response_format(self) -> Optional[ResponseFormat]:
        return self.metadata["response_format"]

    @response_format.setter
    def response_format(self, value: Optional[ResponseFormat]):
        self.metadata["response_format"] = value

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
        if not messages:
            logger.error("Error: No messages in prompt.")
            return None
        response_format = None
        if self.response_format:
            if self.response_format.type != "xml":
                response_format = self.response_format.model_dump(exclude_none=True)

        res = await acompletion(
            model=self.model,
            messages=messages,
            response_format=response_format,
            **lm_config,
        )

        res_text = res.choices[0].message.content
        if not res_text:
            logger.error(res)

        try:
            logger.info(res_text)
            if not self.response_format:
                return res_text
            parsed_outputs: Dict[str, Any]
            if self.response_format.type == "xml":
                parsed_outputs = parse_xml_outputs(res_text)
            else:
                parsed_outputs = json.loads(res_text)
            logger.info("Parsed outputs")
            logger.info(parsed_outputs)
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
            kwargs["_FEWSHOT_"] = format_fewshot(
                fewshot=self.fewshot or [], response_format=self.response_format
            )
        return super().format(**kwargs)

    def reset_copy(self):
        new = self.deepcopy()
        new.fewshot = []
        return new

    def dump(self) -> str:
        response_format_cache = self.response_format
        fewshot_cache = self.fewshot
        self.response_format = (
            self.response_format.model_dump() if self.response_format else None
        )
        self.fewshot = [
            x.model_dump() if isinstance(x, DatasetItem) else x for x in self.fewshot
        ]
        raw = super().dump()
        self.response_format = response_format_cache
        self.fewshot = fewshot_cache
        return raw
