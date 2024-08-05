import logging
from typing import Any, Dict, List, Literal, Optional, Self, Union
from litellm import acompletion
from peter.prompt.utils import format_fewshot_xml
from peter.utils import parse_xml_outputs
import promptfile as pf


class Prompt(pf.PromptConfig):
    def __init__(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(messages, model, **kwargs)
        self._optimized = False

        # Initialize metadata if it doesn't exist
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}

        # Ensure these keys exist in metadata with default values
        self.metadata.setdefault("use_fewshot", False)
        self.metadata.setdefault("fewshot", [])
        self.metadata.setdefault("inputs", {})
        self.metadata.setdefault("outputs", {})

    @property
    def use_fewshot(self) -> bool:
        return self.metadata["use_fewshot"]

    @use_fewshot.setter
    def use_fewshot(self, value: bool):
        self.metadata["use_fewshot"] = value

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
                logging.error(
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
            logging.error(res)

        if not self.outputs_desc:
            logging.info(f"Generated: {res_text}")
            return res_text

        try:
            logging.debug(res_text)
            parsed_outputs = parse_xml_outputs(res_text)
            logging.debug("Parsed outputs")
            logging.debug(parsed_outputs)
            return parsed_outputs
        except Exception as e:
            logging.error(f"Error parsing outputs: {e}")
            return res_text

    @classmethod
    def from_filename(cls, name: str) -> "Prompt":
        config = super().from_filename(name)
        return cls(config.messages, config.model, **config.metadata)

    @classmethod
    def load_file(cls, file_path: str) -> "Prompt":
        content = super().load_file(file_path)
        return cls(content.messages, content.model, **content.metadata)

    def format(self, **kwargs) -> Self:
        if self.use_fewshot:
            kwargs["_FEWSHOT_"] = format_fewshot_xml(self.fewshot or [])
        return super().format(**kwargs)

    def reset_copy(self):
        new = self.deepcopy()
        new.fewshot = []
        return new
