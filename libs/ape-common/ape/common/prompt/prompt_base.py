import json
import os
from typing import Any, Dict, List, Optional, Union
import litellm
from litellm import acompletion
from litellm._logging import verbose_logger as litellm_logger
from pydantic import ConfigDict, BaseModel
from openai.types.chat.completion_create_params import ChatCompletionMessageParam
from openai.lib._parsing._completions import type_to_response_format_param
import promptfile as pf

from .cost_tracker import CostTracker
from .utils import format_fewshot
from ape.common.types import DatasetItem, ResponseFormat
from ape.common.utils import parse_xml_outputs, logger


litellm_logger.disabled = True
litellm.suppress_debug_info = True


class Prompt(pf.Prompt):
    """
    A class representing a prompt configuration for language models.

    This class extends the PromptConfig class from the promptfile library and adds
    additional functionality for managing prompts, including response formats,
    few-shot examples, and cost tracking.

    Attributes:
        _optimized (bool): A flag indicating whether the prompt has been optimized.

    """

    _optimized = False
    messages: List[ChatCompletionMessageParam]
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the Prompt object.

        Args:
            **data: Keyword arguments to initialize the prompt configuration.
        """
        super().__init__(
            model=data.pop("model", None),
            messages=data.pop("messages", []),
            metadata={**data.pop("metadata", {}), **data},
        )
        self._ensure_metadata()

    def _ensure_metadata(self):
        """
        Ensure that the metadata dictionary has all necessary default keys.
        """
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata.setdefault("response_format", None)
        self.metadata.setdefault("fewshot", [])
        self.metadata.setdefault("inputs", {})
        self.metadata.setdefault("outputs", {})
        self.metadata.setdefault("temperature", 0.0)
        self.metadata.setdefault("name", None)
        # self.fewshot_config = None # TODO: implement fewshot config = {input_ignore_key: [], output_ignore_key: []} into input variables of Prompt Init

    @property
    def name(self) -> Optional[str]:
        """Get the name of the prompt."""
        return self.metadata["name"]

    @name.setter
    def name(self, value: Optional[str]):
        """Set the name of the prompt."""
        self.metadata["name"] = value

    @property
    def temperature(self) -> Optional[float]:
        """Get the temperature of the prompt."""
        return self.metadata["temperature"]

    @temperature.setter
    def temperature(self, value: Optional[float]):
        """Set the temperature of the prompt."""
        self.metadata["temperature"] = value

    @property
    def response_format(self) -> Optional[ResponseFormat]:
        """Get the response format."""
        return self.metadata["response_format"]

    @response_format.setter
    def response_format(self, value: Optional[ResponseFormat]):
        """Set the response format."""
        self.metadata["response_format"] = value

    @property
    def fewshot(self) -> Optional[List[DatasetItem]]:
        """Get the few-shot examples."""
        return self.metadata["fewshot"]

    @fewshot.setter
    def fewshot(self, value: Optional[List[Dict[str, Any]]]):
        """Set the few-shot examples."""
        self.metadata["fewshot"] = value

    @property
    def inputs_desc(self) -> Optional[Dict[str, str]]:
        """Get the input descriptions."""
        return self.metadata["inputs"]

    @inputs_desc.setter
    def inputs_desc(self, value: Optional[Dict[str, str]]):
        """Set the input descriptions."""
        self.metadata["inputs"] = value

    @property
    def outputs_desc(self) -> Optional[Dict[str, str]]:
        """Get the output descriptions."""
        return self.metadata["outputs"]

    @outputs_desc.setter
    def outputs_desc(self, value: Optional[Dict[str, str]]):
        """Set the output descriptions."""
        self.metadata["outputs"] = value

    def set_optimized(self, value: bool):
        """Set the optimization status of the prompt."""
        self._optimized = value

    def is_optimized(self) -> bool:
        """Check if the prompt has been optimized."""
        return self._optimized

    async def __call__(
        self,
        lm_config: Optional[Dict[str, Any]] = None,
        num_retries: int = 3,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Call the language model with the configured prompt.

        Args:
            lm_config (Optional[Dict[str, Any]]): Configuration for the language model.
            num_retries (int): Number of retries for API calls. Defaults to 3.
            **kwargs: Additional keyword arguments for the prompt.

        Returns:
            Union[str, Dict[str, Any]]: The response from the language model.

        Raises:
            Exception: If the API call fails after all retries.
        """
        if lm_config is None:
            lm_config = {}

        if "temperature" not in lm_config:
            lm_config["temperature"] = self.temperature

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
        model = self.model
        try:
            res = await acompletion(
                model=model,
                messages=messages,
                response_format=self.response_format,
                num_retries=num_retries,
                **lm_config,
            )
        except Exception as e:
            logger.error(f"Failed to complete after 3 attempts: {e}")
            raise e

        cost = res._hidden_params.get("response_cost", None)
        if cost:
            await CostTracker.add_cost(cost=cost, label=self.name)

        res_text = res.choices[0].message.content
        if not res_text:
            logger.error(res)

        try:
            # logger.info(res_text)
            if not self.response_format:
                return res_text
            if self.response_format["type"] == "text":
                return res_text
            parsed_outputs: Dict[str, Any]
            if isinstance(self.response_format, BaseModel):
                parsed_outputs = json.loads(res_text)
                parsed_outputs = self.response_format.model_validate(parsed_outputs)
            else:
                parsed_outputs = json.loads(res_text)
            return parsed_outputs
        except Exception as e:
            logger.error(f"Error parsing outputs: {e}")
            logger.error(res_text)
            return res_text

    @classmethod
    def load(cls, content: str) -> "Prompt":
        """
        Load a Prompt object from a string content.

        Args:
            content (str): The string content to load the prompt from.

        Returns:
            Prompt: A new Prompt object.
        """
        config = super().load(content)
        instance = cls(**config.model_dump())
        return instance

    @classmethod
    def load_file(cls, file_path: str) -> "Prompt":
        """
        Load a Prompt object from a file.

        Args:
            file_path (str): The path to the file to load the prompt from.

        Returns:
            Prompt: A new Prompt object.
        """
        _prompt = super().load_file(file_path)

        instance = cls(**_prompt.model_dump())
        instance.name = os.path.basename(file_path).split(".prompt")[0]
        return instance

    def format(self, **kwargs) -> "Prompt":
        """
        Format the prompt with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the prompt with.

        Returns:
            Prompt: The formatted Prompt object.
        """
        if self.fewshot:
            # input_key_ignore = None
            # output_key_ignore = None
            # if fewshot_config:
            #     input_key_ignore = fewshot_config.get("input_key_ignore", None)
            #     output_key_ignore = fewshot_config.get("output_key_ignore", None)
            kwargs["_FEWSHOT_"] = format_fewshot(
                fewshot=self.fewshot or [],
                response_format=self.response_format,
                # input_key_ignore=input_key_ignore,
                # output_key_ignore=output_key_ignore
            )
        return super().format(**kwargs)

    def reset_copy(self):
        """
        Create a reset copy of the Prompt object with empty few-shot examples.

        Returns:
            Prompt: A new Prompt object with reset few-shot examples.
        """
        new = self.deepcopy()
        new.fewshot = []
        return new

    def dump(self) -> str:
        """
        Dump the Prompt object to a string representation.

        Returns:
            str: A string representation of the Prompt object.
        """

        response_format_cache = None
        if isinstance(self.response_format, BaseModel):
            response_format_cache = self.response_format
            self.response_format = type_to_response_format_param(self.response_format)

        raw = super().dump()
        if response_format_cache:
            self.response_format = response_format_cache
        return raw
