import json
import os
from typing import Any, Dict, List, Optional, Union
import litellm
from litellm import acompletion
from litellm._logging import verbose_logger as litellm_logger
from pydantic import ConfigDict
import promptfile as pf

from ape.prompt.cost_tracker import CostTracker
from ape.prompt.utils import format_fewshot
from ape.types.dataset_item import DatasetItem
from ape.types.response_format import (
    ResponseFormat,
)
from ape.utils import parse_xml_outputs, logger


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

litellm_logger.disabled = True
litellm.suppress_debug_info = True


class Prompt(pf.PromptConfig):
    """
    A class representing a prompt configuration for language models.

    This class extends the PromptConfig class from the promptfile library and adds
    additional functionality for managing prompts, including response formats,
    few-shot examples, and cost tracking.

    Attributes:
        temperature (float): The temperature setting for the language model. Defaults to 0.0.
        _optimized (bool): A flag indicating whether the prompt has been optimized.

    """

    temperature: float = 0.0
    _optimized = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the Prompt object.

        Args:
            **data: Keyword arguments to initialize the prompt configuration.
        """
        super().__init__(**data)
        self._ensure_metadata()
        self._restructure_models()

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

    def _restructure_models(self):
        """
        Restructure the response format and few-shot examples to appropriate types.
        """
        if self.response_format:
            self.response_format = ResponseFormat(**self.response_format)
        if self.fewshot:
            self.fewshot = [
                DatasetItem(**x) if isinstance(x, dict) else x for x in self.fewshot
            ]

    @property
    def name(self) -> Optional[str]:
        """Get the name of the prompt."""
        return self.metadata["name"]

    @name.setter
    def name(self, value: Optional[str]):
        """Set the name of the prompt."""
        self.metadata["name"] = value

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
        model = self.model
        if self.response_format:
            if self.response_format.type != "xml":
                if self.response_format.type == "json_schema":
                    model = "gpt-4o-2024-08-06"
                response_format = self.response_format.model_dump(exclude_none=True)

        try:
            res = await acompletion(
                model=model,
                messages=messages,
                response_format=response_format,
                num_retries=num_retries,
                **lm_config,
            )
        except Exception as e:
            logger.error(f"Failed to complete after 3 attempts: {e}")
            raise e

        cost = res._hidden_params["response_cost"]
        CostTracker.add_cost(cost=cost, label=self.name)

        res_text = res.choices[0].message.content
        if not res_text:
            logger.error(res)

        try:
            # logger.info(res_text)
            if not self.response_format:
                return res_text
            parsed_outputs: Dict[str, Any]
            if self.response_format.type == "xml":
                parsed_outputs = parse_xml_outputs(res_text)
            else:
                parsed_outputs = json.loads(res_text)
            # logger.info("Parsed outputs")
            # logger.info(parsed_outputs)
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
    def from_filename(cls, name: str) -> "Prompt":
        """
        Create a Prompt object from a filename.

        Args:
            name (str): The name of the file to load the prompt from.

        Returns:
            Prompt: A new Prompt object.
        """
        config = super().from_filename(name)
        instance = cls(**config.model_dump())
        instance.name = name.split(".prompt")[0]
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
            kwargs["_FEWSHOT_"] = format_fewshot(
                fewshot=self.fewshot or [], response_format=self.response_format
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
