import asyncio
import random
from typing import Dict, List, Optional

from ape.common.prompt import Prompt
from ape.common.prompt.utils import format_fewshot
from ape.common.types.response_format import (
    ResponseFormat,
)
from ape.common.utils import logger
from ape.common.types import DatasetItem

from ape.core.core_prompts import ApeCorePrompts
from ape.core.proposer.dataset_summary_generator import create_dataset_summary
from ape.core.proposer.utils import (
    extract_prompt,
    get_response_format_instructions,
)
from ape.core.proposer.propose_base import Proposer

# Hardcoded variables
# MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive. You can add some hand-crafted examples to help the LLM understand the task better.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
}


class GroundedProposer(Proposer):
    """
    A class for generating grounded proposals based on various criteria and configurations.

    This class extends the Proposer base class and provides methods for generating
    instruction proposals using dataset summaries, task demonstrations, instruction history,
    and other configurable parameters.

    Attributes:
        use_dataset_summary (bool): Whether to use dataset summary in proposals.
        use_task_demos (bool): Whether to use task demonstrations in proposals.
        use_tip (bool): Whether to include tips in proposals.
        trainset (List[DatasetItem]): The training dataset.
        view_data_batch_size (int): The batch size for viewing data.
        describe_prompt (Prompt): The prompt for describing tasks.
        generate_instructions (Prompt): The prompt for generating instructions.
        data_summary (Optional[str]): The summary of the dataset.
    """

    def __init__(
        self,
        trainset: List[DatasetItem],
        use_dataset_summary=True,
        use_task_demos=True,
        use_tip=True,
        view_data_batch_size=10,
    ):
        """
        Initialize the GroundedProposer.

        Args:
            trainset (List[DatasetItem]): The training dataset.
            use_dataset_summary (bool, optional): Whether to use dataset summary. Defaults to True.
            use_task_demos (bool, optional): Whether to use task demonstrations. Defaults to True.
            use_tip (bool, optional): Whether to include tips. Defaults to True.
            view_data_batch_size (int, optional): The batch size for viewing data. Defaults to 10.
        """
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.use_tip = use_tip

        self.trainset: List[DatasetItem] = trainset
        self.view_data_batch_size = view_data_batch_size
        self.describe_prompt = ApeCorePrompts.get("describe-prompt")
        self.generate_instructions = ApeCorePrompts.get("gen-instructions")
        self.data_summary = None

    async def prepare_dataset_summary(
        self,
        view_data_batch_size=10,
    ):
        """
        Prepare a summary of the dataset.

        Args:
            view_data_batch_size (int, optional): The batch size for viewing data. Defaults to 10.
        """
        self.view_data_batch_size = view_data_batch_size
        logger.info("Preparing dataset summary")
        self.data_summary = await create_dataset_summary(
            trainset=self.trainset,
            view_data_batch_size=self.view_data_batch_size,
        )
        logger.info(f"DATA SUMMARY: {self.data_summary}")

    async def propose_prompts(
        self,
        N: int,
        T: float,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[List[DatasetItem]]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> List[Prompt]:
        """
        Propose a set of new instructions for the task based on specified criteria.

        Args:
            N (int): Number of proposals to generate.
            T (float): Temperature for generation.
            task_description (Optional[str], optional): Description of the task. Defaults to None.
            prompt_desc (Optional[str], optional): Description of the prompt. Defaults to None.
            base_prompt (Optional[Prompt], optional): Base prompt to start from. Defaults to None.
            fewshot_candidates (Optional[List[List[DatasetItem]]], optional): Candidates for few-shot learning. Defaults to None.
            inputs_desc (Optional[Dict[str, str]], optional): Description of inputs. Defaults to None.
            outputs_desc (Optional[Dict[str, str]], optional): Description of outputs. Defaults to None.
            response_format (Optional[ResponseFormat], optional): Format for the response. Defaults to None.

        Returns:
            List[Prompt]: A list of proposed prompts.
        """
        if not self.data_summary and self.trainset:
            await self.prepare_dataset_summary()

        if not fewshot_candidates:
            fewshot_candidates = [
                random.sample(self.trainset, min(len(self.trainset), 3)) for _ in range(N)
            ]

        proposed_instructions = []

        _tasks = [
            self.propose_one(
                index=i,
                task_description=task_description,
                base_prompt=base_prompt,
                fewshot=fewshot_candidates[i],
                prompt_desc=prompt_desc,
                T=T,
                inputs_desc=inputs_desc,
                outputs_desc=outputs_desc,
                response_format=response_format,
            )
            for i in range(len(fewshot_candidates))
        ]

        proposed_instructions = await asyncio.gather(*_tasks)

        return proposed_instructions

    async def propose_one(
        self,
        T: float,
        index: int,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot: Optional[List[DatasetItem]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
        retry_count: int = 0,
    ) -> Prompt:
        """
        Propose a single instruction based on the given criteria.

        Args:
            T (float): Temperature for generation.
            task_description (Optional[str], optional): Description of the task. Defaults to None.
            prompt_desc (Optional[str], optional): Description of the prompt. Defaults to None.
            base_prompt (Optional[Prompt], optional): Base prompt to start from. Defaults to None.
            fewshot (Optional[List[DatasetItem]], optional): Dataset for few-shot learning. Defaults to None.
            inputs_desc (Optional[Dict[str, str]], optional): Description of inputs. Defaults to None.
            outputs_desc (Optional[Dict[str, str]], optional): Description of outputs. Defaults to None.
            response_format (Optional[ResponseFormat], optional): Format for the response. Defaults to None.
            retry_count (int, optional): Number of retry attempts. Defaults to 0.

        Returns:
            Prompt: A proposed prompt.
        """

        if self.use_tip:
            selected_tip = list(TIPS.values())[index % len(TIPS)]
        else:
            selected_tip = ""

        if self.use_task_demos and fewshot:
            logger.info("Formatting fewshot for generation")
            task_fewshot = format_fewshot(fewshot=fewshot, response_format=response_format)
        else:
            task_fewshot = "-"

        if response_format:
            response_format_instructions = get_response_format_instructions(response_format)
        else:
            response_format_instructions = "-"

        output = await self.generate_instructions(
            lm_config=dict(temperature=T),
            task_description=task_description,
            dataset_desc=self.data_summary if self.use_dataset_summary else "-",
            task_fewshot=task_fewshot,
            prompt_desc=prompt_desc if prompt_desc else "-",
            basic_prompt=base_prompt.dump(),
            tip=selected_tip if self.use_tip else "-",
            inputs_desc=inputs_desc if inputs_desc else "-",
            outputs_desc=outputs_desc if outputs_desc else "-",
            response_format_instructions=response_format_instructions,
        )

        try:
            extracted_prompt = extract_prompt(output)
            new_prompt = Prompt.load(extracted_prompt)
            if not new_prompt.messages:
                raise ValueError("Generated prompt has no messages")
            new_prompt.model = base_prompt.model
            new_prompt.name = base_prompt.name
            new_prompt.inputs_desc = inputs_desc
            new_prompt.outputs_desc = outputs_desc
            new_prompt.response_format = response_format

            return new_prompt

        except Exception as e:
            logger.error(f"Error in propose_one: {e}")
            logger.error(f"Output: {output}")
            if retry_count < 3:
                logger.warning(f"Retry attempt {retry_count + 1} for propose_one")
                return await self.propose_one(
                    # trial_logs=trial_logs,
                    index=index,
                    T=T,
                    task_description=task_description,
                    prompt_desc=prompt_desc,
                    base_prompt=base_prompt,
                    fewshot=fewshot,
                    inputs_desc=inputs_desc,
                    outputs_desc=outputs_desc,
                    response_format=response_format,
                    retry_count=retry_count + 1,
                )
            else:
                logger.error("Max retries reached. Returning base prompt.")
                return base_prompt
