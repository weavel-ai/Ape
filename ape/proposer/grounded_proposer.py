import asyncio
import random
from typing import Any, Dict, List, Optional, Union

from ape.prompt.prompt_base import Prompt
from ape.prompt.utils import format_fewshot
from ape.proposer.dataset_summary_generator import create_dataset_summary
from ape.proposer.utils import (
    create_history_string,
    extract_prompt,
    get_response_format_instructions,
)
from ape.proposer.propose_base import Proposer
from ape.types.response_format import (
    ResponseFormat,
)
from ape.utils import logger
from ape.types import Dataset

# Hardcoded variables
MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
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
        use_instruct_history (bool): Whether to use instruction history in proposals.
        use_tip (bool): Whether to include tips in proposals.
        set_tip_randomly (bool): Whether to randomly select tips.
        set_history_randomly (bool): Whether to randomly decide on using instruction history.
        trainset (Dataset): The training dataset.
        prompt_model (str): The name of the prompt model to use.
        view_data_batch_size (int): The batch size for viewing data.
        describe_prompt (Prompt): The prompt for describing tasks.
        generate_instructions (Prompt): The prompt for generating instructions.
        data_summary (Optional[str]): The summary of the dataset.
    """

    def __init__(
        self,
        prompt_model: str,
        trainset: Dataset,
        use_dataset_summary=True,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        view_data_batch_size=10,
    ):
        """
        Initialize the GroundedProposer.

        Args:
            prompt_model (str): The name of the prompt model to use.
            trainset (Dataset): The training dataset.
            use_dataset_summary (bool, optional): Whether to use dataset summary. Defaults to True.
            use_task_demos (bool, optional): Whether to use task demonstrations. Defaults to True.
            use_instruct_history (bool, optional): Whether to use instruction history. Defaults to True.
            use_tip (bool, optional): Whether to include tips. Defaults to True.
            set_tip_randomly (bool, optional): Whether to randomly select tips. Defaults to True.
            set_history_randomly (bool, optional): Whether to randomly decide on using instruction history. Defaults to True.
            view_data_batch_size (int, optional): The batch size for viewing data. Defaults to 10.
        """
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly = set_tip_randomly
        self.set_history_randomly = set_history_randomly

        self.trainset: Dataset = trainset
        self.prompt_model = prompt_model
        self.view_data_batch_size = view_data_batch_size
        self.describe_prompt = Prompt.from_filename("describe-prompt")
        self.generate_instructions = Prompt.from_filename("gen-instructions")
        self.describe_prompt.model = prompt_model
        self.generate_instructions.model = prompt_model
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
            prompt_model=self.prompt_model,
        )
        logger.info(f"DATA SUMMARY: {self.data_summary}")

    async def propose_prompts(
        self,
        trial_logs: Dict[str, Any],
        N: int,
        T: float,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[Dataset]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
        tip=None,
    ) -> List[Prompt]:
        """
        Propose a set of new instructions for the task based on specified criteria.

        Args:
            trial_logs (Dict[str, Any]): Logs from previous trials.
            N (int): Number of proposals to generate.
            T (float): Temperature for generation.
            task_description (Optional[str], optional): Description of the task. Defaults to None.
            prompt_desc (Optional[str], optional): Description of the prompt. Defaults to None.
            base_prompt (Optional[Prompt], optional): Base prompt to start from. Defaults to None.
            fewshot_candidates (Optional[List[Dataset]], optional): Candidates for few-shot learning. Defaults to None.
            inputs_desc (Optional[Dict[str, str]], optional): Description of inputs. Defaults to None.
            outputs_desc (Optional[Dict[str, str]], optional): Description of outputs. Defaults to None.
            response_format (Optional[ResponseFormat], optional): Format for the response. Defaults to None.
            tip (optional): Tip for generation. Defaults to None.

        Returns:
            List[Prompt]: A list of proposed prompts.
        """
        if not self.data_summary and self.trainset:
            await self.prepare_dataset_summary()

        if not fewshot_candidates:
            # random sample from trainset
            fewshot_candidates = [
                random.sample(self.trainset, min(len(self.trainset), 3))
                for _ in range(N)
            ]

        proposed_instructions = []

        if self.set_tip_randomly:
            logger.info(
                "Using a randomly generated configuration for our grounded proposer."
            )
            # Randomly select the tip
            selected_tip_key = random.choice(list(TIPS.keys()))
            selected_tip = TIPS[selected_tip_key]
            self.use_tip = bool(
                selected_tip,
            )
            logger.info(f"Selected tip: {selected_tip_key}")

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = random.random() < 0.5
            self.use_instruct_history = use_history
            logger.info(f"Use history T/F: {self.use_instruct_history}")

        _tasks = [
            self.propose_one(
                task_description=task_description,
                base_prompt=base_prompt,
                fewshot=fewshot_candidates[i],
                prompt_desc=prompt_desc,
                trial_logs=trial_logs,
                T=T,
                inputs_desc=inputs_desc,
                outputs_desc=outputs_desc,
                tip=selected_tip,
                response_format=response_format,
            )
            for i in range(len(fewshot_candidates))
        ]

        proposed_instructions = await asyncio.gather(*_tasks)

        return proposed_instructions

    async def propose_one(
        self,
        trial_logs: Dict[str, Any],
        T: float,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot: Optional[Dataset] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
        tip=None,
    ) -> Prompt:
        """
        Propose a single instruction based on the given criteria.

        Args:
            trial_logs (Dict[str, Any]): Logs from previous trials.
            T (float): Temperature for generation.
            task_description (Optional[str], optional): Description of the task. Defaults to None.
            prompt_desc (Optional[str], optional): Description of the prompt. Defaults to None.
            base_prompt (Optional[Prompt], optional): Base prompt to start from. Defaults to None.
            fewshot (Optional[Dataset], optional): Dataset for few-shot learning. Defaults to None.
            inputs_desc (Optional[Dict[str, str]], optional): Description of inputs. Defaults to None.
            outputs_desc (Optional[Dict[str, str]], optional): Description of outputs. Defaults to None.
            response_format (Optional[ResponseFormat], optional): Format for the response. Defaults to None.
            tip (optional): Tip for generation. Defaults to None.

        Returns:
            Prompt: A proposed prompt.
        """
        # Create an instruction history string for our predictor
        instruction_history = create_history_string(
            base_prompt,
            trial_logs,
            MAX_INSTRUCT_IN_HISTORY,
        )
        logger.info(f"Create instruction history: {instruction_history}")

        if self.use_task_demos and fewshot:
            logger.info("Formatting fewshot for generation")
            task_fewshot = format_fewshot(
                fewshot=fewshot, response_format=response_format
            )
            logger.info(f"Formatted fewshot: {task_fewshot}")
        else:
            task_fewshot = "-"

        if response_format:
            response_format_instructions = get_response_format_instructions(
                response_format
            )
        else:
            response_format_instructions = "-"

        # logger.info("Formatted prompt for generation")
        # logger.info(
        #     self.generate_instructions.format(
        #         task_description=task_description or "-",
        #         dataset_desc=self.data_summary if self.use_dataset_summary else "-",
        #         task_fewshot=task_fewshot,
        #         previous_prompts=(
        #             instruction_history if self.use_instruct_history else "-"
        #         ),
        #         prompt_desc=prompt_desc if prompt_desc else "-",
        #         basic_prompt=base_prompt.dump(),
        #         tip=tip if self.use_tip else "-",
        #         inputs_desc=inputs_desc if inputs_desc else "-",
        #         outputs_desc=outputs_desc if outputs_desc else "-",
        #         response_format_instructions=response_format_instructions,
        #     ).dump()
        # )

        output = await self.generate_instructions(
            lm_config=dict(temperature=T),
            task_description=task_description,
            dataset_desc=self.data_summary if self.use_dataset_summary else "-",
            task_fewshot=task_fewshot,
            previous_prompts=instruction_history if self.use_instruct_history else "-",
            prompt_desc=prompt_desc if prompt_desc else "-",
            basic_prompt=base_prompt.dump(),
            tip=tip if self.use_tip else "-",
            inputs_desc=inputs_desc if inputs_desc else "-",
            outputs_desc=outputs_desc if outputs_desc else "-",
            response_format_instructions=response_format_instructions,
        )

        try:
            # logger.info("output")
            # logger.info(output)
            extracted_prompt = extract_prompt(output)
            # logger.info("Extracted prompt")
            # logger.info(extracted_prompt)

            new_prompt = Prompt.load(extracted_prompt)
            if not new_prompt.messages:
                new_prompt = await self.propose_one(
                    trial_logs=trial_logs,
                    T=T,
                    task_description=task_description,
                    prompt_desc=prompt_desc,
                    base_prompt=base_prompt,
                    fewshot=fewshot,
                    inputs_desc=inputs_desc,
                    outputs_desc=outputs_desc,
                    response_format=response_format,
                    tip=tip,
                )
            # logger.info("New prompt")
            new_prompt.name = base_prompt.name
            new_prompt.model = self.prompt_model
            new_prompt.inputs_desc = inputs_desc
            new_prompt.outputs_desc = outputs_desc
            new_prompt.response_format = response_format
            # logger.info(new_prompt)
            # logger.info(type(new_prompt))

            return new_prompt
        except Exception as e:
            logger.error(f"Error extracting prompt.\n{e}")
            logger.error(output)
            return base_prompt
