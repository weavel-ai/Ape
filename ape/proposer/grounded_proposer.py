import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional, Union

from ape.prompt.prompt_base import Prompt
from ape.prompt.utils import format_fewshot
from ape.proposer.dataset_summary_generator import create_dataset_summary
from ape.proposer.utils import create_history_string, extract_prompt
from ape.proposer.propose_base import Proposer
from ape.types.response_format import ResponseFormat, ResponseFormatType
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
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly = set_tip_randomly
        self.set_history_randomly = set_history_randomly

        self.trainset = trainset
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
        self.view_data_batch_size = view_data_batch_size
        self.data_summary = await create_dataset_summary(
            trainset=self.trainset,
            view_data_batch_size=self.view_data_batch_size,
            prompt_model=self.prompt_model,
        )
        logger.debug(f"DATA SUMMARY: {self.data_summary}")

    async def propose_prompts(
        self,
        task_description: str,
        trial_logs: Dict[str, Any],
        N: int,
        T: float,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[Dataset]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: ResponseFormat = ResponseFormat(type=ResponseFormatType.XML),
        tip=None,
    ) -> List[Prompt]:
        """This method is responsible for returning the full set of new instructions for our task, given the specified criteria."""

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
            logger.debug(
                "Using a randomly generated configuration for our grounded proposer."
            )
            # Randomly select the tip
            selected_tip_key = random.choice(list(TIPS.keys()))
            selected_tip = TIPS[selected_tip_key]
            self.use_tip = bool(
                selected_tip,
            )
            logger.debug(f"Selected tip: {selected_tip_key}")

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = random.random() < 0.5
            self.use_instruct_history = use_history
            logger.debug(f"Use history T/F: {self.use_instruct_history}")

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
        task_description: str,
        trial_logs: Dict[str, Any],
        T: float,
        prompt_desc: Optional[str] = None,
        base_prompt: Optional[Prompt] = None,
        fewshot: Optional[Dataset] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: ResponseFormat = ResponseFormat(type=ResponseFormatType.XML),
        tip=None,
    ) -> Prompt:
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        # Create an instruction history string for our predictor
        instruction_history = create_history_string(
            base_prompt,
            trial_logs,
            MAX_INSTRUCT_IN_HISTORY,
        )

        # task_fewshot = ""
        # curr_fewshots_num = 0

        # for example in fewshot_candidates[fewshot_i]:
        #     if "augmented" in example.keys():
        #         fields_to_use = {**prompt.inputs_desc, **prompt.outputs_desc}
        #         example_string = create_example_string(fields_to_use, example)
        #         task_fewshot += f"{example_string}\n"
        #         curr_fewshots_num += 1
        #         if curr_fewshots_num >= max_demos:
        #             break
        if self.use_task_demos and fewshot:
            task_fewshot = format_fewshot(
                fewshot=fewshot, response_format=response_format
            )
        else:
            task_fewshot = "-"

        # if self.prompt_aware:
        #     output = await self.describe_prompt(prompt=prompt.dump())

        #     prompt_description = (
        #         output if isinstance(output, str) else output.get("description", "-")
        #     )
        #     logger.info(f"Prompt description: {prompt_description}")

        logger.debug("Formatted prompt for generation")
        logger.debug(
            self.generate_instructions.format(
                task_description=task_description,
                dataset_desc=self.data_summary if self.use_dataset_summary else "-",
                task_fewshot=task_fewshot,
                previous_prompts=(
                    instruction_history if self.use_instruct_history else "-"
                ),
                prompt_desc=prompt_desc if prompt_desc else "-",
                basic_prompt=base_prompt.dump(),
                tip=tip if self.use_tip else "-",
                inputs_desc=inputs_desc if inputs_desc else "-",
                outputs_desc=outputs_desc if outputs_desc else "-",
            ).dump()
        )

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
        )

        try:
            base_prompt = extract_prompt(output)
            logger.debug("Extracted prompt")
            logger.debug(base_prompt)

            return Prompt.load(output)
        except Exception as e:
            logger.error(f"Error extracting prompt.\n{e}")
            logger.error(output)
            return base_prompt
