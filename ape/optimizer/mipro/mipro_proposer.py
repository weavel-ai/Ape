from ape.optimizer.mipro.mipro_base import MIPROBase
from ape.proposer.grounded_proposer import GroundedProposer
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.response_format import ResponseFormat
from ape.utils.logging import logger
from typing import Dict, List, Optional


class MIPROProposer(MIPROBase):
    """
    A class for generating prompt candidates using the MIPRO (Model-based Instruction Prompt Optimization) approach.
    """

    async def generate_candidates(
        self,
        trainset: Dataset,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[Dataset]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> List[Prompt]:
        """
        Generate a list of prompt candidates based on the given parameters.

        Args:
            trainset (Dataset): The dataset used for training.
            task_description (Optional[str]): A description of the task.
            prompt_desc (Optional[str]): A description of the prompt.
            prompt (Optional[Prompt]): A base prompt to start from.
            fewshot_candidates (Optional[List[Dataset]]): A list of few-shot example datasets.
            inputs_desc (Optional[Dict[str, str]]): A description of the input fields.
            outputs_desc (Optional[Dict[str, str]]): A description of the output fields.
            response_format (Optional[ResponseFormat]): The desired format for the response.

        Returns:
            List[Prompt]: A list of generated prompt candidates.
        """
        logger.info("Initializing GroundedProposer")
        proposer = GroundedProposer(
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=self.view_data_batch_size,
            set_history_randomly=True,
            set_tip_randomly=True,
        )

        proposer.program_aware = True
        proposer.use_tip = True
        proposer.use_instruct_history = False
        proposer.set_history_randomly = False

        logger.info(f"Generating {self.num_candidates} instruction candidates")
        prompt_candidates = await proposer.propose_prompts(
            prompt_desc=prompt_desc,
            base_prompt=prompt,
            fewshot_candidates=fewshot_candidates,
            N=self.num_candidates,
            T=self.init_temperature,
            task_description=task_description,
            trial_logs={},
            inputs_desc=inputs_desc,
            outputs_desc=outputs_desc,
            response_format=response_format,
        )

        return prompt_candidates
