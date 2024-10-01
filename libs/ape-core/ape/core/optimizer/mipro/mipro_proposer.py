from typing import Dict, List, Optional
from ape.common.prompt import Prompt
from ape.common.types import DatasetItem, ResponseFormat
from ape.common.utils import logger

from ape.core.optimizer.mipro.mipro_base import MIPROBase
from ape.core.proposer.grounded_proposer import GroundedProposer


class MIPROProposer(MIPROBase):
    """
    A class for generating prompt candidates using the MIPRO (Model-based Instruction Prompt Optimization) approach.
    """

    async def generate_candidates(
        self,
        trainset: List[DatasetItem],
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[List[DatasetItem]]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> List[Prompt]:
        """
        Generate a list of prompt candidates based on the given parameters.

        Args:
            trainset (List[DatasetItem]): The dataset used for training.
            task_description (Optional[str]): A description of the task.
            prompt_desc (Optional[str]): A description of the prompt.
            prompt (Optional[Prompt]): A base prompt to start from.
            fewshot_candidates (Optional[List[List[DatasetItem]]]): A list of few-shot example datasets.
            inputs_desc (Optional[Dict[str, str]]): A description of the input fields.
            outputs_desc (Optional[Dict[str, str]]): A description of the output fields.
            response_format (Optional[ResponseFormat]): The desired format for the response.

        Returns:
            List[Prompt]: A list of generated prompt candidates.
        """
        logger.info("Initializing GroundedProposer")
        proposer = GroundedProposer(
            trainset=trainset,
            view_data_batch_size=self.view_data_batch_size,
        )

        proposer.use_tip = True

        logger.info(f"Generating {self.num_candidates} instruction candidates")
        prompt_candidates = await proposer.propose_prompts(
            prompt_desc=prompt_desc,
            base_prompt=prompt,
            fewshot_candidates=fewshot_candidates,
            N=self.num_candidates,
            T=self.init_temperature,
            task_description=task_description,
            inputs_desc=inputs_desc,
            outputs_desc=outputs_desc,
            response_format=response_format,
        )

        return prompt_candidates
