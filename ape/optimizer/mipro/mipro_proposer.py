from ape.optimizer.mipro.mipro_base import MIPROBase
from ape.proposer.grounded_proposer import GroundedProposer
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.utils.logging import logger
from typing import Dict, List, Optional


class MIPROProposer(MIPROBase):
    async def generate_candidates(
        self,
        trainset: Dataset,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        fewshot_candidates: Optional[List[Dataset]] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
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
        )

        return prompt_candidates
