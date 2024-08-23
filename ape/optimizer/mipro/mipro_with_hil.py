import json
import random
from typing import Dict, List, Optional, Tuple, Union

import optuna
from pydantic import ConfigDict
from ape.optimizer.mipro.mipro_base import MIPROBase
from ape.optimizer.mipro.mipro_proposer import MIPROProposer
from ape.optimizer.utils import reformat_prompt
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.dataset_item import DatasetItem
from ape.optimizer import OptunaSingletonStorage
from ape.types.response_format import (
    ResponseFormat,
)


class MIPROWithHIL(MIPROBase):
    """MIPRO optimizer with Human-In-the-Loop capabilities."""

    instruction_candidates: List[Prompt] = []
    fewshot_candidates: List[Dataset] = []
    storage: optuna.storages.RDBStorage = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, db_url: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = OptunaSingletonStorage.get_instance(db_url)

    async def create_or_load_study(
        self,
        study_name: str,
        trainset: Optional[Dataset] = None,
        task_description: Optional[str] = None,
        prompt_desc: Optional[str] = None,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
        response_format: Optional[ResponseFormat] = None,
        base_prompt: Optional[Prompt] = None,
        max_demos: int = 5,
    ) -> Tuple[optuna.Study, bool]:
        """
        Create or load an Optuna study for prompt optimization.

        Args:
            study_name (str): Name of the study.
            trainset (Optional[Dataset]): Training dataset.
            task_description (Optional[str]): Description of the task.
            prompt_desc (Optional[str]): Description of the prompt.
            inputs_desc (Optional[Dict[str, str]]): Description of inputs.
            outputs_desc (Optional[Dict[str, str]]): Description of outputs.
            response_format (Optional[ResponseFormat]): Format of the response.
            base_prompt (Optional[Prompt]): Base prompt to start with.
            max_demos (int): Maximum number of demonstrations.

        Returns:
            Tuple[optuna.Study, bool]: The created or loaded study and a boolean indicating if it's a new study.
        """
        if trainset is None:
            trainset = []

        if self.storage is None:
            raise ValueError("Storage is not set up.")

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True),
        )

        is_new_study = len(study.trials) == 0

        if is_new_study:
            # if not trainset:
            #     raise ValueError("Trainset is required for a new study.")
            # if not task_description:
            #     raise ValueError("Task description is required for a new study.")
            if base_prompt and response_format:
                base_prompt = await reformat_prompt(
                    prompt=base_prompt, response_format=response_format
                )

            proposer = MIPROProposer(**self.model_dump())
            self.instruction_candidates = await proposer.generate_candidates(
                prompt=base_prompt,
                trainset=trainset,
                task_description=task_description,
                prompt_desc=prompt_desc,
                inputs_desc=inputs_desc,
                outputs_desc=outputs_desc,
                response_format=response_format,
            )
            study.set_user_attr(
                "prompt_candidates",
                json.dumps([p.dump() for p in self.instruction_candidates]),
            )
            self.fewshot_candidates = [
                [
                    d if isinstance(d, DatasetItem) else DatasetItem(**d)
                    for d in random.sample(trainset, min(max_demos, len(trainset)))
                ]
                for _ in range(self.num_candidates)
            ]

            study.set_user_attr(
                "fewshot_candidates",
                json.dumps(
                    [
                        [d.model_dump() for d in fewshot_set]
                        for fewshot_set in self.fewshot_candidates
                    ]
                ),
            )
        else:
            prompt_candidates_json = study.user_attrs["prompt_candidates"]
            fewshot_candidates_json = study.user_attrs["fewshot_candidates"]
            self.instruction_candidates = [
                Prompt.load(p) for p in json.loads(prompt_candidates_json)
            ]
            self.fewshot_candidates = [
                [DatasetItem(**d) for d in fewshot_set]
                for fewshot_set in json.loads(fewshot_candidates_json)
            ]

        return study, is_new_study

    def suggest_next_prompt(self, study: optuna.Study) -> Tuple[optuna.Trial, Prompt]:
        """
        Suggests the next prompt for the optimization study.

        Args:
            study (optuna.Study): The optimization study.

        Returns:
            Tuple[optuna.Trial, Prompt]: A tuple containing the suggested trial and prompt.
        """
        trial = study.ask()
        if len(self.instruction_candidates) == 0:
            raise ValueError("No instruction candidates available.")
        instruction_idx = trial.suggest_categorical(
            "instruction",
            range(len(self.instruction_candidates)),
        )
        prompt = self.instruction_candidates[instruction_idx]
        if len(self.fewshot_candidates) > 0:
            fewshot_idx = trial.suggest_categorical(
                "fewshot", range(len(self.fewshot_candidates))
            )
            prompt.fewshot = self.fewshot_candidates[fewshot_idx]

        return trial, prompt

    def complete_trial(
        self, study: optuna.Study, trial: Union[optuna.Trial, int], score: float
    ):
        """
        Completes a trial by telling the study the score obtained.

        Args:
            study (optuna.Study): The study object.
            trial (Union[optuna.Trial, int]): The trial object or trial number.
            score (float): The score obtained for the trial.
        """
        study.tell(trial, score)

    async def get_best_prompt(self, study: optuna.Study) -> Prompt:
        """
        Retrieves the best prompt based on the given Optuna study.

        Args:
            study (optuna.Study): The Optuna study object.

        Returns:
            Prompt: The best prompt object.

        """
        best_trial = study.best_trial
        instruction_idx = best_trial.params["instruction"]
        fewshot_idx = best_trial.params["fewshot"]
        prompt = self.instruction_candidates[instruction_idx]
        prompt.fewshot = self.fewshot_candidates[fewshot_idx]

        return prompt
