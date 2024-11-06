import asyncio
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import optuna

from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.prompt import Prompt
from ape.common.prompt.utils import format_fewshot
from ape.common.types import MetricResult, DatasetItem
from ape.common.utils import logger
from ape.core.core_prompts import ApeCorePrompts
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import OptunaTrainerReport
from ape.core.utils import extract_prompt, get_response_format_instructions, run_async


class DspyMiproTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        random_seed: int = 42,
        num_candidates: int = 10,
        max_steps: int = 20,
        minibatch_size: int = 25,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 2,
        success_score: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the DspyMiproTrainer.

        Args:
            generator (BaseGenerator): Generator for producing model outputs.
            metric (BaseMetric): Metric for evaluating model outputs.
            global_metric (BaseGlobalMetric): Global metric for overall evaluation.
            random_seed (int, optional): Seed for reproducibility. Defaults to 42.
            num_candidates (int, optional): Number of candidate prompts to generate. Defaults to 10.
            max_steps (int, optional): Maximum number of optimization steps. Defaults to 30.
            minibatch_size (int, optional): Size of minibatches for evaluation. Defaults to 25.
            max_bootstrapped_demos (int, optional): Maximum number of bootstrapped demos. Defaults to 5.
            max_labeled_demos (int, optional): Maximum number of labeled demos. Defaults to 2.
            success_score (float, optional): Score threshold for sampling. Defaults to 1.0.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            generator=generator,
            metric=metric,
            global_metric=global_metric,
            **kwargs,
        )
        self.random_seed = random_seed
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.minibatch_size = minibatch_size
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.success_score = success_score

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.generate_instructions_by_prompting: Prompt = ApeCorePrompts.get("gen-instructions")
        logger.debug(f"DspyMiproTrainer initialized with random_seed: {self.random_seed}")

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, OptunaTrainerReport]:
        """
        Optimize the given prompt using Optuna.

        Args:
            prompt (Prompt): The base prompt to optimize.
            trainset (List[DatasetItem]): The training dataset.
            valset (List[DatasetItem]): The validation dataset.

        Returns:
            Tuple[Prompt, OptunaTrainerReport]: The best performing prompt and the optimization report.
        """
        logger.debug("Starting train method")
        report = OptunaTrainerReport(scores=[], trial_logs={}, best_score=0.0)

        if self.metric_description is None:
            logger.debug("Generating metric description")
            self.metric_description = await self._generate_metric_description()
        if self.task_description is None:
            logger.debug("Generating task description")
            self.task_description = await self._generate_task_description(
                prompt=prompt, trainset=trainset
            )

        messages_str = ""
        for message in prompt.messages:
            messages_str += message["content"]

        if "{_FEWSHOT_}" not in messages_str:
            logger.debug("Generating fewshot placeholder")
            prompt = await self.generate_fewshot_placeholder(prompt)

        logger.debug("Evaluating initial prompt")
        preds, eval_results, global_result = await self._evaluate(trainset, prompt)
        report.best_score = global_result.score
        report.trial_logs = []

        logger.debug("Creating fewshot demo sets")
        fewshot_candidates, fewshot_candidate_indices = await self.create_n_fewshot_demo_sets(
            trainset, preds, eval_results
        )

        logger.debug("Generating instruction candidates")
        instruction_candidates = await self.generate_instruction_candidates(
            base_prompt=prompt,
            trainset=trainset,
            num_candidates=self.num_candidates,
        )

        best_score = global_result.score
        best_prompt = prompt.deepcopy()

        trial_logs: Dict[int, Dict[str, Any]] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_prompt, best_score, trial_logs

            logger.debug(f"Starting trial {trial.number}")
            trial_logs[trial.number] = {}

            instruction_idx = trial.suggest_categorical(
                "instruction", range(len(instruction_candidates))
            )
            fewshot_idx = trial.suggest_categorical("fewshot", range(len(fewshot_candidates)))

            trial_logs[trial.number].update(
                {
                    "instruction": instruction_idx,
                    "fewshot": fewshot_idx,
                }
            )

            selected_instruction_candidate = instruction_candidates[instruction_idx]
            selected_fewshot = fewshot_candidates[fewshot_idx]
            selected_fewshot_indices = fewshot_candidate_indices[fewshot_idx]
            candidate_prompt = prompt.deepcopy()
            candidate_prompt.messages = selected_instruction_candidate.messages
            candidate_prompt.fewshot = selected_fewshot

            try:
                trainset_without_fewshot = [
                    trainset[i] for i in range(len(trainset)) if i not in selected_fewshot_indices
                ]
                logger.debug(f"Evaluating candidate prompt for trial {trial.number}")
                preds, eval_results, global_result = run_async(
                    self._evaluate(
                        trainset_without_fewshot,
                        candidate_prompt,
                    )
                )
                score = global_result.score
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                trial_logs[trial.number]["evaluation_error"] = str(e)
                return float("-inf")

            trial_logs[trial.number].update(
                {
                    "score": score,
                    "num_eval_calls": len(trainset_without_fewshot),
                }
            )

            if score > best_score:
                logger.info(f"New best score: {score} (trial {trial.number})")
                best_score = score
                best_prompt = candidate_prompt.deepcopy()
                trial_logs[trial.number]["best_score_update"] = True
            else:
                trial_logs[trial.number]["best_score_update"] = False

            report.trial_logs = trial_logs
            if self.testmode:
                _, _, val_global_result  = run_async(self._evaluate(valset, candidate_prompt))
                report.scores.append({"step": trial.number, "score": score, "val_score": val_global_result.score})
            else:
                report.scores.append({"step": trial.number, "score": score})

            if score >= 1.0:
                logger.info(f"Perfect score achieved in trial {trial.number}")
                trial.study.stop()

            return score

        logger.debug("Creating Optuna study")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed, multivariate=True),
        )

        logger.debug(f"Starting optimization with {self.max_steps} trials")
        await asyncio.to_thread(study.optimize, objective, n_trials=self.max_steps)

        report.best_score = best_score
        logger.info(f"Optimization completed. Best score: {best_score}")
        return best_prompt, report

    async def create_n_fewshot_demo_sets(
        self, trainset: List[DatasetItem], predictions: List[Any], eval_results: List[MetricResult]
    ) -> Tuple[List[List[DatasetItem]], List[List[int]]]:
        logger.debug("Creating fewshot demo sets")
        candidates = []
        candidate_indices = []

        # Add no-shot candidate
        candidates.append([])
        candidate_indices.append([])

        # Add sampled-fewshot candidate
        sampled_fewshot, sampled_indices = self.sample_fewshot(trainset, self.max_labeled_demos)
        candidates.append(sampled_fewshot)
        candidate_indices.append(sampled_indices)
        # Add bootstrapped candidates
        for i in range(
            self.num_candidates - 2
        ):  # -2 because we already added no-shot and sampled-fewshot
            logger.debug(f"Creating bootstrapped candidate {i+1}")
            max_bootstrapped = random.randint(1, self.max_bootstrapped_demos)
            max_labeled = random.randint(1, self.max_labeled_demos)
            samples, indices = await self.sample(
                trainset, predictions, eval_results, max_bootstrapped, max_labeled
            )
            candidates.append(samples)
            candidate_indices.append(indices)

        logger.debug(f"Created {len(candidates)} fewshot demo sets")
        return candidates, candidate_indices

    def sample_fewshot(
        self, trainset: List[DatasetItem], num_samples: int
    ) -> Tuple[List[DatasetItem], List[int]]:
        logger.debug(f"Sampling {num_samples} fewshot examples")
        sampled_indices = random.sample(range(len(trainset)), min(num_samples, len(trainset)))
        return [trainset[i] for i in sampled_indices], sampled_indices
    async def sample(
        self,
        trainset: List[DatasetItem],
        predictions: List[Any],
        eval_results: List[MetricResult],
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
    ) -> Tuple[List[DatasetItem], List[int]]:
        logger.debug(
            f"Sampling with max_bootstrapped_demos={max_bootstrapped_demos}, max_labeled_demos={max_labeled_demos}"
        )
        bootstrapped_samples = []
        labeled_samples = []
        bootstrapped_indices = []
        labeled_indices = []

        success_indices = [
            i for i, result in enumerate(eval_results) if result.score >= self.success_score
        ]

        if success_indices:
            bootstrapped_indices = random.sample(
                success_indices, min(max_bootstrapped_demos, len(success_indices))
            )
            bootstrapped_samples = [
                DatasetItem(inputs=trainset[i]["inputs"], outputs=predictions[i])
                for i in bootstrapped_indices
            ]

        # Select labeled demos from the remaining data (excluding bootstrapped demos)
        remaining_indices = list(set(range(len(trainset))) - set(bootstrapped_indices))
        if remaining_indices:
            labeled_indices = random.sample(
                remaining_indices, min(max_labeled_demos, len(remaining_indices))
            )
            labeled_samples = [
                DatasetItem(inputs=trainset[i]["inputs"], outputs=trainset[i]["outputs"])
                for i in labeled_indices
            ]

        logger.debug(
            f"Sampled {len(bootstrapped_samples)} bootstrapped and {len(labeled_samples)} labeled samples"
        )
        return bootstrapped_samples + labeled_samples, bootstrapped_indices + labeled_indices

    def random_sample(
        self,
        dataset: List[int],
        num_shots: int,
        replace: bool = False,
        weights: Optional[List[float]] = None,
        delta: float = 1e-5,
    ) -> List[int]:
        logger.debug(f"Random sampling {num_shots} shots from dataset of size {len(dataset)}")
        if len(dataset) == 0:
            return []

        if not replace and num_shots > len(dataset):
            num_shots = len(dataset)

        if weights is not None:
            weights = np.array(weights)
            weights = weights + delta
            if weights.sum() == 0:
                raise ValueError("Sum of weights cannot be zero.")
            weights = weights / weights.sum()

        indices = np.random.choice(len(dataset), size=num_shots, replace=replace, p=weights)
        return [dataset[i] for i in indices]

    async def generate_instruction_candidates(
        self,
        trainset: List[DatasetItem],
        base_prompt: Prompt,
        num_candidates: int,
    ) -> List[Prompt]:
        """
        Generate a set of new prompt candidates based on the base prompt using prompt engineering techniques.
        """
        logger.debug(f"Generating {num_candidates} instruction candidates")

        TIPS = {
            "creative": "Don't be afraid to be creative when creating the new instruction!",
            "simple": "Keep the instruction clear and concise.",
            "description": "Make sure your instruction is very informative and descriptive. You can add some hand-crafted examples to help the LLM understand the task better.",
            "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
            "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
        }

        async def propose_one(index: int) -> Prompt:
            logger.debug(f"Proposing instruction candidate {index+1}")
            selected_tip = list(TIPS.values())[index % len(TIPS)]
            logger.debug(f"Selected tip for candidate {index+1}: {selected_tip}")

            fewshot = random.sample(trainset, min(len(trainset), 3))
            logger.debug(f"Sampled {len(fewshot)} examples for fewshot")
            task_fewshot = format_fewshot(
                fewshot=fewshot, response_format=base_prompt.response_format
            )
            logger.debug("Formatted fewshot examples")

            response_format_instructions = get_response_format_instructions(
                base_prompt.response_format
            )
            logger.debug("Generated response format instructions")

            logger.debug("Calling generate_instructions_by_prompting")
            output = await self.generate_instructions_by_prompting(
                task_description="",
                dataset_desc=self.dataset_summary,
                task_fewshot=task_fewshot,
                prompt_desc=self.task_description,
                basic_prompt=str(base_prompt.messages),
                tip=selected_tip,
                inputs_desc=base_prompt.inputs_desc if base_prompt.inputs_desc else "-",
                outputs_desc=base_prompt.outputs_desc if base_prompt.outputs_desc else "-",
                response_format_instructions=response_format_instructions,
            )
            logger.debug("Received output from generate_instructions_by_prompting")

            try:
                logger.debug("Attempting to extract and load new prompt")
                new_prompt_message = output["messages"]
                new_prompt = base_prompt.deepcopy()
                new_prompt.messages = new_prompt_message
                logger.debug("Successfully created new prompt")

                return new_prompt

            except Exception as e:
                logger.error(f"Error in propose_one for candidate {index+1}: {e}")
                logger.error(f"Output: {output}")
                logger.debug("Returning base prompt due to error")
                return base_prompt

        logger.debug(f"Creating {num_candidates} tasks for propose_one")
        tasks = [propose_one(i) for i in range(num_candidates)]
        logger.debug("Awaiting completion of all propose_one tasks")
        proposed_instructions = await asyncio.gather(*tasks)

        logger.debug(f"Generated {len(proposed_instructions)} instruction candidates")
        logger.debug("Returning proposed instructions")
        return proposed_instructions
