import asyncio
import json
import random
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple

import optuna

from ape.common.prompt.prompt_base import Prompt, format_fewshot
from ape.common.prompt.utils import format_fewshot
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generate import BaseGenerate
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.utils import logger
from ape.core.optimizer.utils import run_async
from ape.core.core_prompts import ApeCorePrompts
from ape.core.proposer.utils import extract_prompt, get_response_format_instructions
from ape.core.v2.trainer.base import BaseTrainer
from ape.core.v2.types.report import OptunaTrainerReport

class DspyMiproTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerate,
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
            generator (BaseGenerate): Generator for producing model outputs.
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
        report = OptunaTrainerReport(scores=[], trial_logs={}, best_score=0.0)

        if self.metric_description is None:
            self.metric_description = await self._generate_metric_description()
        if self.task_description is None:
            self.task_description = await self._generate_task_description(
                prompt=prompt, trainset=trainset
            )
        
        messages_str = ""
        for message in prompt.messages:
            messages_str += message["content"]
            
        if "{_FEWSHOT_}" not in messages_str:
            prompt = await self.generate_fewshot_placeholder(prompt)

        preds, eval_results, global_result = await self._evaluate(trainset, prompt)
        report.best_score = global_result.score
        report.trial_logs = []

        fewshot_candidates, fewshot_candidate_indices = await self.create_n_fewshot_demo_sets(
            trainset, preds, eval_results
        )

        instruction_candidates = (
            await self.generate_instruction_candidates( 
                base_prompt=prompt,
                trainset=trainset,
                num_candidates=self.num_candidates,
            )
        )

        best_score = global_result.score
        best_prompt = prompt.deepcopy()

        trial_logs: Dict[int, Dict[str, Any]] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_prompt, best_score, trial_logs

            trial_logs[trial.number] = {}

            instruction_idx = trial.suggest_categorical(
                "instruction", range(len(instruction_candidates))
            )
            fewshot_idx = trial.suggest_categorical(
                "fewshot", range(len(fewshot_candidates))
            )

            trial_logs[trial.number].update({
                "instruction": instruction_idx,
                "fewshot": fewshot_idx,
            })

            selected_instruction_candidate = instruction_candidates[instruction_idx]
            selected_fewshot = fewshot_candidates[fewshot_idx]
            selected_fewshot_indices = fewshot_candidate_indices[fewshot_idx]
            candidate_prompt = prompt.deepcopy()
            candidate_prompt.messages = selected_instruction_candidate.messages
            candidate_prompt.fewshot = selected_fewshot

            try:
                trainset_without_fewshot = [trainset[i] for i in range(len(trainset)) if i not in selected_fewshot_indices]
                preds, eval_results, global_result = run_async(
                    self._evaluate(
                        random.sample(trainset_without_fewshot, min(self.minibatch_size, len(trainset_without_fewshot))),
                        candidate_prompt,
                    )
                )
                score = global_result.score
            except Exception as e:
                trial_logs[trial.number]["evaluation_error"] = str(e)
                return float("-inf")

            trial_logs[trial.number].update({
                "score": score,
                "num_eval_calls": min(self.minibatch_size, len(trainset)),
            })

            if score > best_score:
                best_score = score
                best_prompt = candidate_prompt.deepcopy()
                trial_logs[trial.number]["best_score_update"] = True
            else:
                trial_logs[trial.number]["best_score_update"] = False

            report.trial_logs = trial_logs
            report.scores.append({"step": trial.number, "score": score})

            if score >= 1.0:
                trial.study.stop()

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed, multivariate=True),
        )

        study.optimize(objective, n_trials=self.max_steps)
        report.best_score = best_score
        return best_prompt, report

    async def create_n_fewshot_demo_sets(
        self, trainset: List[DatasetItem], predictions: List[Any], eval_results: List[MetricResult]
    ) -> Tuple[List[List[DatasetItem]], List[List[int]]]:
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
        for _ in range(self.num_candidates - 2):  # -2 because we already added no-shot and sampled-fewshot
            max_bootstrapped = random.randint(1, self.max_bootstrapped_demos)
            max_labeled = random.randint(1, self.max_labeled_demos)
            samples, indices = await self.sample(trainset, predictions, eval_results, max_bootstrapped, max_labeled)
            candidates.append(samples)
            candidate_indices.append(indices)

        return candidates, candidate_indices
    
    def sample_fewshot(self, trainset: List[DatasetItem], num_samples: int) -> Tuple[List[DatasetItem], List[int]]:
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

        failed_indices = [
            i for i, result in enumerate(eval_results) if result.score < self.success_score
        ]

        if failed_indices:
            weights = [1 - eval_results[i].score for i in failed_indices]
            labeled_indices = self.random_sample(
                failed_indices,
                num_shots=min(max_labeled_demos, len(failed_indices)),
                replace=False,
                weights=weights,
            )
            labeled_samples = [
                DatasetItem(inputs=trainset[i]["inputs"], outputs=trainset[i]["outputs"])
                for i in labeled_indices
            ]

        return bootstrapped_samples + labeled_samples, bootstrapped_indices + labeled_indices

    def random_sample(
        self,
        dataset: List[int],
        num_shots: int,
        replace: bool = False,
        weights: Optional[List[float]] = None,
        delta: float = 1e-5,
    ) -> List[int]:
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

        TIPS = {
            "creative": "Don't be afraid to be creative when creating the new instruction!",
            "simple": "Keep the instruction clear and concise.",
            "description": "Make sure your instruction is very informative and descriptive. You can add some hand-crafted examples to help the LLM understand the task better.",
            "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
            "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
        }

        async def propose_one(index: int) -> Prompt:
            selected_tip = list(TIPS.values())[index % len(TIPS)]

            fewshot = random.sample(trainset, min(len(trainset), 3))
            task_fewshot = format_fewshot(
                fewshot=fewshot, response_format=base_prompt.response_format
            )

            response_format_instructions = get_response_format_instructions(
                base_prompt.response_format
            )

            output = await self.generate_instructions_by_prompting(
                task_description="",
                dataset_desc=self.dataset_summary,
                task_fewshot=task_fewshot,
                prompt_desc=self.task_description,
                basic_prompt=base_prompt.dump(),
                tip=selected_tip,
                inputs_desc=base_prompt.inputs_desc if base_prompt.inputs_desc else "-",
                outputs_desc=base_prompt.outputs_desc if base_prompt.outputs_desc else "-",
                response_format_instructions=response_format_instructions,
            )

            try:
                extracted_prompt = extract_prompt(output)
                new_prompt_message = Prompt.load(extracted_prompt)
                if not new_prompt_message.messages:
                    raise ValueError("Generated prompt has no messages")
                new_prompt = base_prompt.deepcopy()
                new_prompt.messages = new_prompt_message.messages

                return new_prompt

            except Exception as e:
                logger.error(f"Error in propose_one: {e}")
                logger.error(f"Output: {output}")
                return base_prompt

        tasks = [propose_one(i) for i in range(num_candidates)]
        proposed_instructions = await asyncio.gather(*tasks)

        return proposed_instructions