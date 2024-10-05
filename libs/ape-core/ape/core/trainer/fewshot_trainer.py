import asyncio
import copy
import random
from typing import Any, List, Optional, Tuple

import numpy as np

from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.prompt import Prompt
from ape.common.types import MetricResult, DatasetItem
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import FewShotTrainerReport


class FewShotTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 5,
        random_seed: int = 42,
        success_score: float = 1.0,
        num_candidates: int = 10,
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.random_seed = random_seed
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.success_score = success_score
        self.num_candidates = num_candidates
        
        random.seed(random_seed)

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, FewShotTrainerReport]:
        report = FewShotTrainerReport(scores=[], choices=[], best_params={}, best_score=0.0)
        messages_str = ""
        for message in prompt.messages:
            messages_str += message["content"]
            
        if "{_FEWSHOT_}" not in messages_str:
            prompt = await self.generate_fewshot_placeholder(prompt)
            
        best_score = float("-inf")
        best_fewshot = []

        preds, eval_results, _ = await self._evaluate(trainset, prompt)

        fewshot_candidates, fewshot_candidate_indices = await self.create_n_fewshot_demo_sets(trainset, preds, eval_results)

        async def run_iteration(step: int, candidate: List[DatasetItem], indices: List[int]):
            temp_prompt = copy.deepcopy(prompt)
            temp_prompt.fewshot = candidate
            trainset_without_fewshot = [trainset[i] for i in range(len(trainset)) if i not in indices]
            _, _, global_result = await self._evaluate(trainset_without_fewshot, temp_prompt)

            report.scores.append({"step": step, "score": global_result.score})
            report.choices.append({"step": step, "fewshot": candidate})

            print(f"Step {step} completed. Score: {global_result.score}")
            return global_result.score, candidate

        results = await asyncio.gather(*[run_iteration(i, candidate, indices) for i, (candidate, indices) in enumerate(zip(fewshot_candidates, fewshot_candidate_indices))])

        for score, fewshot in results:
            if score > best_score:
                best_score = score
                best_fewshot = fewshot

        prompt.fewshot = best_fewshot
        report.best_params = {"fewshot": best_fewshot}
        report.best_score = best_score
        return prompt, report

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
            # shuffle the trainset
            shuffled_trainset = copy.deepcopy(trainset)
            random.shuffle(shuffled_trainset)
            samples, indices = await self.sample(shuffled_trainset, predictions, eval_results, max_bootstrapped, max_labeled)
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

        # Sample bootstrapped demos
        success_indices = [
            i for i, result in enumerate(eval_results) if result.score >= self.success_score
        ]

        if len(success_indices) > 0:
            bootstrapped_indices = random.sample(
                success_indices, min(max_bootstrapped_demos, len(success_indices))
            )
            bootstrapped_samples = [
                DatasetItem(inputs=trainset[i]["inputs"], outputs=predictions[i])
                for i in bootstrapped_indices
            ]

        # Sample labeled demos
        failed_indices = [
            i for i, result in enumerate(eval_results) if result.score < self.success_score
        ]

        if len(failed_indices) > 0:
            # Calculate weights based on (1 - score)
            weights = [1 - eval_results[i].score for i in failed_indices]

            # Use random_sample function with weights
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