import asyncio
import copy
import random
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple
from ape.common.prompt.prompt_base import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generate import BaseGenerate
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.core.v2.paraphraser.base import BaseParaphraser
from ape.core.v2.trainer.base import BaseTrainer
from ape.core.v2.types.report import FewShotTrainerReport

class FewShotTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerate,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        paraphraser: Optional[BaseParaphraser] = None,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 5,
        random_seed: int = 42,
        score_threshold: float = 1.0,
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.paraphraser = paraphraser
        self.paraphraser = paraphraser  
        self.random_seed = random_seed
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.score_threshold = score_threshold
        random.seed(random_seed)
        
    async def fit(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, FewShotTrainerReport]:
        report = FewShotTrainerReport(scores=[], choices=[], best_params={})
        best_score = float('-inf')
        best_fewshot = []
        
        preds, eval_results, global_result = await self._evaluate_dataset(trainset, prompt)

        async def run_iteration(step: int, preds, eval_results):
            max_bootstrapped = random.randint(1, self.max_bootstrapped_demos)
            max_labeled = random.randint(1, self.max_labeled_demos)
            
            samples, indices = await self.sample(trainset, preds, eval_results, max_bootstrapped, max_labeled)            # Exclude sampled examples from validation
            validation_set = [item for i, item in enumerate(trainset) if i not in indices]
            
            temp_prompt = copy.deepcopy(prompt)
            temp_prompt.fewshot = samples
            
            _, eval_results, global_score = await self._evaluate_dataset(validation_set, temp_prompt)
            
            report.scores.append({"step": step, "score": global_score.score})
            report.choices.append({"step": step, "indices": indices})
            
            print(f"Step {step} completed. Score: {global_score.score}")
            return global_score.score, samples

        results = await asyncio.gather(*[run_iteration(i, preds, eval_results) for i in range(10)])

        for score, fewshot in results:
            if score > best_score:
                best_score = score
                best_fewshot = fewshot

        prompt.fewshot = best_fewshot
        return prompt, report

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
        success_indices = [i for i, result in enumerate(eval_results) if result.score >= self.score_threshold]
        
        if len(success_indices) > 0:
            bootstrapped_indices = random.sample(success_indices, min(max_bootstrapped_demos, len(success_indices)))
            bootstrapped_samples = [
                DatasetItem(
                    inputs=trainset[i]["inputs"],
                    outputs=predictions[i]
                ) for i in bootstrapped_indices
            ]
        
        # Sample labeled demos
        failed_indices = [i for i, result in enumerate(eval_results) if result.score < self.score_threshold]
        
        if len(failed_indices) > 0:
            # Calculate weights based on (1 - score)
            weights = [1 - eval_results[i].score for i in failed_indices]
            
            # Use random_sample function with weights
            labeled_indices = self.random_sample(
                failed_indices,
                num_shots=min(max_labeled_demos, len(failed_indices)),
                replace=False,
                weights=weights
            )
            
            labeled_samples = [
                DatasetItem(
                    inputs=trainset[i]["inputs"],
                    outputs=trainset[i]["outputs"]
                ) for i in labeled_indices
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
