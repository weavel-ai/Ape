from typing import List
from ape.global_metric.global_metric_base import GlobalMetric
from ape.types import EvaluationResult


class AverageGlobalMetric(GlobalMetric):
    async def compute(self, results: List[EvaluationResult]) -> float:
        """
        Compute the average of local scores as the global metric.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations.

        Returns:
            float: The average score.
        """
        if not results:
            return 0.0
        return sum(result.score for result in results) / len(results)
