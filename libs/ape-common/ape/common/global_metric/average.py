from typing import List

from ape.common.types import EvaluationResult, GlobalMetricResult
from .global_metric_base import BaseGlobalMetric


class AverageGlobalMetric(BaseGlobalMetric):
    async def compute(self, results: List[EvaluationResult]) -> GlobalMetricResult:
        """
        Compute the average of local scores as the global metric.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations.

        Returns:
            GlobalMetricResult: The average score.
        """
        if not results:
            return GlobalMetricResult(score=0.0)
        return GlobalMetricResult(score=sum(result.score for result in results) / len(results))
