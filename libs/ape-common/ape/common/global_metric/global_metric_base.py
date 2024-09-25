from abc import ABC, abstractmethod
import asyncio
from typing import List

from ape.common.types import EvaluationResult, GlobalMetricResult


class BaseGlobalMetric(ABC):
    @abstractmethod
    async def compute(self, results: List[EvaluationResult]) -> GlobalMetricResult:
        """
        Compute the global metric.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations.

        Returns:
            GlobalMetricResult: The computed global metric value.
        """
        pass

    async def __call__(self, results: List[EvaluationResult]) -> GlobalMetricResult:
        """
        Unified method to compute the global metric, handling both sync and async implementations.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations. use results[i].intermediate_values to get the local metric results.

        Returns:
            GlobalMetricResult: The computed global metric value.
        """
        result = self.compute(results)
        if asyncio.iscoroutine(result):
            return await result
        return result
