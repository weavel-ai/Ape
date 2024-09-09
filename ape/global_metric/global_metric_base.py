from abc import ABC, abstractmethod
from typing import List
import asyncio

from ape.types import EvaluationResult


class GlobalMetric(ABC):
    @abstractmethod
    async def compute(self, results: List[EvaluationResult]) -> float:
        """
        Compute the global metric.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations.

        Returns:
            float: The computed global metric value.
        """
        pass

    async def __call__(self, results: List[EvaluationResult]) -> float:
        """
        Unified method to compute the global metric, handling both sync and async implementations.

        Args:
            results (List[EvaluationResult]): The results from BaseMetric evaluations. use results[i].trace to get the trace of the local metric.

        Returns:
            float: The computed global metric value.
        """
        result = self.compute(results)
        if asyncio.iscoroutine(result):
            return await result
        return result
