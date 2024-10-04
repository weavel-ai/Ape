from abc import ABC, abstractmethod
import asyncio
from typing import List

from ape.common.types import MetricResult, GlobalMetricResult


class BaseGlobalMetric(ABC):
    @abstractmethod
    async def compute(self, results: List[MetricResult]) -> GlobalMetricResult:
        """
        Compute the global metric. This method can be implemented as either synchronous or asynchronous.

        Args:
            results (List[MetricResult]): The results from BaseMetric evaluations.

        Returns:
            GlobalMetricResult: The computed global metric value.
        """
        pass

    async def __call__(self, results: List[MetricResult]) -> GlobalMetricResult:
        """
        Unified method to compute the global metric, handling both sync and async implementations.

        Args:
            results (List[MetricResult]): The results from BaseMetric evaluations. use results[i].metadata to get the local metric results.

        Returns:
            GlobalMetricResult: The computed global metric value.
        """
        result = self.compute(results)
        if asyncio.iscoroutine(result):
            return await result
        return result
