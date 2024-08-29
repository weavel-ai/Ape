from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, inputs: Dict[str, Any], gold: Any, pred: Any, trace: Optional[Dict] = None) -> float:
        """
        Compute the metric. This method can be implemented as either synchronous or asynchronous.

        Args:
            gold (Any): The ground truth.
            pred (Any): The prediction.
            trace (Optional[Dict]): Additional trace information.

        Returns:
            float: The computed metric value.
        """
        pass

    async def __call__(
        self, inputs: Dict[str, Any], gold: Any, pred: Any, trace: Optional[Dict] = None
    ) -> float:
        """
        Unified method to compute the metric, handling both sync and async implementations.

        Args:
            inputs (Dict[str, Any]): The inputs.
            gold (Any): The ground truth.
            pred (Any): The prediction.
            trace (Optional[Dict]): Additional trace information.

        Returns:
            float: The computed metric value.
        """
        result = self.compute(inputs, gold, pred, trace)
        if asyncio.iscoroutine(result):
            return await result
        return result
