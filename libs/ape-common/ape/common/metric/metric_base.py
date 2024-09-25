from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import asyncio

from ape.common.types import MetricResult


class BaseMetric(ABC):
    @abstractmethod
    def compute(
        self,
        pred: Union[str, Dict[str, Any]],
        gold: Union[str, Dict[str, Any]],
        inputs: Dict[str, Any] = {},
        trace: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> MetricResult:
        """
        Compute the metric. This method can be implemented as either synchronous or asynchronous.

        Args:
            pred (Union[str, Dict[str, Any]]): The prediction to evaluate.
            gold (Union[str, Dict[str, Any]]): The ground truth to compare against.
            inputs (Dict[str, Any]): Additional input data that may be required for computation.
            trace (Optional[Dict]): Additional trace information for debugging or logging purposes.
            metadata (Optional[Dict]): Additional metadata that may be relevant to the metric computation.

        Returns:
            MetricResult: An object containing the computed score and any intermediate values.

        Note:
            The implementation of this method should handle the comparison between `pred` and `gold`,
            potentially using the information in `inputs`, `trace`, and `metadata` to inform the computation.
            It should return a MetricResult object that encapsulates the result of the metric calculation.
        """
        pass

    async def __call__(
        self,
        *,
        pred: Union[str, Dict[str, Any]],
        gold: Union[str, Dict[str, Any]],
        inputs: Dict[str, Any] = {},
        trace: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> MetricResult:
        """
        Unified method to compute the metric, handling both sync and async implementations.

        Args:
            inputs (Dict[str, Any]): The inputs.
            gold (Any): The ground truth.
            pred (Any): The prediction.
            trace (Optional[Dict]): Additional trace information.
            metadata (Optional[Dict]): Additional metadata.

        Returns:
            MetricResult: An object containing the score and intermediate values.
        """
        result = self.compute(pred=pred, gold=gold, inputs=inputs, trace=trace, metadata=metadata)
        if asyncio.iscoroutine(result):
            return await result
        return result
