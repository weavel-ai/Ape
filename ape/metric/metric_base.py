from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional, Union
import asyncio

from ape.types import Dataset, MetricResult, EvaluationResult, GlobalMetricResult

class BaseMetric(ABC):
    @abstractmethod
    def compute(
        self,
        inputs: Dict[str, Any],
        gold: Any,
        pred: Any,
        trace: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> MetricResult:
        """
        Compute the metric. This method can be implemented as either synchronous or asynchronous.

        Args:
            inputs (Dict[str, Any]): The inputs.
            gold (Any): The ground truth.
            pred (Any): The prediction.
            trace (Optional[Dict]): Additional trace information.
            metadata (Optional[Dict]): Additional metadata.
        Returns:
            MetricResult: An object containing the score and intermediate values.
        """
        pass

    async def __call__(
        self,
        inputs: Dict[str, Any],
        gold: Any,
        pred: Any,
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
        result = self.compute(inputs, gold, pred, trace, metadata)
        if asyncio.iscoroutine(result):
            return await result
        return result

class GlobalMetric(ABC):
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

class AverageGlobalMetric(GlobalMetric):
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

class EvaluationConfig(BaseModel):
    testset: Dataset
    metric: BaseMetric
    global_metric: Optional[GlobalMetric] = AverageGlobalMetric()
    display_progress: bool = False
    display_table: Union[bool, int] = False
    max_errors: int = 15
    return_outputs: bool = False
    batch_size: int = 50
    return_all_scores: bool = False
    return_global_metric_metadata: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)