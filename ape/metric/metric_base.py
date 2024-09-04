from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Literal, Optional, Union
import asyncio

from ape.types import Dataset


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
        self, inputs: Dict[str, Any], gold: Any, pred: Any, trace: Optional[Dict] = None, metadata: Optional[Dict] = None
    ) -> Union[float, Dict[str, Any]]:
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

class EvaluationResult(BaseModel):
    example: Union[dict, str]
    prediction: Union[dict, str]
    score: Optional[float] = 0.0
    eval_results: Optional[Dict[str, Any]] = None


class ExtraMetric(ABC):
    @abstractmethod
    async def compute(self, results: List[EvaluationResult]) -> float:
        """
        Compute the metric. This method can be implemented as either synchronous or asynchronous.

        Args:
            results (List[EvaluationResult]): The results.
        """
        pass
    
    async def __call__(self, results: List[EvaluationResult]) -> float:
        """
        Unified method to compute the metric, handling both sync and async implementations.

        Args:
            results (List[EvaluationResult]): The results. use EvaluationResult.eval_results Dict to calculate final evaluation score.

        Returns:
            float: The computed metric value.   
        """
        result = self.compute(results)
        if asyncio.iscoroutine(result):
            return await result
        return result
    

class EvaluationConfig(BaseModel):
    testset: Dataset
    metric: Optional[BaseMetric] = None
    metric_type: Optional[Literal["average", "global"]] = "average"
    global_extra_metric: Optional[ExtraMetric] = None
    display_progress: bool = False
    display_table: Union[bool, int] = False
    max_errors: int = 15
    return_outputs: bool = False
    batch_size: int = 50
    return_all_scores: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)