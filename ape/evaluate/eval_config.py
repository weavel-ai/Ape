from pydantic import BaseModel, ConfigDict
from typing import Optional, Union

from ape.types import Dataset
from ape.metric import BaseMetric
from ape.global_metric import GlobalMetric, AverageGlobalMetric


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

    model_config = ConfigDict(arbitrary_types_allowed=True)
