from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict

from ape.common.metric import BaseMetric
from ape.common.global_metric import BaseGlobalMetric, AverageGlobalMetric
from ape.common.types import DatasetItem


class EvaluationConfig(BaseModel):
    testset: List[DatasetItem]
    metric: BaseMetric
    global_metric: Optional[BaseGlobalMetric] = AverageGlobalMetric()
    display_progress: bool = False
    display_table: Union[bool, int] = False
    max_errors: int = 15
    return_outputs: bool = False
    batch_size: int = 50
    return_all_scores: bool = False
    return_global_metric_metadata: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
