from pydantic import BaseModel
from typing import Dict, Optional

from ape.common.evaluate.evaluate import Evaluate
from ape.common.generate.generate import Generate
from ape.common.generate.generate_base import BaseGenerate
from ape.common.metric import BaseMetric
from ape.common.global_metric import BaseGlobalMetric


class MIPROBase(BaseModel):
    """
    Base class for MIPRO (Model-based Instruction Prompt Optimization) implementations.

    Attributes:
        task_model (Optional[str]): The model used for executing the task. Default is "gpt-4o-mini".
        teacher_settings (Dict): Settings for the teacher model. Default is an empty dictionary.
        num_candidates (int): Number of candidate prompts to generate. Default is 10.
        metric (Optional[BaseMetric]): The metric object used for evaluation. Default is None.
        init_temperature (float): Initial temperature for sampling. Default is 1.0.
        verbose (bool): Whether to print verbose output. Default is False.
        track_stats (bool): Whether to track statistics during optimization. Default is True.
        log_dir (Optional[str]): Directory for logging. Default is None.
        view_data_batch_size (int): Batch size for viewing data. Default is 10.
        minibatch_size (int): Size of minibatches for evaluation. Default is 25.
        update_prompt_after_full_eval (bool): Whether to only update the best prompt after full evaluation. Default is True.
        minibatch_full_eval_steps (int): Number of steps for full evaluation on minibatches. Default is 10.
    """

    task_model: Optional[str] = "gpt-4o-mini"
    teacher_settings: Dict = {}
    num_candidates: int = 10
    generate: Optional[BaseGenerate] = Generate()
    evaluate: Optional[Evaluate] = None  # If provided, 'metric' and 'global_metric' will be ignored
    metric: Optional[BaseMetric] = None
    global_metric: Optional[BaseGlobalMetric] = None
    init_temperature: float = 1.0
    verbose: bool = False
    track_stats: bool = True
    log_dir: Optional[str] = None
    view_data_batch_size: int = 10
    minibatch_size: int = 25
    update_prompt_after_full_eval: bool = True
    minibatch_full_eval_steps: int = 10

    class Config:
        arbitrary_types_allowed = True
