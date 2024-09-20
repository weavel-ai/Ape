from typing import List, Tuple, Optional
from ape.types import DatasetItem
from ape.prompt.prompt_base import Prompt
from ape.evaluate import Evaluate

async def instruction_iteration(
    base_prompt: Prompt,
    trainset: List[DatasetItem],
    evaluator: Evaluate,
    task_description: Optional[str] = None,
    metric_description: Optional[str] = None,
    iteration_count: int = 10,
) -> Tuple[Prompt, List[str]]:
    pass
    # TODO: make experiment list 
    
    experiment_prompts: List[Prompt] = []
    
    # Minibatch testing? rank the experiment results