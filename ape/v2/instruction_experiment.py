import asyncio
from typing import List, Tuple
from pydantic import BaseModel
from ape.prompt.prompt_base import Prompt
from ape.evaluate import Evaluate
from ape.types.v2 import InstructionExperimentResult, ValueAnalysisResult
from ape.types import GlobalMetricResult
from ape.v2.instruction_tools import (
    direction_generator_prompting_base,
    direction_generator_eval_base,
    update_proposer,
    update_splitter,
    update_applier,
    split_analyzer,
    paraphraser,
    paraphrase_analyzer,
)

from ape.v2.general_tools import (
    value_analyzer,
    dataset_summarizer,
    task_description_generator,
    metric_description_generator,
)

class ParaphraseResult(BaseModel):
    is_success: bool
    paraphrased_prompt: Prompt
    update_log: str

async def _paraphrase(
    metric_description: str,
    experiment_description: str,
    base_prompt: Prompt,
    base_prompt_trainset_result: GlobalMetricResult,
    new_prompt: Prompt,
    previous_paraphrase_results: str,
    evaluator: Evaluate,
) -> ParaphraseResult:
    paraphrased_prompts: List[Tuple[Prompt, str]] = await asyncio.gather(*[
        paraphraser(
            experiment_description=experiment_description,
            base_prompt=base_prompt,
            new_prompt=new_prompt,
            previous_paraphrase_results=previous_paraphrase_results,
        ) for _ in range(5)
    ])
    
    # control the number of evaluation data
    paraphrased_evaluation_results: List[Tuple[float, GlobalMetricResult]] = await asyncio.gather(*[
        evaluator(
            prompt=paraphrased_prompt[0],
            return_global_metric_metadata=True,
        ) for paraphrased_prompt in paraphrased_prompts
    ])
    
    paraphrased_values: List[ValueAnalysisResult] = await asyncio.gather(*[
        value_analyzer(
            metric_description=metric_description,
            experiment_description=experiment_description,
            new_prompt=paraphrased_prompt[0],
            base_result=base_prompt_trainset_result,
            new_result=paraphrased_evaluation_results[i][1],
        ) for i, paraphrased_prompt in enumerate(paraphrased_prompts)
    ])
    
    success_index = [i for i, result in enumerate(paraphrased_values) if result.is_success]
    success_index.sort(key=lambda x: paraphrased_values[x].value, reverse=True)
    
    paraphrase_analysis = await paraphrase_analyzer(
        experiment_description=experiment_description,
        paraphrase_results=[
            (prompt[1], result[1], value) for prompt, result, value in zip(paraphrased_prompts, paraphrased_evaluation_results, paraphrased_values)
        ]
    )

    if len(success_index) == 0:
        return ParaphraseResult(
            is_success=False,
            paraphrase_prompt=new_prompt,
            update_log=paraphrase_analysis.failure_analysis
        )
    
    else:
        best_prompt = paraphrased_prompts[success_index[0]][0]
        return ParaphraseResult(
            is_success=True,
            paraphrase_prompt=best_prompt,
            update_log=paraphrase_analysis.success_analysis + "\n" + paraphrase_analysis.failure_analysis
        )
    
    

async def instruction_experiment(
    base_prompt: Prompt,
    task_description: str,
    metric_description: str,
    tip: str,
    trainset_evaluator: Evaluate,
    base_prompt_trainset_result: GlobalMetricResult,
) -> InstructionExperimentResult:
    
    experiment_direction = await direction_generator_eval_base(
        task_description=task_description,
        base_prompt=base_prompt,
        trainset_result=base_prompt_trainset_result,
        metric_description=metric_description,
        tip=tip
    )
    
    updated_prompt = await update_proposer(
        task_description=task_description,
        base_prompt=base_prompt,
        direction_description=experiment_direction,
    )
    
    # update_prompt possibility evaluate
    update_prompt_possibility = False
    
    if not update_prompt_possibility:
        return InstructionExperimentResult(
            experiment_success=False
        )
    
    best_prompt = updated_prompt
    # paraphrase 3 epoch
    paraphrase_epoch = 3
    for i in range(paraphrase_epoch):
        result = await _paraphrase(
            metric_description=metric_description,
            experiment_description=experiment_direction,
            base_prompt=base_prompt,
            base_prompt_trainset_result=base_prompt_trainset_result,
            new_prompt=best_prompt,
            previous_paraphrase_results=previous_paraphrase_results
        )
        
        if not result.is_success:
            return InstructionExperimentResult(
                experiment_success=True,
                best_prompt=best_prompt,
                experiment_direction=experiment_direction,
            )
        
        best_prompt = result.paraphrase_prompt
        previous_paraphrase_results += result.update_log
        
    return InstructionExperimentResult(
        experiment_success=True,
        best_prompt=best_prompt
    )