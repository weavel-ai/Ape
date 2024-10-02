from abc import ABC, abstractmethod
import asyncio
import inspect
import json
import random
from typing import Dict, Any, List, Optional, Tuple

from ape.common.prompt.prompt_base import Prompt
from ape.common.types.dataset_item import DatasetItem
from ape.common.types.eval_result import GlobalMetricResult, MetricResult
from ape.common.generate import BaseGenerate
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.core.v2.types.report import BaseReport

class BaseTrainer(ABC):
    def __init__(
        self,
        generator: BaseGenerate,
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
        task_description: Optional[str] = None,
        metric_description: Optional[str] = None,
        **kwargs,
    ):
        self.generator = generator
        self.metric = metric
        self.global_metric = global_metric
        self.task_description = task_description
        self.metric_description = metric_description
        self.dataset_summary = None
        
    @abstractmethod
    async def fit(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, BaseReport]:
        """
        Train model

        Args:
            prompt (Prompt): Prompt
            trainset (List[DatasetItem]): Training dataset
            valset (List[DatasetItem]): Validation dataset

        Returns:
            Tuple[Prompt, BaseReport]: Trained prompt and report
        """
        pass

    async def __call__(self, prompt: Prompt, trainset: List[DatasetItem], valset: List[DatasetItem]) -> Tuple[Prompt, BaseReport]:
        return await self.train(prompt, trainset, valset)
    
    async def _evaluate_dataset(
        self,
        dataset: List[DatasetItem],
        prompt: Prompt
    ) -> Tuple[List[Any], List[MetricResult], GlobalMetricResult]:
        """
        Evaluate a dataset using the generator and metric.

        Args:
            dataset (List[DatasetItem]): The dataset to evaluate.
            prompt (Prompt): The prompt to use for generation.

        Returns:
            GlobalMetricResult: The aggregated metric result for the dataset.
        """
        # Asynchronously generate predictions and compute metrics for each item
        generate_tasks = [
            self.generator.generate(
                messages=prompt.format(**item.inputs).messages,
                model=prompt.model
            ) for item in dataset
        ]
        preds = await asyncio.gather(*generate_tasks)

        metric_tasks = [
            self.metric.compute(
                inputs=item.inputs,
                pred=pred,
                gold=item.outputs
            ) for item, pred in zip(dataset, preds)
        ]
        eval_results = await asyncio.gather(*metric_tasks)

        # Compute the global metric
        global_score = await self.global_metric.compute(eval_results)
        return preds, eval_results, global_score
    
    async def _generate_task_description(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
    ) -> str:
        describe_prompt = Prompt.from_filename("describe-prompt")
        
        base_prompt_messages = [
            json.dumps(message) for message in prompt.messages
        ]
        base_prompt_messages_str = "\n".join(base_prompt_messages)
        
        temperature = describe_prompt.temperature
        
        for attempt in range(3):
            try:
                # Generate dataset summary
                dataset_summary = await self._dataset_summarizer(trainset)
                self.dataset_summary = dataset_summary
                
                # Describe the prompt
                prompt_description = await describe_prompt(
                    lm_config=dict(temperature=temperature),
                    prompt=base_prompt_messages_str, 
                    dataset_description=dataset_summary
                )
                
                prompt_description = prompt_description["description"]
                
                return prompt_description
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Error generating task description: {e}")
                    return ""
                print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1

    
    async def _generate_metric_description(
        self,
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
    ) -> str:
        
        gen_metric_description: Prompt = Prompt.from_filename("gen-metric-description")
        gen_metric_description_with_global_metric: Prompt = Prompt.from_filename("gen-metric-description-with-global-metric")
    
        compute_function = getattr(metric, "compute", None)
        compute_function_source_code = inspect.getsource(compute_function)
        
        if global_metric:
            global_metric_compute_function = getattr(global_metric, "compute", None)
            global_metric_compute_function_source_code = inspect.getsource(global_metric_compute_function)
            
            # get Prompt gen-metric-description-with-global-metric.prompt
            metric_str = await gen_metric_description_with_global_metric(
                **{
                    "metric_sourcecode": compute_function_source_code,
                    "global_metric_sourcecode": global_metric_compute_function_source_code,
                }
            )

        else:
            metric_str = await gen_metric_description(
                **{
                    "metric_sourcecode": compute_function_source_code,
                }
            )
        return metric_str

    async def _dataset_summarizer(
        self,
        trainset: List[DatasetItem],
        view_data_batch_size: Optional[int] = 10,
    ) -> str:
        upper_lim = min(len(trainset), view_data_batch_size)
        
        descriptor = Prompt.from_filename("dataset-descriptor")
        temperature = 0.7
        
        for attempt in range(3):
            try:
                formatted_examples = self._format_examples(trainset, upper_lim)
                
                output = await descriptor(
                    lm_config=dict(temperature=temperature),
                    examples=formatted_examples,
                )
                res = ""
                for line in output["observations"]:
                    res += line + "\n"
                return res
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e
                print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1
                
    def _format_examples(self, trainset: List[DatasetItem], batch_size: int) -> str:
        formatted_examples = ""
        random_samples = random.sample(trainset, min(batch_size, len(trainset)))
        
        for idx, item in enumerate(random_samples):
            formatted_examples += f"### Demo {idx+1} ###\n"
            
            if isinstance(item, DatasetItem):
                inputs, outputs = item.inputs, item.outputs
            else:
                inputs, outputs = item.get("inputs", {}), item.get("outputs", {})
            
            formatted_examples += "**Inputs**\n"
            for key, value in inputs.items():
                formatted_examples += f"{key.capitalize()}:\n{value}\n"
            
            formatted_examples += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"
        
        return formatted_examples
