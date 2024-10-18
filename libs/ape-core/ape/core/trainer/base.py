import asyncio
import copy
import inspect
import json
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.prompt import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.utils.logging import logger
from ape.core.core_prompts import ApeCorePrompts
from ape.core.types.report import BaseReport
from ape.core.utils import extract_prompt


class BaseTrainer(ABC):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
        task_description: Optional[str] = None,
        metric_description: Optional[str] = None,
        testmode: Optional[bool] = False,
        **kwargs,
    ):
        self.generate = generator
        self.metric = metric
        self.global_metric = global_metric
        self.task_description = task_description
        self.metric_description = metric_description
        self.dataset_summary = None
        self.testmode = testmode

    @abstractmethod
    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, BaseReport]:
        """
        Train the prompt

        Args:
            prompt (Prompt): Prompt
            trainset (List[DatasetItem]): Training dataset
            valset (List[DatasetItem]): Validation dataset

        Returns:
            Tuple[Prompt, BaseReport]: Trained prompt and report
        """
        pass

    async def __call__(
        self, prompt: Prompt, trainset: List[DatasetItem], valset: List[DatasetItem]
    ) -> Tuple[Prompt, BaseReport]:
        return await self.train(prompt=prompt, trainset=trainset, valset=valset)

    async def _evaluate(
        self, dataset: List[DatasetItem], prompt: Prompt
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
            self.generate(
                prompt=prompt,
                inputs=item["inputs"],
            )
            for item in dataset
        ]
        preds = []
        for i in range(0, len(generate_tasks), 50):
            preds.extend(await asyncio.gather(*generate_tasks[i:i+50]))

        metric_tasks = [
            self.metric(
                dataset_item=item,
                pred=pred,
            )
            for item, pred in zip(dataset, preds)
        ]
        eval_results = []
        for i in range(0, len(metric_tasks), 50):
            eval_results.extend(await asyncio.gather(*metric_tasks[i:i+50]))

        # Compute the global metric
        global_score = await self.global_metric(eval_results)
        return preds, eval_results, global_score

    async def _generate_task_description(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
    ) -> str:
        describe_prompt = ApeCorePrompts.get("describe-prompt")

        temperature = describe_prompt.temperature

        for attempt in range(3):
            try:
                # Generate dataset summary
                dataset_summary = await self._dataset_summarizer(trainset)
                self.dataset_summary = dataset_summary

                # Describe the prompt
                prompt_description = await describe_prompt(
                    lm_config=dict(temperature=temperature),
                    prompt=str(prompt.messages),
                    dataset_description=dataset_summary,
                )

                prompt_description = prompt_description["description"]

                return prompt_description
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.exception("Error generating task description")
                    return ""
                logger.warning(
                    f"Error generating task description: {e}. Retrying... (Attempt {attempt + 1}/3)"
                )
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1

    async def _generate_metric_description(
        self,
    ) -> str:

        gen_metric_description: Prompt = ApeCorePrompts.get("gen-metric-description")
        gen_metric_description_with_global_metric: Prompt = ApeCorePrompts.get(
            "gen-metric-description-with-global-metric"
        )

        compute_function = getattr(self.metric, "compute", None)
        compute_function_source_code = inspect.getsource(compute_function)

        if self.global_metric:
            global_metric_compute_function = getattr(self.global_metric, "compute", None)
            global_metric_compute_function_source_code = inspect.getsource(
                global_metric_compute_function
            )

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

        descriptor = ApeCorePrompts.get("dataset-descriptor")
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
                logger.warning(
                    f"Error summarizing dataset: {e}. Retrying... (Attempt {attempt + 1}/3)"
                )
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1

    def _format_examples(self, trainset: List[DatasetItem], batch_size: int) -> str:
        formatted_examples = ""
        random_samples = random.sample(trainset, min(batch_size, len(trainset)))

        for idx, item in enumerate(random_samples):
            formatted_examples += f"### Demo {idx+1} ###\n"

            inputs, outputs = item["inputs"], item["outputs"]

            formatted_examples += "**Inputs**\n"
            for key, value in inputs.items():
                formatted_examples += f"{key.capitalize()}:\n{value}\n"

            formatted_examples += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"

        return formatted_examples

    async def generate_fewshot_placeholder(self, prompt: Prompt) -> Prompt:
        fewshot_placeholder_generator = ApeCorePrompts.get("gen-fewshot-placeholder")

        retry_count = 0
        while retry_count < 5:
            try:
                new_prompt_raw = await fewshot_placeholder_generator(prompt=str(prompt.messages))
                new_prompt_messages = new_prompt_raw["messages"]
                new_prompt = copy.deepcopy(prompt)
                new_prompt.messages = new_prompt_messages
                return new_prompt
            except Exception as exc:
                logger.warning(
                    f"Error generating fewshot placeholder: {exc}. Retrying... (Attempt {retry_count + 1}/5)"
                )
                retry_count += 1
                if retry_count == 5:
                    raise ValueError(
                        f"Failed to generate fewshot placeholder after 5 attempts: {str(exc)}"
                    ) from exc
