import asyncio
import copy
import json
import random
from collections import deque
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple
from ape.common.prompt.prompt_base import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.utils.logging import logger
from ape.core.utils import extract_prompt, reformat_prompt
from ape.core.core_prompts import ApeCorePrompts
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import TextGradientTrainerReport


class TextGradientTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        batch_size: int = 4,
        early_stopping_rounds: int = 10,
        random_seed: int = 42,
        max_proposals_per_step: int = 5,
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.max_proposals_per_step = max_proposals_per_step

        self.text_gradient_generator_prompt = ApeCorePrompts.get("text-gradient-generator")
        self.text_gradient_applier_prompt = ApeCorePrompts.get("text-gradient-applier")

        random.seed(random_seed)

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, TextGradientTrainerReport]:
        """
        Train the model using text gradients.

        Args:
            prompt (Prompt): The initial prompt to start training.
            trainset (List[DatasetItem]): The training dataset.
            valset (List[DatasetItem]): The validation dataset.

        Returns:
            Tuple[Prompt, Optional[TextGradientTrainerReport]]: The best prompt and an optional training report.
        """

        # Initialize Text Gradient Trainer Report
        report = TextGradientTrainerReport(
            scores=[],
            text_gradients=[],
            best_score=0.0,
        )

        # Step 1: Shuffle the training set without modifying the original list
        shuffled_trainset = copy.deepcopy(trainset)
        random.shuffle(shuffled_trainset)

        # Step 2: Initialize best_prompt and failed_count
        best_prompt = prompt
        prompt_history_queue = deque(maxlen=4)

        _, best_trainset_results, best_trainset_global_result = await self._evaluate(
            prompt=best_prompt, dataset=trainset
        )
        best_trainset_score = best_trainset_global_result.score

        # Iterate over the shuffled training set in batches
        for batch_start in tqdm(
            range(0, len(shuffled_trainset), self.batch_size), desc="Training Batches"
        ):
            batch = shuffled_trainset[batch_start : batch_start + self.batch_size]

            # Step 3: Run generator in parallel for the current batch
            best_batch_preds, best_batch_eval_results, best_batch_global_result = (
                await self._evaluate(batch, best_prompt)
            )
            best_batch_score = best_batch_global_result.score

            best_batch_results: List[Tuple[DatasetItem, Any, MetricResult]] = []
            for item, pred, eval_result in zip(batch, best_batch_preds, best_batch_eval_results):
                best_batch_results.append((item, pred, eval_result))

            # Initialize retry mechanism
            retry_count = 0
            success = False
            if best_batch_score == 1.0:
                logger.debug("Batch score is 1.0, skipping text gradient generation")
                success = True
                
            text_gradient_history = {}
            while retry_count < self.max_proposals_per_step and not success:
                text_gradients = await asyncio.gather(
                    *[
                        self._text_gradient_generator(
                            prompt=best_prompt,
                            inputs=item["inputs"],
                            outputs=item["outputs"],
                            generator_output=pred,
                            metric_result=eval_result,
                            text_gradient_history=text_gradient_history
                        )
                        for (item, pred, eval_result) in best_batch_results
                        if eval_result.score < 1.0 # TODO: fix the score threshold into dynamic
                    ]
                )  
                text_gradients = [tg for tg in text_gradients if tg]  # Filter out empty gradients
                logger.debug(f"Generated {len(text_gradients)} text gradients")

                if not text_gradients:
                    logger.debug("No valid text gradients generated. Skipping to next batch.")
                    break  # Exit retry loop if no gradients are generated

                # Step 10: Apply text gradients in batches
                new_prompt = await self._text_gradient_applier(
                    prompt=best_prompt,
                    text_gradient="\n".join(text_gradients),
                    prompt_history=prompt_history_queue,
                )

                # Evaluate new_prompt on the current batch
                new_batch_preds, new_batch_eval_results, new_batch_global_result = (
                    await self._evaluate(batch, new_prompt)
                )
                new_batch_score = new_batch_global_result.score

                new_batch_results = []
                for item, pred, eval_result in zip(
                    batch, new_batch_preds, new_batch_eval_results
                ):
                    new_batch_results.append((item, pred, eval_result))

                if new_batch_score > best_batch_score:
                    # Evaluate new_prompt on trainset
                    new_trainset_preds, new_trainset_eval_results, new_trainset_global_result = (
                        await self._evaluate(prompt=new_prompt, dataset=trainset)
                    )
                    new_trainset_score = new_trainset_global_result.score

                    prompt_history_queue.append(
                        {"prompt": new_prompt, "score": new_trainset_score}
                    )

                    # Compare new score with the current best score
                    if new_trainset_score > best_trainset_score:
                        logger.debug(
                            f"Trial {retry_count + 1}: Score Improved: {best_trainset_score} -> {new_trainset_score}"
                        )
                        best_prompt = new_prompt
                        best_trainset_score = new_trainset_score
                        best_batch_results = new_batch_results

                        # Update report with text_gradients
                        report.text_gradients.extend(text_gradients)
                        success = True  # Mark as successful update
                    else:
                        logger.debug(
                            f"Trial {retry_count + 1}: Score Not Improved: {best_trainset_score} -> {new_trainset_score}"
                        )
                        # Increment retry_count
                        retry_count += 1

                        if retry_count >= self.max_proposals_per_step:
                            logger.debug(
                                f"Maximum retries ({self.max_proposals_per_step}) reached for this batch."
                            )
                        else:
                            logger.debug("Retrying with a new proposal...")
                else:
                    logger.debug(
                        f"Trial {retry_count + 1}: Score Not Improved in batch: {best_batch_score} -> {new_batch_score}"
                    )
                    prompt_history_queue.append({"prompt": new_prompt, "score": 0.0})
                    retry_count += 1
                    if retry_count >= self.max_proposals_per_step:
                        logger.debug(
                            f"Maximum retries ({self.max_proposals_per_step}) reached for this batch."
                        )
                    else:
                        logger.debug("Retrying with a new proposal...")

            # Update report with new score
            if self.testmode:
                _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                report.scores.append({"step": len(report.scores), "score": best_trainset_score, "val_score": val_global_result.score})
            else:
                report.scores.append({"step": len(report.scores), "score": best_trainset_score})
                
            if best_trainset_score == 1.0:
                logger.debug("Score reached 1.0")
                report.best_score = 1.0
                return best_prompt, report

        _, _, trainset_global_result = await self._evaluate(trainset, best_prompt)
        trainset_score = trainset_global_result.score
        report.best_score = trainset_score
        return best_prompt, report
    
    async def _text_gradient_generator(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any],
        outputs: Any,
        generator_output: Any,
        metric_result: MetricResult,
        text_gradient_history: Dict[str, List[str]],
    ) -> Any:
        """
        Generate text gradient based on inputs, outputs, generator outputs, and metric results.
        """
        retry_count = 0
        text_gradient_history_str = ""
        text_gradient_history_list = text_gradient_history.get(str(inputs), [])
        for tg in text_gradient_history_list:
            text_gradient_history_str += f"{tg}\n"
        
        while retry_count < 3:
            try:
                text_gradient = await self.text_gradient_generator_prompt(
                    task_description=self.task_description,
                    metric_description=self.metric_description,
                    base_prompt=str(prompt.messages),
                    inputs=str(inputs),
                    outputs=str(outputs),
                    generator_output=str(generator_output),
                    metric_result=str(metric_result),
                    feedback_history=text_gradient_history_str,
                    _retry_count=retry_count
                )
                text_gradient = text_gradient.strip()
                if not text_gradient.startswith("{"):
                    text_gradient = "{" + text_gradient

                text_gradient = json.loads(text_gradient)["feedback"]
                if str(inputs) not in text_gradient_history:
                    text_gradient_history[str(inputs)] = []
                text_gradient_history[str(inputs)].append(text_gradient)
                return text_gradient
            except Exception as e:
                logger.warning(f"Error generating text gradient: {e}")
                retry_count += 1
                if retry_count >= 3:
                    return ""

    async def _text_gradient_applier(
        self, prompt: Prompt, text_gradient: str, prompt_history: List[Dict[str, Any]]
    ) -> Prompt:
        """
        Apply text gradient to the prompt to obtain a new prompt.
        """
        retry_count = 0

        prompt_history_str = ""
        for ph in prompt_history:
            prompt_history_str += f"Prompt: {str(ph['prompt'].messages)}"
            prompt_history_str += f"Score: {ph['score']}\n\n"

        while retry_count < 3:
            try:
                new_prompt_raw = await self.text_gradient_applier_prompt(
                    task_description=self.task_description,
                    base_prompt=str(prompt.messages),
                    feedback=text_gradient,
                    prompt_history=prompt_history_str,
                    _retry_count=retry_count
                )

                new_prompt_message = new_prompt_raw["messages"]
                new_prompt = prompt.deepcopy()
                new_prompt.messages = new_prompt_message

                # add "json" if response_format is not None & response_format.type is "json_object" & "json" not in messages
                messages = [json.dumps(message) for message in new_prompt.messages]
                messages_str = "\n".join(messages)
                if (
                    new_prompt.response_format is not None
                    and new_prompt.response_format["type"] == "json_object"
                    and "json" not in messages_str
                ):
                    # add "json" to the messages
                    new_prompt = await reformat_prompt(new_prompt, new_prompt.response_format)

                return new_prompt
            except Exception as e:
                logger.warning(f"Error applying text gradient: {e}")
                retry_count += 1
