import asyncio
import copy
import json
import random
from collections import deque
from pprint import pprint
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from ape.common.prompt.prompt_base import Prompt
from ape.common.proposer.utils import extract_prompt
from ape.common.types.dataset_item import DatasetItem
from ape.common.types.metric import GlobalMetricResult, MetricResult
from ape.common.generate import BaseGenerate
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.core.v2.paraphraser.base import BaseParaphraser
from ape.core.v2.trainer.base import BaseTrainer
from ape.core.v2.types.report import TextGradientTrainerReport
from ape.core.optimizer.utils import reformat_prompt

class TextGradientTrainerV4(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerate,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        paraphraser: Optional[BaseParaphraser] = None,
        batch_size: int = 4,
        early_stopping_rounds: int = 10,
        random_seed: int = 42,
        max_proposals_per_step: int = 5,
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.paraphraser = paraphraser
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.paraphraser = paraphraser  
        self.random_seed = random_seed
        self.max_proposals_per_step = max_proposals_per_step
        
        self.text_gradient_generator_prompt = Prompt.from_filename("text-gradient-generator")
        self.text_gradient_applier_prompt = Prompt.from_filename("text-gradient-applier")
        
        self.buffer_trainset = deque(maxlen=20)
        random.seed(random_seed)
        
    async def fit(
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
        report = TextGradientTrainerReport(scores=[], text_gradients=[])

        # Step 1: Shuffle the training set without modifying the original list
        shuffled_trainset = copy.deepcopy(trainset)
        random.shuffle(shuffled_trainset)

        # Step 2: Initialize best_prompt and failed_count
        best_prompt = prompt
        step_failed_count = 0
        # text_gradient_result_queue = deque(maxlen=4)
        prompt_history_queue = deque(maxlen=4)

        ## Step 8: Compute valset_score
        # valset_score = await self._evaluate_dataset(valset, best_prompt)

        # Iterate over the shuffled training set in batches
        for batch_start in tqdm(range(0, len(shuffled_trainset), self.batch_size), desc="Training Batches"):
            batch = shuffled_trainset[batch_start:batch_start + self.batch_size]

            # Step 3: Run generator in parallel for the current batch
            best_batch_preds = await asyncio.gather(*[
                self.generator.generate(
                    messages=best_prompt.format(**item.inputs).messages,
                    model=best_prompt.model
                ) for item in batch
            ])

            # Step 5: Run metric in parallel for the generated predictions
            best_batch_eval_results: List[MetricResult] = await asyncio.gather(*[
                self.metric.compute(
                    inputs=item.inputs,
                    pred=pred,
                    gold=item.outputs
                ) for item, pred in zip(batch, best_batch_preds)
            ])
            
            best_batch_score: GlobalMetricResult = await self.global_metric.compute(best_batch_eval_results)

            # Step 6: Store (output, eval_result) in buffer_trainset_result
            best_batch_results: List[Tuple[DatasetItem, Any, MetricResult]] = []
            for item, pred, eval_result in zip(batch, best_batch_preds, best_batch_eval_results):
                best_batch_results.append((item, pred, eval_result))
            
            # Compute buffer_trainset_score using global_metric
            best_buffer_trainset_results = [er for (_, _, er) in self.buffer_trainset]
            best_buffer_trainset_score = await self.global_metric.compute(best_buffer_trainset_results + best_batch_eval_results)
            
            # Initialize retry mechanism
            retry_count = 0
            success = False

            while retry_count < self.max_proposals_per_step and not success:
                # Step 9: Compute text gradients for the buffered training set
                text_gradients = await asyncio.gather(*[
                    self._text_gradient_generator(
                        prompt=best_prompt,
                        inputs=item.inputs,
                        outputs=item.outputs,
                        generator_output=pred,
                        metric_result=eval_result,
                        # text_gradient_momentum=list(text_gradient_result_queue)
                    ) for (item, pred, eval_result) in best_batch_results if eval_result.score < 1.0
                ]) # TODO: fix the score threshold into dynamic)
                text_gradients = [tg for tg in text_gradients if tg]  # Filter out empty gradients
                print(f"Generated {len(text_gradients)} text gradients")

                if not text_gradients:
                    print("No valid text gradients generated. Skipping to next batch.")
                    break  # Exit retry loop if no gradients are generated

                # Step 10: Apply text gradients in batches
                if self.paraphraser is not None:
                    best_prompt, best_batch_results, best_buffer_trainset_results = await self.paraphraser(
                        prompt=best_prompt,
                        trainset=trainset,
                        valset=valset,
                        buffer_trainset=list(self.buffer_trainset),
                        text_gradient="\n".join(text_gradients)
                    )
                    success = True
                else:
                    # Apply paraphraser in this iteration
                    new_prompt = await self._text_gradient_applier(
                        prompt=best_prompt, 
                        text_gradient="\n".join(text_gradients), 
                        prompt_history=list(prompt_history_queue)
                    )
                    
                    # Evaluate new_prompt on the current batch
                    new_batch_preds = await asyncio.gather(*[
                        self.generator.generate(
                            messages=new_prompt.format(**item.inputs).messages,
                            model=new_prompt.model
                        ) for item in batch
                    ])

                    new_batch_eval_results = await asyncio.gather(*[
                        self.metric.compute(
                            inputs=item.inputs,
                            pred=pred,
                            gold=item.outputs
                        ) for item, pred in zip(batch, new_batch_preds)
                    ])
                    
                    new_batch_results = []
                    for item, pred, eval_result in zip(batch, new_batch_preds, new_batch_eval_results):
                        new_batch_results.append((item, pred, eval_result))

                    # Compute new global metric score for the batch
                    new_batch_score = await self.global_metric.compute(new_batch_eval_results)
                    
                    if new_batch_score.score > best_batch_score.score:
                        
                        # Step 11: Evaluate new_prompt on buffer_trainset
                        new_buffer_trainset_preds = await asyncio.gather(*[
                            self.generator.generate(
                                messages=new_prompt.format(**item.inputs).messages,
                                model=new_prompt.model
                            ) for (item, _, _) in self.buffer_trainset
                        ])

                        new_buffer_trainset_eval_results = await asyncio.gather(*[
                            self.metric.compute(
                                inputs=item.inputs,
                                pred=pred,
                                gold=item.outputs
                            ) for ((item, _, _), pred) in zip(self.buffer_trainset, new_buffer_trainset_preds)
                        ])

                        # Compute new global metric score
                        new_buffer_trainset_score = await self.global_metric.compute(
                            new_buffer_trainset_eval_results + new_batch_eval_results
                        )
                        
                        # text_gradient_result_queue.append(
                        #     {
                        #         "text_gradient": "\n".join(text_gradients),
                        #         "score": new_buffer_trainset_score.score
                        #     }
                        # )
                        prompt_history_queue.append(
                            {
                                "prompt": new_prompt,
                                "score": new_buffer_trainset_score.score
                            }
                        )

                        # Step 12: Compare new score with the current best score
                        if new_buffer_trainset_score.score > best_buffer_trainset_score.score:
                            print(f"Trial {retry_count + 1}: Score Improved: {best_buffer_trainset_score.score} -> {new_buffer_trainset_score.score}")
                            # Update buffers with new predictions and metric results
                            best_prompt = new_prompt
                            best_buffer_trainset_score = new_buffer_trainset_score
                            best_batch_results = new_batch_results

                            # Clear existing buffer and repopulate with new predictions
                            for i, (item, _, _) in enumerate(self.buffer_trainset):
                                self.buffer_trainset[i] = (item, new_buffer_trainset_preds[i], new_buffer_trainset_eval_results[i])
                            
                            # Update report with text_gradients
                            report.text_gradients.extend(text_gradients)
                            success = True  # Mark as successful update
                            step_failed_count = 0
                        else:
                            print(f"Trial {retry_count + 1}: Score Not Improved: {best_buffer_trainset_score.score} -> {new_buffer_trainset_score.score}")
                            # Step 13: Increment retry_count
                            retry_count += 1

                            if retry_count >= self.max_proposals_per_step:
                                print(f"Maximum retries ({self.max_proposals_per_step}) reached for this batch.")
                                step_failed_count += 1
                                if step_failed_count > self.early_stopping_rounds:
                                    print("Early Stopping Triggered")
                                    return best_prompt, report
                            else:
                                print("Retrying with a new proposal...")
                    else:
                        print(f"Trial {retry_count + 1}: Score Not Improved in batch: {best_batch_score.score} -> {new_batch_score.score}")
                        # text_gradient_result_queue.append(
                        #     {
                        #         "text_gradient": "\n".join(text_gradients),
                        #         "score": 0.0
                        #     }
                        # )
                        prompt_history_queue.append(
                            {
                                "prompt": new_prompt,
                                "score": 0.0
                            }
                        )
                        retry_count += 1
                        if retry_count >= self.max_proposals_per_step:
                            print(f"Maximum retries ({self.max_proposals_per_step}) reached for this batch.")
                            step_failed_count += 1
                            if step_failed_count > self.early_stopping_rounds:
                                print("Early Stopping Triggered")
                                return best_prompt, report
                        else:
                            print("Retrying with a new proposal...")                            

            # add new samples to the buffer
            self._manage_buffer(best_batch_results)
            print(f"Buffer Trainset Updated: {len(self.buffer_trainset)}")
            
            # Update report with new score
            report.scores.append({
                "step": len(report.scores),
                "score": best_buffer_trainset_score.score
            })
            if best_buffer_trainset_score.score == 1.0:
                # test valset and early stop if the score is 1.0
                _, _, valset_score = await self._evaluate_dataset(valset, best_prompt)
                if valset_score.score == 1.0:
                    return best_prompt, report

        # Step 14: All data processed, return the best_prompt found and the report
        return best_prompt, report
    
    def _manage_buffer(self, new_samples: List[Tuple[Any, Any, Any]]):
        """
        Manage the buffer_trainset to ensure it contains at most max_buffer_size samples
        with a balanced ratio of success and failure samples.

        Args:
            new_samples (List[Tuple[Any, Any, Any]]): New samples to add to the buffer.
        """
        # Add new samples to the buffer
        for sample in new_samples:
            self.buffer_trainset.append(sample)

        # If buffer size exceeds max_buffer_size, remove oldest samples to maintain balance
        while len(self.buffer_trainset) > 20:
            # Count current successes and failures
            num_success = sum(1 for s in self.buffer_trainset if s[2].score >= 0.5)
            num_failure = sum(1 for s in self.buffer_trainset if s[2].score < 0.5)

            # Determine desired counts
            desired_success = 10
            desired_failure = 10

            # Calculate excess
            excess_success = num_success - desired_success
            excess_failure = num_failure - desired_failure

            # Decide which class to remove from
            if excess_success > 0:
                # Remove the oldest success sample
                for idx, sample in enumerate(self.buffer_trainset):
                    if sample[2].score >= 0.5:
                        del self.buffer_trainset[idx]
                        break
            elif excess_failure > 0:
                # Remove the oldest failure sample
                for idx, sample in enumerate(self.buffer_trainset):
                    if sample[2].score < 0.5:
                        del self.buffer_trainset[idx]
                        break
            else:
                # If no class is in excess, remove the oldest sample
                self.buffer_trainset.popleft()
    
    async def _text_gradient_generator(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any],
        outputs: Any,
        generator_output: Any,
        metric_result: MetricResult,
        # text_gradient_momentum: List[Dict[str, Any]]
    ) -> Any:
        """
        Generate text gradient based on inputs, outputs, generator outputs, and metric results.
        """
        # TODO: apply EXPEL-style text gradient generator in this function
        retry_count = 0
        # text_gradient_momentum_str = ""
        # for tg in text_gradient_momentum:
        #     text_gradient_momentum_str += f"Feedback: {tg['text_gradient']}, Score: {tg['score']}\n"
        prompt_messages = [
            json.dumps(message) for message in prompt.messages
        ]
        prompt_messages_str = "\n".join(prompt_messages)
        self.text_gradient_generator_prompt.model = prompt.model
        while retry_count < 3:
            try:
                text_gradient = await self.text_gradient_generator_prompt(
                    task_description=self.task_description,
                    metric_description=self.metric_description,
                    base_prompt=prompt_messages_str,
                    inputs=str(inputs),
                    outputs=str(outputs),
                    generator_output=str(generator_output),
                    metric_result=str(metric_result),
                    # feedback_history=text_gradient_momentum_str
                )
                text_gradient = text_gradient.strip()
                if not text_gradient.startswith("{"):
                    text_gradient = "{" + text_gradient
                
                text_gradient = json.loads(text_gradient)["feedback"]
                
                
                return text_gradient
            except Exception as e:
                print(f"Error: {e}")
                retry_count += 1
                if retry_count >= 3:
                    return ""

    async def _text_gradient_applier(
        self,
        prompt: Prompt,
        text_gradient: str,
        prompt_history: List[Dict[str, Any]]
    ) -> Prompt:
        """
        Apply text gradient to the prompt to obtain a new prompt.
        """
        retry_count = 0
        prompt_messages = [
            json.dumps(message) for message in prompt.messages
        ]
        prompt_messages_str = "\n".join(prompt_messages)
        
        prompt_history_str = ""
        self.text_gradient_applier_prompt.model = prompt.model
        for ph in prompt_history:
            prompt_history_str += f"Prompt: ```"
            history_prompt_messages = [
                json.dumps(message) for message in ph['prompt'].messages
            ]
            history_prompt_messages_str = "\n".join(history_prompt_messages)
            prompt_history_str += f"{history_prompt_messages_str}```\nScore: {ph['score']}\n\n"
        
        while retry_count < 3:
            try:
                new_prompt_str = await self.text_gradient_applier_prompt(
                    task_description=self.task_description,
                    base_prompt=prompt_messages_str,
                    feedback=text_gradient,
                    prompt_history=prompt_history_str
                )
                new_prompt_str = new_prompt_str.strip()
                if not new_prompt_str.startswith("```prompt"):
                    new_prompt_str = "```prompt\n" + new_prompt_str
                
                extracted_prompt = extract_prompt(new_prompt_str)
                new_prompt = Prompt.load(extracted_prompt)
                new_prompt.name = prompt.name
                new_prompt.response_format = prompt.response_format
                new_prompt.model = prompt.model
                
                # add "json" if response_format is not None & response_format.type is "json_object" & "json" not in messages
                messages = [
                    json.dumps(message) for message in new_prompt.messages
                ]
                messages_str = "\n".join(messages)
                if (
                    new_prompt.response_format is not None 
                    and new_prompt.response_format.type == "json_object" 
                    and "json" not in messages_str
                ):
                    # add "json" to the messages
                    new_prompt = await reformat_prompt(new_prompt, new_prompt.response_format)
                
                # Print text gradient in blue
                print("\033[94m" + text_gradient + "\033[0m")
                
                # Print new prompt's system message content in red
                for message in new_prompt.messages:
                    if message['role'] == 'system':
                        print("\033[91m" + message['content'] + "\033[0m")
                        break
                
                return new_prompt
            except Exception as e:
                print(f"Error: {e}")
                retry_count += 1
                
