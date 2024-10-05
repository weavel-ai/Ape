import asyncio
import json
import random
import numpy as np
from typing import Any, Dict, List, Tuple

import optuna

from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.prompt.prompt_base import Prompt, format_fewshot
from ape.common.prompt.utils import format_fewshot
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.core.core_prompts import ApeCorePrompts
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import OptunaTrainerReport
from ape.core.utils import extract_prompt, get_response_format_instructions, run_async


class OptunaTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        random_seed: int = 42,
        num_candidates: int = 10,
        max_steps: int = 30,
        minibatch_size: int = 25,
        **kwargs,
    ):
        """
        Initialize the OptunaTrainer.

        Args:
            generator (BaseGenerator): Generator for producing model outputs.
            metric (BaseMetric): Metric for evaluating model outputs.
            global_metric (BaseGlobalMetric): Global metric for overall evaluation.
            random_seed (int, optional): Seed for reproducibility. Defaults to 42.
            num_candidates (int, optional): Number of candidate prompts to generate. Defaults to 10.
            init_temperature (float, optional): Initial temperature for sampling. Defaults to 1.0.
            verbose (bool, optional): Verbosity flag. Defaults to False.
            track_stats (bool, optional): Whether to track statistics. Defaults to True.
            view_data_batch_size (int, optional): Batch size for viewing data. Defaults to 10.
            minibatch_size (int, optional): Size of minibatches for evaluation. Defaults to 25.
            minibatch_full_eval_steps (int, optional): Number of steps for full evaluation on minibatches. Defaults to 10.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            generator=generator,
            metric=metric,
            global_metric=global_metric,
            **kwargs,
        )
        self.random_seed = random_seed
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.minibatch_size = minibatch_size

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Load prompts for generating metric descriptions and merging
        self.merge_prompts: Prompt = ApeCorePrompts.get("gen-merged-prompt")
        self.generate_instructions_by_eval: Prompt = ApeCorePrompts.get("gen-instruction-with-eval")
        self.generate_instructions_by_prompting: Prompt = ApeCorePrompts.get("gen-instructions")

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, OptunaTrainerReport]:
        """
        Optimize the given prompt using Optuna.

        Args:
            prompt (Prompt): The base prompt to optimize.
            trainset (List[DatasetItem]): The training dataset.
            valset (List[DatasetItem]): The validation dataset.

        Returns:
            Tuple[Prompt, OptunaTrainerReport]: The best performing prompt and the optimization report.
        """
        report = OptunaTrainerReport(scores=[], trial_logs={}, best_score=0.0)

        # Generate metric description1
        if self.metric_description is None:
            self.metric_description = await self._generate_metric_description()
        if self.task_description is None:
            self.task_description = await self._generate_task_description(
                prompt=prompt, trainset=trainset
            )

        # Initialize evaluation on train set
        preds, eval_results, global_result = await self._evaluate(trainset, prompt)
        report.best_score = global_result
        report.trial_logs = []

        # Generate candidate prompts
        eval_based_candidates = await self.generate_prompt_candidates_by_eval_result(
            base_prompt=prompt,
            evaluation_result=(preds, eval_results, global_result),
        )

        prompt_engineering_based_candidates = (
            await self.generate_prompt_candidates_by_prompt_engineering(
                base_prompt=prompt,
                trainset=trainset,
                num_candidates=self.num_candidates,
            )
        )

        best_score = global_result.score
        best_prompt = prompt.deepcopy()

        # Initialize trial_logs and total_eval_calls
        trial_logs: Dict[int, Dict[str, Any]] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_prompt, best_score, trial_logs, eval_based_candidates, prompt_engineering_based_candidates

            trial_logs[trial.number] = {}

            # Suggest index for instruction candidate
            eval_based_candidate_idx = trial.suggest_categorical(
                "eval_based_candidate_idx", range(len(eval_based_candidates))
            )
            trial_logs[trial.number]["eval_based_candidate_idx"] = eval_based_candidate_idx

            # Select the candidate prompt
            selected_eval_based_candidate = eval_based_candidates[eval_based_candidate_idx]

            prompt_engineering_candidate_idx = trial.suggest_categorical(
                "prompt_engineering_candidate_idx", range(len(prompt_engineering_based_candidates))
            )
            trial_logs[trial.number][
                "prompt_engineering_candidate_idx"
            ] = prompt_engineering_candidate_idx

            selected_prompt_engineering_candidate = prompt_engineering_based_candidates[
                prompt_engineering_candidate_idx
            ]

            # Merge the selected candidate with the base prompt
            max_retries = 3
            retry_count = 0

            basic_prompt_messages = [json.dumps(message) for message in prompt.messages]
            basic_prompt_messages_str = "\n".join(basic_prompt_messages)
            selected_eval_based_candidate_messages = [
                json.dumps(message) for message in selected_eval_based_candidate.messages
            ]
            selected_eval_based_candidate_messages_str = "\n".join(
                selected_eval_based_candidate_messages
            )
            selected_prompt_engineering_candidate_messages = [
                json.dumps(message) for message in selected_prompt_engineering_candidate.messages
            ]
            selected_prompt_engineering_candidate_messages_str = "\n".join(
                selected_prompt_engineering_candidate_messages
            )

            while retry_count < max_retries:
                try:
                    merged_prompt_text = run_async(
                        self.merge_prompts(
                            basic_prompt=basic_prompt_messages_str,
                            instruction_improved_prompt=selected_eval_based_candidate_messages_str,
                            format_improved_prompt=selected_prompt_engineering_candidate_messages_str,
                        )
                    )
                    if not merged_prompt_text.startswith("```prompt"):
                        merged_prompt_text = "```prompt\n" + merged_prompt_text

                    merged_prompt_message = extract_prompt(merged_prompt_text)
                    merged_prompt_message = Prompt.load(merged_prompt_message)

                    merged_prompt = prompt.deepcopy()
                    merged_prompt.messages = merged_prompt_message.messages

                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e

            # Update the candidate prompt
            candidate_prompt: Prompt = prompt.deepcopy()
            candidate_prompt.messages = merged_prompt.messages

            # Evaluate the candidate prompt on the train set
            try:
                preds, eval_results, global_result = run_async(
                    self._evaluate(
                        random.sample(trainset, min(self.minibatch_size, len(trainset))),
                        candidate_prompt,
                    )
                )
                score = global_result.score
            except Exception as e:
                # If evaluation fails, assign a very low score
                trial_logs[trial.number]["evaluation_error"] = str(e)
                return float("-inf")

            # Update trial logs
            trial_logs[trial.number].update(
                {
                    "score": score,
                    "num_eval_calls": min(self.minibatch_size, len(trainset)),
                }
            )

            # Update best prompt if necessary
            if score > best_score:
                best_score = score
                best_prompt = candidate_prompt.deepcopy()
                trial_logs[trial.number]["best_score_update"] = True
            else:
                trial_logs[trial.number]["best_score_update"] = False

            # Update report
            report.trial_logs = trial_logs
            report.scores.append(
                {
                    "step": trial.number,
                    "score": score,
                }
            )

            # If score meets or exceeds the goal, stop the study
            if score >= 1.0:
                trial.study.stop()

            return score

        # Initialize Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed, multivariate=True),
        )

        # Optimize the study
        study.optimize(objective, n_trials=self.max_steps)
        report.best_score = best_score
        return best_prompt, report

    async def generate_prompt_candidates_by_eval_result(
        self,
        base_prompt: Prompt,
        evaluation_result: Tuple[List[Any], List[MetricResult], GlobalMetricResult],
    ) -> List[Prompt]:
        """
        Generate a set of new prompt candidates based on the base prompt and its evaluation.
        """
        preds, eval_results, global_result = evaluation_result
        evaluation_result_str = (
            f"final score : {global_result.score}\nmetadata : {global_result.trace}"
        )

        TIPS = {
            "none": "Make it better",
            "one_side_first": "Try to optimize a certain part of the evaluation score first. Be extreme at times.",
            "double_down": "Find what the prompt is already doing well and double down on it. Be extreme at times",
            "weakness_first": "Find what the prompt is already doing poorly based on the evaluation score and metric and try to improve that. Be extreme at times",
            "think": "Try to think about what the evaluation score and metric are actually measuring. Then, try to optimize for that. Be extreme at times",
        }

        async def generate_new_instruction(index: int) -> Prompt:
            selected_tip = list(TIPS.values())[index % len(TIPS)]
            base_prompt_messages = [json.dumps(message) for message in base_prompt.messages]
            base_prompt_messages_str = "\n".join(base_prompt_messages)

            new_instruction_text = await self.generate_instructions_by_eval(
                base_prompt=base_prompt_messages_str,
                evaluation_result=evaluation_result_str,
                evaluation_function=self.metric_description,
                tip=selected_tip,
                response_format=str(base_prompt.response_format),
                human_tip="",
            )

            if "```prompt" not in new_instruction_text:
                new_instruction_text = "```prompt\n" + new_instruction_text
            extracted_prompt = extract_prompt(new_instruction_text)

            new_prompt_message = Prompt.load(extracted_prompt)
            new_prompt = base_prompt.deepcopy()
            new_prompt.messages = new_prompt_message.messages

            return new_prompt

        tasks = [generate_new_instruction(i) for i in range(self.num_candidates)]
        proposed_instructions = await asyncio.gather(*tasks)

        return proposed_instructions

    async def generate_prompt_candidates_by_prompt_engineering(
        self,
        trainset: List[DatasetItem],
        base_prompt: Prompt,
        num_candidates: int,
    ) -> List[Prompt]:
        """
        Generate a set of new prompt candidates based on the base prompt using prompt engineering techniques.
        """

        TIPS = {
            "creative": "Don't be afraid to be creative when creating the new instruction!",
            "simple": "Keep the instruction clear and concise.",
            "description": "Make sure your instruction is very informative and descriptive. You can add some hand-crafted examples to help the LLM understand the task better.",
            "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
            "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
        }

        async def propose_one(index: int) -> Prompt:
            selected_tip = list(TIPS.values())[index % len(TIPS)]

            fewshot = random.sample(trainset, min(len(trainset), 3))
            task_fewshot = format_fewshot(
                fewshot=fewshot, response_format=base_prompt.response_format
            )

            response_format_instructions = get_response_format_instructions(
                base_prompt.response_format
            )

            output = await self.generate_instructions_by_prompting(
                task_description="",
                dataset_desc=self.dataset_summary,
                task_fewshot=task_fewshot,
                prompt_desc=self.task_description,
                basic_prompt=base_prompt.dump(),
                tip=selected_tip,
                inputs_desc=base_prompt.inputs_desc if base_prompt.inputs_desc else "-",
                outputs_desc=base_prompt.outputs_desc if base_prompt.outputs_desc else "-",
                response_format_instructions=response_format_instructions,
            )

            try:
                extracted_prompt = extract_prompt(output)
                new_prompt_message = Prompt.load(extracted_prompt)
                if not new_prompt_message.messages:
                    raise ValueError("Generated prompt has no messages")
                new_prompt = base_prompt.deepcopy()
                new_prompt.messages = new_prompt_message.messages

                return new_prompt

            except Exception as e:
                return base_prompt

        tasks = [propose_one(i) for i in range(num_candidates)]
        proposed_instructions = await asyncio.gather(*tasks)

        return proposed_instructions
