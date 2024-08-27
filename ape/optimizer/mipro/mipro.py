import asyncio
import json
import os
from collections import defaultdict
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from ape.evaluate.evaluate import Evaluate
from ape.optimizer.mipro.mipro_base import MIPROBase
from ape.optimizer.mipro.mipro_proposer import MIPROProposer
from ape.optimizer.utils import (
    create_n_fewshot_demo_sets,
    eval_candidate_prompt,
    get_prompt_with_highest_avg_score,
    reformat_prompt,
    save_candidate_prompt,
)
from ape.utils import run_async, logger
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.response_format import ResponseFormat

BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 0
MAX_REFLECTION_SAMPLES: int = 40

class MIPRO(MIPROBase):
    """
    MIPRO (Multi-prompt Instruction PRoposal Optimizer) class for optimizing prompts.

    This class implements the MIPRO algorithm, which generates and evaluates multiple prompt
    candidates using optuna for hyperparameter optimization. It supports minibatch evaluation,
    few-shot examples, and various optimization strategies.

    For more details, see the MIPRO paper: https://arxiv.org/abs/2406.11695
    """

    def _get_batch_size(self, minibatch: bool, trainset: Dataset) -> int:
        return self.minibatch_size if minibatch else len(trainset)

    def _display_warning_and_confirm(
        self,
        trainset: Dataset,
        max_steps: int,
        minibatch: bool,
        requires_permission_to_run: bool,
    ) -> bool:
        """
        Display a warning about projected LM calls and costs, and optionally ask for user confirmation.

        Args:
            trainset (Dataset): The training dataset.
            max_steps (int): Maximum number of optimization steps.
            minibatch (bool): Whether minibatch optimization is used.
            requires_permission_to_run (bool): Whether user confirmation is required.

        Returns:
            bool: True if the user confirms or confirmation is not required, False otherwise.
        """
        console = Console()

        estimated_prompt_model_calls = 10 + self.num_candidates + 1

        if not minibatch:
            estimated_task_model_calls = len(trainset) * max_steps
            task_model_line = f"- Task Model: {len(trainset)} examples in train set * {max_steps} batches * # of LM calls in your program = ({estimated_task_model_calls} * # of LM calls in your program) task model calls"
        else:
            estimated_task_model_calls = self.minibatch_size * max_steps + (
                len(trainset) * (max_steps // self.minibatch_full_eval_steps)
            )
            task_model_line = f"- Task Model: {self.minibatch_size} examples in minibatch * {max_steps} batches + {len(trainset)} examples in train set * {max_steps // self.minibatch_full_eval_steps} full evals = {estimated_task_model_calls} task model calls"

        warning_text = Text.from_markup(
            f"[bold yellow]WARNING: Projected Language Model (LM) Calls[/bold yellow]\n\n"
            f"Please be advised that based on the parameters you have set, the maximum number of LM calls is projected as follows:\n\n"
            f"[cyan]- Prompt Model: 10 data summarizer calls + {self.num_candidates} lm calls in program = {estimated_prompt_model_calls} prompt model calls[/cyan]\n"
            f"[cyan]{task_model_line}[/cyan]\n\n"
            f"[bold yellow]Estimated Cost Calculation:[/bold yellow]\n"
            f"Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token) \n"
            f"            + (Number of calls to prompt model * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).\n\n"
            f"For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task and prompt models you intend to use. "
            f"If the projected costs exceed your budget or expectations, you may consider:\n"
            f"- Reducing the number of trials (`max_steps`), the size of the trainset, or the number of LM calls in your program.\n"
            f"- Using a cheaper task model to optimize the prompt."
        )

        console.print(Panel(warning_text, title="Cost Warning", expand=False))

        if requires_permission_to_run:
            return Confirm.ask("Do you wish to continue?")
        return True

    async def process_sample(self, best_prompt, sample):
        try:
            prediction = await best_prompt(
                lm_config=dict(temperature=0.0), **sample.inputs
            )
            if self.metric:
                metric_val = await self.metric(sample.inputs, prediction, None)
                if isinstance(metric_val, (bool, float, int)):
                    if metric_val == False or (
                        isinstance(metric_val, (float, int)) and metric_val < 0.5
                    ):
                        return sample, prediction
                else:
                    logger.warning(f"Unexpected metric value type: {type(metric_val)}")
        except Exception as e:
            logger.error(f"Error processing sample or calculating metric: {e}")
        return None

    async def generate_reflexion(self, sample, answer, reasoning):
        reflexion_prompt = Prompt.from_filename("reflexion")
        reflexion = await reflexion_prompt(
            lm_config=dict(temperature=0.0),
            question=sample.inputs["question"],
            llm_output=answer,
            llm_reasoning=reasoning,
            ground_truth_answer=sample.outputs["answer"],
            ground_truth_reasoning=sample.outputs["gold_reasoning"],
        )
        if not isinstance(reflexion, dict):
            try:
                reflexion = json.loads(reflexion)
            except json.JSONDecodeError:
                reflexion = {
                    "question": "Error parsing reflexion",
                    "insight": "Unable to generate insight",
                }
        return f"- Question: {reflexion.get('question', '')}\n  Insight: {reflexion.get('insight', '')}\n\n"

    async def optimize(
        self,
        student: Prompt,
        *,
        task_description: str = "",
        trainset: Dataset,
        testset: Optional[Dataset] = None,
        max_steps: int = 30,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 2,
        goal_score: float = 100,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 9,
        minibatch: bool = True,
        requires_permission_to_run: bool = True,
        response_format: Optional[ResponseFormat] = None,
        log_dir: str,
        reflect: bool = False,
        inputs_desc: Optional[Dict[str, str]] = None,
        outputs_desc: Optional[Dict[str, str]] = None,
    ) -> Optional[Prompt]:
        """
        Optimize the given prompt using MIPRO (Multi-prompt Instruction PRoposal Optimizer).

        This method generates and evaluates multiple prompt candidates, using optuna for hyperparameter optimization.
        It supports minibatch evaluation, fewshot examples, and various optimization strategies.
        Args:
            student (Prompt): The initial prompt to be optimized.
            task_description (str): Description of the task. Defaults to an empty string. Will be used to generate instructions.
            inputs_desc (Optional[Dict[str, str]]): Description of input fields. Will be used to generate instructions.
            outputs_desc (Optional[Dict[str, str]]): Description of output fields. Will be used to generate instructions.
            trainset (Dataset): The dataset used for training.
            testset (Optional[Dataset]): The dataset used for testing. If None, trainset is used.
            max_steps (int): Maximum number of optimization steps. Defaults to 30.
            max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations. Defaults to 5.
            max_labeled_demos (int): Maximum number of labeled demonstrations. Defaults to 2.
            goal_score (float): The target score to achieve. Defaults to 100.
            eval_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for evaluation.
            seed (int): Random seed for reproducibility. Defaults to 9.
            minibatch (bool): Whether to use minibatch evaluation. Defaults to True.
            requires_permission_to_run (bool): Whether user confirmation is required before running. Defaults to True.
            response_format (Optional[ResponseFormat]): The desired format for the response.
            log_dir (str): Directory to save logs and results.
            reflect (bool): Whether to perform reflection on the best performing prompt. Defaults to False.
        Returns:
            Optional[Prompt]: The best performing prompt, or None if optimization is aborted.
        """
        eval_kwargs = eval_kwargs or {}
        if not self._display_warning_and_confirm(
            trainset, max_steps, minibatch, requires_permission_to_run
        ):
            logger.info("Optimization aborted by the user.")
            return None

        random.seed(seed)
        np.random.seed(seed)
        testset = testset or trainset

        if response_format is not None:
            student = await reformat_prompt(
                prompt=student, response_format=response_format
            )

        evaluate: Evaluate = Evaluate(
            testset=testset, metric=self.metric, **eval_kwargs
        )

        max_bootstrapped_demos_for_candidate_gen: int = (
            BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0
            else max_bootstrapped_demos
        )
        max_labeled_demos_for_candidate_gen: int = (
            LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0
            else max_labeled_demos
        )

        try:
            fewshot_candidates: Optional[List[Dataset]] = (
                await create_n_fewshot_demo_sets(
                    student=student,
                    num_candidate_sets=self.num_candidates,
                    trainset=trainset,
                    max_labeled_demos=max_labeled_demos_for_candidate_gen,
                    max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
                    metric=self.metric,
                    teacher_settings=self.teacher_settings,
                    seed=seed,
                )
            )
            if log_dir:
                with open(
                    os.path.join(log_dir, "fewshot_examples_to_save.pkl"), "wb"
                ) as file:
                    pickle.dump(fewshot_candidates, file)
        except Exception as e:
            logger.error(f"Error generating fewshot examples: {e}")
            logger.error("Running without fewshot examples.")
            fewshot_candidates = None

        proposer: MIPROProposer = MIPROProposer(**self.model_dump())
        prompt_candidates: List[Prompt] = await proposer.generate_candidates(
            prompt=student,
            trainset=trainset,
            task_description=task_description,
            fewshot_candidates=fewshot_candidates,
            response_format=response_format,
            inputs_desc=inputs_desc,
            outputs_desc=outputs_desc,
        )

        if log_dir:
            with open(os.path.join(log_dir, "instructions_to_save.pkl"), "wb") as file:
                pickle.dump(prompt_candidates, file)

        instruction_candidates: List[List[Dict[str, str]]] = [
            prompt.messages for prompt in prompt_candidates
        ]

        if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
            fewshot_candidates = None

        best_score: float = float("-inf")
        best_prompt: Optional[Prompt] = None
        trial_logs: Dict[int, Dict[str, Any]] = {}
        total_eval_calls: int = 0
        param_score_dict: Dict[str, List[Tuple[float, Prompt]]] = defaultdict(list)
        fully_evaled_param_combos: Dict[str, Dict[str, Union[Prompt, float]]] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_prompt, best_score, trial_logs, total_eval_calls, param_score_dict, fully_evaled_param_combos

            logger.info(f"Starting trial num: {trial.number}")
            trial_logs[trial.number] = {}

            candidate_prompt: Prompt = student.deepcopy()

            instruction_idx: int = trial.suggest_categorical(
                "instruction", range(len(instruction_candidates))
            )
            chosen_params: List[int] = [instruction_idx]

            if fewshot_candidates:
                fewshot_idx: int = trial.suggest_categorical(
                    "fewshot", range(len(fewshot_candidates))
                )
                chosen_params.append(fewshot_idx)
                candidate_prompt.fewshot = fewshot_candidates[fewshot_idx]

            trial_logs[trial.number].update(
                {
                    "instruction": instruction_idx,
                    "fewshot": fewshot_idx if fewshot_candidates else None,
                }
            )

            candidate_prompt.messages = instruction_candidates[instruction_idx]

            trial_logs[trial.number]["prompt_path"] = save_candidate_prompt(
                candidate_prompt, log_dir, trial.number
            )

            batch_size: int = self._get_batch_size(minibatch, trainset)
            score: float = run_async(
                eval_candidate_prompt(batch_size, trainset, candidate_prompt, evaluate)
            )

            categorical_key: str = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append((score, candidate_prompt))

            trial_logs[trial.number].update(
                {
                    "num_eval_calls": batch_size,
                    "full_eval": batch_size >= len(trainset),
                    "score": score,
                    "pruned": False,
                    "total_eval_calls_so_far": total_eval_calls + batch_size,
                }
            )
            total_eval_calls += batch_size

            if (
                score > best_score
                and trial_logs[trial.number]["full_eval"]
                and not minibatch
            ):
                best_score = score
                best_prompt = candidate_prompt.deepcopy()

            if minibatch and trial.number % self.minibatch_full_eval_steps == 0:
                trial_logs[trial.number]["mb_score"] = score
                trial_logs[trial.number]["mb_prompt_path"] = trial_logs[trial.number][
                    "prompt_path"
                ]

                highest_mean_prompt, combo_key = get_prompt_with_highest_avg_score(
                    param_score_dict, fully_evaled_param_combos
                )
                full_train_score: float = run_async(
                    eval_candidate_prompt(
                        len(trainset), trainset, highest_mean_prompt, evaluate
                    )
                )

                fully_evaled_param_combos[combo_key] = {
                    "program": highest_mean_prompt,
                    "score": full_train_score,
                }
                total_eval_calls += len(trainset)
                trial_logs[trial.number].update(
                    {
                        "total_eval_calls_so_far": total_eval_calls,
                        "full_eval": True,
                        "prompt_path": save_candidate_prompt(
                            prompt=highest_mean_prompt,
                            log_dir=log_dir,
                            trial_num=trial.number,
                            note="full_eval",
                        ),
                        "score": full_train_score,
                    }
                )

                if full_train_score > best_score:
                    best_score = full_train_score
                    best_prompt = highest_mean_prompt.deepcopy()

            if score >= goal_score:
                trial.study.stop()

            return score

        study: optuna.Study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        )
        study.optimize(objective, n_trials=max_steps)

        if best_prompt is not None and self.track_stats:
            best_prompt.metadata["trial_logs"] = trial_logs
            best_prompt.metadata["score"] = best_score
            best_prompt.metadata["total_eval_calls"] = total_eval_calls

        if log_dir:
            with open(
                os.path.join(log_dir, "best_prompt.prompt"), "w", encoding="utf-8"
            ) as f:
                f.write(best_prompt.dump())

            with open(os.path.join(log_dir, "optuna_study.pkl"), "wb") as file:
                pickle.dump(study, file)
        
        with open(os.path.join(log_dir, "best_prompt.prompt"), "r", encoding="utf-8") as f:
            best_prompt = Prompt.load(f.read())

        if reflect:
            failed_samples = [
                result
                for result in await asyncio.gather(
                    *[self.process_sample(best_prompt, sample) for sample in trainset]
                )
                if result is not None
            ]
            
            if len(failed_samples) > MAX_REFLECTION_SAMPLES:
                failed_samples = random.sample(failed_samples, MAX_REFLECTION_SAMPLES)

            if failed_samples:
                extract_keys = Prompt.from_filename("extract-keys")
                keys = await extract_keys(
                    response=failed_samples[0]
                )
                
                answer_key = keys.get("answer", None)
                reasoning_key = keys.get("reasoning", None)
                
                if answer_key is None or reasoning_key is None:
                    logger.warning("No answer or reasoning key found in the response. Skipping reflexion.")
                    return best_prompt
                
                all_reflections = await asyncio.gather(
                    *[
                        self.generate_reflexion(sample, prediction[answer_key], prediction[reasoning_key])
                        for sample, prediction in failed_samples
                    ]
                )
                refine_reflection = Prompt.from_filename("refine-reflexion")
                reflections = await refine_reflection(reflections=all_reflections)
                reflexion_message = (
                    "\n\nYou have failed to solve the problem in the past. "
                    "Here are the questions and insights on how to solve them.\n"
                    "ONLY IF a question asking for a similar problem is asked, "
                    "learn from the insights and solve the problem correctly.\n"
                    "<START OF INSIGHT>\n"
                    f"{''.join(reflections)}"
                    "\n<END OF INSIGHT>\n"
                    "Remember to apply these insights only when encountering similar problem types. "
                    "Do not use them if the question is not related to the examples provided.\n\n"
                )
                best_prompt.messages[0]["content"] += reflexion_message

                logger.info("Prompt after reflexion:")
                logger.info(best_prompt)
        
        if log_dir:
            with open(os.path.join(log_dir, "best_prompt_after_reflexion.prompt"), "w", encoding="utf-8") as f:
                f.write(best_prompt.dump())

        return best_prompt
