from collections import defaultdict
import os
import pickle
import random
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm

from ape.common.utils import logger
from ape.common.evaluate.evaluate import Evaluate
from ape.common.metric import (
    BaseMetric,
    CosineSimilarityMetric,
    JsonMatchMetric,
    SemanticF1Metric,
)
from ape.common.global_metric import BaseGlobalMetric
from ape.common.prompt import Prompt
from ape.common.types import DatasetItem, ResponseFormat

from ape.core.core_prompts import ApeCorePrompts
from .mipro_base import MIPROBase
from .mipro_proposer import MIPROProposer
from ..utils import (
    run_async,
    eval_candidate_prompt,
    find_best_fewshot,
    reformat_prompt,
    save_candidate_prompt,
)
from ...proposer.evaluation_driven_proposer import EvaluationDrivenProposer
from ...proposer.utils import extract_prompt

BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 0


# Instruction Generation. Later Add Optuna and GroundedProposer to do (N+1) x (M+1) search.
class MIPROInstruct(MIPROBase):
    """
    MIPROInstruct class for optimizing prompts.

    This class is a modified vesion of MIPRO with a focus on generating better instructions. It supports minibatch evaluation,
    few-shot examples, and various optimization strategies.
    """

    gen_metric_description: Prompt = ApeCorePrompts.get("gen-metric-description")
    gen_metric_description_with_global_metric: Prompt = ApeCorePrompts.get(
        "gen-metric-description-with-global-metric"
    )
    merge_prompts: Prompt = ApeCorePrompts.get("gen-merged-prompt")

    def __init__(
        self,
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
        teacher_settings: Dict[str, Any] = {},
        num_candidates: int = 10,
        init_temperature: float = 1.0,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: Optional[str] = None,
        view_data_batch_size: int = 10,
        minibatch_size: int = 25,
        minibatch_full_eval_steps: int = 10,
    ):
        """
        Initialize the MIPROInstruct optimizer.

        Args:
            metric (BaseMetric): The metric object used for evaluation.
            global_metric (Optional[BaseGlobalMetric]): The global metric object for overall evaluation.
            teacher_settings (Dict[str, Any]): Settings for the teacher model.
            num_candidates (int): Number of candidate prompts to generate.
            init_temperature (float): Initial temperature for sampling.
            verbose (bool): Whether to print verbose output.
            track_stats (bool): Whether to track statistics during optimization.
            log_dir (Optional[str]): Directory for logging.
            view_data_batch_size (int): Batch size for viewing data.
            minibatch_size (int): Size of minibatches for evaluation.
            update_prompt_after_full_eval (bool): Whether to update the best prompt after full evaluation.
            minibatch_full_eval_steps (int): Number of steps for full evaluation on minibatches.
        """

        super().__init__(
            teacher_settings=teacher_settings,
            num_candidates=num_candidates,
            metric=metric,
            global_metric=global_metric,
            init_temperature=init_temperature,
            verbose=verbose,
            track_stats=track_stats,
            log_dir=log_dir,
            view_data_batch_size=view_data_batch_size,
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=minibatch_full_eval_steps,
        )

    def _get_batch_size(self, minibatch: bool, trainset: List[DatasetItem]) -> int:
        return self.minibatch_size if minibatch else len(trainset)

    def _display_warning_and_confirm(
        self,
        trainset: List[DatasetItem],
        testset: List[DatasetItem],
        max_steps: int,
        minibatch: bool,
        requires_permission_to_run: bool,
    ) -> bool:
        """
        Display a warning about projected LM calls and costs, and optionally ask for user confirmation.

        Args:
            trainset (List[DatasetItem]): The training dataset.
            max_steps (int): Maximum number of optimization steps.
            minibatch (bool): Whether minibatch optimization is used.
            requires_permission_to_run (bool): Whether user confirmation is required.

        Returns:
            bool: True if the user confirms or confirmation is not required, False otherwise.
        """
        console = Console()

        estimated_prompt_model_calls = 10 + self.num_candidates + 1

        # TODO: Modify to match the actual number of LM calls in the program.
        if not minibatch:
            estimated_task_model_calls = len(testset) * max_steps
            task_model_line = f"- Task Model: {len(testset)} examples in test set * {max_steps} batches * # of LM calls in your program = ({estimated_task_model_calls} * # of LM calls in your program) task model calls"
        else:
            # if self.update_prompt_after_full_eval:
            #     estimated_task_model_calls = self.minibatch_size * max_steps + (
            #         len(trainset) * (max_steps // self.minibatch_full_eval_steps)
            #     )
            #     task_model_line = f"- Task Model: {self.minibatch_size} examples in minibatch * {max_steps} batches + {len(trainset)} examples in train set * {max_steps // self.minibatch_full_eval_steps} full evals = {estimated_task_model_calls} task model calls"
            # else:
            estimated_task_model_calls = self.minibatch_size * max_steps
            task_model_line = f"- Task Model: {self.minibatch_size} examples in minibatch * {max_steps} batches = {estimated_task_model_calls} task model calls"

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

    async def generate_metric_description(self) -> str:
        """
        Generate a description of the metric.
        """
        try:
            if isinstance(self.metric, CosineSimilarityMetric):
                metric_str = "Measures how similar the predicted text is to the correct answer by comparing their vector representations. A higher score means the prediction is more similar to the gold."
            elif isinstance(self.metric, JsonMatchMetric):
                metric_str = "Compares JSON format predictions with the correct JSON answer. It checks if each key-value pair in the prediction matches the ground truth exactly. The score reflects how many pairs match correctly."
            elif isinstance(self.metric, SemanticF1Metric):
                metric_str = """\
        Evaluates how well the prediction captures the meaning of the correct answer:
        1. Extracts key statements from both the prediction and ground truth.
        2. Checks how many statements from the prediction are found in the ground truth (Precision).
        3. Checks how many statements from the ground truth are found in the prediction (Recall).
        4. Calculates the F1 score, which balances Precision and Recall. A higher score indicates better semantic matching."""
            else:
                compute_function = getattr(self.metric, "compute", None)
                compute_function_source_code = inspect.getsource(compute_function)

                if self.global_metric:
                    global_metric_compute_function = getattr(self.global_metric, "compute", None)
                    global_metric_compute_function_source_code = inspect.getsource(
                        global_metric_compute_function
                    )

                    # get Prompt gen-metric-description-with-global-metric.prompt
                    metric_str = await self.gen_metric_description_with_global_metric(
                        **{
                            "metric_sourcecode": compute_function_source_code,
                            "global_metric_sourcecode": global_metric_compute_function_source_code,
                        }
                    )

                else:
                    metric_str = await self.gen_metric_description(
                        **{
                            "metric_sourcecode": compute_function_source_code,
                        }
                    )
                return metric_str
        except Exception as e:
            logger.error(f"Error generating metric description: {e}")
            return ""

    async def generate_instructions(
        self,
        student: Prompt,
        *,
        trainset: List[DatasetItem],
        testset: Optional[List[DatasetItem]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = {},
        seed: int = 9,
        response_format: Optional[ResponseFormat] = None,
        log_dir: str,
    ) -> Optional[Prompt]:
        """
        Optimize the given prompt using MIPRO (Multi-prompt Instruction PRoposal Optimizer).

        This method generates and evaluates multiple prompt candidates, using optuna for hyperparameter optimization.
        It supports minibatch evaluation, fewshot examples, and various optimization strategies.

        Returns:
            Optional[Prompt]: The best performing prompt, or None if optimization is aborted.
        """

        random.seed(seed)
        np.random.seed(seed)
        testset = testset or trainset

        if response_format is not None:
            student = await reformat_prompt(prompt=student, response_format=response_format)

        metric_str = await self.generate_metric_description()

        evaluate: Evaluate = Evaluate(
            testset=testset,
            metric=self.metric,
            global_metric=self.global_metric,
            **eval_kwargs,
        )

        # This is the proposer.
        proposer = EvaluationDrivenProposer(
            trainset=trainset,
            view_data_batch_size=self.view_data_batch_size,
        )

        logger.info(f"Generating {self.num_candidates} explore instruction candidates")

        # Generate N candidates using EvaluationDrivenProposer.
        explore_prompt_candidates = await proposer.propose_prompts(
            base_prompt=student,
            N=self.num_candidates,
            T=self.init_temperature,
            evaluate=evaluate,
            metric=metric_str,
        )

        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, "instructions_to_save.pkl"), "wb") as file:
                pickle.dump(explore_prompt_candidates, file)

        instruction_candidates: List[List[Dict[str, str]]] = [
            prompt.messages for prompt in explore_prompt_candidates
        ]

        return instruction_candidates

    async def optimize(
        self,
        student: Prompt,
        *,
        task_description: str = "",
        trainset: List[DatasetItem],
        testset: Optional[List[DatasetItem]] = None,
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
        metric_threshold: Optional[float] = None,
    ) -> Optional[Prompt]:
        """
        Optimize the given prompt using MIPRO (Multi-prompt Instruction PRoposal Optimizer).

        This method generates and evaluates multiple prompt candidates, using optuna for hyperparameter optimization.
        It supports minibatch evaluation, fewshot examples, and various optimization strategies.

        Returns:
            Optional[Prompt]: The best performing prompt, or None if optimization is aborted.
        """
        eval_kwargs = eval_kwargs or {}
        if not self._display_warning_and_confirm(
            trainset, testset, max_steps, minibatch, requires_permission_to_run
        ):
            logger.info("Optimization aborted by the user.")
            return None

        random.seed(seed)
        np.random.seed(seed)
        if testset is None:
            testset = trainset

        if response_format is not None:
            student = await reformat_prompt(prompt=student, response_format=response_format)

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

        trainset_evaluate: Evaluate = Evaluate(
            testset=trainset, metric=self.metric, global_metric=self.global_metric, **eval_kwargs
        )

        testset_evaluate: Evaluate = Evaluate(
            testset=testset, metric=self.metric, global_metric=self.global_metric, **eval_kwargs
        )

        logger.info("Start Find Best Fewshot")

        # find best few-shot
        best_fewshot, best_score = await find_best_fewshot(
            student=student,
            num_candidate_sets=self.num_candidates,
            trainset=trainset,
            max_labeled_demos=max_labeled_demos_for_candidate_gen,
            max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
            metric=self.metric,
            teacher_settings=self.teacher_settings,
            seed=seed,
            evaluate=testset_evaluate,
            batch_size=self.minibatch_size,
            metric_threshold=metric_threshold,
        )

        logger.info("Start Propose Instruction Candidates from Evaluation Result")

        score_based_proposer = EvaluationDrivenProposer(
            trainset=trainset,
            view_data_batch_size=self.view_data_batch_size,
            use_tip=True,
        )

        logger.info(f"Generating {self.num_candidates} explore instruction candidates")

        metric_str = await self.generate_metric_description()

        # Generate N candidates using EvaluationDrivenProposer.
        score_based_prompt_candidates = await score_based_proposer.propose_prompts(
            base_prompt=student,
            N=self.num_candidates,
            T=self.init_temperature,
            evaluate=trainset_evaluate,
            metric=metric_str,
            task_description=task_description,
        )

        score_based_instruction_candidates: List[List[Dict[str, str]]] = [
            prompt.messages for prompt in score_based_prompt_candidates
        ]

        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, "score_based_instructions_to_save.pkl"), "wb") as file:
                pickle.dump(score_based_prompt_candidates, file)

        logger.info("Start Propose Instruction Candidates from Prompt Engineering Techniques")

        format_based_proposer: MIPROProposer = MIPROProposer(**self.model_dump())
        format_based_prompt_candidates: List[Prompt] = (
            await format_based_proposer.generate_candidates(
                prompt=student,
                trainset=trainset,
                task_description=task_description,
                response_format=response_format,
            )
        )

        if log_dir:
            with open(os.path.join(log_dir, "format_based_instructions_to_save.pkl"), "wb") as file:
                pickle.dump(format_based_prompt_candidates, file)

        format_based_instruction_candidates: List[List[Dict[str, str]]] = [
            prompt.messages for prompt in format_based_prompt_candidates
        ]

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

            score_based_instruction_idx: int = trial.suggest_categorical(
                "score_based_instruction", range(len(score_based_instruction_candidates))
            )
            chosen_params: List[int] = [score_based_instruction_idx]

            format_based_instruction_idx: int = trial.suggest_categorical(
                "format_based_instruction", range(len(format_based_instruction_candidates))
            )
            chosen_params.append(format_based_instruction_idx)

            trial_logs[trial.number].update(
                {
                    "score_based_instruction": score_based_instruction_idx,
                    "format_based_instruction": format_based_instruction_idx,
                }
            )

            score_based_candidate_prompt = score_based_instruction_candidates[
                score_based_instruction_idx
            ]
            format_based_candidate_prompt = format_based_instruction_candidates[
                format_based_instruction_idx
            ]

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    merged_candidate_prompt_text = run_async(
                        self.merge_prompts(
                            **{
                                "basic_prompt": student.messages,
                                "instruction_improved_prompt": score_based_candidate_prompt,
                                "format_improved_prompt": format_based_candidate_prompt,
                            }
                        )
                    )

                    merged_candidate_prompt = extract_prompt(merged_candidate_prompt_text)
                    merged_candidate_prompt = Prompt.load(merged_candidate_prompt)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e
                    logger.warning(f"Attempt {retry_count} failed. Retrying...")

            candidate_prompt.messages = merged_candidate_prompt.messages
            candidate_prompt.fewshot = best_fewshot

            # print(candidate_prompt)

            trial_logs[trial.number]["prompt_path"] = save_candidate_prompt(
                candidate_prompt, log_dir, trial.number
            )

            batch_size: int = self._get_batch_size(minibatch, testset)
            score: float = run_async(
                eval_candidate_prompt(batch_size, testset, candidate_prompt, testset_evaluate)
            )

            categorical_key: str = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append((score, candidate_prompt))

            trial_logs[trial.number].update(
                {
                    "num_eval_calls": batch_size,
                    "full_eval": batch_size >= len(testset),
                    "score": score,
                    "pruned": False,
                    "total_eval_calls_so_far": total_eval_calls + batch_size,
                }
            )
            total_eval_calls += batch_size

            if score > best_score:
                print(
                    f"Best Score Updated to Score-based : {score_based_instruction_idx}, Format-based : {format_based_instruction_idx}"
                )
                best_score = score
                best_prompt = candidate_prompt.deepcopy()

            if score >= goal_score:
                trial.study.stop()

            return score

        logger.info("Start Optuna Study to find best combination")

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
            with open(os.path.join(log_dir, "best_prompt.prompt"), "w", encoding="utf-8") as f:
                f.write(best_prompt.dump())

            with open(os.path.join(log_dir, "optuna_study.pkl"), "wb") as file:
                pickle.dump(study, file)

        return best_prompt
