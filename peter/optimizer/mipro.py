import asyncio
import os
import sys
import textwrap
import threading
import optuna
from collections import defaultdict
import logging
import pickle
import random
from typing import Any, Awaitable, Callable, Dict, Literal, Optional, Sequence
import numpy as np
from pydantic import BaseModel
from rich.logging import RichHandler
import tqdm
from peter.evaluate.evaluate import Evaluate
from peter.optimizer.utils import (
    create_n_fewshot_demo_sets,
    eval_candidate_prompt,
    get_prompt_with_highest_avg_score,
    reformat_prompt_xml_style,
    save_candidate_prompt,
)
from peter.proposer.grounded_proposer import GroundedProposer
from peter.utils import run_async
import weavel.types
from rich import print
import logging

from peter.optimizer.optimizer_base import Optimizer
from peter.prompt.prompt_base import Prompt
from peter.types import Dataset


BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0

MB_FULL_EVAL_STEPS = 10
MINIBATCH_SIZE = 25  # 50


class MIPRO(Optimizer):
    """MIPRO: Multi-prompt Instruction PRoposal Optimizer

    Args:
        Optimizer (_type_): _description_
    """

    def __init__(
        self,
        prompt_model: Optional[str] = None,
        task_model: Optional[str] = None,
        teacher_settings={},
        num_candidates=10,
        metric: Callable[..., Awaitable[Any]] = None,
        init_temperature=1.0,
        verbose=False,
        track_stats=True,
        log_dir=None,
        view_data_batch_size=10,
        minibatch_size=MINIBATCH_SIZE,
        minibatch_full_eval_steps=MB_FULL_EVAL_STEPS,
    ):
        self.n = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.task_model = task_model
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.view_data_batch_size = view_data_batch_size
        self.teacher_settings = teacher_settings
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps

    def _get_batch_size(
        self,
        minibatch,
        trainset,
    ):
        if minibatch:
            return self.minibatch_size
        else:
            return len(trainset)

    async def optimize(
        self,
        student: Prompt,
        *,
        trainset: Dataset,
        valset=None,
        num_batches=30,
        max_bootstrapped_demos=5,
        max_labeled_demos=2,
        eval_kwargs={},
        seed=9,
        minibatch=True,
        prompt_aware_proposer=True,
        requires_permission_to_run=True,
        log_dir,
    ):
        # Define ANSI escape codes for colors
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        ENDC = "\033[0m"  # Resets the color to default

        random.seed(seed)
        valset = valset or trainset
        estimated_prompt_model_calls = 10 + self.n + 1  # num data summary calls + N + 1

        prompt_model_line = f"""[yellow]- Prompt Model: [blue][bold]10[/bold][/blue] data summarizer calls + [blue][bold]{self.n}[/bold][/blue] lm calls in program = [blue][bold]{estimated_prompt_model_calls}[/bold][/blue] prompt model calls[/yellow]"""

        estimated_task_model_calls_wo_module_calls = 0
        task_model_line = ""
        if not minibatch:
            estimated_task_model_calls_wo_module_calls = (
                len(trainset) * num_batches
            )  # M * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{num_batches}{ENDC}{YELLOW} batches * {BLUE}{BOLD}# of LM calls in your program{ENDC}{YELLOW} = ({BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls} * # of LM calls in your program{ENDC}{YELLOW}) task model calls{ENDC}"""
        else:
            estimated_task_model_calls_wo_module_calls = (
                self.minibatch_size * num_batches
                + (len(trainset) * (num_batches // self.minibatch_full_eval_steps))
            )  # B * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{self.minibatch_size}{ENDC}{YELLOW} examples in minibatch * {BLUE}{BOLD}{num_batches}{ENDC}{YELLOW} batches + {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{num_batches // self.minibatch_full_eval_steps}{ENDC}{YELLOW} full evals = {BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls}{ENDC}{YELLOW} task model calls{ENDC}"""

        user_message = textwrap.dedent(
            f"""\
            {YELLOW}{BOLD}WARNING: Projected Language Model (LM) Calls{ENDC}

            Please be advised that based on the parameters you have set, the maximum number of LM calls is projected as follows:

            
            {prompt_model_line}
            {task_model_line}

            {YELLOW}{BOLD}Estimated Cost Calculation:{ENDC}

            {YELLOW}Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token) 
                        + (Number of calls to prompt model * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).{ENDC}

            For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task
            and prompt models you intend to use. If the projected costs exceed your budget or expectations, you may consider:

            {YELLOW}- Reducing the number of trials (`num_batches`), the size of the trainset, or the number of LM calls in your program.{ENDC}
            {YELLOW}- Using a cheaper task model to optimize the prompt.{ENDC}"""
        )

        # user_confirmation_message = textwrap.dedent(
        #     f"""\
        #     To proceed with the execution of this program, please confirm by typing {BLUE}'y'{ENDC} for yes or {BLUE}'n'{ENDC} for no.

        #     If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False` when calling compile.{ENDC}

        #     {YELLOW}Awaiting your input...{ENDC}
        # """
        # )

        sys.stdout.flush()  # Flush the output buffer to force the message to print

        run = True
        # if requires_permission_to_run:
        #     print(user_confirmation_message)
        #     user_input = input("Do you wish to continue? (y/n): ").strip().lower()
        #     if user_input != "y":
        #         print("Compilation aborted by the user.")
        #         run = False

        if run:
            # Reformat the prompt to use xml format.
            student = await reformat_prompt_xml_style(student)
            prompt_string = student.dump() if prompt_aware_proposer else None

            logging.debug("Reformatted student prompt")
            logging.debug(prompt_string)

            # Setup our proposer
            proposer = GroundedProposer(
                trainset=trainset,
                prompt_model=self.prompt_model,
                prompt_string=prompt_string,
                view_data_batch_size=self.view_data_batch_size,
                prompt_aware=prompt_aware_proposer,
                set_history_randomly=True,
                set_tip_randomly=True,
            )

            # Setup random seeds
            random.seed(seed)
            np.random.seed(seed)

            # Log current file to log_dir
            curr_file = os.path.abspath(__file__)
            # save_file_to_log_dir(curr_file, self.log_dir)

            # Set up prompt and evaluation function
            prompt = student.deepcopy()
            evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)

            # Determine the number of fewshot examples to use to generate demos for prompt
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                max_bootstrapped_demos_for_candidate_gen = (
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
                )
                max_labeled_demos_for_candidate_gen = (
                    LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
                )
            else:
                max_bootstrapped_demos_for_candidate_gen = max_bootstrapped_demos
                max_labeled_demos_for_candidate_gen = max_labeled_demos

            # Generate N few shot example sets
            try:
                fewshot_candidates = await create_n_fewshot_demo_sets(
                    student=prompt,
                    num_candidate_sets=self.n,
                    trainset=trainset,
                    max_labeled_demos=max_labeled_demos_for_candidate_gen,
                    max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
                    metric=self.metric,
                    teacher_settings=self.teacher_settings,
                    seed=seed,
                )
                # Save the candidate fewshots generated
                if self.log_dir:
                    fp = os.path.join(self.log_dir, "fewshot_examples_to_save.pickle")
                    with open(fp, "wb") as file:
                        pickle.dump(fewshot_candidates, file)
            except Exception as e:
                print(f"[red]Error generating fewshot examples: {e}[/red]")
                print("Running without fewshot examples.")
                fewshot_candidates = None

            # Generate N candidate prompts
            proposer.program_aware = prompt_aware_proposer
            proposer.use_tip = True
            proposer.use_instruct_history = False
            proposer.set_history_randomly = False
            prompt_candidates = await proposer.propose_prompts(
                prompt=prompt,
                fewshot_candidates=fewshot_candidates,
                N=self.n,
                T=self.init_temperature,
                trial_logs={},
            )
            prompt_candidates[0].messages = prompt.messages

            # instruction_candidates[1][0] = "Given the question, and context, respond with the number of the document that is most relevant to answering the question in the field 'Answer' (ex. Answer: '3')."

            # Save the candidate instructions generated
            if log_dir:
                fp = os.path.join(log_dir, "instructions_to_save.pickle")
                with open(fp, "wb") as file:
                    pickle.dump(prompt_candidates, file)

            instruction_candidates = [prompt.messages for prompt in prompt_candidates]

            # If we're doing zero-shot, reset demo_candidates to none
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                fewshot_candidates = None

            # Initialize variables to store the best program and its score
            best_score = float("-inf")
            best_prompt: Prompt = None
            trial_logs = {}
            total_eval_calls = 0
            param_score_dict = defaultdict(
                list
            )  # Dictionaries of paramater combinations we've tried, and their associated scores
            fully_evaled_param_combos = (
                {}
            )  # List of the parameter combinations we've done full evals of

            # Define our trial objective
            def create_objective(
                baseline_prompt: Prompt,
                instruction_candidates: list[dict[Literal["role", "content"], str]],
                fewshot_candidates: (
                    list[dict[str, any]] | list[weavel.types.DatasetItem]
                ),
                evaluate: Callable[..., Awaitable[Any]],
                trainset: list[dict[str, any]] | list[weavel.types.DatasetItem],
            ):
                def objective(
                    trial: optuna.Trial,
                ) -> Awaitable[float | Sequence[float]]:
                    nonlocal best_prompt, best_score, trial_logs, total_eval_calls  # Allow access to the outer variables

                    # Kick off trial
                    logging.info(f"Starting trial num: {trial.number}")
                    trial_logs[trial.number] = {}

                    logging.debug("Baseline prompt")
                    logging.debug(baseline_prompt.dump())
                    # Create a new candidate prompt
                    candidate_prompt = baseline_prompt.deepcopy()
                    logging.debug("Initial candidate prompt")
                    logging.debug(candidate_prompt.dump())

                    # Choose set of instructions & demos to use for each predictor
                    chosen_params = []

                    # Suggest the index of the instruction / demo candidate to use in our trial
                    instruction_idx = trial.suggest_categorical(
                        "instruction",
                        range(len(instruction_candidates)),
                    )
                    chosen_params.append(instruction_idx)
                    if fewshot_candidates:
                        fewshot_idx = trial.suggest_categorical(
                            "fewshot",
                            range(len(fewshot_candidates)),
                        )
                        chosen_params.append(fewshot_idx)

                    # Log the selected instruction / demo candidate
                    trial_logs[trial.number]["instruction"] = instruction_idx
                    if fewshot_candidates:
                        trial_logs[trial.number]["fewshot"] = fewshot_idx

                    logging.info(f"instruction_idx {instruction_idx}")
                    if fewshot_candidates:
                        logging.info(f"fewshot_idx {fewshot_idx}")

                    # Set the instruction
                    selected_instruction = instruction_candidates[instruction_idx]
                    candidate_prompt.messages = selected_instruction

                    # Set the fewshot
                    if fewshot_candidates:
                        logging.critical("Fewshot candidates")
                        logging.critical(fewshot_candidates)
                        candidate_prompt.fewshot = fewshot_candidates[fewshot_idx]

                    # Log assembled program
                    # print("CANDIDATE PROMPT:")
                    # print(candidate_prompt.dump())
                    # print("...")

                    # Save the candidate prompt
                    trial_logs[trial.number]["prompt_path"] = save_candidate_prompt(
                        candidate_prompt,
                        log_dir,
                        trial.number,
                    )

                    trial_logs[trial.number]["num_eval_calls"] = 0

                    # Evaluate the candidate program with relevant batch size
                    batch_size = self._get_batch_size(minibatch, trainset)
                    logging.debug("Candidate prompt")
                    logging.debug(candidate_prompt.dump())
                    score = run_async(
                        eval_candidate_prompt(
                            batch_size,
                            trainset,
                            candidate_prompt,
                            evaluate,
                        )
                    )

                    # Print out a full trace of the program in use
                    # print("FULL TRACE")
                    # full_trace = get_task_model_history_for_full_example(
                    #     candidate_program, self.task_model, trainset, evaluate,
                    # )
                    # print("...")

                    # Log relevant information
                    # print(f"Score {score}")
                    categorical_key = ",".join(map(str, chosen_params))
                    param_score_dict[categorical_key].append(
                        (score, candidate_prompt),
                    )
                    trial_logs[trial.number]["num_eval_calls"] = batch_size
                    trial_logs[trial.number]["full_eval"] = batch_size >= len(trainset)
                    # trial_logs[trial.number]["eval_example_call"] = full_trace
                    trial_logs[trial.number]["score"] = score
                    trial_logs[trial.number]["pruned"] = False
                    total_eval_calls += trial_logs[trial.number]["num_eval_calls"]
                    trial_logs[trial.number][
                        "total_eval_calls_so_far"
                    ] = total_eval_calls
                    trial_logs[trial.number]["prompt"] = candidate_prompt.deepcopy()

                    # Update the best program if the current score is better, and if we're not using minibatching
                    best_score_updated = False
                    if (
                        score > best_score
                        and trial_logs[trial.number]["full_eval"]
                        and not minibatch
                    ):
                        # print("Updating best score")
                        best_score = score
                        best_prompt = candidate_prompt.deepcopy()
                        best_score_updated = True

                    # If we're doing minibatching, check to see if it's time to do a full eval
                    if minibatch and trial.number % self.minibatch_full_eval_steps == 0:

                        # Save old information as the minibatch version
                        trial_logs[trial.number]["mb_score"] = score
                        trial_logs[trial.number]["mb_prompt_path"] = trial_logs[
                            trial.number
                        ]["prompt_path"]

                        # Identify our best program (based on mean of scores so far, and do a full eval on it)
                        highest_mean_prompt, combo_key = (
                            get_prompt_with_highest_avg_score(
                                param_score_dict, fully_evaled_param_combos
                            )
                        )
                        full_train_score = run_async(
                            eval_candidate_prompt(
                                len(trainset),
                                trainset,
                                highest_mean_prompt,
                                evaluate,
                            )
                        )

                        # Log relevant information
                        fully_evaled_param_combos[combo_key] = {
                            "program": highest_mean_prompt,
                            "score": full_train_score,
                        }
                        total_eval_calls += len(trainset)
                        trial_logs[trial.number][
                            "total_eval_calls_so_far"
                        ] = total_eval_calls
                        trial_logs[trial.number]["full_eval"] = True
                        trial_logs[trial.number]["prompt_path"] = save_candidate_prompt(
                            prompt=highest_mean_prompt,
                            log_dir=log_dir,
                            trial_num=trial.number,
                            note="full_eval",
                        )
                        trial_logs[trial.number]["score"] = full_train_score

                        if full_train_score > best_score:
                            # print(f"UPDATING BEST SCORE WITH {full_train_score}")
                            best_score = full_train_score
                            best_prompt = highest_mean_prompt.deepcopy()
                            best_score_updated = True

                    # If the best score was updated, do a full eval on the dev set
                    if best_score_updated:
                        full_dev_score = run_async(
                            evaluate(
                                best_prompt,
                                devset=valset,
                                display_table=0,
                            )
                        )

                    return score

                return objective

            # Run the trial
            objective_function = create_objective(
                prompt,
                instruction_candidates,
                fewshot_candidates,
                evaluate,
                trainset,
            )

            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            score = study.optimize(objective_function, n_trials=num_batches)

            if best_prompt is not None and self.track_stats:
                best_prompt.trial_logs = trial_logs
                best_prompt.score = best_score
                best_prompt.prompt_model_total_calls = self.prompt_model_total_calls
                best_prompt.total_calls = self.total_calls

            # program_file_path = os.path.join(self.log_dir, 'best_program.pickle')
            if log_dir:
                prompt_file_path = os.path.join(log_dir, "best_prompt.prompt")
                with open(prompt_file_path, "w") as f:
                    f.write(best_prompt.dump())

                optuna_study_file_path = os.path.join(log_dir, "optuna_study.pickle")
                with open(optuna_study_file_path, "wb") as file:
                    pickle.dump(study, file)

            return best_prompt

        return student
