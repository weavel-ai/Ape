import os
import sys
import textwrap
import optuna
from collections import defaultdict
import pickle
import random
from typing import Any, Awaitable, Callable, List, Literal, Sequence
import numpy as np
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
from ape.proposer.grounded_proposer import GroundedProposer
from ape.utils import run_async, logger
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.response_format import (
    ResponseFormat,
    ResponseFormatJSON,
    ResponseFormatType,
)


BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0


class MIPRO(MIPROBase):
    """MIPRO: Multi-prompt Instruction PRoposal Optimizer

    Args:
        Optimizer (_type_): _description_
    """

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
        task_description: str = "",
        trainset: Dataset,
        valset=None,
        max_steps=30,
        max_bootstrapped_demos=5,
        max_labeled_demos=2,
        goal_score=96,
        eval_kwargs={},
        seed=9,
        minibatch=True,
        prompt_aware_proposer=True,
        requires_permission_to_run=True,
        response_format: ResponseFormat = ResponseFormatJSON,
        log_dir: str,
    ):
        # Define ANSI escape codes for colors
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        ENDC = "\033[0m"  # Resets the color to default

        random.seed(seed)
        valset = valset or trainset
        estimated_prompt_model_calls = (
            10 + self.num_candidates + 1
        )  # num data summary calls + N + 1

        prompt_model_line = f"""[yellow]- Prompt Model: [blue][bold]10[/bold][/blue] data summarizer calls + [blue][bold]{self.num_candidates}[/bold][/blue] lm calls in program = [blue][bold]{estimated_prompt_model_calls}[/bold][/blue] prompt model calls[/yellow]"""

        estimated_task_model_calls_wo_module_calls = 0
        task_model_line = ""
        if not minibatch:
            estimated_task_model_calls_wo_module_calls = (
                len(trainset) * max_steps
            )  # M * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{max_steps}{ENDC}{YELLOW} batches * {BLUE}{BOLD}# of LM calls in your program{ENDC}{YELLOW} = ({BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls} * # of LM calls in your program{ENDC}{YELLOW}) task model calls{ENDC}"""
        else:
            estimated_task_model_calls_wo_module_calls = (
                self.minibatch_size * max_steps
                + (len(trainset) * (max_steps // self.minibatch_full_eval_steps))
            )  # B * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{self.minibatch_size}{ENDC}{YELLOW} examples in minibatch * {BLUE}{BOLD}{max_steps}{ENDC}{YELLOW} batches + {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{max_steps // self.minibatch_full_eval_steps}{ENDC}{YELLOW} full evals = {BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls}{ENDC}{YELLOW} task model calls{ENDC}"""

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
            student = await reformat_prompt(
                prompt=student, response_format=response_format
            )

            logger.info("Reformatted student prompt")

            # Setup our proposer
            proposer = GroundedProposer(
                trainset=trainset,
                prompt_model=self.prompt_model,
                view_data_batch_size=self.view_data_batch_size,
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
                # log time
                logger.info("Started generating fewshot examples")
                fewshot_candidates = await create_n_fewshot_demo_sets(
                    student=prompt,
                    num_candidate_sets=self.num_candidates,
                    trainset=trainset,
                    max_labeled_demos=max_labeled_demos_for_candidate_gen,
                    max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
                    metric=self.metric,
                    teacher_settings=self.teacher_settings,
                    seed=seed,
                )
                # Save the candidate fewshots generated
                if log_dir:
                    fp = os.path.join(log_dir, "fewshot_examples_to_save.pkl")
                    with open(fp, "wb") as file:
                        pickle.dump(fewshot_candidates, file)
                logger.info(fewshot_candidates)
            except Exception as e:
                logger.error(f"Error generating fewshot examples: {e}")
                logger.error("Running without fewshot examples.")
                fewshot_candidates = None

            # Generate N candidate prompts
            proposer = MIPROProposer(**self.model_dump())
            prompt_candidates = await proposer.generate_candidates(
                prompt=prompt,
                trainset=trainset,
                task_description=task_description,
                fewshot_candidates=fewshot_candidates,
            )
            # proposer.program_aware = prompt_aware_proposer
            # proposer.use_tip = True
            # proposer.use_instruct_history = False
            # proposer.set_history_randomly = False
            # logger.info("Started generating instructions")
            # prompt_candidates = await proposer.propose_prompts(
            #     task_description=task_description,
            #     prompt=prompt,
            #     fewshot_candidates=fewshot_candidates,
            #     N=self.num_candidates,
            #     T=self.init_temperature,
            #     trial_logs={},
            # )
            prompt_candidates[0].messages = prompt.messages

            # instruction_candidates[1][0] = "Given the question, and context, respond with the number of the document that is most relevant to answering the question in the field 'Answer' (ex. Answer: '3')."

            # Save the candidate instructions generated
            if log_dir:
                fp = os.path.join(log_dir, "instructions_to_save.pkl")
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
                fewshot_candidates: List[Dataset],
                evaluate: Callable[..., Awaitable[Any]],
                trainset: Dataset,
            ):
                def objective(
                    trial: optuna.Trial,
                ) -> Awaitable[float | Sequence[float]]:
                    nonlocal best_prompt, best_score, trial_logs, total_eval_calls  # Allow access to the outer variables

                    # Kick off trial
                    logger.info(f"Starting trial num: {trial.number}")
                    trial_logs[trial.number] = {}

                    logger.info("Baseline prompt")
                    logger.info(baseline_prompt.dump())
                    # Create a new candidate prompt
                    candidate_prompt = baseline_prompt.deepcopy()
                    logger.info("Initial candidate prompt")
                    logger.info(candidate_prompt.dump())

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

                    logger.info(f"instruction_idx {instruction_idx}")
                    if fewshot_candidates:
                        logger.info(f"fewshot_idx {fewshot_idx}")

                    # Set the instruction
                    selected_instruction = instruction_candidates[instruction_idx]
                    candidate_prompt.messages = selected_instruction

                    # Set the fewshot
                    if fewshot_candidates:
                        logger.info("Fewshot candidates")
                        logger.info(fewshot_candidates)
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
                    logger.info("Candidate prompt")
                    logger.info(candidate_prompt.dump())
                    logger.info(f"Started evals for trial {trial.number}")
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

                    if score >= goal_score:
                        trial.study.stop()

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

            logger.info("Started optimization")
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            score = study.optimize(objective_function, n_trials=max_steps)

            if best_prompt is not None and self.track_stats:
                best_prompt.trial_logs = trial_logs
                best_prompt.score = best_score

            # program_file_path = os.path.join(self.log_dir, 'best_program.pkl')
            if log_dir:
                prompt_file_path = os.path.join(log_dir, "best_prompt.prompt")
                with open(prompt_file_path, "w") as f:
                    f.write(best_prompt.dump())

                optuna_study_file_path = os.path.join(log_dir, "optuna_study.pkl")
                with open(optuna_study_file_path, "wb") as file:
                    pickle.dump(study, file)

            return best_prompt

        return student
