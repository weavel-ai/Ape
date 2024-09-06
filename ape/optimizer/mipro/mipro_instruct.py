import os
import pickle
import random
from typing import Any, Dict, List, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from ape.evaluate.evaluate import Evaluate
from ape.optimizer.mipro.mipro_base import MIPROBase
from ape.optimizer.utils import reformat_prompt
from ape.proposer.instruct_by_score import InstructByScore
from ape.utils import logger
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.response_format import ResponseFormat

BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT: int = 0

# Instruction Generation. Later Add Optuna and GroundedProposer to do (N+1) x (M+1) search.
class MIPROInstruct(MIPROBase):
    """
    MIPROInstruct class for optimizing prompts.

    This class is a modified vesion of MIPRO with a focus on generating better instructions. It supports minibatch evaluation,
    few-shot examples, and various optimization strategies.
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

        # TODO: Modify to match the actual number of LM calls in the program.
        if not minibatch:
            estimated_task_model_calls = len(trainset) * max_steps
            task_model_line = f"- Task Model: {len(trainset)} examples in train set * {max_steps} batches * # of LM calls in your program = ({estimated_task_model_calls} * # of LM calls in your program) task model calls"
        else:
            if self.update_prompt_after_full_eval:
                estimated_task_model_calls = self.minibatch_size * max_steps + (
                    len(trainset) * (max_steps // self.minibatch_full_eval_steps)
                )
                task_model_line = f"- Task Model: {self.minibatch_size} examples in minibatch * {max_steps} batches + {len(trainset)} examples in train set * {max_steps // self.minibatch_full_eval_steps} full evals = {estimated_task_model_calls} task model calls"
            else:
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

    async def generate_instructions(
        self,
        student: Prompt,
        *,
        trainset: Dataset,
        testset: Optional[Dataset] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 9,
        response_format: Optional[ResponseFormat] = None,
        log_dir: str,
        metric: Optional[str] = None,
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
            student = await reformat_prompt(
                prompt=student, response_format=response_format
            )

        evaluate: Evaluate = Evaluate(
            testset=testset,
            metric=self.metric,
            global_metric=self.global_metric,
            **eval_kwargs,
        )
        
        # This is the proposer.
        proposer = InstructByScore(
            prompt_model=self.prompt_model,
            trainset=trainset,
            view_data_batch_size=self.view_data_batch_size,
        )

        proposer.program_aware = True
        proposer.use_tip = True
        proposer.use_instruct_history = False
        proposer.set_history_randomly = False

        logger.info(f"Generating {self.num_candidates} instruction candidates")
        
        # Generate N candidates using InstructByScore.
        prompt_candidates = await proposer.propose_prompts(
            base_prompt=student,
            N=self.num_candidates,
            T=self.init_temperature,
            trial_logs={},
            evaluate=evaluate,
            metric=metric,
        )

        if log_dir:
            with open(os.path.join(log_dir, "instructions_to_save.pkl"), "wb") as file:
                pickle.dump(prompt_candidates, file)

        instruction_candidates: List[List[Dict[str, str]]] = [
            prompt.messages for prompt in prompt_candidates
        ]

        return instruction_candidates
