import asyncio
import random
import threading
from typing import Any, Dict, Optional, Union
from pydantic import Field, ConfigDict

from ape.metric.metric_base import BaseMetric
from ape.optimizer.sampled_fewshot import SampledFewshot
from ape.optimizer.optimizer_base import Optimizer
from ape.prompt.prompt_base import Prompt
from ape.utils import logger
from ape.types import DataItem, Dataset


class BootstrapFewShot(Optimizer):
    """
    A class for optimizing prompts using bootstrapped few-shot examples.

    This optimizer uses a combination of labeled and bootstrapped examples to create
    an optimized prompt. It iteratively improves the prompt by bootstrapping examples
    from the training set.

    Attributes:
        metric (Optional[BaseMetric]): The metric used to evaluate bootstrapped examples.
        metric_threshold (Optional[float]): The threshold for the metric to consider an example successful.
        teacher_settings (Dict): Settings for the teacher prompt.
        max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations to include.
        max_labeled_demos (int): Maximum number of labeled demonstrations to include.
        max_rounds (int): Maximum number of bootstrapping rounds.
        max_errors (int): Maximum number of errors allowed before raising an exception.
        error_count (int): Current count of errors encountered.
        error_lock (threading.Lock): Lock for thread-safe error counting.
        bootstrapped_fewshot (Optional[Dataset]): Dataset of successfully bootstrapped examples.
        validation (Optional[Dataset]): Dataset used for validation.
    """

    metric: Optional[BaseMetric] = None
    metric_threshold: Optional[float] = None
    teacher_settings: Dict = {}
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 3
    max_rounds: int = 1
    max_errors: int = 5
    error_count: int = 0
    error_lock: threading.Lock = Field(default_factory=threading.Lock, exclude=True)
    bootstrapped_fewshot: Optional[Dataset] = None
    validation: Optional[Dataset] = None

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the BootstrapFewShot optimizer."""
        super().__init__(**data)
        self.error_lock = threading.Lock()

    async def optimize(
        self,
        student: Prompt,
        *,
        teacher: Optional[Prompt] = None,
        trainset: Dataset,
        **kwargs,
    ) -> Prompt:
        """
        Optimize the student prompt using bootstrapped few-shot examples.

        Args:
            student (Prompt): The student prompt to be optimized.
            teacher (Optional[Prompt]): The teacher prompt used for bootstrapping.
            trainset (Dataset): The dataset used for training and bootstrapping.

        Returns:
            Prompt: The optimized student prompt.
        """
        self.trainset = trainset

        await self._prepare_student_and_teacher(student, teacher)
        await self._bootstrap()

        self.student = self._train()
        self.student.set_optimized(True)

        return self.student

    async def _prepare_student_and_teacher(self, student: Prompt, teacher: Prompt):
        """
        Prepare the student and teacher prompts for optimization.

        Args:
            student (Prompt): The student prompt to be optimized.
            teacher (Prompt): The teacher prompt used for bootstrapping.
        """
        self.student = student.reset_copy()
        self.teacher = (
            teacher.deepcopy() if teacher is not None else student.reset_copy()
        )

        assert self.student.is_optimized() is False, "Student must be unoptimized."

        if self.max_labeled_demos and self.teacher.is_optimized() is False:
            optimizer = SampledFewshot(k=self.max_labeled_demos)
            self.teacher = await optimizer.optimize(
                self.teacher.reset_copy(), trainset=self.trainset
            )

    async def _bootstrap(self, *, max_bootstraps=None):
        """
        Bootstrap examples from the training set.

        Args:
            max_bootstraps (Optional[int]): Maximum number of examples to bootstrap.
        """
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos

        bootstrapped = {}
        self.bootstrapped_fewshot = []

        for round_idx in range(self.max_rounds):
            tasks = []
            for example_idx, example in enumerate(self.trainset):
                if len(bootstrapped) >= max_bootstraps:
                    break

                if example_idx not in bootstrapped:
                    tasks.append(self._bootstrap_one_example(example, round_idx))

            results = await asyncio.gather(*tasks)

            for example_idx, success in enumerate(results):
                if success:
                    bootstrapped[example_idx] = True

            if len(bootstrapped) >= max_bootstraps:
                break

        logger.info(
            f"Bootstrapped {len(bootstrapped)} full traces after {len(self.trainset)} examples in round {round_idx}.",
        )

        self.validation = [
            x for idx, x in enumerate(self.trainset) if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)

        # NOTE: Can't yet use evaluate because we need to trace *per example*
        # evaluate = Evaluate(program=self.teacher, metric=self.metric, num_threads=12)
        # score = evaluate(self.metric, display_table=False, display_progress=True)

    async def _bootstrap_one_example(self, example: DataItem, round_idx=0):
        """
        Bootstrap a single example from the training set.

        Args:
            example (DataItem): The example to bootstrap.
            round_idx (int): The current bootstrapping round index.

        Returns:
            bool: True if the example was successfully bootstrapped, False otherwise.
        """
        teacher = self.teacher
        cache = []

        try:
            lm_config = (
                {"temperature": 0.7 + 0.001 * round_idx}
                if round_idx > 0
                else {"temperature": 1.0}
            )

            cache = teacher.fewshot
            teacher.fewshot = [x for x in teacher.fewshot if x != example]
            inputs = (
                example.inputs
                if hasattr(example, "inputs")
                else example.get("inputs", {})
            )
            outputs = (
                example.outputs
                if hasattr(example, "outputs")
                else example.get("outputs", {})
            )
            prediction: Union[str, Dict[str, Any]] = await teacher(
                lm_config=lm_config, **inputs
            )
            teacher.fewshot = cache

            if self.metric is not None:
                metric_val = await self.metric(inputs, outputs, prediction, None)
                if self.metric_threshold:
                    success = metric_val >= self.metric_threshold
                else:
                    success = metric_val
            else:
                success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            if current_error_count >= self.max_errors:
                raise e
            logger.error(
                f"Failed to run or to evaluate example {example} with {self.metric}.\n Error: {e}."
            )

        if success:
            self.bootstrapped_fewshot.append(example)

        return success

    def _train(self):
        """
        Train the student prompt using the bootstrapped and raw few-shot examples.

        Returns:
            Prompt: The trained student prompt.
        """
        rng = random.Random(0)
        raw_fewshot = self.validation
        augmented_fewshot = self.bootstrapped_fewshot[: self.max_bootstrapped_demos]

        sample_size = min(
            self.max_labeled_demos - len(augmented_fewshot), len(raw_fewshot)
        )
        sample_size = max(sample_size, 0)

        raw_fewshot = rng.sample(raw_fewshot, sample_size)

        self.student.fewshot = raw_fewshot + augmented_fewshot

        return self.student
