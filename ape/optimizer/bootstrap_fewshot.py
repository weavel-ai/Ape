import asyncio
import random
import threading
from typing import Any, Awaitable, Callable, Dict, Optional, Union
from pydantic import Field, ConfigDict

from ape.base import lm as base_lm
from ape.metric.metric_base import BaseMetric
from ape.optimizer.fewshot_optimizer import FewShotOptimizer
from ape.optimizer.optimizer_base import Optimizer
from ape.prompt.prompt_base import Prompt
from ape.utils import logger
from ape.types import DataItem, Dataset, DatasetItem


class BootstrapFewShot(Optimizer):
    """
    An Optimizer class that composes a set of demos/examples to go into a predictor's prompt.
    These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

    Parameters
    ----------
    metric: BaseMetric
        A function that compares an expected value and predicted value, outputting the result of that comparison.
    metric_threshold: optional float, default `None`
        If the metric yields a numerical value, then check it against this threshold when
        deciding whether or not to accept a bootstrap example.
    teacher_settings: dict, optional
        Settings for the `teacher` model.
    max_bootstrapped_demos: int, default 4
        Maximum number of bootstrapped demonstrations to include
    max_labeled_demos: int, default 3
        Maximum number of labeled demonstrations to include.
    max_rounds: int, default 1
        Number of iterations to attempt generating the required bootstrap examples. If unsuccessful after `max_rounds`, the program ends.
    max_errors: int, default 5
        Maximum number of errors until program ends.
    """

    metric: BaseMetric = None
    metric_threshold: Optional[float] = None
    teacher_settings: Dict = {}
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 3
    max_rounds: int = 1
    max_errors: int = 5
    error_count: int = 0
    error_lock: Any = Field(default_factory=threading.Lock, exclude=True)
    bootstrapped_fewshot: Optional[Dataset] = None
    validation: Optional[Dataset] = None

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    async def optimize(
        self,
        student: Prompt,
        *,
        teacher: Optional[Prompt] = None,
        trainset: Dataset,
    ) -> Prompt:
        self.trainset = trainset

        await self._prepare_student_and_teacher(student, teacher)
        await self._bootstrap()

        self.student = self._train()
        self.student._optimized = True

        # set assert_failures and suggest_failures as attributes of student w/ value 0
        self.student._assert_failures = 0
        self.student._suggest_failures = 0

        return self.student

    async def _prepare_student_and_teacher(self, student: Prompt, teacher: Prompt):
        self.student = student.reset_copy()
        self.teacher = (
            teacher.deepcopy() if teacher is not None else student.reset_copy()
        )

        assert (
            getattr(self.student, "_compiled", False) is False
        ), "Student must be uncompiled."

        if (
            self.max_labeled_demos
            and getattr(self.teacher, "_compiled", False) is False
        ):
            optimizer = FewShotOptimizer(k=self.max_labeled_demos)
            self.teacher = await optimizer.optimize(
                self.teacher.reset_copy(), trainset=self.trainset
            )

    async def _bootstrap(self, *, max_bootstraps=None):
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
        teacher = self.teacher  # .deepcopy()
        cache = []

        try:
            lm = (
                base_lm.copy(temperature=0.7 + 0.001 * round_idx)
                if round_idx > 0
                else base_lm
            )

            cache = teacher.fewshot
            teacher.fewshot = [x for x in teacher.fewshot if x != example]
            inputs = (
                example.inputs
                if hasattr(example, "inputs")
                else example.get("inputs", {})
            )
            prediction: Union[str, Dict[str, Any]] = await teacher(
                lm_config=lm, **inputs
            )

            teacher.fewshot = cache

            if self.metric:
                metric_val = await self.metric(example.outputs, prediction, None)

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
