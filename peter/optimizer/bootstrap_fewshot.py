import asyncio
import logging
import random
import threading
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import tqdm
from peter.base import lm as base_lm
from peter.optimizer.fewshot_optimizer import FewShotOptimizer
from peter.optimizer.optimizer_base import Optimizer
from peter.prompt.prompt_base import Prompt
from peter.types import DataItem, Dataset


class BootstrapFewShot(Optimizer):
    def __init__(
        self,
        metric: Callable[..., Union[Any, Awaitable[Any]]] = None,
        metric_threshold=None,
        teacher_settings: Optional[Dict] = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=5,
    ):
        """
        A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
        These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

        Parameters
        ----------
        metric: Callable
            A function that compares an expected value and predicted value, outputting the result of that comparison.
        metric_threshold: optional float, default `None`
            If the metric yields a numerical value, then check it against this threshold when
            deciding whether or not to accept a bootstrap example.
        teacher_settings: dict, optional
            Settings for the `teacher` model.
        max_bootstrapped_demos: int, default 4
            Maximum number of bootstrapped demonstrations to include
        max_labeled_demos: int, default 16
            Maximum number of labeled demonstrations to include.
        max_rounds: int, default 1
            Number of iterations to attempt generating the required bootstrap examples. If unsuccessful after `max_rounds`, the program ends.
        max_errors: int, default 5
            Maximum number of errors until program ends.
        """
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = {} if teacher_settings is None else teacher_settings
        self.validation = []

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    async def optimize(
        self,
        student: Prompt,
        *,
        teacher: Optional[Prompt] = None,
        trainset: Dataset,
    ):
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
            for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
                if len(bootstrapped) >= max_bootstraps:
                    break

                if example_idx not in bootstrapped:
                    success = await self._bootstrap_one_example(example, round_idx)

                    if success:
                        bootstrapped[example_idx] = True

        logging.debug(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx + 1} examples in round {round_idx}.",
        )

        # Unbootstrapped training examples

        self.validation = [
            x for idx, x in enumerate(self.trainset) if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)

        self.validation = self.validation

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
                # Check if the metric function is a coroutine function
                if asyncio.iscoroutinefunction(self.metric):
                    metric_val = await self.metric(example, prediction, None)
                else:
                    metric_val = self.metric(example, prediction, None)

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
            logging.error(
                f"Failed to run or to evaluate example {example} with {self.metric} due to {e}."
            )

        return success

    def _train(self):
        raw_fewshot = self.validation
        self.student.fewshot = raw_fewshot

        return self.student
