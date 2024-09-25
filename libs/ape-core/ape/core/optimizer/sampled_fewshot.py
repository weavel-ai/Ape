import random
from typing import List
from ape.core.optimizer.optimizer_base import Optimizer
from ape.common.prompt import Prompt
from ape.common.types import DatasetItem


class SampledFewshot(Optimizer):
    """
    A class for optimizing prompts by sampling few-shot examples from a dataset.

    This optimizer selects a subset of examples from the training set to use as few-shot examples
    in the prompt. The selection can be either random or sequential.

    Attributes:
        k (int): The maximum number of few-shot examples to include in the prompt. Defaults to 16.
    """

    k: int = 16

    async def optimize(
        self, student: Prompt, *, trainset: List[DatasetItem], randomize: bool = True, **kwargs
    ) -> Prompt:
        """
        Optimize the given prompt by sampling few-shot examples from the training set.

        Args:
            student (Prompt): The initial prompt to be optimized.
            trainset (List[DatasetItem]): The dataset to sample few-shot examples from.
            randomize (bool): Whether to randomly sample examples or take the first k. Defaults to True.

        Returns:
            Prompt: The optimized prompt with sampled few-shot examples.
        """
        self.student = student.reset_copy()
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        if randomize:
            self.student.fewshot = random.sample(self.trainset, min(self.k, len(self.trainset)))
        else:
            self.student.fewshot = self.trainset[: min(self.k, len(self.trainset))]

        return self.student
