import random
from ape.optimizer.optimizer_base import Optimizer
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset


class FewShotOptimizer(Optimizer):
    k: int = 16

    async def optimize(self, student: Prompt, trainset: Dataset, sample: bool = True):
        self.student = student.reset_copy()
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        if sample:
            self.student.fewshot = rng.sample(
                self.trainset, min(self.k, len(self.trainset))
            )
        else:
            self.student.fewshot = self.trainset[: min(self.k, len(self.trainset))]

        return self.student
