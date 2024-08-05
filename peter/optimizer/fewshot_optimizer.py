import random
from peter.optimizer.optimizer_base import Optimizer
from peter.prompt.prompt_base import Prompt
from peter.types import Dataset


class FewShotOptimizer(Optimizer):
    def __init__(self, k=16):
        self.k: int = k

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
