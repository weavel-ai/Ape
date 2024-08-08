from ape.prompt.prompt_base import Prompt
from ape.types import Dataset


class Optimizer:
    def __init__(self):
        self.student: Prompt
        self.trainset: Dataset

    async def optimize():
        pass
