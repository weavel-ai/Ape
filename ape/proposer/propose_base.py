from abc import ABC, abstractmethod


class Proposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def propose_prompts(self):
        pass

    def propose_one(self):
        pass
