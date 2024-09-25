from abc import ABC, abstractmethod


class Proposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def propose_prompts(self):
        pass

    async def propose_one(self):
        pass
