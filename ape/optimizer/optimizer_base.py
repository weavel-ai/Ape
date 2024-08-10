from typing import Optional
from pydantic import BaseModel
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset


class Optimizer(BaseModel):
    student: Optional[Prompt] = None
    teacher: Optional[Prompt] = None
    trainset: Optional[Dataset] = None

    async def optimize():
        pass
