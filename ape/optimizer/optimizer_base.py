from typing import Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset


class Optimizer(BaseModel, ABC):
    """
    Abstract base class for prompt optimization algorithms.

    Attributes:
        student (Optional[Prompt]): The student prompt to be optimized.
        teacher (Optional[Prompt]): An optional teacher prompt used in some optimization strategies.
        trainset (Optional[Dataset]): The dataset used for training/optimization.
    """

    student: Optional[Prompt] = None
    teacher: Optional[Prompt] = None
    trainset: Optional[Dataset] = None

    @abstractmethod
    async def optimize(self, student: Prompt, *, trainset: Dataset, **kwargs) -> Prompt:
        """
        Abstract method to optimize a given prompt.

        Args:
            student (Prompt): The initial prompt to be optimized.
            trainset (Dataset): The dataset used for optimization.
            **kwargs: Additional keyword arguments specific to the optimization algorithm.

        Returns:
            Prompt: The optimized prompt.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")
