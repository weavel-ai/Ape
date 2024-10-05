from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from ape.common.prompt.prompt_base import Prompt
from ape.common.types.dataset_item import DatasetItem

class BaseParaphraser(ABC):
    @abstractmethod
    async def paraphrase(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
        buffer_trainset: List[DatasetItem],
        text_gradient: str,
        config: Dict[str, Any],
    ) -> Tuple[Prompt, List[DatasetItem], List[DatasetItem], List[DatasetItem], str]:
        """
        Paraphrase the given text

        Args:
            prompt (Prompt): Prompt
            config (Dict[str, Any]): Paraphrasing parameters

        Returns:
            Prompt: Paraphrased prompt
        """
        pass

    async def __call__(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
        buffer_trainset: List[DatasetItem],
        text_gradient: str,
        config: Dict[str, Any],
    ) -> Tuple[Prompt, List[DatasetItem], List[DatasetItem], List[DatasetItem], str]:
        return await self.paraphrase(prompt, trainset, valset, buffer_trainset, text_gradient, config)
