from typing import Any, Dict, Optional, List, Union

import numpy as np
from litellm import aembedding
from .metric_base import BaseMetric


class CosineSimilarityMetric(BaseMetric):
    """
    A metric class that computes the cosine similarity between two text inputs using embeddings.

    Attributes:
        model (str): The name of the embedding model to use. Defaults to "text-embedding-3-large".
    """

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize the CosineSimilarityMetric.

        Args:
            model (str): The name of the embedding model to use. Defaults to "text-embedding-3-large".
        """
        self.model = model

    async def compute(
        self, inputs, gold: str, pred: str, trace: Optional[Dict] = None
    ) -> float:
        """
        Compute the cosine similarity between the gold standard and prediction texts.

        Args:
            gold (str): The gold standard text.
            pred (str): The prediction text.
            trace (Optional[Dict]): Additional trace information (not used in this implementation).

        Returns:
            float: The cosine similarity score between 0 and 1, or 0 if the similarity is negative.

        Note:
            This method converts inputs to strings if they aren't already,
            generates embeddings for both inputs, and then computes their cosine similarity.
        """
        if not isinstance(gold, str):
            gold = str(gold)
        if not isinstance(pred, str):
            pred = str(pred)
            
        def get_embedding(result):
            if hasattr(result.data[0], 'embedding'):
                return result.data[0].embedding
            elif isinstance(result.data[0], dict) and 'embedding' in result.data[0]:
                return result.data[0]['embedding']
            else:
                raise ValueError("Embedding not found in the expected format")

        gold_embedding = await aembedding(model=self.model, input=gold)
        gold_embedding = get_embedding(gold_embedding)

        pred_embedding = await aembedding(model=self.model, input=pred)
        pred_embedding = get_embedding(pred_embedding)

        similarity = np.dot(gold_embedding, pred_embedding) / (
            np.linalg.norm(gold_embedding) * np.linalg.norm(pred_embedding)
        )
        if similarity <= 0:
            return 0.0
        else:
            return similarity
