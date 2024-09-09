from typing import Any, Dict, Optional
import numpy as np
from litellm import aembedding
from ape.metric.metric_base import BaseMetric


class CosineSimilarityMetric(BaseMetric):
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model

    async def compute(
        self,
        inputs: Dict[str, Any],
        gold: Any,
        pred: Any,
        trace: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> float:
        try:
            if not isinstance(gold, str):
                gold = str(gold)
            if not isinstance(pred, str):
                pred = str(pred)

            def get_embedding(result):
                if hasattr(result.data[0], "embedding"):
                    return result.data[0].embedding
                elif isinstance(result.data[0], dict) and "embedding" in result.data[0]:
                    return result.data[0]["embedding"]
                else:
                    raise ValueError("Embedding not found in the expected format")

            gold_embedding = await aembedding(model=self.model, input=gold)
            gold_embedding = get_embedding(gold_embedding)

            pred_embedding = await aembedding(model=self.model, input=pred)
            pred_embedding = get_embedding(pred_embedding)

            similarity = np.dot(gold_embedding, pred_embedding) / (
                np.linalg.norm(gold_embedding) * np.linalg.norm(pred_embedding)
            )
            score = max(0.0, similarity)

            return score
        except Exception as e:
            if trace is not None:
                trace["error"] = str(e)
            return 0.0
