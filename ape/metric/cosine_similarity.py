from typing import Any, Dict, Optional, List, Union

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .metric_base import BaseMetric

openai_client = AsyncOpenAI()

class CosineSimilarityMetric(BaseMetric):
    async def compute(self, gold: str, pred: str, trace: Optional[Dict] = None) -> float:
        if not isinstance(gold, str):
            gold = str(gold)
        if not isinstance(pred, str):
            pred = str(pred)
            
        gold_embedding = await openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=gold
        )
        gold_embedding = gold_embedding.data[0].embedding
        
        pred_embedding = await openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=pred
        )
        pred_embedding = pred_embedding.data[0].embedding
        
        similarity = np.dot(gold_embedding, pred_embedding) / (np.linalg.norm(gold_embedding) * np.linalg.norm(pred_embedding))
        if similarity <= 0:
            return 0.0
        else:
            return similarity
        
    