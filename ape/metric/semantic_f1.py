from typing import Any, Dict, Optional, List, Union

from pysbd import Segmenter
from pysbd.utils import TextSpan

from ape.prompt.prompt_base import Prompt
from .metric_base import BaseMetric
from ape.types import DataItem, DatasetItem

# TODO: add Cost Tracking
# TOOD: fix BaseMetric.compute to get inputs

semantic_analysis = Prompt.from_filename("statement_analysis")
semantic_precision = Prompt.from_filename("semantic_precision")
semantic_recall = Prompt.from_filename("semantic_recall")

def _segment_text(text: str) -> str:
    segmenter = Segmenter(language="en", clean=False, char_span=True)
    sentences: List[TextSpan] = segmenter.segment(text)
    sentences: List[str] = [sent.sent for sent in sentences]
    sentences = [sent.strip() for sent in sentences if sent.strip().endswith('.')]
    sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
    return sentences

async def _extract_statements(question: str, answer: str, sentences: str) -> List[str]:
    res = await semantic_analysis(
        lm_config={
            "timeout": 120
        },
        **{
            "question": question,
            "answer": answer,
            "sentences": sentences
        }
    )
    res = res["analysis"]
    statements = []
    for analysis in res:
        statements.extend(analysis["simpler_statements"])
    return statements

async def _compute_precision(
    gold: str,
    prediction_statements: List[str]
) -> float:
    res = await semantic_precision(
        lm_config={
            "timeout": 120
        },
        **{
            "ground_truth": gold,
            "statements": prediction_statements
        }
    )
    answer = res["answer"]
    length = len(answer)
    score = sum([1 for x in answer if x["verdict"] == 1])
    score = score / length if length > 0 else 0
    return score

async def _compute_recall(
    pred: str,
    gold_statements: List[str]
) -> float:
    res = await semantic_recall(
        lm_config={
            "timeout": 120
        },
        **{
            "prediction": pred,
            "ground_truth_statements": gold_statements
        }
    )
    answer = res["answer"]
    length = len(answer)
    score = sum([1 for x in answer if x["verdict"] == 1])
    score = score / length if length > 0 else 0
    return score

class SemanticF1Metric(BaseMetric):
    def __init__(self, question_key: str):
        self.inputs_question_key = question_key.lower().replace(' ', '_')
        
    async def compute(self, inputs: Dict[str, Any], gold: str, pred: str, trace: Optional[Dict] = None) -> float:
        inputs = {k.lower().replace(' ', '_'): v for k, v in inputs.items()}
        question = inputs[self.inputs_question_key]
        
        prediction_sentences = _segment_text(pred)
        gold_sentences = _segment_text(gold)
        
        prediction_statements = await _extract_statements(question, pred, prediction_sentences)
        gold_statements = await _extract_statements(question, gold, gold_sentences)
    
        semantic_precision = await _compute_precision(gold, prediction_statements)
        semantic_recall = await _compute_recall(pred, gold_statements)
        return (2 * semantic_precision * semantic_recall) / (semantic_precision + semantic_recall) if (semantic_precision + semantic_recall) > 0 else 0
    
    