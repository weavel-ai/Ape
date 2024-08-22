from typing import Any, Dict, Optional, List, Union

from pysbd import Segmenter
from pysbd.utils import TextSpan

from ape.prompt.prompt_base import Prompt
from .metric_base import BaseMetric


class SemanticF1Metric(BaseMetric):
    """
    A metric class that computes the Semantic F1 score between a prediction and a gold standard.

    This metric uses semantic analysis to extract statements from both the prediction and gold standard,
    and then computes precision and recall based on these statements. The final F1 score is calculated
    using the harmonic mean of precision and recall.

    Attributes:
        inputs_question_key (str): The key to access the question in the inputs dictionary.
        semantic_analysis (Prompt): A prompt for extracting statements from text.
        semantic_precision (Prompt): A prompt for computing semantic precision.
        semantic_recall (Prompt): A prompt for computing semantic recall.
        segmenter (Segmenter): A text segmenter for breaking text into sentences.

    Methods:
        compute: Computes the Semantic F1 score.
        _segment_text: Segments text into sentences.
        _extract_statements: Extracts statements from text using semantic analysis.
        _compute_precision: Computes semantic precision.
        _compute_recall: Computes semantic recall.
    """

    def __init__(self, question_key: str):
        """
        Initialize the SemanticF1Metric.

        Args:
            question_key (str): The key to access the question in the inputs dictionary.
        """
        self.inputs_question_key = question_key.lower().replace(" ", "_")
        self.semantic_analysis = Prompt.from_filename("statement_analysis")
        self.semantic_precision = Prompt.from_filename("semantic_precision")
        self.semantic_recall = Prompt.from_filename("semantic_recall")
        self.segmenter = Segmenter(language="en", clean=False, char_span=True)

    async def compute(
        self, inputs: Dict[str, Any], gold: str, pred: str, trace: Optional[Dict] = None
    ) -> float:
        """
        Compute the Semantic F1 score between the prediction and gold standard.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing the question.
            gold (str): The gold standard text.
            pred (str): The prediction text.
            trace (Optional[Dict]): Additional trace information (not used in this implementation).

        Returns:
            float: The computed Semantic F1 score between 0 and 1.
        """
        inputs = {k.lower().replace(" ", "_"): v for k, v in inputs.items()}
        question = inputs[self.inputs_question_key]

        prediction_sentences = self._segment_text(pred)
        gold_sentences = self._segment_text(gold)

        prediction_statements = await self._extract_statements(
            question, pred, prediction_sentences
        )
        gold_statements = await self._extract_statements(question, gold, gold_sentences)

        semantic_precision = await self._compute_precision(gold, prediction_statements)
        semantic_recall = await self._compute_recall(pred, gold_statements)
        return (
            (2 * semantic_precision * semantic_recall)
            / (semantic_precision + semantic_recall)
            if (semantic_precision + semantic_recall) > 0
            else 0
        )

    def _segment_text(self, text: str) -> str:
        """
        Segment the input text into sentences.

        Args:
            text (str): The input text to be segmented.

        Returns:
            str: A string of numbered sentences, each on a new line.
        """
        sentences: List[TextSpan] = self.segmenter.segment(text)
        sentences: List[str] = [sent.sent for sent in sentences]
        sentences = [sent.strip() for sent in sentences if sent.strip().endswith(".")]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        return sentences

    async def _extract_statements(
        self, question: str, answer: str, sentences: str
    ) -> List[str]:
        """
        Extract statements from the text using semantic analysis.

        Args:
            question (str): The question related to the text.
            answer (str): The full answer text.
            sentences (str): The segmented sentences from the answer.

        Returns:
            List[str]: A list of extracted statements.
        """
        res = await self.semantic_analysis(
            lm_config={"timeout": 120},
            **{"question": question, "answer": answer, "sentences": sentences},
        )
        res = res["analysis"]
        statements = []
        for analysis in res:
            statements.extend(analysis["simpler_statements"])
        return statements

    async def _compute_precision(
        self, gold: str, prediction_statements: List[str]
    ) -> float:
        """
        Compute the semantic precision of the prediction statements against the gold standard.

        Args:
            gold (str): The gold standard text.
            prediction_statements (List[str]): The list of statements extracted from the prediction.

        Returns:
            float: The computed precision score between 0 and 1.
        """
        res = await self.semantic_precision(
            lm_config={"timeout": 120},
            **{"ground_truth": gold, "statements": prediction_statements},
        )
        answer = res["answer"]
        length = len(answer)
        score = sum([1 for x in answer if x["verdict"] == 1])
        score = score / length if length > 0 else 0
        return score

    async def _compute_recall(self, pred: str, gold_statements: List[str]) -> float:
        """
        Compute the semantic recall of the prediction against the gold standard statements.

        Args:
            pred (str): The prediction text.
            gold_statements (List[str]): The list of statements extracted from the gold standard.

        Returns:
            float: The computed recall score between 0 and 1.
        """
        res = await self.semantic_recall(
            lm_config={"timeout": 120},
            **{"prediction": pred, "ground_truth_statements": gold_statements},
        )
        answer = res["answer"]
        length = len(answer)
        score = sum([1 for x in answer if x["verdict"] == 1])
        score = score / length if length > 0 else 0
        return score
