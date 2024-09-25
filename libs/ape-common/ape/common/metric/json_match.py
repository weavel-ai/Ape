from typing import Any, Dict, Optional, List
from ape.common.metric import BaseMetric
from ape.common.types import MetricResult
from ape.common.utils import logger
from ape.common.metric_prompts import ApeMetricPrompts


class JsonMatchMetric(BaseMetric):
    """
    A metric class that computes the similarity between two JSON-like structures.

    This metric compares two dictionaries or lists, considering the structure and content
    of nested objects. It can optionally consider list order and ignore specified keys.

    Attributes:
        consider_list_order (bool): If True, list order is considered in comparisons.
            Defaults to False.
        ignore_keys (List[str]): A list of keys to ignore during comparison.
            Keys are normalized to lowercase with spaces replaced by underscores.
        binary_judge (Callable): An LLM-based binary judge for scoring non-exact matches.

    Methods:
        compute(pred: Dict[str, Any], gold: Dict[str, Any], inputs: Dict[str, Any] = {},
                trace: Optional[Dict] = None, metadata: Optional[Dict] = None) -> MetricResult:
            Computes the similarity score between the gold standard and prediction.

    The similarity score is calculated as the ratio of correctly matched fields to total fields,
    with nested structures being recursively compared. For non-exact matches, it falls back to
    using an LLM-based binary judge for scoring.
    """

    def __init__(self, consider_list_order: bool = False, ignore_keys: Optional[List[str]] = None):
        self.binary_judge = ApeMetricPrompts.get("binary-judge")
        self.consider_list_order = consider_list_order
        self.ignore_keys = (
            [key.lower().replace(" ", "_") for key in ignore_keys] if ignore_keys else []
        )

    async def compute(
        self,
        pred: Dict[str, Any],
        gold: Dict[str, Any],
        inputs: Dict[str, Any] = {},
        trace: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> MetricResult:
        """
        Compute the similarity score between the gold standard and prediction.

        Args:
            pred (Dict[str, Any]): The prediction to compare against the gold standard.
            gold (Dict[str, Any]): The gold standard data item.
            inputs (Dict[str, Any]): Additional input information (not used in this implementation).
            trace (Optional[Dict]): Additional trace information (not used in this implementation).
            metadata (Optional[Dict]): Additional metadata (not used in this implementation).

        Returns:
            MetricResult: The computed similarity score and any intermediate values.

        This method normalizes both inputs, compares their structures recursively,
        and returns a MetricResult representing the overall similarity. It handles nested
        dictionaries and lists, applying special comparison logic based on the
        `consider_list_order` attribute.
        """

        async def compare_lists(list1, list2):
            """
            Compare two lists and return a similarity score.

            Args:
                list1 (List): The first list to compare.
                list2 (List): The second list to compare.

            Returns:
                float: A similarity score between 0 and 1.
            """
            if not list1 and not list2:
                return 1.0
            if not list1 or not list2:
                return 0.0

            if self.consider_list_order:
                if len(list1) != len(list2):
                    return 0.0
                total_score = 0.0
                for a, b in zip(list1, list2):
                    if isinstance(a, dict) and isinstance(b, dict):
                        score = await compare_dicts(a, b)
                    else:
                        score = 1.0 if a == b else 0.0
                    total_score += score
                return total_score / len(list1)
            else:
                # For unordered lists, handle dictionaries by matching each dict in list1 to the best match in list2
                if any(isinstance(item, dict) for item in list1 + list2):
                    matched_indices = set()
                    total_score = 0.0
                    for a in list1:
                        best_score = 0.0
                        best_idx = -1
                        for idx, b in enumerate(list2):
                            if idx in matched_indices:
                                continue
                            if isinstance(a, dict) and isinstance(b, dict):
                                score = await compare_dicts(a, b)
                            else:
                                score = 1.0 if a == b else 0.0
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        if best_score > 0.0:
                            matched_indices.add(best_idx)
                        total_score += best_score
                    total_score += (
                        len(list2) - len(matched_indices)
                    ) * 0.0  # Unmatched items contribute 0
                    total_items = max(len(list1), len(list2))
                    return total_score / total_items if total_items > 0 else 0.0
                else:
                    # If no dictionaries, proceed with set comparison
                    set1 = set(list1)
                    set2 = set(list2)

                    num_identical = len(set1.intersection(set2))
                    total_unique = len(set1.union(set2))

                    return num_identical / total_unique

        async def compare_dicts(dict1, dict2):
            """
            Compare two dictionaries and return a similarity score.

            Args:
                dict1 (Dict): The first dictionary to compare.
                dict2 (Dict): The second dictionary to compare.

            Returns:
                float: A similarity score between 0 and 1.
            """
            total_fields = 0
            correct_fields = 0
            for key in dict1:
                if key in self.ignore_keys:
                    continue  # Skip keys that should be ignored
                if key in dict2:
                    try:
                        total_fields += 1
                        if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                            score = await compare_lists(dict1[key], dict2[key])
                        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                            score = await compare_dicts(dict1[key], dict2[key])
                        elif dict1[key] == dict2[key]:
                            score = 1.0
                        else:
                            # Use LLM judge as a fallback for non-matching fields
                            res: dict = await self.binary_judge(
                                ground_truth=dict1[key], prediction=dict2[key]
                            )
                            score = res.get("score", 0)
                        correct_fields += score
                    except Exception as err:
                        logger.error(f"Error in JsonMatchMetric compare_dicts: {err}")
                        continue
            return correct_fields / total_fields if total_fields > 0 else 0

        try:
            # Normalize keys
            gold_dict = {k.lower().replace(" ", "_"): v for k, v in gold.items()}
            pred_dict = {k.lower().replace(" ", "_"): v for k, v in pred.items()}

            accuracy = await compare_dicts(gold_dict, pred_dict)
            return MetricResult(
                score=accuracy,
            )

        except Exception as e:
            logger.error(f"Error in JsonMatchMetric: {e}")
            return MetricResult(score=0.0, intermediate_values={"error": str(e)})
