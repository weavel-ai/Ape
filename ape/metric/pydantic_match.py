from ape.prompt.prompt_base import Prompt
from .metric_base import BaseMetric
from ape.types import DataItem, DatasetItem
from typing import Any, Dict, Optional, List
import json  # Module for JSON serialization
from pydantic import BaseModel  # Import for Pydantic model handling


class PydanticMatchMetric(BaseMetric):
    """
    A metric class that computes the similarity between two Pydantic models or JSON-like structures.

    This metric compares two dictionaries, lists, or Pydantic models, considering the structure and content
    of nested objects. It can optionally consider list order and ignore specified keys.

    Attributes:
        consider_list_order (bool): If True, list order is considered in comparisons.
            Defaults to False.
        ignore_keys (List[str]): A list of keys to ignore during comparison.
            Keys are normalized to lowercase with spaces replaced by underscores.

    Methods:
        compute(gold: DataItem, pred: Any, trace: Optional[Dict] = None) -> float:
            Computes the similarity score between the gold standard and prediction.

    The similarity score is calculated as the ratio of correctly matched fields to total fields,
    with nested structures being recursively compared. For non-exact matches, it falls back to
    using an LLM-based binary judge for scoring.
    """

    def __init__(
        self, consider_list_order: bool = False, ignore_keys: Optional[List[str]] = None
    ):
        self.binary_judge = Prompt.from_filename("binary_judge")
        self.consider_list_order = (
            consider_list_order  # Flag to determine whether to consider list order
        )
        self.ignore_keys = (
            [key.lower().replace(" ", "_") for key in ignore_keys]
            if ignore_keys
            else []
        )  # List of keys to ignore during comparison

    async def compute(
        self, inputs, gold: DataItem, pred: Any, trace: Optional[Dict] = None
    ) -> float:
        """
        Compute the similarity score between the gold standard and prediction.

        Args:
            gold (DataItem): The gold standard data item.
            pred (Any): The prediction to compare against the gold standard.
            trace (Optional[Dict]): Additional trace information (not used in this implementation).

        Returns:
            float: The computed similarity score between 0 and 1.

        This method normalizes both inputs, compares their structures recursively,
        and returns a float representing the overall similarity. It handles nested
        dictionaries and lists, applying special comparison logic based on the
        `consider_list_order` attribute.
        """

        async def compare_values(gold_value, pred_value):
            """Recursively compares values based on their types."""
            if isinstance(gold_value, dict) and isinstance(pred_value, dict):
                return await compare_dicts(gold_value, pred_value)
            elif isinstance(gold_value, list) and isinstance(pred_value, list):
                return await compare_lists(gold_value, pred_value)
            elif gold_value == pred_value:
                return 1.0
            else:
                res: dict = await self.binary_judge(
                    ground_truth=gold_value, prediction=pred_value
                )
                return res.get("score", 0)

        async def compare_dicts(dict1, dict2):
            """Compares two dictionaries while ignoring specified keys."""
            total_fields = 0
            correct_fields = 0
            for key in dict1:
                if key in self.ignore_keys:
                    continue  # Skip keys that should be ignored
                if key in dict2:
                    try:
                        total_fields += 1
                        score = await compare_values(dict1[key], dict2[key])
                        correct_fields += score
                    except Exception as e:
                        continue
            return correct_fields / total_fields if total_fields > 0 else 0

        async def compare_lists(list1, list2):
            """Compares two lists, considering order if specified."""
            if not list1 and not list2:
                return 1.0
            if not list1 or not list2:
                return 0.0

            if not self.consider_list_order:
                # Separate dictionaries from other elements
                dicts1 = set(
                    json.dumps(item, sort_keys=True)
                    for item in list1
                    if isinstance(item, dict)
                )
                dicts2 = set(
                    json.dumps(item, sort_keys=True)
                    for item in list2
                    if isinstance(item, dict)
                )

                non_dicts1 = set(item for item in list1 if not isinstance(item, dict))
                non_dicts2 = set(item for item in list2 if not isinstance(item, dict))

                # Combine sets of dictionaries and non-dictionaries
                set1 = dicts1 | non_dicts1
                set2 = dicts2 | non_dicts2

                num_identical = len(set1.intersection(set2))
                total_unique = len(set1.union(set2))

                return num_identical / total_unique

            else:
                # When considering list order, compare elements by their indices
                return (
                    sum(await compare_values(g1, g2) for g1, g2 in zip(list1, list2))
                    / len(list1)
                    if len(list1) == len(list2)
                    else 0.0
                )

        try:
            # Handle the possibility that gold or pred could be Pydantic models
            if isinstance(gold, DatasetItem) or isinstance(gold, BaseModel):
                gold = gold.model_dump()
            if isinstance(pred, BaseModel):
                pred = pred.model_dump()

            # Normalize keys
            gold_dict = {k.lower().replace(" ", "_"): v for k, v in gold.items()}
            pred_dict = {k.lower().replace(" ", "_"): v for k, v in pred.items()}

            accuracy = await compare_dicts(gold_dict, pred_dict)
            return accuracy

        except Exception as e:
            return 0
