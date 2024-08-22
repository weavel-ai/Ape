from typing import Any, Dict, Optional, List, Union
from ape.prompt.prompt_base import Prompt
from .metric_base import BaseMetric
from ape.types import DataItem, DatasetItem

binary_judge = Prompt.from_filename("binary_judge")

class JsonMatchMetric(BaseMetric):
    def __init__(self, consider_list_order: bool = False, ignore_keys: Optional[List[str]] = None):
        self.consider_list_order = consider_list_order  # Flag to determine whether to consider list order
        self.ignore_keys = [key.lower().replace(' ', '_') for key in ignore_keys] if ignore_keys else []  # List of keys to ignore during comparison
        
    async def compute(self, gold: DataItem, pred: Dict[str, Any], trace: Optional[Dict] = None) -> float:
        
        def compare_lists(list1, list2):
            if not list1 and not list2:
                return 1.0
            if not list1 or not list2:
                return 0.0
            
            if self.consider_list_order:
                # If list order should be considered, compare elements by their indices
                return sum(1.0 for a, b in zip(list1, list2) if a == b) / len(list1) if len(list1) == len(list2) else 0.0
            else:
                # If list order should not be considered, compare as sets
                set1 = set(list1)
                set2 = set(list2)
                
                num_identical = len(set1.intersection(set2))
                total_unique = len(set1.union(set2))
                
                return num_identical / total_unique

        async def compare_dicts(dict1, dict2):
            total_fields = 0
            correct_fields = 0
            for key in dict1:
                if key in self.ignore_keys:
                    continue  # Skip keys that should be ignored
                if key in dict2:
                    try:
                        total_fields += 1
                        if isinstance(dict1[key], list):
                            score = compare_lists(dict1[key], dict2[key])
                        elif dict1[key] == dict2[key]:
                            score = 1.0
                        else:
                            # Use LLM judge as a fallback for non-matching fields
                            res: dict = await binary_judge(ground_truth=dict1[key], prediction=dict2[key])
                            score = res.get("score", 0)
                        correct_fields += score
                    except Exception:
                        continue
            return correct_fields / total_fields if total_fields > 0 else 0

        try:
            gold = gold.model_dump() if isinstance(gold, DatasetItem) else gold
            pred = pred.model_dump() if isinstance(pred, DatasetItem) else pred
            
            # Normalize keys
            gold_dict = {k.lower().replace(' ', '_'): v for k, v in gold.items()}
            pred_dict = {k.lower().replace(' ', '_'): v for k, v in pred.items()}
            
            accuracy = await compare_dicts(gold_dict, pred_dict)
            return accuracy

        except Exception as e:
            return 0
