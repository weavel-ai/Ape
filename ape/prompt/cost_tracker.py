from typing import Dict, Optional


class CostTracker:
    """
    A singleton class for tracking costs across different categories.

    This class provides methods to add costs, retrieve total cost,
    get a breakdown of costs by category, and reset the tracker.

    Attributes:
        total_cost (float): The total accumulated cost.
        cost_breakdown (Dict[str, float]): A dictionary mapping categories to their respective costs.

    Usage:
        tracker = CostTracker()
        tracker.add_cost(10.5, "API")
        total = tracker.get_total_cost()
        breakdown = tracker.get_cost_breakdown()
    """

    _instance: Optional["CostTracker"] = None

    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def __init__(self) -> None:
        self._initialize()

    def _initialize(self) -> None:
        self.total_cost: float = 0.0
        self.cost_breakdown: Dict[str, float] = {}

    def add_cost(self, cost: float, details: str) -> None:
        self.total_cost += cost
        if details in self.cost_breakdown:
            self.cost_breakdown[details] += cost
        else:
            self.cost_breakdown[details] = cost

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_cost_breakdown(self) -> Dict[str, float]:
        return self.cost_breakdown

    def reset(self) -> None:
        self._initialize()
