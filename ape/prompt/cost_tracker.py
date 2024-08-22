from typing import Dict, Optional


class CostTracker:
    """
    A singleton class for tracking costs across different categories.

    This class provides methods to add costs, retrieve total cost,
    get a breakdown of costs by category, and reset the tracker.
    It ensures that only one instance of the cost tracker exists throughout the application.

    Attributes:
        _instance (Optional[CostTracker]): The single instance of the CostTracker class.
        total_cost (float): The total accumulated cost across all categories.
        cost_breakdown (Dict[str, float]): A dictionary mapping category labels to their respective costs.

    Methods:
        __new__(cls) -> CostTracker: Ensures only one instance of CostTracker is created.
        __init__(self) -> None: Initializes the CostTracker instance.
        _initialize(self) -> None: Initializes or resets the cost tracking attributes.
        add_cost(cls, cost: float, label: str) -> None: Adds a cost to a specific category.
        get_total_cost(cls) -> float: Returns the total accumulated cost.
        get_cost_breakdown(cls) -> Dict[str, float]: Returns the breakdown of costs by category.
        reset(cls) -> None: Resets the cost tracker to its initial state.
    """

    _instance: Optional["CostTracker"] = None

    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        self.total_cost: float = 0.0
        self.cost_breakdown: Dict[str, float] = {}

    @classmethod
    def add_cost(cls, cost: float, label: str) -> None:
        instance = cls()
        instance.total_cost += cost
        instance.cost_breakdown[label] = instance.cost_breakdown.get(label, 0) + cost

    @classmethod
    def get_total_cost(cls) -> float:
        return cls()._instance.total_cost

    @classmethod
    def get_cost_breakdown(cls) -> Dict[str, float]:
        return cls()._instance.cost_breakdown

    @classmethod
    def reset(cls) -> None:
        cls()._initialize()
