from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, Optional
import asyncio
from contextvars import ContextVar
from uuid import uuid4


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
        get_context_cost(cls) -> Dict[str, float]: Returns the cost for the current context.
        get_total_cost(cls) -> float: Returns the total accumulated cost.
        get_cost_breakdown(cls) -> Dict[str, float]: Returns the breakdown of costs by category.
        reset(cls) -> None: Resets the cost tracker to its initial state.
        set_context(cls, context_uuid: str) -> None: Sets the context for cost tracking.
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
        self.context_uuid: ContextVar[Optional[str]] = ContextVar("context_uuid", default=None)
        self.cost_queue: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.cost_breakdown: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    @classmethod
    async def add_cost(cls, cost: float, label: str) -> None:
        instance = cls()
        current_context_uuid = instance.context_uuid.get()
        if current_context_uuid is None:
            async with instance._lock:
                instance.total_cost += cost
                instance.cost_breakdown[label] = instance.cost_breakdown.get(label, 0) + cost
        else:
            await instance.cost_queue[current_context_uuid].put((cost, label))

    @classmethod
    def get_context_cost(cls) -> Dict[str, float]:
        instance = cls()
        current_context_uuid = instance.context_uuid.get()
        if current_context_uuid is None:
            return instance.cost_breakdown

        cost_queue = instance.cost_queue[current_context_uuid]
        cost_dict = defaultdict(float)
        while not cost_queue.empty():
            cost, label = cost_queue.get_nowait()
            cost_dict[label] += cost
        # delete queue
        instance.cost_queue.pop(current_context_uuid)

        # instance.cost_breakdown[current_context_uuid] = cost_dict
        # instance.total_cost += sum(cost_dict.values())
        return dict(cost_dict)

    @classmethod
    def get_total_cost(cls) -> float:
        return cls()._instance.total_cost

    @classmethod
    def get_cost_breakdown(cls) -> Dict[str, float]:
        return dict(cls()._instance.cost_breakdown)

    @classmethod
    def reset(cls) -> None:
        cls()._initialize()

    @classmethod
    def delete_context(cls, context_uuid: str) -> None:
        instance = cls()
        instance.cost_queue.pop(context_uuid, None)

    @classmethod
    @asynccontextmanager
    async def set_context(cls, context_uuid: str):
        instance = cls()
        token = instance.context_uuid.set(context_uuid)
        try:
            yield
        finally:
            instance.context_uuid.reset(token)


class CostTrackerContext:
    def __init__(self):
        self.context_uuid = str(uuid4())
        self._cm = None

    async def __aenter__(self):
        self._cm = CostTracker.set_context(self.context_uuid)
        async with self._cm:
            return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._cm.__anext__()  # await the async generator
        CostTracker.delete_context(self.context_uuid)
