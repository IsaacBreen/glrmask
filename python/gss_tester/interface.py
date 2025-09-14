from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Callable, Set, Iterable, Dict, Any
import json

T = TypeVar('T')  # Stack item type
Acc = TypeVar('Acc') # Accumulator type

class GSS(ABC, Generic[T, Acc]):
    """Abstract Base Class for a Graph-Structured Stack (GSS)."""

    @classmethod
    @abstractmethod
    def initial(cls, acc_default_factory: Callable[[], Acc]) -> 'GSS[T, Acc]':
        """Creates an initial GSS, typically with one empty stack."""
        pass

    @abstractmethod
    def push(self, value: T) -> 'GSS[T, Acc]':
        """Pushes a value onto all active stack heads, returning a new GSS state."""
        pass

    @abstractmethod
    def pop(self, value: T) -> 'GSS[T, Acc]':
        """
        For all stacks ending in `value`, creates new stacks by removing that value.
        Returns a new GSS state containing only these popped stacks.
        """
        pass

    @abstractmethod
    def apply(self, func: Callable[[Acc], Acc]) -> 'GSS[T, Acc]':
        """Applies a function to each accumulator, returning a new GSS state."""
        pass

    @abstractmethod
    def peek(self) -> Set[T]:
        """Returns the set of all values at the top of any stack."""
        pass

    @staticmethod
    @abstractmethod
    def merge(gss_list: Iterable['GSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'GSS[T, Acc]':
        """Merges multiple GSS instances into one, combining accumulators for identical stacks."""
        pass

    @abstractmethod
    def to_json_serializable(self) -> Any:
        """Returns a JSON-serializable representation of the GSS state for comparison."""
        pass

    def __str__(self) -> str:
        """Provides a default string representation for debugging."""
        try:
            data = self.to_json_serializable()
            return json.dumps(data, indent=2, sort_keys=True)
        except Exception:
            return super().__str__()

    def __eq__(self, other):
        """Defines equality based on the canonical JSON representation."""
        if not isinstance(other, GSS):
            return NotImplemented
        return self.to_json_serializable() == other.to_json_serializable()
