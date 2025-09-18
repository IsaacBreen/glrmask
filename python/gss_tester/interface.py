from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Callable, Set, Iterable, Dict, Any, Optional, Protocol
import json

class Mergeable(Protocol):
    """Protocol for accumulator types that can be merged."""
    @abstractmethod
    def merge(self, other: 'Mergeable') -> 'Mergeable':
        ...

T = TypeVar('T')  # Stack item type
Acc = TypeVar('Acc', bound=Mergeable) # Accumulator type

class GSS(ABC, Generic[T, Acc]):
    """Abstract Base Class for a Graph-Structured Stack (GSS)."""

    @classmethod
    @abstractmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> 'GSS[T, Acc]':
        """Creates a GSS from a list of explicit stacks."""
        pass

    @abstractmethod
    def push(self, value: T) -> 'GSS[T, Acc]':
        """Pushes a value onto all active stack heads, returning a new GSS state."""
        pass

    @abstractmethod
    def pop(self) -> 'GSS[T, Acc]':
        """
        For all active stacks, creates new stacks by removing the top value.
        Returns a new GSS state containing the popped stacks.
        """
        pass

    def popn(self, n: int) -> 'GSS[T, Acc]':
        """
        For all active stacks, creates new stacks by removing the top `n` values.
        Returns a new GSS state containing the popped stacks.
        """
        gss = self
        for _ in range(n):
            gss = gss.pop()
        return gss

    @abstractmethod
    def is_empty(self) -> bool:
        """Checks if the GSS contains only the initial empty stack."""
        pass

    @abstractmethod
    def isolate(self, value: Optional[T]) -> 'GSS[T, Acc]':
        """
        Keeps only the stacks that have `value` at the top.
        If `value` is None, it keeps only the empty stacks.
        Returns a new GSS state containing only these stacks.
        """
        pass

    @abstractmethod
    def apply(self, func: Callable[[Acc], Acc]) -> 'GSS[T, Acc]':
        """Applies a function to each accumulator, returning a new GSS state."""
        pass

    @abstractmethod
    def prune(self, predicate: Callable[[Acc], bool]) -> 'GSS[T, Acc]':
        """
        Removes stacks from the GSS based on a predicate on their accumulator.
        If `predicate(acc)` returns False, the stack is removed.
        """
        pass

    @abstractmethod
    def peek(self) -> Set[T]:
        """Returns the set of all values at the top of any stack."""
        pass

    @abstractmethod
    def reduce_acc(self) -> Optional[Acc]:
        """
        Merges the accumulators of all active stacks into a single optional value.
        Returns None if there are no active stacks.
        """
        pass

    @abstractmethod
    def to_reference_impl(self) -> 'GSS[T, Acc]':
        """
        Converts the GSS to its canonical ReferenceGSS representation.
        This involves merging accumulators for identical stacks.
        """
        pass

    @staticmethod
    @abstractmethod
    def merge(gss_list: Iterable['GSS[T, Acc]']) -> 'GSS[T, Acc]':
        """Merges multiple GSS instances into one, combining accumulators for identical stacks."""
        pass

    def to_json_serializable(self) -> Any:
        """Returns a JSON-serializable representation of the GSS state for comparison."""
        # to_reference_impl is expected to return a ReferenceGSS instance.
        ref_impl = self.to_reference_impl()
        # We can now call the ReferenceGSS's specific implementation.
        return ref_impl.to_json_serializable()

    def __str__(self) -> str:
        """Provides a default string representation for debugging."""
        # The canonical JSON representation requires a merge function, so it cannot be
        # used for a simple string conversion.
        return super().__str__()
