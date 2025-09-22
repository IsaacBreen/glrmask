from __future__ import annotations
from abc import ABC, abstractmethod
from functools import reduce
from typing import TypeVar, Generic, List, Tuple, Callable, Set, Iterable, Optional, Protocol, Type, Dict


class Mergeable(Protocol):
    """Protocol for accumulator types that can be merged."""
    @abstractmethod
    def merge(self, other: 'Mergeable') -> 'Mergeable':
        ...

class MergeableInt(int):
    """
    An integer that is mergeable (for testing `Acc` typevars) and
    returns itself from arithmetic operations to satisfy `Callable[[Acc], Acc]`.
    """
    def merge(self, other: 'MergeableInt') -> 'MergeableInt':
        return MergeableInt(super().__add__(other))

    def __add__(self, other: int) -> 'MergeableInt':
        if isinstance(other, int):
            return MergeableInt(super().__add__(other))
        return NotImplemented

T = TypeVar('T')  # Stack item type
Acc = TypeVar('Acc', bound=Mergeable) # Accumulator type
NewAcc = TypeVar('NewAcc', bound=Mergeable) # New accumulator type for apply/apply_and_prune
GSSType = TypeVar("GSSType", bound="GSS[Any, Any]")

class GSS(ABC, Generic[T, Acc]):
    """Abstract Base Class for a Graph-Structured Stack (GSS)."""

    @classmethod
    def empty(cls: Type[GSSType]) -> GSSType:
        """Creates an empty GSS with no active stacks."""
        return cls.from_stacks([])

    @classmethod
    @abstractmethod
    def from_stacks(cls: Type[GSSType], stacks: List[Tuple[List[T], Acc]]) -> GSSType:
        """Creates a GSS from a list of explicit stacks."""
        pass

    @abstractmethod
    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Converts the GSS to its canonical representation as a sorted list of stacks.
        This involves merging accumulators for identical stacks and sorting.
        """
        pass

    @abstractmethod
    def push(self: GSSType, value: T) -> GSSType:
        """Pushes a value onto all active stack heads, returning a new GSS state."""
        pass

    @classmethod
    def push_many(cls: Type[GSSType], items: Iterable[Tuple[GSSType, T]]) -> GSSType:
        """Pushes multiple values onto all active stack heads, returning a new GSS state."""
        dest = cls.empty()
        for gss_item, value in items:
            dest = dest.merge(gss_item.push(value))
        return dest

    @abstractmethod
    def pop(self: GSSType) -> GSSType:
        """
        For all active stacks, creates new stacks by removing the top value.
        Returns a new GSS state containing the popped stacks.
        """
        pass

    def popn(self: GSSType, n: int) -> GSSType:
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
        """Checks if the GSS contains no active stacks."""
        pass

    @abstractmethod
    def isolate(self: GSSType, value: Optional[T]) -> GSSType:
        """
        Keeps only the stacks that have `value` at the top.
        If `value` is None, it keeps only the empty stacks.
        Returns a new GSS state containing only these stacks.
        """
        pass

    def isolate_many(self: GSSType, values: Iterable[Optional[T]]) -> GSSType:
        """
        Keeps only the stacks that have any of the `values` at the top.
        If `None` is in `values`, it keeps only the empty stacks as well.
        Returns a new GSS state containing only these stacks.
        """
        dest = self.empty()
        for v in values:
            dest = dest.merge(self.isolate(v))
        return dest

    @abstractmethod
    def apply(self, func: Callable[[Acc], NewAcc]) -> GSS[T, NewAcc]:
        """Applies a function to each accumulator, returning a new GSS state."""
        pass

    @abstractmethod
    def prune(self: GSSType, predicate: Callable[[Acc], bool]) -> GSSType:
        """
        Removes stacks from the GSS based on a predicate on their accumulator.
        If `predicate(acc)` returns False, the stack is removed.
        """
        pass

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]]) -> GSS[T, NewAcc]:
        """
        Combined transform that applies a mutate-and-prune function to each stack's accumulator.
        The `mutator` must return:
          - a new (or unchanged) accumulator to keep the stack, or
          - None to prune the stack.

        Default implementation composes prune + apply, with memoization to avoid
        invoking `mutator` more than once per accumulator instance when possible.
        Concrete implementations can override for a faster single-pass transform.
        """
        cache: Dict[int, Optional[NewAcc]] = {}

        def decide(a: Acc) -> Optional[NewAcc]:
            k = id(a)
            if k in cache:
                return cache[k]
            r = mutator(a)
            cache[k] = r
            return r

        pruned = self.prune(lambda a: decide(a) is not None)
        def map_acc(a: Acc) -> NewAcc:
            r = decide(a)
            if r is None:
                raise AssertionError("This should not be reached if prune worked correctly")
            return r
        return pruned.apply(map_acc)
    @abstractmethod
    def merge(self: GSSType, other: GSSType) -> GSSType:
        """Merges this GSS with another, combining accumulators for identical stacks."""
        pass

    @classmethod
    def merge_many(cls: Type[GSSType], gss_list: Iterable[GSSType]) -> GSSType:
        """
        Merges multiple GSS instances into one.
        This default implementation uses functools.reduce with the instance `merge` method.
        """
        # Start with an empty GSS of the target class `cls` to ensure the correct
        # return type and to handle an empty gss_list.
        initial = cls.empty()
        return reduce(lambda acc_gss, next_gss: acc_gss.merge(next_gss), gss_list, initial)

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

    def to_reference_impl(self) -> 'GSS[T, Acc]':
        """
        Converts the GSS to its canonical ReferenceGSS representation.
        This involves merging accumulators for identical stacks.
        """
        from .implementations.reference_impl import ReferenceGSS
        return ReferenceGSS.from_stacks(self.to_stacks())

    def __str__(self) -> str:
        """Provides a human-readable string representation for debugging."""
        return f"{self.__class__.__name__}({self.to_stacks()})"

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the GSS."""
        return f"{self.__class__.__name__}({self.to_stacks()!r})"
