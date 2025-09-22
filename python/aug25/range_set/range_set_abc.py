from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

T = TypeVar("T")


class RangeSet(ABC, Generic[T]):
    """
    Abstract base class for range-based sets of comparable elements.
    Implementations represent sets as normalized, sorted, disjoint closed intervals.
    """

    @property
    @abstractmethod
    def intervals(self) -> Tuple[Tuple[T, T], ...]:
        """
        The normalized, sorted, disjoint closed intervals representing the set.
        Implementations should guarantee:
        - intervals are sorted by start
        - no overlaps and no gaps of size 0 (i.e., [a,b] and [b+1,c] should be merged where applicable)
        """
        raise NotImplementedError

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if the set is empty."""
        raise NotImplementedError

    @abstractmethod
    def contains(self, x: T) -> bool:
        """Return True if x is contained in the set."""
        raise NotImplementedError

    @abstractmethod
    def union(self, other: "RangeSet[T]") -> "RangeSet[T]":
        """Return the union of two RangeSets."""
        raise NotImplementedError

    @abstractmethod
    def intersection(self, other: "RangeSet[T]") -> "RangeSet[T]":
        """Return the intersection of two RangeSets."""
        raise NotImplementedError

    @abstractmethod
    def difference(self, other: "RangeSet[T]") -> "RangeSet[T]":
        """Return the set difference self \\ other."""
        raise NotImplementedError
