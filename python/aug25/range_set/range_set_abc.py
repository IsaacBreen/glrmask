from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, TypeVar

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
    def __len__(self) -> int:
        """Return the total number of elements in the set."""
        raise NotImplementedError

    def __or__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return self.union(other)

    def __ror__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return self.union(other)

    def __ior__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        self.union_update(other)
        return self

    def __iand__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        self.intersection_update(other)
        return self

    def __isub__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        self.difference_update(other)
        return self

    def __and__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return self.intersection(other)

    def __rand__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return self.intersection(other)

    def __sub__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return self.difference(other)

    def __rsub__(self, other: "RangeSet[T]") -> "RangeSet[T]":
        if other is None:
            return self
        return other.difference(self)

    def issubset(self, other: "RangeSet[T]") -> bool:
        """Return True if self is a subset of other."""
        return self.difference(other).is_empty()

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
    def union_update(self, other: "RangeSet[T]") -> None:
        """Update self with the union of self and other."""
        raise NotImplementedError

    @abstractmethod
    def intersection_update(self, other: "RangeSet[T]") -> None:
        """Update self with the intersection of self and other."""
        raise NotImplementedError

    @abstractmethod
    def difference_update(self, other: "RangeSet[T]") -> None:
        """Update self with the set difference self \\ other."""
        raise NotImplementedError

    @abstractmethod
    def difference(self, other: "RangeSet[T]") -> "RangeSet[T]":
        """Return the set difference self \\ other."""
        raise NotImplementedError

    @classmethod
    def union_many(cls, sets: Iterable["RangeSet[T]"]) -> "RangeSet[T]":
        """Return the union of many RangeSets."""
        result: RangeSet[T] = cls.empty()
        for s in sets:
            result = result.union(s)
        return result

    @abstractmethod
    def to_ranges(self) -> List[Tuple[T]]:
        """Returns the intervals as a list of lists for JSON serialization."""
        raise NotImplementedError

    @abstractmethod
    def to_indices(self) -> List[T]:
        """Returns the elements of the set as a list."""
        raise NotImplementedError

    @abstractmethod
    def iter_indices(self) -> Iterable[T]:
        """Iterates over all individual indices in the set."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json(cls, data: List[List[T]]) -> "RangeSet[T]":
        """Creates a RangeSet from a list of [start, end] lists (JSON format)."""
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> List[List[T]]:
        """Converts the RangeSet to a list of [start, end] lists (JSON format)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_ranges(cls, ranges: List[List[T]]) -> "RangeSet[T]":
        """Creates a RangeSet from a list of [start, end] lists."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_indices(cls, indices: Iterable[T]) -> "RangeSet[T]":
        """Creates a RangeSet from an iterable of individual indices."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls) -> "RangeSet[T]":
        """Creates an empty RangeSet."""
        raise NotImplementedError
