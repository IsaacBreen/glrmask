from __future__ import annotations
from typing import Iterable, Optional, Tuple, List, Set

from .range_set_abc import RangeSet

class SetRangeSet(RangeSet[int]):
    """
    A RangeSet implementation backed by a standard Python set.
    This implementation is simple but can be inefficient in terms of memory
    and performance for large, dense ranges.
    """
    __slots__ = ('_elements',)

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        self._elements: Set[int] = set()
        if intervals:
            for start, end in intervals:
                self._elements.update(range(start, end + 1))

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        """
        Reconstructs intervals from the set of elements. This can be slow.
        """
        if not self._elements:
            return tuple()
        
        indices_sorted = sorted(self._elements)
        intervals_list: List[Tuple[int, int]] = []
        start = indices_sorted[0]
        prev = start
        for i in indices_sorted[1:]:
            if i == prev + 1:
                prev = i
            else:
                intervals_list.append((start, prev))
                start = i
                prev = i
        intervals_list.append((start, prev))
        return tuple(intervals_list)

    def to_ranges(self) -> List[Tuple[int]]:
        return list(self.intervals)

    def to_indices(self) -> List[int]:
        return sorted(self._elements)

    def iter_indices(self) -> Iterable[int]:
        """Iterates over all individual indices in the set, in sorted order."""
        yield from sorted(self._elements)

    def iter_ranges(self) -> Iterable[Tuple[int, int]]:
        """Iterates over all [start, end] intervals in the set."""
        yield from self.intervals

    def contains(self, x: int) -> bool:
        return x in self._elements

    def _get_other_elements(self, other: RangeSet[int]) -> Set[int]:
        if isinstance(other, SetRangeSet):
            return other._elements
        # This can be very slow if other contains large ranges
        return set(other.iter_indices())

    def _new_from_set(self, elements: Set[int]) -> "SetRangeSet":
        new_set = SetRangeSet()
        new_set._elements = elements
        return new_set

    def union(self, other: RangeSet[int]) -> "SetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._new_from_set(self._elements.union(other_elements))

    def intersection(self, other: RangeSet[int]) -> "SetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._new_from_set(self._elements.intersection(other_elements))

    def difference(self, other: RangeSet[int]) -> "SetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._new_from_set(self._elements.difference(other_elements))

    def issuperset(self, other: RangeSet[int]) -> bool:
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._elements.issuperset(other_elements)

    def issubset(self, other: RangeSet[int]) -> bool:
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._elements.issubset(other_elements)

    def isdisjoint(self, other: RangeSet[int]) -> bool:
        if not isinstance(other, RangeSet):
            return NotImplemented
        other_elements = self._get_other_elements(other)
        return self._elements.isdisjoint(other_elements)

    def union_update(self, other: RangeSet[int]) -> None:
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        self._elements.update(self._get_other_elements(other))

    def intersection_update(self, other: RangeSet[int]) -> None:
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        self._elements.intersection_update(self._get_other_elements(other))

    def difference_update(self, other: RangeSet[int]) -> None:
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        self._elements.difference_update(self._get_other_elements(other))

    def is_empty(self) -> bool:
        return not self._elements

    def __len__(self) -> int:
        return len(self._elements)

    def __repr__(self) -> str:
        return f"SetRangeSet({self.intervals!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, SetRangeSet):
            return self._elements == other._elements
        if isinstance(other, RangeSet):
            # Fallback for other RangeSet types
            return self._elements == set(other.iter_indices())
        return NotImplemented

    def __hash__(self) -> int:
        # A set is not hashable. We must use something immutable.
        # Hashing intervals is consistent with other implementations.
        return hash(self.intervals)

    @classmethod
    def from_ranges(cls, ranges: List[List[int]]) -> 'SetRangeSet':
        return cls(iter(map(tuple, ranges)))

    @classmethod
    def from_indices(cls, indices: Iterable[int]) -> 'SetRangeSet':
        new_set = cls()
        new_set._elements = set(indices)
        return new_set

    @classmethod
    def empty(cls) -> 'SetRangeSet':
        return cls()

    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'SetRangeSet':
        return cls.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return [list(r) for r in self.intervals]
