from __future__ import annotations
from typing import Iterable, Optional, Tuple, List

try:
    from pyroaring import BitMap as RoaringBitmap
except ImportError:
    raise ImportError("RoaringRangeSet requires the 'roaringbitmap' library. Please install it with 'pip install pyroaring'.")

from .range_set_abc import RangeSet


class RoaringRangeSet(RangeSet[int]):
    """
    A RangeSet implementation backed by the 'roaringbitmap' library.
    This implementation is very fast and memory-efficient, especially for sparse sets.
    It only supports non-negative integers.
    """
    __slots__ = ('_rb',)

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        self._rb = RoaringBitmap()
        if intervals:
            for start, end in intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                # RoaringBitmap.add_range is [start, end)
                self._rb.add_range(start, end + 1)

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        # This is not the most efficient way, but it's correct.
        # It leverages the logic from other implementations.
        indices = list(self._rb)
        if not indices:
            return tuple()
        
        intervals: List[Tuple[int, int]] = []
        start = indices[0]
        prev = start
        for i in indices[1:]:
            if i == prev + 1:
                prev = i
            else:
                intervals.append((start, prev))
                start = i
                prev = i
        intervals.append((start, prev))
        return tuple(intervals)

    def to_ranges(self) -> List[Tuple[int]]:
        return list(self.intervals)

    def to_indices(self) -> List[int]:
        return list(self._rb)

    def iter_indices(self) -> Iterable[int]:
        """Iterates over all individual indices in the set."""
        yield from self._rb

    def iter_ranges(self) -> Iterable[Tuple[int, int]]:
        """Iterates over all [start, end] intervals in the set."""
        yield from self.intervals

    def contains(self, x: int) -> bool:
        if x < 0:
            return False
        return x in self._rb

    def _new_from_rb(self, rb: RoaringBitmap) -> "RoaringRangeSet":
        new_set = RoaringRangeSet()
        new_set._rb = rb
        return new_set

    def union(self, other: RangeSet[int]) -> "RoaringRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        if isinstance(other, RoaringRangeSet):
            return self._new_from_rb(self._rb | other._rb)
        else:
            # Generic path
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            return self._new_from_rb(self._rb | other_rb)

    def intersection(self, other: RangeSet[int]) -> "RoaringRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        if isinstance(other, RoaringRangeSet):
            return self._new_from_rb(self._rb & other._rb)
        else:
            # Generic path
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            return self._new_from_rb(self._rb & other_rb)

    def difference(self, other: RangeSet[int]) -> "RoaringRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        if isinstance(other, RoaringRangeSet):
            return self._new_from_rb(self._rb - other._rb)
        else:
            # Generic path
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            return self._new_from_rb(self._rb - other_rb)

    def issuperset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a superset of other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, RoaringRangeSet):
            return self._rb.issuperset(other._rb)
        else:
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            return self._rb.issuperset(other_rb)

    def issubset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a subset of other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, RoaringRangeSet):
            return self._rb.issubset(other._rb)
        else:
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            return self._rb.issubset(other_rb)

    def isdisjoint(self, other: RangeSet[int]) -> bool:
        """Return True if self has no elements in common with other."""
        if not isinstance(other, RangeSet):
            return NotImplemented

        if isinstance(other, RoaringRangeSet):
            return self._rb.isdisjoint(other._rb)
        else:
            other_rb = RoaringBitmap(other.iter_indices())
            return self._rb.isdisjoint(other_rb)

    def union_update(self, other: RangeSet[int]) -> None:
        """Update self with the union of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, RoaringRangeSet):
            self._rb |= other._rb
        else:
            # Generic path
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            self._rb |= other_rb

    def intersection_update(self, other: RangeSet[int]) -> None:
        """Update self with the intersection of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, RoaringRangeSet):
            self._rb &= other._rb
        else:
            other_rb = RoaringBitmap()
            for start, end in other.intervals:
                if start < 0:
                    raise ValueError("RoaringRangeSet only supports non-negative integers.")
                other_rb.add_range(start, end + 1)
            self._rb &= other_rb

    def difference_update(self, other: RangeSet[int]) -> None:
        """Update self with the set difference self \\ other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, RoaringRangeSet):
            self._rb -= other._rb
        else:
            other_rb = RoaringBitmap(other.iter_indices())
            self._rb -= other_rb

    def is_empty(self) -> bool:
        return len(self._rb) == 0

    def __len__(self) -> int:
        return len(self._rb)

    def __repr__(self) -> str:
        return f"RoaringRangeSet({self.intervals!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, RoaringRangeSet):
            return self._rb == other._rb
        if isinstance(other, RangeSet):
            return self.intervals == other.intervals
        return NotImplemented

    def __hash__(self) -> int:
        # RoaringBitmap is not hashable.
        return hash(self.intervals)

    @classmethod
    def from_ranges(cls, ranges: List[List[int]]) -> 'RoaringRangeSet':
        return cls(iter(map(tuple, ranges)))

    @classmethod
    def from_indices(cls, indices: Iterable[int]) -> 'RoaringRangeSet':
        if any(i < 0 for i in indices):
            raise ValueError("RoaringRangeSet only supports non-negative integers.")
        
        new_set = cls()
        new_set._rb = RoaringBitmap(indices)
        return new_set

    @classmethod
    def empty(cls) -> 'RoaringRangeSet':
        return cls()

    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'RoaringRangeSet':
        return cls.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return [list(r) for r in self.intervals]
