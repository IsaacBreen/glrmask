from __future__ import annotations
from typing import Iterable, Optional, Tuple, List

try:
    from ranges import Range, RangeSet as ExternalRangeSet
except ImportError:
    raise ImportError("FastPyRangeSet requires the 'python-ranges' library. Please install it with 'pip install python-ranges'.")

from .range_set_abc import RangeSet


class FastPyRangeSet(RangeSet[int]):
    """
    A RangeSet implementation backed by the 'python-ranges' library.
    This implementation is generally fast for a wide variety of use cases.
    """
    __slots__ = ('_rs',)

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        if intervals is None:
            self._rs = ExternalRangeSet()
            return

        # python-ranges uses [start, end) ranges. Our API uses [start, end] (inclusive).
        # So we need to convert: [s, e] -> Range(s, e + 1).
        ranges = [Range(start, end + 1) for start, end in intervals if start <= end]
        self._rs = ExternalRangeSet(ranges)

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        # Convert back from [start, end) to [start, end]
        return tuple((r.start, r.end - 1) for r in self._rs.ranges())

    def to_ranges(self) -> List[Tuple[int]]:
        return list(self.intervals)

    def to_indices(self) -> List[int]:
        indices = []
        for start, end in self.intervals:
            indices.extend(range(start, end + 1))
        return indices

    def contains(self, x: int) -> bool:
        return x in self._rs

    def union(self, other: RangeSet[int]) -> "FastPyRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        new_set = FastPyRangeSet()
        if isinstance(other, FastPyRangeSet):
            new_set._rs = self._rs.union(other._rs)
        else:
            # Generic path for other RangeSet types
            other_as_fast = FastPyRangeSet(other.intervals)
            new_set._rs = self._rs.union(other_as_fast._rs)
        return new_set

    def intersection(self, other: RangeSet[int]) -> "FastPyRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        new_set = FastPyRangeSet()
        if isinstance(other, FastPyRangeSet):
            new_set._rs = self._rs.intersection(other._rs)
        else:
            other_as_fast = FastPyRangeSet(other.intervals)
            new_set._rs = self._rs.intersection(other_as_fast._rs)
        return new_set

    def difference(self, other: RangeSet[int]) -> "FastPyRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented

        new_set = FastPyRangeSet()
        if isinstance(other, FastPyRangeSet):
            new_set._rs = self._rs.difference(other._rs)
        else:
            other_as_fast = FastPyRangeSet(other.intervals)
            new_set._rs = self._rs.difference(other_as_fast._rs)
        return new_set

    def is_empty(self) -> bool:
        return self._rs.is_empty()

    def __len__(self) -> int:
        return self._rs.len()

    def __repr__(self) -> str:
        return f"FastPyRangeSet({self.intervals!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, FastPyRangeSet):
            return self._rs == other._rs
        if isinstance(other, RangeSet):
            return self.intervals == other.intervals
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.intervals)

    @classmethod
    def from_ranges(cls, ranges: List[List[int]]) -> 'FastPyRangeSet':
        return cls(iter(map(tuple, ranges)))

    @classmethod
    def from_indices(cls, indices: Iterable[int]) -> 'FastPyRangeSet':
        indices_sorted = sorted(set(indices))
        if not indices_sorted:
            return cls.empty()
        intervals: List[Tuple[int, int]] = []
        start = indices_sorted[0]
        prev = start
        for i in indices_sorted[1:]:
            if i == prev + 1:
                prev = i
            else:
                intervals.append((start, prev))
                start = i
                prev = i
        intervals.append((start, prev))
        return cls(intervals)

    @classmethod
    def empty(cls) -> 'FastPyRangeSet':
        return cls()

    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'FastPyRangeSet':
        return cls.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return [list(r) for r in self.intervals]
