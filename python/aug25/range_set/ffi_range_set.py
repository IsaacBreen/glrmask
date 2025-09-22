import _sep1 as ffi
from typing import List, Tuple, Iterable, cast

from .range_set_abc import RangeSet
from .rangeset_stats import time_method, time_func, record_metric



class FFIRangeSet(RangeSet[int]):
    """A FFIRangeSet implementation backed by the Rust ffi.Bitset."""

    __slots__ = ('_bitset',)

    def __init__(self, intervals: Iterable[Tuple[int, int]] = ()):
        # ffi.Bitset.from_ranges expects a list of [start, end] lists/tuples
        self._bitset = ffi.Bitset.from_ranges(list(intervals))

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        """Returns the intervals as a tuple of tuples."""
        return tuple(cast(List[Tuple[int, int]], self._bitset.to_ranges()))

    def to_ranges(self) -> List[List[int]]:
        """Returns the intervals as a list of lists for JSON serialization."""
        return [list(r) for r in self._bitset.to_ranges()]

    @time_method
    def to_indices(self) -> List[int]:
        """Returns the elements of the set as a list."""
        res = self._bitset.to_indices()
        try:
            record_metric('FFIRangeSet.to_indices.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def contains(self, x: int) -> bool:
        """Return True if x is contained in the set."""
        res = self._bitset.contains(x)
        record_metric('FFIRangeSet.contains.true' if res else 'FFIRangeSet.contains.false', 1)
        return res

    @time_method
    def union(self, other: RangeSet[int]) -> "FFIRangeSet":
        """Return the union of two RangeSets."""
        if not isinstance(other, FFIRangeSet):
            raise TypeError("other must be a FFIRangeSet")
        try:
            record_metric('FFIRangeSet.union.in_len_a', len(self))
            record_metric('FFIRangeSet.union.in_len_b', len(other))
        except Exception:
            pass
        res = self | other
        try:
            record_metric('FFIRangeSet.union.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def intersection(self, other: RangeSet[int]) -> "FFIRangeSet":
        """Return the intersection of two RangeSets."""
        if not isinstance(other, FFIRangeSet):
            raise TypeError("other must be a FFIRangeSet")
        try:
            record_metric('FFIRangeSet.intersection.in_len_a', len(self))
            record_metric('FFIRangeSet.intersection.in_len_b', len(other))
        except Exception:
            pass
        res = self & other
        try:
            record_metric('FFIRangeSet.intersection.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def difference(self, other: RangeSet[int]) -> "FFIRangeSet":
        """Return the set difference self \\ other."""
        if not isinstance(other, FFIRangeSet):
            raise TypeError("other must be a FFIRangeSet")
        try:
            record_metric('FFIRangeSet.difference.in_len_a', len(self))
            record_metric('FFIRangeSet.difference.in_len_b', len(other))
        except Exception:
            pass
        res = self - other
        try:
            record_metric('FFIRangeSet.difference.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def is_empty(self) -> bool:
        res = self._bitset.is_empty()
        record_metric('FFIRangeSet.is_empty.true' if res else 'FFIRangeSet.is_empty.false', 1)
        return res

    @time_method
    def __len__(self) -> int:
        v = self._bitset.len()
        # Sum of cardinalities observed; you can divide by calls to get an average size
        record_metric('FFIRangeSet.__len__.sum', v)
        return v

    def __repr__(self) -> str:
        return f"FFIRangeSet.from_ranges({self.intervals!r})"

    @time_method
    def __eq__(self, other) -> bool:
        if not isinstance(other, FFIRangeSet):
            raise TypeError("other must be a FFIRangeSet")
        res = self._bitset == other._bitset
        record_metric('FFIRangeSet.__eq__.true' if res else 'FFIRangeSet.__eq__.false', 1)
        return res

    @time_method
    def __hash__(self) -> int:
        # The PyHybridBitset has a __hash__ method
        h = hash(self._bitset)
        record_metric('FFIRangeSet.__hash__', 1)
        return h

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Normalizes a list of [start, end] intervals into a sorted, merged, disjoint list of pairs.
        This can be achieved by creating a temporary FFIRangeSet.
        """
        # The ffi.Bitset constructor handles merging and sorting.
        temp_rs = FFIRangeSet.from_ranges(ranges)
        return temp_rs._bitset.to_ranges()

    @staticmethod
    @time_func('FFIRangeSet.from_ranges')
    def from_ranges(ranges: List[List[int]]) -> 'FFIRangeSet':
        """Creates a FFIRangeSet from a list of [start, end] lists."""
        # return FFIRangeSet(tuple(map(tuple, ranges)))
        self = FFIRangeSet()
        try:
            record_metric('FFIRangeSet.from_ranges.in_ranges_count', len(ranges))
        except Exception:
            pass
        self._bitset = ffi.Bitset.from_ranges(ranges)
        try:
            record_metric('FFIRangeSet.from_ranges.out_len', len(self))
        except Exception:
            pass
        return self

    @staticmethod
    @time_func('FFIRangeSet.from_indices')
    def from_indices(indices: Iterable[int]) -> 'FFIRangeSet':
        """Creates a FFIRangeSet from an iterable of individual indices."""
        new_set = FFIRangeSet.from_ranges([])
        # The FFI function expects a list.
        idx_list = list(indices)
        record_metric('FFIRangeSet.from_indices.in_len', len(idx_list))
        new_set._bitset = ffi.Bitset.from_indices(idx_list)
        try:
            record_metric('FFIRangeSet.from_indices.out_len', len(new_set))
        except Exception:
            pass
        return new_set

    @staticmethod
    @time_func('FFIRangeSet.empty')
    def empty() -> 'FFIRangeSet':
        """Creates an empty FFIRangeSet."""
        record_metric('FFIRangeSet.empty.calls', 1)
        return FFIRangeSet.from_ranges([])

    @staticmethod
    def from_json(data: List[List[int]]) -> 'FFIRangeSet':
        return FFIRangeSet.from_ranges(tuple(map(tuple, data)))

    def to_json(self) -> List[List[int]]:
        return self.to_ranges()
