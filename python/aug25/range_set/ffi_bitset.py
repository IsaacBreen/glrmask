import _sep1 as ffi
from typing import List, Tuple, Iterable

from .range_set_abc import RangeSet
from .rangeset_stats import time_method, time_func, record_metric


class BitsetRangeSet(RangeSet[int]):
    """A BitsetRangeSet implementation backed by the Rust ffi.Bitset."""

    __slots__ = ('_bitset',)

    def __init__(self, intervals: Iterable[Tuple[int, int]] = ()):
        indices = []
        for start, end in intervals:
            indices.extend(range(start, end + 1))
        self._bitset = ffi.Bitset.from_indices(indices)

    @classmethod
    def from_ffi_bitset(cls, bitset: ffi.Bitset) -> 'BitsetRangeSet':
        """Creates a BitsetRangeSet from a PyBitset."""
        self = cls.empty()
        self._bitset = bitset
        return self

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        """Returns the intervals as a tuple of tuples."""
        return tuple(self.iter_ranges())

    def to_ranges(self) -> List[Tuple[int, int]]:
        """Returns the intervals as a list of lists for JSON serialization."""
        return list(self.iter_ranges())

    @time_method
    def to_indices(self) -> List[int]:
        """Returns the elements of the set as a list."""
        res = self._bitset.to_indices()
        try:
            record_metric('BitsetRangeSet.to_indices.out_len', len(res))
        except Exception:
            pass
        return res

    def iter_indices(self) -> Iterable[int]:
        """Iterates over all individual indices in the set."""
        yield from self._bitset

    def iter_ranges(self) -> Iterable[Tuple[int, int]]:
        """Iterates over all [start, end] intervals in the set."""
        indices = self._bitset.to_indices()
        if not indices:
            return

        start = indices[0]
        end = start
        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                yield (start, end)
                start = indices[i]
                end = start
        yield (start, end)

    @time_method
    def contains(self, x: int) -> bool:
        """Return True if x is contained in the set."""
        res = self._bitset.contains(x)
        record_metric('BitsetRangeSet.contains.true' if res else 'BitsetRangeSet.contains.false', 1)
        return res

    @time_method
    def union(self, other: RangeSet[int]) -> "BitsetRangeSet":
        """Return the union of two RangeSets."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        try:
            record_metric('BitsetRangeSet.union.in_len_a', len(self))
            record_metric('BitsetRangeSet.union.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.union(other._bitset)
        res = BitsetRangeSet()
        res._bitset = new_bs
        try:
            record_metric('BitsetRangeSet.union.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def intersection(self, other: RangeSet[int]) -> "BitsetRangeSet":
        """Return the intersection of two RangeSets."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        try:
            record_metric('BitsetRangeSet.intersection.in_len_a', len(self))
            record_metric('BitsetRangeSet.intersection.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.intersection(other._bitset)
        res = BitsetRangeSet()
        res._bitset = new_bs
        try:
            record_metric('BitsetRangeSet.intersection.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def difference(self, other: RangeSet[int]) -> "BitsetRangeSet":
        """Return the set difference self \\ other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        try:
            record_metric('BitsetRangeSet.difference.in_len_a', len(self))
            record_metric('BitsetRangeSet.difference.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.difference(other._bitset)
        res = BitsetRangeSet()
        res._bitset = new_bs
        try:
            record_metric('BitsetRangeSet.difference.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def issuperset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a superset of other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        res = self._bitset.is_superset(other._bitset)
        record_metric('BitsetRangeSet.issuperset.true' if res else 'BitsetRangeSet.issuperset.false', 1)
        return res

    @time_method
    def issubset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a subset of other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        res = self._bitset.is_subset(other._bitset)
        record_metric('BitsetRangeSet.issubset.true' if res else 'BitsetRangeSet.issubset.false', 1)
        return res

    @time_method
    def isdisjoint(self, other: RangeSet[int]) -> bool:
        """Return True if self has no elements in common with other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        res = self._bitset.is_disjoint(other._bitset)
        record_metric('BitsetRangeSet.isdisjoint.true' if res else 'BitsetRangeSet.isdisjoint.false', 1)
        return res

    @time_method
    def union_update(self, other: RangeSet[int]) -> None:
        """Update self with the union of self and other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        self._bitset |= other._bitset

    @time_method
    def intersection_update(self, other: RangeSet[int]) -> None:
        """Update self with the intersection of self and other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        self._bitset &= other._bitset

    @time_method
    def difference_update(self, other: RangeSet[int]) -> None:
        """Update self with the set difference self \\ other."""
        if not isinstance(other, BitsetRangeSet):
            raise TypeError("other must be a BitsetRangeSet")
        self._bitset -= other._bitset

    @time_method
    def is_empty(self) -> bool:
        res = self._bitset.is_empty()
        record_metric('BitsetRangeSet.is_empty.true' if res else 'BitsetRangeSet.is_empty.false', 1)
        return res

    @time_method
    def __len__(self) -> int:
        v = self._bitset.len()
        record_metric('BitsetRangeSet.__len__.sum', v)
        return v

    def __repr__(self) -> str:
        return f"BitsetRangeSet({self.intervals!r})"

    @time_method
    def __eq__(self, other) -> bool:
        if not isinstance(other, BitsetRangeSet):
            return NotImplemented
        res = self._bitset == other._bitset
        record_metric('BitsetRangeSet.__eq__.true' if res else 'BitsetRangeSet.__eq__.false', 1)
        return res

    @time_method
    def __hash__(self) -> int:
        h = hash(self._bitset)
        record_metric('BitsetRangeSet.__hash__', 1)
        return h

    def __getstate__(self):
        """Return a pickleable representation of the RangeSet."""
        return self.to_indices()

    def __setstate__(self, state):
        """Restore the RangeSet from its pickleable representation."""
        self._bitset = ffi.Bitset.from_indices(state)

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Normalizes a list of [start, end] intervals into a sorted, merged, disjoint list of pairs.
        """
        temp_rs = BitsetRangeSet(ranges)
        return temp_rs.to_ranges()

    @staticmethod
    @time_func('BitsetRangeSet.from_ranges')
    def from_ranges(ranges: List[List[int]]) -> 'BitsetRangeSet':
        """Creates a BitsetRangeSet from a list of [start, end] lists."""
        return BitsetRangeSet(ranges)

    @staticmethod
    @time_func('BitsetRangeSet.from_indices')
    def from_indices(indices: Iterable[int]) -> 'BitsetRangeSet':
        """Creates a BitsetRangeSet from an iterable of individual indices."""
        new_set = BitsetRangeSet()
        idx_list = list(indices)
        record_metric('BitsetRangeSet.from_indices.in_len', len(idx_list))
        new_set._bitset = ffi.Bitset.from_indices(idx_list)
        try:
            record_metric('BitsetRangeSet.from_indices.out_len', len(new_set))
        except Exception:
            pass
        return new_set

    @classmethod
    @time_func('BitsetRangeSet.empty')
    def empty(cls) -> 'BitsetRangeSet':
        """Creates an empty BitsetRangeSet."""
        record_metric('BitsetRangeSet.empty.calls', 1)
        return BitsetRangeSet()

    @staticmethod
    def from_json(data: List[List[int]]) -> 'BitsetRangeSet':
        return BitsetRangeSet.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return self.to_ranges()