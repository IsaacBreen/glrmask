import _sep1 as ffi
from typing import List, Tuple, Iterable

from .range_set_abc import RangeSet
from .rangeset_stats import time_method, time_func, record_metric


class FFIBitset(RangeSet[int]):
    """A FFIBitset implementation backed by the Rust ffi.Bitset."""

    __slots__ = ('_bitset',)

    def __init__(self, intervals: Iterable[Tuple[int, int]] = ()):
        indices = []
        for start, end in intervals:
            indices.extend(range(start, end + 1))
        self._bitset = ffi.Bitset.from_indices(indices)

    @classmethod
    def from_ffi_bitset(cls, bitset: ffi.Bitset) -> 'FFIBitset':
        """Creates a FFIBitset from a PyBitset."""
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
            record_metric('FFIBitset.to_indices.out_len', len(res))
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
        record_metric('FFIBitset.contains.true' if res else 'FFIBitset.contains.false', 1)
        return res

    @time_method
    def union(self, other: RangeSet[int]) -> "FFIBitset":
        """Return the union of two RangeSets."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        try:
            record_metric('FFIBitset.union.in_len_a', len(self))
            record_metric('FFIBitset.union.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.union(other._bitset)
        res = FFIBitset()
        res._bitset = new_bs
        try:
            record_metric('FFIBitset.union.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def intersection(self, other: RangeSet[int]) -> "FFIBitset":
        """Return the intersection of two RangeSets."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        try:
            record_metric('FFIBitset.intersection.in_len_a', len(self))
            record_metric('FFIBitset.intersection.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.intersection(other._bitset)
        res = FFIBitset()
        res._bitset = new_bs
        try:
            record_metric('FFIBitset.intersection.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def difference(self, other: RangeSet[int]) -> "FFIBitset":
        """Return the set difference self \\ other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        try:
            record_metric('FFIBitset.difference.in_len_a', len(self))
            record_metric('FFIBitset.difference.in_len_b', len(other))
        except Exception:
            pass
        new_bs = self._bitset.difference(other._bitset)
        res = FFIBitset()
        res._bitset = new_bs
        try:
            record_metric('FFIBitset.difference.out_len', len(res))
        except Exception:
            pass
        return res

    @time_method
    def issuperset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a superset of other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        res = self._bitset.is_superset(other._bitset)
        record_metric('FFIBitset.issuperset.true' if res else 'FFIBitset.issuperset.false', 1)
        return res

    @time_method
    def issubset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a subset of other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        res = self._bitset.is_subset(other._bitset)
        record_metric('FFIBitset.issubset.true' if res else 'FFIBitset.issubset.false', 1)
        return res

    @time_method
    def isdisjoint(self, other: RangeSet[int]) -> bool:
        """Return True if self has no elements in common with other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        res = self._bitset.is_disjoint(other._bitset)
        record_metric('FFIBitset.isdisjoint.true' if res else 'FFIBitset.isdisjoint.false', 1)
        return res

    @time_method
    def union_update(self, other: RangeSet[int]) -> None:
        """Update self with the union of self and other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        self._bitset |= other._bitset

    @time_method
    def intersection_update(self, other: RangeSet[int]) -> None:
        """Update self with the intersection of self and other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        self._bitset &= other._bitset

    @time_method
    def difference_update(self, other: RangeSet[int]) -> None:
        """Update self with the set difference self \\ other."""
        if not isinstance(other, FFIBitset):
            raise TypeError("other must be a FFIBitset")
        self._bitset -= other._bitset

    @time_method
    def is_empty(self) -> bool:
        res = self._bitset.is_empty()
        record_metric('FFIBitset.is_empty.true' if res else 'FFIBitset.is_empty.false', 1)
        return res

    @time_method
    def __len__(self) -> int:
        v = self._bitset.len()
        record_metric('FFIBitset.__len__.sum', v)
        return v

    def __repr__(self) -> str:
        return f"FFIBitset({self.intervals!r})"

    @time_method
    def __eq__(self, other) -> bool:
        if not isinstance(other, FFIBitset):
            return NotImplemented
        res = self._bitset == other._bitset
        record_metric('FFIBitset.__eq__.true' if res else 'FFIBitset.__eq__.false', 1)
        return res

    @time_method
    def __hash__(self) -> int:
        h = hash(self._bitset)
        record_metric('FFIBitset.__hash__', 1)
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
        temp_rs = FFIBitset(ranges)
        return temp_rs.to_ranges()

    @staticmethod
    @time_func('FFIBitset.from_ranges')
    def from_ranges(ranges: List[List[int]]) -> 'FFIBitset':
        """Creates a FFIBitset from a list of [start, end] lists."""
        return FFIBitset(ranges)

    @staticmethod
    @time_func('FFIBitset.from_indices')
    def from_indices(indices: Iterable[int]) -> 'FFIBitset':
        """Creates a FFIBitset from an iterable of individual indices."""
        new_set = FFIBitset()
        idx_list = list(indices)
        record_metric('FFIBitset.from_indices.in_len', len(idx_list))
        new_set._bitset = ffi.Bitset.from_indices(idx_list)
        try:
            record_metric('FFIBitset.from_indices.out_len', len(new_set))
        except Exception:
            pass
        return new_set

    @classmethod
    @time_func('FFIBitset.empty')
    def empty(cls) -> 'FFIBitset':
        """Creates an empty FFIBitset."""
        record_metric('FFIBitset.empty.calls', 1)
        return FFIBitset()

    @staticmethod
    def from_json(data: List[List[int]]) -> 'FFIBitset':
        return FFIBitset.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return self.to_ranges()