import _sep1 as ffi
from typing import List, Tuple, Iterable, cast

class RangeSet:
    """A RangeSet implementation backed by the Rust ffi.Bitset."""

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

    def __or__(self, other: 'RangeSet') -> 'RangeSet':
        new_set = RangeSet()
        new_set._bitset = self._bitset.union(other._bitset)
        return new_set

    def __and__(self, other: 'RangeSet') -> 'RangeSet':
        new_set = RangeSet()
        new_set._bitset = self._bitset.intersection(other._bitset)
        return new_set

    def __sub__(self, other: 'RangeSet') -> 'RangeSet':
        new_set = RangeSet()
        new_set._bitset = self._bitset.difference(other._bitset)
        return new_set
    
    def is_empty(self) -> bool:
        return self._bitset.is_empty()

    def __len__(self) -> int:
        return self._bitset.len()

    def __repr__(self) -> str:
        return f"RangeSet({self.intervals!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, RangeSet):
            return NotImplemented
        return self._bitset == other._bitset

    def __hash__(self) -> int:
        # The PyHybridBitset has a __hash__ method
        return hash(self._bitset)

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Normalizes a list of [start, end] intervals into a sorted, merged, disjoint list of pairs.
        This can be achieved by creating a temporary RangeSet.
        """
        # The ffi.Bitset constructor handles merging and sorting.
        temp_rs = RangeSet(ranges)
        return temp_rs._bitset.to_ranges()

    @staticmethod
    def from_json(data: List[List[int]]) -> 'RangeSet':
        return RangeSet(tuple(map(tuple, data)))

    def to_json(self) -> List[List[int]]:
        return self.to_ranges()
