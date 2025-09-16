from typing import List, Tuple, Optional, Iterable, Dict
import _sep1 as ffi

class RangeSet:
    def __init__(self, intervals: Optional[Tuple[Tuple[int, int], ...]] = None):
        self.intervals = intervals or tuple()

    @staticmethod
    def from_json(data: Optional[List[List[int]]]) -> 'RangeSet':
        if not data:
            return RangeSet()
        # Assuming data is already normalized from JSON
        return RangeSet(tuple(tuple(item) for item in data))

    @staticmethod
    def from_ranges(ranges: List[Tuple[int, int]]) -> 'RangeSet':
        return RangeSet(tuple(ranges))

    def to_ranges(self) -> List[List[int]]:
        return [list(item) for item in self.intervals]

    def contains(self, item: int) -> bool:
        # This is a placeholder and not efficient.
        # The real RangeSet might have a more optimized implementation.
        return any(start <= item < end for start, end in self.intervals)

class GraphProvider:
    def get_root(self, state_id: int) -> int: ...
    def is_end(self, node: int) -> bool: ...
    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]: ...
