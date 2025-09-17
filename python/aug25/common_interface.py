from typing import Protocol, Iterable, Optional, Tuple, List


class RangeSet:
    """
    Represents a set of integers as a sorted, disjoint list of closed intervals.
    """
    __slots__ = ('intervals',)

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        if intervals:
            self.intervals = self._normalize(intervals)
        else:
            self.intervals = tuple()

    @staticmethod
    def _normalize(intervals: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        """
        Normalizes a list of [start, end] intervals into a sorted, merged, disjoint tuple of pairs.
        """
        items = sorted(intervals)
        if not items:
            return tuple()

        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                ce = max(ce, ne)
            else:
                merged.append((cs, ce))
                cs, ce = ns, ne
        merged.append((cs, ce))
        return tuple(merged)

    @staticmethod
    def _merge_unsorted(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Same as normalize but returns a list. Used by optimizer.
        """
        items = sorted(intervals)
        if not items:
            return []

        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                ce = max(ce, ne)
            else:
                merged.append((cs, ce))
                cs, ce = ns, ne
        merged.append((cs, ce))
        return merged

    @staticmethod
    def from_ranges(ranges: List[List[int]]) -> 'RangeSet':
        """Creates a RangeSet from a list of [start, end] lists."""
        return RangeSet(tuple(map(tuple, ranges)))

    def to_ranges(self) -> List[List[int]]:
        """Converts the RangeSet to a list of [start, end] lists."""
        return [list(interval) for interval in self.intervals]

    @staticmethod
    def from_numpy(bv) -> 'RangeSet':
        """Creates a RangeSet from a numpy array of booleans."""
        intervals = []
        in_range = False
        start = 0
        for i in range(len(bv)):
            if bv[i] and not in_range:
                start = i
                in_range = True
            elif not bv[i] and in_range:
                intervals.append((start, i - 1))
                in_range = False
        if in_range:
            intervals.append((start, len(bv) - 1))
        return RangeSet(intervals)

    def __eq__(self, other):
        if not isinstance(other, RangeSet):
            return NotImplemented
        return self.intervals == other.intervals

    def __hash__(self):
        return hash(self.intervals)

    def __repr__(self):
        return f"RangeSet({self.intervals!r})"


class GraphProvider(Protocol):
    """
    A common interface for different graph-based model implementations to be
    used in equivalence checking and other analysis tools.
    """
    def get_root(self, state_id: int) -> int:
        """Get the root node index for a given tokenizer state ID."""
        ...

    def is_end(self, node: int) -> bool:
        """Check if a given node is an accepting/end state."""
        ...

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
        """
        For a given node and token, iterate over all possible transitions.
        Yields tuples of (pop_count, state_id_or_none, destination_node_index).
        - pop_count: Number of items to pop from the GSS.
        - state_id_or_none: The state ID required at the top of the GSS for this
          transition, or None for an epsilon transition on the GSS.
        - destination_node_index: The index of the node to transition to.
        """
        ...
