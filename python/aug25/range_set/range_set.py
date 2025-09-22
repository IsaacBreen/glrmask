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
    def from_indices(indices: Iterable[int]) -> 'RangeSet':
        """Creates a RangeSet from an iterable of individual indices."""
        indices_sorted = sorted(set(indices))
        if not indices_sorted:
            return RangeSet.empty()
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
        return RangeSet(intervals)

    @staticmethod
    def empty() -> 'RangeSet':
        """Creates an empty RangeSet."""
        return RangeSet()

    def to_indices(self) -> List[int]:
        """Converts the RangeSet to a list of individual indices."""
        result = []
        for start, end in self.intervals:
            result.extend(range(start, end + 1))
        return result

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

    # New utilities for set-like operations
    def is_empty(self) -> bool:
        """Return True if no indices are present."""
        return not self.intervals

    def contains(self, x: int) -> bool:
        """Return True if x is contained in the set."""
        for s, e in self.intervals:
            if s <= x <= e:
                return True
            if x < s:
                return False
        return False

    def union(self, other: 'RangeSet') -> 'RangeSet':
        """Return the union of two RangeSets."""
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        return RangeSet(self.intervals + other.intervals)

    def intersection(self, other: 'RangeSet') -> 'RangeSet':
        """Return the intersection of two RangeSets."""
        if self.is_empty() or other.is_empty():
            return RangeSet.empty()
        a = list(self.intervals)
        b = list(other.intervals)
        i = j = 0
        res: List[Tuple[int, int]] = []
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            s = max(s1, s2)
            e = min(e1, e2)
            if s <= e:
                res.append((s, e))
            if e1 < e2:
                i += 1
            else:
                j += 1
        return RangeSet(res)

    def difference(self, other: 'RangeSet') -> 'RangeSet':
        """Return the set difference self \ other."""
        if self.is_empty():
            return RangeSet.empty()
        if other.is_empty():
            return self
        a = list(self.intervals)
        b = list(other.intervals)
        res: List[Tuple[int, int]] = []
        j = 0
        for s1, e1 in a:
            curr = s1
            while j < len(b) and b[j][1] < curr:
                j += 1
            k = j
            while k < len(b) and b[k][0] <= e1:
                s2, e2 = b[k]
                if s2 > curr:
                    res.append((curr, s2 - 1))
                if e2 + 1 > curr:
                    curr = e2 + 1
                if curr > e1:
                    break
                k += 1
            if curr <= e1:
                res.append((curr, e1))
        return RangeSet(res)
