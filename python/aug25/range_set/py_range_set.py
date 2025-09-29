from __future__ import annotations
from typing import Iterable, Optional, Tuple, List
from .range_set_abc import RangeSet

class PyRangeSet(RangeSet[int]):
    """
    Represents a set of integers as a sorted, disjoint list of closed intervals.
    Implements the generic RangeSet[int] interface.
    """
    __slots__ = ('_intervals',)

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        if intervals:
            self._intervals = self._normalize(intervals)
        else:
            self._intervals = tuple()

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        return self._intervals

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
    def from_ranges(ranges: List[List[int]]) -> 'PyRangeSet':
        """Creates a PyRangeSet from a list of [start, end] lists."""
        return PyRangeSet(iter(map(tuple, ranges)))

    def to_ranges(self) -> List[Tuple[int]]:
        """Converts the PyRangeSet to a list of [start, end] lists."""
        return [tuple(interval) for interval in self.intervals]

    @staticmethod
    def from_json(data: List[List[int]]) -> 'PyRangeSet':
        """Creates a PyRangeSet from a list of [start, end] lists (JSON format)."""
        return PyRangeSet(iter(map(tuple, data)))

    def to_json(self) -> List[List[int]]:
        """Converts the PyRangeSet to a list of [start, end] lists (JSON format)."""
        return self.to_ranges()

    @staticmethod
    def from_indices(indices: Iterable[int]) -> 'PyRangeSet':
        """Creates a PyRangeSet from an iterable of individual indices."""
        indices_sorted = sorted(set(indices))
        if not indices_sorted:
            return PyRangeSet.empty()
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
        return PyRangeSet(intervals)

    @classmethod
    def empty(cls) -> 'PyRangeSet':
        """Creates an empty PyRangeSet."""
        return PyRangeSet()

    def to_indices(self) -> List[int]:
        """Converts the PyRangeSet to a list of individual indices."""
        result = []
        for start, end in self.intervals:
            result.extend(range(start, end + 1))
        return result

    def iter_indices(self) -> Iterable[int]:
        """Iterates over all individual indices in the set."""
        for start, end in self.intervals:
            yield from range(start, end + 1)

    def iter_ranges(self) -> Iterable[Tuple[int, int]]:
        """Iterates over all [start, end] intervals in the set."""
        yield from self.intervals

    @staticmethod
    def from_numpy(bv) -> 'PyRangeSet':
        """Creates a PyRangeSet from a numpy array of booleans."""
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
        return PyRangeSet(intervals)

    def __eq__(self, other):
        if isinstance(other, PyRangeSet):
            return self.intervals == other.intervals
        if isinstance(other, RangeSet):
            # Compare by normalized intervals for any RangeSet implementation
            return self.intervals == other.intervals
        return NotImplemented

    def __hash__(self):
        return hash(self.intervals)

    def __repr__(self):
        return f"PyRangeSet({self.intervals!r})"

    # New utilities for set-like operations
    def is_empty(self) -> bool:
        """Return True if no indices are present."""
        return not self.intervals

    def __len__(self) -> int:
        """Return the total number of elements in the set."""
        return sum(e - s + 1 for s, e in self._intervals)

    def contains(self, x: int) -> bool:
        """Return True if x is contained in the set."""
        for s, e in self.intervals:
            if s <= x <= e:
                return True
            if x < s:
                return False
        return False

    def union(self, other: 'RangeSet[int]') -> 'PyRangeSet':
        """Return the union of two RangeSets."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        if self.is_empty():
            return PyRangeSet(other.intervals)
        if other.is_empty():
            return self
        return PyRangeSet(self.intervals + other.intervals)

    def intersection(self, other: 'RangeSet[int]') -> 'PyRangeSet':
        """Return the intersection of two RangeSets."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        if self.is_empty() or other.is_empty():
            return PyRangeSet.empty()
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
        return PyRangeSet(res)

    def isdisjoint(self, other: 'RangeSet[int]') -> bool:
        """Return True if self has no elements in common with other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        if self.is_empty() or other.is_empty():
            return True
        a = list(self.intervals)
        b = list(other.intervals)
        i = j = 0
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            if max(s1, s2) <= min(e1, e2):
                return False  # Found an intersection
            if e1 < e2:
                i += 1
            else:
                j += 1
        return True

    def issubset(self, other: 'RangeSet[int]') -> bool:
        """Return True if self is a subset of other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        other_intervals = other.intervals
        
        i = 0  # pointer for self.intervals
        j = 0  # pointer for other.intervals
        
        while i < len(self.intervals):
            s1, e1 = self.intervals[i]
            
            # Find the first interval in other that could cover s1
            while j < len(other_intervals) and other_intervals[j][1] < s1:
                j += 1
            
            if j == len(other_intervals):
                # No more intervals in other, but we still have intervals in self
                return False
            
            s2, e2 = other_intervals[j]
            
            # Check if s1 is covered
            if s1 < s2:
                return False # Gap, s1 is not in any interval of other
            
            # s1 is covered. Now check up to e1.
            current_s = s1
            while current_s <= e1:
                if current_s > e2:
                    j += 1
                    if j == len(other_intervals): return False
                    s2, e2 = other_intervals[j]
                    if current_s < s2: return False # Gap
                    continue

                covered_until = e2
                
                if e1 <= covered_until:
                    current_s = e1 + 1 # break inner loop
                else:
                    current_s = covered_until + 1
                    j += 1
                    if j == len(other_intervals): return False
                    s2_next, e2_next = other_intervals[j]
                    if s2_next != current_s: return False # Gap
                    s2, e2 = s2_next, e2_next
            i += 1
        return True

    def difference(self, other: 'RangeSet[int]') -> 'PyRangeSet':
        """Return the set difference self \\ other as a PyRangeSet."""
        if self.is_empty():
            return PyRangeSet.empty()
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
        return PyRangeSet(res)

    def union_update(self, other: 'RangeSet[int]') -> None:
        """Update self with the union of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        if other.is_empty():
            return
        self._intervals = self._normalize(self.intervals + other.intervals)

    def intersection_update(self, other: 'RangeSet[int]') -> None:
        """Update self with the intersection of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        if self.is_empty():
            return
        if other.is_empty():
            self._intervals = tuple()
            return
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
        self._intervals = tuple(res)

    def difference_update(self, other: 'RangeSet[int]') -> None:
        """Update self with the set difference self \\ other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")
        if self.is_empty() or other.is_empty():
            return
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
        self._intervals = tuple(res)
