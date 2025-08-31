from typing import Any, Dict, List, Tuple, Optional, Iterable, Sequence
from dataclasses import dataclass
import bisect

class GraphProvider:
    def get_root(self, state_id: int) -> int:
        raise NotImplementedError
    def is_end(self, node: int) -> bool:
        raise NotImplementedError
    def iter_edges(self, node: int, token: int):
        """
        Yield (pop_count: int, state_id_or_none: Optional[int], dest_node: int) for edges whose token filter passes.
        Implementations for precompute3 can prefilter with their token bitsets; for precompute2 leave filtering to caller.
        """
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class RangeSet:
    """
    Efficient, normalized (sorted, disjoint, inclusive) intervals for large, sparse token sets.
    Internally represented as a tuple of (start, end) pairs, both inclusive.
    """
    intervals: Tuple[Tuple[int, int], ...]

    @staticmethod
    def empty() -> "RangeSet":
        return RangeSet(())

    @staticmethod
    def from_ranges(ranges: Iterable[Sequence[int]]) -> "RangeSet":
        normalized = RangeSet._merge_unsorted(ranges)
        return RangeSet(tuple(normalized))

    @staticmethod
    def from_indices(indices: Iterable[int]) -> "RangeSet":
        ranges = [(x, x) for x in indices]
        normalized = RangeSet._merge_unsorted(ranges)
        return RangeSet(tuple(normalized))

    @staticmethod
    def from_json(ranges_json: Optional[List[List[int]]]) -> "RangeSet":
        if not ranges_json:
            return RangeSet.empty()
        return RangeSet.from_ranges(ranges_json)

    def is_empty(self) -> bool:
        return not self.intervals

    def __bool__(self) -> bool:
        return not self.is_empty()

    def contains(self, x: int) -> bool:
        a = self.intervals
        if not a:
            return False
        starts = [s for s, _ in a]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0:
            return False
        s, e = a[i]
        return s <= x <= e

    def union(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty(): return other
        if other.is_empty(): return self
        merged: List[Tuple[int, int]] = []
        i, j = 0, 0
        a, b = self.intervals, other.intervals
        def append_or_merge(start: int, end: int) -> None:
            if not merged:
                merged.append((start, end))
                return
            ps, pe = merged[-1]
            if start <= pe + 1:
                merged[-1] = (ps, max(pe, end))
            else:
                merged.append((start, end))
        while i < len(a) and j < len(b):
            if a[i][0] <= b[j][0]:
                append_or_merge(a[i][0], a[i][1]); i += 1
            else:
                append_or_merge(b[j][0], b[j][1]); j += 1
        while i < len(a):
            append_or_merge(a[i][0], a[i][1]); i += 1
        while j < len(b):
            append_or_merge(b[j][0], b[j][1]); j += 1
        return RangeSet(tuple(merged))

    def intersection(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty() or other.is_empty(): return RangeSet.empty()
        i, j = 0, 0
        a, b = self.intervals, other.intervals
        out: List[Tuple[int, int]] = []
        while i < len(a) and j < len(b):
            s1, e1 = a[i]; s2, e2 = b[j]
            start = max(s1, s2); end = min(e1, e2)
            if start <= end: out.append((start, end))
            if e1 < e2: i += 1
            else: j += 1
        return RangeSet(tuple(out)) if out else RangeSet.empty()

    def to_json(self) -> List[List[int]]:
        return [[s, e] for s, e in self.intervals]

    def __str__(self) -> str:
        if self.is_empty(): return "{}"
        parts = [f"{s}-{e}" if s != e else str(s) for s, e in self.intervals]
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
        items = sorted([(int(s), int(e)) for s, e in ranges if s is not None and e is not None])
        if not items: return []
        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1: ce = max(ce, ne)
            else: merged.append((cs, ce)); cs, ce = ns, ne
        merged.append((cs, ce))
        return merged
