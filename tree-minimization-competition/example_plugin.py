"""
Example Plugin Submission for the Precompute2 Trie Competition.

This plugin demonstrates two supported APIs used by the harness (`trie_stuff.py`):

1) New graph API for robust equivalence (preferred):
   - build(roots_map, arena): initialize internal structure
   - get_root(structure, state_id) -> node
   - iter_edges(structure, node, token) -> iterator of (pop_count, state_id or None, dest)
   - is_end(structure, node) -> bool

   The harness will build a token-normalized, epsilon-free NFA from your graph by
   following only the edges whose per-edge token bitvector contains the queried token.

2) Legacy API (kept for backward compatibility with legacy scoring mode):
   - get_bv(structure, state_id, path) -> JSON list of [start, end] (RangeSet JSON)
   - stats(structure) -> dict with keys 'nodes' and 'edges'

This example plugin is a simple pass-through that uses the original arena as-is,
so it is not memory efficient. A competitive entry would instead build a compact
representation and implement iter_edges/is_end accordingly.
"""

import collections
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------

TrieNodeIndex = int
TokenizerStateID = int
StateID = int
NormalizedPath = List[Tuple[int, StateID]]
LLMTokenBVJSON = List[List[int]]


# -----------------------------------------------------------------------------
# Self-Contained RangeSet Implementation (with 'contains')
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RangeSet:
    intervals: Tuple[Tuple[int, int], ...]

    @staticmethod
    def empty() -> "RangeSet":
        return RangeSet(())

    @staticmethod
    def from_ranges(ranges: Iterable[Sequence[int]]) -> "RangeSet":
        normalized = RangeSet._merge_unsorted(ranges)
        return RangeSet(tuple(normalized))

    @staticmethod
    def from_json(ranges_json: Optional[LLMTokenBVJSON]) -> "RangeSet":
        if not ranges_json:
            return RangeSet.empty()
        return RangeSet.from_ranges(ranges_json)

    def is_empty(self) -> bool:
        return not self.intervals

    def contains(self, x: int) -> bool:
        a = self.intervals
        if not a:
            return False
        # binary search on starts
        starts = [s for s, _ in a]
        import bisect as _bis
        i = _bis.bisect_right(starts, x) - 1
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
                merged.append((start, end)); return
            ps, pe = merged[-1]
            if start <= pe + 1: merged[-1] = (ps, max(pe, end))
            else: merged.append((start, end))
        while i < len(a) and j < len(b):
            if a[i][0] <= b[j][0]: append_or_merge(a[i][0], a[i][1]); i += 1
            else: append_or_merge(b[j][0], b[j][1]); j += 1
        while i < len(a): append_or_merge(a[i][0], a[i][1]); i += 1
        while j < len(b): append_or_merge(b[j][0], b[j][1]); j += 1
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
        return RangeSet(tuple(out))

    def difference(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty(): return RangeSet.empty()
        if other.is_empty(): return self
        a, b = self.intervals, other.intervals
        out: List[Tuple[int, int]] = []
        j = 0
        for s1, e1 in a:
            while j < len(b) and b[j][1] < s1: j += 1
            cur_start = s1
            while j < len(b) and b[j][0] <= e1:
                bs, be = b[j]
                if bs > cur_start: out.append((cur_start, min(e1, bs - 1)))
                if be >= e1: cur_start = e1 + 1; break
                else: cur_start = be + 1; j += 1
            if cur_start <= e1: out.append((cur_start, e1))
        return RangeSet(tuple(out))

    def to_json(self) -> LLMTokenBVJSON:
        return [[s, e] for s, e in self.intervals]

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
        items = [(int(s), int(e)) for s, e in ranges if s is not None and e is not None]
        if not items: return []
        items.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1: ce = max(ce, ne)
            else: merged.append((cs, ce)); cs, ce = ns, ne
        merged.append((cs, ce))
        return merged


# -----------------------------------------------------------------------------
# Plugin Data Structure
# -----------------------------------------------------------------------------

class MyTrie:
    """A simple container for the trie data passed by the scorer."""
    def __init__(self, roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]], arena: Dict[TrieNodeIndex, Dict]):
        self.roots: Dict[TokenizerStateID, TrieNodeIndex] = dict(roots_map)
        self.arena: Dict[TrieNodeIndex, Dict] = arena

        # Pre-calculate size metrics for the `stats` function (legacy).
        self.node_count = len(arena)
        self.edge_count = 0
        for node in arena.values():
            for _edge_key, dest_map in node.get("children", []):
                self.edge_count += len(dest_map)


# -----------------------------------------------------------------------------
# Graph API implementation (new, preferred)
# -----------------------------------------------------------------------------

def build(roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]], arena: Dict) -> MyTrie:
    """
    API function: Builds and returns the plugin's internal data structure.
    The scorer calls this once at the beginning.
    """
    print("Example plugin: Building data structure...")
    t = MyTrie(roots_map, arena)
    print("Example plugin: Build complete.")
    return t


def get_root(structure: MyTrie, state_id: TokenizerStateID) -> TrieNodeIndex:
    """
    API function (new): Returns the root node index corresponding to the tokenizer state_id.
    """
    return int(structure.roots.get(int(state_id), -1))


def iter_edges(structure: MyTrie, node: TrieNodeIndex, token: int) -> Iterator[Tuple[int, Optional[int], TrieNodeIndex]]:
    """
    API function (new): Yields outgoing edges for a given node, filtered by the provided token.
    Each yielded edge is a tuple (pop_count, state_id or None, dest_node).
    """
    n = structure.arena.get(int(node))
    if not n:
        return
    children = n.get("children", []) or []
    for (pop_count, sid_opt), dest_map in children:
        for dest_idx, edge_bv in dest_map:
            rs: RangeSet = edge_bv  # scorer converts to RangeSet on load
            if rs.contains(int(token)):
                yield (int(pop_count), int(sid_opt) if sid_opt is not None else None, int(dest_idx))


def is_end(structure: MyTrie, node: TrieNodeIndex) -> bool:
    """
    API function (new): Returns whether the given node is an accepting (end) node.
    """
    n = structure.arena.get(int(node))
    if not n:
        return False
    return bool((n.get("value", {}) or {}).get("end", False))


# -----------------------------------------------------------------------------
# Legacy logic kept for compatibility with legacy scoring mode
# -----------------------------------------------------------------------------

def _update_visited_bv(store: Dict[Any, RangeSet], key: Any, incoming_bv: RangeSet) -> Optional[RangeSet]:
    if key in store:
        existing = store[key]
        diff = incoming_bv.difference(existing)
        if diff.is_empty(): return None
        store[key] = existing.union(diff)
        return diff
    else:
        store[key] = incoming_bv
        return incoming_bv


def _find_end_bv_from_node_via_none_edges(
    start_node_index: TrieNodeIndex,
    initial_bv: RangeSet,
    arena: Dict
) -> RangeSet:
    """Finds BVs for all paths from a start node to an `end` node via (k, None) edges."""
    if initial_bv.is_empty():
        return RangeSet.empty()

    total_end_bv = RangeSet.empty()
    q: Deque[Tuple[TrieNodeIndex, RangeSet]] = collections.deque([(start_node_index, initial_bv)])
    visited: Dict[TrieNodeIndex, RangeSet] = {}

    while q:
        node_idx, current_bv = q.popleft()
        node = arena.get(node_idx)
        if not node: continue

        if (node.get("value", {}) or {}).get("end"):
            total_end_bv = total_end_bv.union(current_bv)

        for (_pop, state_id_opt), dest_map in node.get("children", []):
            if state_id_opt is not None: continue

            for dest_idx, edge_bv in dest_map:
                new_bv = current_bv.intersection(edge_bv)
                if new_bv.is_empty(): continue
                diff = _update_visited_bv(visited, dest_idx, new_bv)
                if diff:
                    q.append((dest_idx, diff))
    return total_end_bv


def _get_bv_for_normalized_path_internal(
    root_index: TrieNodeIndex,
    path: NormalizedPath,
    arena: Dict
) -> RangeSet:
    """Computes the token bitvector for a given normalized path."""
    root_node = arena.get(root_index)
    if not root_node: return RangeSet.empty()

    initial_bv = (root_node.get("value", {}) or {}).get("live_tokens") or RangeSet.empty()
    if initial_bv.is_empty() and path: return RangeSet.empty()

    final_bv = RangeSet.empty()
    q: Deque[Tuple[TrieNodeIndex, int, int, RangeSet]] = collections.deque()
    q.append((root_index, 0, 0, initial_bv))
    visited: Dict[Tuple[TrieNodeIndex, int, int], RangeSet] = {(root_index, 0, 0): initial_bv}

    while q:
        node_idx, path_idx, k_so_far, current_bv = q.popleft()

        if path_idx == len(path):
            end_bv = _find_end_bv_from_node_via_none_edges(node_idx, current_bv, arena)
            final_bv = final_bv.union(end_bv)
            continue

        target_k, target_sid = path[path_idx]
        node = arena.get(node_idx)
        if not node: continue

        for (pop_count, state_id_opt), dest_map in node.get("children", []):
            new_k = k_so_far + pop_count
            for dest_idx, edge_bv in dest_map:
                new_bv = current_bv.intersection(edge_bv)
                if new_bv.is_empty(): continue

                if state_id_opt is not None:
                    if new_k == target_k and state_id_opt == target_sid:
                        next_key = (dest_idx, path_idx + 1, 0)
                        diff = _update_visited_bv(visited, next_key, new_bv)
                        if diff: q.append((dest_idx, path_idx + 1, 0, diff))
                else:
                    if new_k <= target_k:
                        cont_key = (dest_idx, path_idx, new_k)
                        diff = _update_visited_bv(visited, cont_key, new_bv)
                        if diff: q.append((dest_idx, path_idx, new_k, diff))
    return final_bv


def get_bv(structure: MyTrie, state_id: TokenizerStateID, path: NormalizedPath) -> LLMTokenBVJSON:
    """
    API function (legacy): Returns the token bitvector for a path from a given state_id.
    The return value must be JSON-serializable (list of [start, end] pairs).
    """
    if not isinstance(structure, MyTrie):
        raise TypeError(f"Expected structure to be MyTrie, but got {type(structure)}")

    root_index = structure.roots.get(state_id)
    if root_index is None:
        return []  # No root for this state_id, so the path is invalid.

    final_bv = _get_bv_for_normalized_path_internal(root_index, path, structure.arena)
    return final_bv.to_json()


def stats(structure: MyTrie) -> Dict[str, Any]:
    """
    API function: Returns a dictionary of statistics about the internal data structure.
    The 'nodes' and 'edges' keys are used for scoring (legacy).
    """
    if not isinstance(structure, MyTrie):
        return {"nodes": 0, "edges": 0, "error": "Structure not initialized or wrong type."}

    return {
        "nodes": structure.node_count,
        "edges": structure.edge_count,
        "comment": "This is a simple pass-through implementation; size is not optimized."
    }
