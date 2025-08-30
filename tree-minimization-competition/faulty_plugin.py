"""
FAULTY Example Plugin Submission for the Precompute2 Trie Competition.

This plugin demonstrates the required API but contains a DELIBERATE BUG in its
core traversal logic. It is designed to fail the correctness checks performed
by the scoring harness (`trie_stuff.py`).

The bug is located in the `_get_bv_for_normalized_path_internal` function.
Specifically, when matching a path segment against a stateful edge (an edge with
a `state_id`), it fails to include that edge's own `pop_count` in the total `k`
for the segment. This will cause it to return incorrect results for many paths.

This file can be used to test the scoring harness itself or as an example of
what a failing submission might look like.
"""

import collections
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# -----------------------------------------------------------------------------
# Type Aliases (for clarity, matching the scorer)
# -----------------------------------------------------------------------------

TrieNodeIndex = int
TokenizerStateID = int
StateID = int
NormalizedPath = List[Tuple[int, StateID]]
LLMTokenBVJSON = List[List[int]]


# -----------------------------------------------------------------------------
# Self-Contained RangeSet Implementation
#
# It is highly recommended to copy the reference RangeSet implementation from
# the scorer script (`trie_stuff.py`) directly into your plugin. This ensures
# that your bitvector operations are bit-for-bit identical to the reference
# implementation, avoiding subtle bugs and correctness failures.
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

    def is_empty(self) -> bool:
        return not self.intervals

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


# -----------------------------------------------------------------------------
# Plugin Data Structure
# -----------------------------------------------------------------------------

class MyTrie:
    """A simple container for the trie data passed by the scorer."""
    def __init__(self, roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]], arena: Dict[TrieNodeIndex, Dict]):
        self.roots: Dict[TokenizerStateID, TrieNodeIndex] = dict(roots_map)
        self.arena: Dict[TrieNodeIndex, Dict] = arena
        self.node_count = len(arena)
        self.edge_count = sum(len(dest_map) for node in arena.values() for _, dest_map in node.get("children", []))


# -----------------------------------------------------------------------------
# Core Logic (WITH A DELIBERATE BUG)
# -----------------------------------------------------------------------------

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
    """
    Computes the token bitvector for a given normalized path.
    THIS IMPLEMENTATION CONTAINS A BUG.
    """
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
            for dest_idx, edge_bv in dest_map:
                new_bv = current_bv.intersection(edge_bv)
                if new_bv.is_empty(): continue

                if state_id_opt is not None:
                    # !!! --- BUG IS HERE --- !!!
                    # The correct logic is `if (k_so_far + pop_count) == target_k`.
                    # By omitting `+ pop_count`, we fail to account for the pop count
                    # of the stateful edge that completes the path segment. This will
                    # cause mismatches for any path segment where the final edge has
                    # a non-zero pop_count.
                    if k_so_far == target_k and state_id_opt == target_sid:
                        next_key = (dest_idx, path_idx + 1, 0)
                        diff = _update_visited_bv(visited, next_key, new_bv)
                        if diff: q.append((dest_idx, path_idx + 1, 0, diff))
                else:
                    # This part is correct: accumulate k along (k, None) edges.
                    new_k = k_so_far + pop_count
                    if new_k <= target_k:
                        cont_key = (dest_idx, path_idx, new_k)
                        diff = _update_visited_bv(visited, cont_key, new_bv)
                        if diff: q.append((dest_idx, path_idx, new_k, diff))
    return final_bv


# -----------------------------------------------------------------------------
# COMPETITION API IMPLEMENTATION
# -----------------------------------------------------------------------------

def build(roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]], arena: Dict) -> MyTrie:
    """
    API function: Builds and returns the plugin's internal data structure.
    """
    print("Faulty plugin: Building data structure...")
    instance = MyTrie(roots_map, arena)
    print("Faulty plugin: Build complete.")
    return instance


def get_bv(structure: MyTrie, state_id: TokenizerStateID, path: NormalizedPath) -> LLMTokenBVJSON:
    """
    API function: Returns the token bitvector for a path from a given state_id.
    This will return incorrect results due to the bug in its helper function.
    """
    if not isinstance(structure, MyTrie):
        raise TypeError(f"Expected structure to be MyTrie, but got {type(structure)}")

    root_index = structure.roots.get(state_id)
    if root_index is None:
        return []

    # Use our internal (faulty) logic to compute the result.
    final_bv = _get_bv_for_normalized_path_internal(root_index, path, structure.arena)

    return final_bv.to_json()


def stats(structure: MyTrie) -> Dict[str, Any]:
    """
    API function: Returns a dictionary of statistics about the internal data structure.
    """
    if not isinstance(structure, MyTrie):
        return {"nodes": 0, "edges": 0, "error": "Structure not initialized or wrong type."}

    return {
        "nodes": structure.node_count,
        "edges": structure.edge_count,
        "comment": "This is a faulty pass-through implementation; size is not optimized."
    }
