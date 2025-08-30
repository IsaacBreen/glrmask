import argparse
import bisect
import collections
import gzip
import importlib
import importlib.util
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# ----------------------------
# Type aliases for clarity
# ----------------------------

TrieNodeIndex = int
TokenizerStateID = int
StateID = int
EdgeKey = Tuple[int, Optional[StateID]]  # (pop_count, state_id or None)
TrieNode = Dict[str, Any]
Arena = Dict[TrieNodeIndex, TrieNode]
NormalizedPath = List[Tuple[int, StateID]]  # List of (k, state_id)
LLMTokenBVJSON = List[List[int]]  # JSON encoding for BVs: List of [start, end] inclusive ranges


# ----------------------------
# Efficient RangeSet for token bitvectors
# ----------------------------

@dataclass(frozen=True, slots=True)
class RangeSet:
    """
    Efficient, normalized (sorted, disjoint, inclusive) intervals for large, sparse token sets.
    Designed for very few, very large intervals (acts like a compact bitset of contiguous runs).
    Internally represented as a tuple of (start, end) pairs, both inclusive.
    """
    intervals: Tuple[Tuple[int, int], ...]  # sorted, non-overlapping, inclusive

    # ---------- Constructors ----------

    @staticmethod
    def empty() -> "RangeSet":
        return RangeSet(())

    @staticmethod
    def from_ranges(ranges: Iterable[Sequence[int]]) -> "RangeSet":
        """
        Build from an iterable of [start, end] pairs (inclusive). Input may be unsorted and overlapping.
        """
        normalized = RangeSet._merge_unsorted(ranges)
        return RangeSet(tuple(normalized))

    @staticmethod
    def from_json(ranges_json: Optional[LLMTokenBVJSON]) -> "RangeSet":
        """Convenience: build from JSON-encoded list of [start, end] pairs."""
        if not ranges_json:
            return RangeSet.empty()
        return RangeSet.from_ranges(ranges_json)

    # ---------- Basic predicates ----------

    def is_empty(self) -> bool:
        return not self.intervals

    def __bool__(self) -> bool:
        return not self.is_empty()

    def contains(self, x: int) -> bool:
        """
        Test whether integer x is in the set. Uses binary search over disjoint intervals.
        """
        a = self.intervals
        if not a:
            return False
        # Search by interval starts
        starts = [s for s, _ in a]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0:
            return False
        s, e = a[i]
        return s <= x <= e

    # ---------- Set operations (all return new RangeSet) ----------

    def union(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        # Merge like a merge-sorted union with coalescing
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
                append_or_merge(a[i][0], a[i][1])
                i += 1
            else:
                append_or_merge(b[j][0], b[j][1])
                j += 1

        while i < len(a):
            append_or_merge(a[i][0], a[i][1])
            i += 1
        while j < len(b):
            append_or_merge(b[j][0], b[j][1])
            j += 1

        return RangeSet(tuple(merged))

    def intersection(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty() or other.is_empty():
            return RangeSet.empty()

        i, j = 0, 0
        a, b = self.intervals, other.intervals
        out: List[Tuple[int, int]] = []

        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            start = s1 if s1 >= s2 else s2
            end = e1 if e1 <= e2 else e2
            if start <= end:
                out.append((start, end))
            if e1 < e2:
                i += 1
            else:
                j += 1

        if not out:
            return RangeSet.empty()
        return RangeSet(tuple(out))

    def difference(self, other: "RangeSet") -> "RangeSet":
        """
        Returns self - other.
        """
        if self.is_empty():
            return RangeSet.empty()
        if other.is_empty():
            return self

        a, b = self.intervals, other.intervals
        out: List[Tuple[int, int]] = []
        j = 0

        for s1, e1 in a:
            # skip b intervals that end before s1
            while j < len(b) and b[j][1] < s1:
                j += 1

            cur_start = s1
            while j < len(b) and b[j][0] <= e1:
                bs, be = b[j]
                if bs > cur_start:
                    out.append((cur_start, min(e1, bs - 1)))
                if be >= e1:
                    cur_start = e1 + 1
                    break
                else:
                    cur_start = be + 1
                    j += 1

            if cur_start <= e1:
                out.append((cur_start, e1))

        if not out:
            return RangeSet.empty()
        return RangeSet(tuple(out))

    def symmetric_difference(self, other: "RangeSet") -> "RangeSet":
        return self.difference(other).union(other.difference(self))

    # ---------- Utilities ----------

    def to_json(self) -> LLMTokenBVJSON:
        return [[s, e] for s, e in self.intervals]

    def __str__(self) -> str:
        if self.is_empty():
            return "{}"
        parts = []
        for s, e in self.intervals:
            if s == e:
                parts.append(str(s))
            else:
                parts.append(f"{s}-{e}")
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
        items = [(int(s), int(e)) for s, e in ranges if s is not None and e is not None]
        if not items:
            return []
        items.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                ce = ce if ce >= ne else ne
            else:
                merged.append((cs, ce))
                cs, ce = ns, ne
        merged.append((cs, ce))
        return merged


# ----------------------------
# Formatting helper
# ----------------------------

def format_bitvector(bv: Optional[RangeSet]) -> str:
    if not bv:
        return "{}"
    return str(bv)


# ----------------------------
# Trie Utilities (printing)
# ----------------------------

def print_trie_recursive(
    node_index: TrieNodeIndex,
    arena: Arena,
    indent: str = "",
    visited: Optional[Set[TrieNodeIndex]] = None
) -> None:
    """
    Recursively prints the structure of a Trie from a given node index.
    Copies the visited set per-branch to avoid suppressing shared subtrees.
    """
    if visited is None:
        visited = set()

    if node_index in visited:
        print(f"{indent}Node {node_index} (visited, cycle)")
        return

    visited.add(node_index)
    node = arena.get(node_index)
    if not node:
        print(f"{indent}Node {node_index} (not found in arena)")
        return

    node_value = node.get("value", {}) or {}
    end_str = "END" if node_value.get("end") else "internal"
    live_tokens: RangeSet = node_value.get("live_tokens") or RangeSet.empty()
    print(f"{indent}Node {node_index} [{end_str}] (live_tokens: {format_bitvector(live_tokens)})")

    children = node.get("children", []) or []
    if not children:
        return

    for edge_key_json, dest_map_json in children:
        pop_count, state_id_opt = edge_key_json
        edge_key_str = f"pop={pop_count}, state_id={state_id_opt if state_id_opt is not None else 'Any'}"
        print(f"{indent}  - Edge({edge_key_str}):")

        for dest_index, edge_bv in dest_map_json:
            edge_bv_rs: RangeSet = edge_bv
            print(f"{indent}    -> Dest: {dest_index} (tokens: {format_bitvector(edge_bv_rs)})")
            print_trie_recursive(dest_index, arena, indent + "       | ", visited.copy())


# ----------------------------------
# Core Equivalence / Traversal Logic (legacy, kept for backward compatibility)
# ----------------------------------

def _update_visited_bv(store: Dict[Any, RangeSet], key: Any, incoming_bv: RangeSet) -> Optional[RangeSet]:
    """
    Maintains a map of key -> accumulated BV.
    Returns the 'new portion' (difference) to propagate if any; otherwise None.
    """
    if key in store:
        existing = store[key]
        diff = incoming_bv.difference(existing)
        if diff.is_empty():
            return None
        store[key] = existing.union(diff)
        return diff
    else:
        store[key] = incoming_bv
        return incoming_bv


def find_end_bv_from_node_via_none_edges(
    start_node_index: TrieNodeIndex,
    initial_bv: RangeSet,
    arena: Arena
) -> RangeSet:
    """
    Finds the union of BVs for all paths from a start node to any `end` node
    that consist solely of `(k, None)` edges. Intersects edge BVs along the way.
    """
    if initial_bv.is_empty():
        return RangeSet.empty()

    total_end_bv: RangeSet = RangeSet.empty()
    q: Deque[Tuple[TrieNodeIndex, RangeSet]] = collections.deque()
    q.append((start_node_index, initial_bv))

    visited: Dict[TrieNodeIndex, RangeSet] = {}

    while q:
        node_idx, current_bv = q.popleft()
        node = arena.get(node_idx)
        if not node:
            continue

        node_val = node.get("value", {}) or {}
        if node_val.get("end"):
            total_end_bv = total_end_bv.union(current_bv)

        for edge_key_json, dest_map_json in node.get("children", []) or []:
            _pop_count, state_id_opt = edge_key_json
            if state_id_opt is not None:
                continue  # Only traverse (k, None) edges here

            for dest_idx, edge_bv in dest_map_json:
                edge_bv_rs: RangeSet = edge_bv
                new_bv = current_bv.intersection(edge_bv_rs)
                if new_bv.is_empty():
                    continue

                diff = _update_visited_bv(visited, dest_idx, new_bv)
                if diff is not None and not diff.is_empty():
                    q.append((dest_idx, diff))

    return total_end_bv


def get_bv_for_normalized_path(
    root_index: TrieNodeIndex,
    path: NormalizedPath,
    arena: Arena
) -> RangeSet:
    """
    For a given normalized path, computes the union of LLM token bitvectors for all
    possible ways to traverse that path in the trie.
    A normalized path is a list of (k, state_id) segments where k accumulates pop_counts
    from (k, None) edges until a (k', Some(state_id)) matches, then k resets to 0.
    """
    root_node = arena.get(root_index)
    if not root_node:
        return RangeSet.empty()

    initial_bv: RangeSet = (root_node.get("value", {}) or {}).get("live_tokens") or RangeSet.empty()
    if initial_bv.is_empty() and path:
        return RangeSet.empty()

    final_bv: RangeSet = RangeSet.empty()
    q: Deque[Tuple[TrieNodeIndex, int, int, RangeSet]] = collections.deque()
    q.append((root_index, 0, 0, initial_bv))

    # Visited key: (node_idx, path_idx, accumulated_k)
    visited: Dict[Tuple[TrieNodeIndex, int, int], RangeSet] = {(root_index, 0, 0): initial_bv}

    while q:
        node_idx, path_idx, k_so_far, current_bv = q.popleft()

        # If we've matched the full path, gather end BVs along None edges
        if path_idx == len(path):
            end_bv = find_end_bv_from_node_via_none_edges(node_idx, current_bv, arena)
            final_bv = final_bv.union(end_bv)
            continue

        target_k, target_sid = path[path_idx]
        node = arena.get(node_idx)
        if not node:
            continue

        for edge_key_json, dest_map_json in node.get("children", []) or []:
            pop_count, state_id_opt = edge_key_json
            new_k = k_so_far + pop_count

            for dest_idx, edge_bv in dest_map_json:
                edge_bv_rs: RangeSet = edge_bv
                new_bv = current_bv.intersection(edge_bv_rs)
                if new_bv.is_empty():
                    continue

                if state_id_opt is not None:
                    # Advance only if this edge completes the current (k, sid) segment
                    if new_k == target_k and state_id_opt == target_sid:
                        next_key = (dest_idx, path_idx + 1, 0)
                        diff = _update_visited_bv(visited, next_key, new_bv)
                        if diff is not None and not diff.is_empty():
                            q.append((dest_idx, path_idx + 1, 0, diff))
                else:
                    # Accumulate k along None edges
                    if new_k <= target_k:
                        cont_key = (dest_idx, path_idx, new_k)
                        diff = _update_visited_bv(visited, cont_key, new_bv)
                        if diff is not None and not diff.is_empty():
                            q.append((dest_idx, path_idx, new_k, diff))

    return final_bv


def sample_normalized_path(
    root_index: TrieNodeIndex,
    max_len: int,
    arena: Arena,
    rng: Optional[random.Random] = None
) -> Optional[NormalizedPath]:
    """
    Samples a single normalized path by performing a random walk from the root.
    Returns None if the path becomes invalid (BV becomes empty before a valid end).
    """
    if rng is None:
        rng = random

    current_node_idx = root_index
    path: NormalizedPath = []
    current_k = 0

    root_node = arena.get(root_index)
    if not root_node:
        return None

    current_bv: RangeSet = ((root_node.get("value", {}) or {}).get("live_tokens")) or RangeSet.empty()

    # Loop until the normalized path reaches max_len, with a safety break
    # for long chains of (k, None) edges. This matches the Rust reference logic.
    max_total_edges = max_len * 100  # A reasonable upper bound
    edges_walked = 0

    while len(path) < max_len:
        if edges_walked >= max_total_edges:
            break  # Safety break
        edges_walked += 1

        node = arena.get(current_node_idx)
        if not node:
            return None

        can_terminate = bool((node.get("value", {}) or {}).get("end", False))
        edges: List[Tuple[EdgeKey, TrieNodeIndex, RangeSet]] = []
        for ek, d_map in node.get("children", []) or []:
            for d_idx, e_bv in d_map:
                edges.append((ek, d_idx, e_bv))
        can_continue = bool(edges)

        if not can_continue:
            return path if can_terminate else None

        # Small chance to terminate when allowed
        if can_terminate and rng.random() < 0.2:
            return path

        edge_key_json, dest_idx, edge_bv = rng.choice(edges)

        # Intersect tokens; if empty, path is invalid
        current_bv = current_bv.intersection(edge_bv)
        if current_bv.is_empty():
            return None

        pop_count, state_id_opt = edge_key_json
        current_k += pop_count
        if state_id_opt is not None:
            path.append((current_k, state_id_opt))
            current_k = 0

        current_node_idx = dest_idx

    return path


def are_precompute2_trees_equivalent(
    root_a: TrieNodeIndex,
    arena_a: Arena,
    root_b: TrieNodeIndex,
    arena_b: Arena
) -> bool:
    """
    Checks for semantic equivalence between two precompute2 trees using stochastic sampling.
    Preserves the original algorithm and observable behavior.
    """
    if root_a == root_b and arena_a is arena_b:
        return True

    NUM_SAMPLES = 100
    MAX_PATH_LEN = 32

    # Sample from A, check in B
    for i in range(NUM_SAMPLES):
        path = sample_normalized_path(root_a, MAX_PATH_LEN, arena_a)
        if path is not None:
            bv_a = get_bv_for_normalized_path(root_a, path, arena_a)
            if bv_a.is_empty() and i > 0:
                continue
            bv_b = get_bv_for_normalized_path(root_b, path, arena_b)
            if bv_a != bv_b:
                print("\n--- Precompute2 Equivalence Mismatch (legacy) ---")
                print("Path sampled from Tree A:")
                print(f"  Path: {path}")
                print(f"  BV from A: {format_bitvector(bv_a)}")
                print(f"  BV from B: {format_bitvector(bv_b)}")
                print(f"  Difference (A ^ B): {format_bitvector(bv_a.symmetric_difference(bv_b))}")
                return False

    # Sample from B, check in A
    for i in range(NUM_SAMPLES):
        path = sample_normalized_path(root_b, MAX_PATH_LEN, arena_b)
        if path is not None:
            bv_b = get_bv_for_normalized_path(root_b, path, arena_b)
            if bv_b.is_empty() and i > 0:
                continue
            bv_a = get_bv_for_normalized_path(root_a, path, arena_a)
            if bv_a != bv_b:
                print("\n--- Precompute2 Equivalence Mismatch (legacy) ---")
                print("Path sampled from Tree B:")
                print(f"  Path: {path}")
                print(f"  BV from A: {format_bitvector(bv_a)}")
                print(f"  BV from B: {format_bitvector(bv_b)}")
                print(f"  Difference (A ^ B): {format_bitvector(bv_a.symmetric_difference(bv_b))}")
                return False

    return True


# ----------------------------
# I/O and CLI: load/convert
# ----------------------------

def _convert_node_bvs_inplace(arena: Arena) -> None:
    """
    Convert all BV lists in the arena into RangeSet instances, in-place.
    Mutates each node's 'value.live_tokens' and each edge's BV.
    """
    for node in arena.values():
        val = node.get("value", {}) or {}
        if "live_tokens" in val:
            val["live_tokens"] = RangeSet.from_json(val.get("live_tokens"))
        else:
            val["live_tokens"] = RangeSet.empty()
        node["value"] = val

        children = node.get("children", []) or []
        new_children = []
        for edge_key_json, dest_map_json in children:
            # edge_key_json: [pop_count, state_id or None]
            new_dest_map = []
            for dest_idx, edge_bv in dest_map_json:
                new_dest_map.append((int(dest_idx), RangeSet.from_json(edge_bv)))
            # normalize edge key to tuple
            if isinstance(edge_key_json, (list, tuple)) and len(edge_key_json) == 2:
                pop_count = int(edge_key_json[0])
                state_id = None if edge_key_json[1] is None else int(edge_key_json[1])
                new_children.append(((pop_count, state_id), new_dest_map))
            else:
                raise ValueError(f"Invalid edge key format: {edge_key_json!r}")
        node["children"] = new_children


def load_precompute2(path: Path) -> Tuple[List[Tuple[TokenizerStateID, TrieNodeIndex]], Arena]:
    """
    Loads a precomputed trie (precompute2 format) from a gzipped JSON file.
    Returns (roots_map, arena) where:
      - roots_map: list of (tokenizer_state_id, root_node_index)
      - arena: dict from node index to node object
    """
    try:
        with gzip.open(str(path), "rt", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse: {path} ({e})") from e

    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("Expected top-level JSON to be a list of two elements: [roots_map, arena_data]")

    roots_map_json, arena_data = data

    # Parse arena into a dictionary for easy lookup by integer index
    try:
        arena: Arena = {int(index): node_data for index, node_data in arena_data.get("values", [])}
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing arena data: {e}") from e

    # Normalize roots_map: ensure ints
    try:
        roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]] = [
            (int(tok_state_id), int(root_idx)) for tok_state_id, root_idx in roots_map_json
        ]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing roots map: {e}") from e

    # Convert all BVs to RangeSet for performance
    _convert_node_bvs_inplace(arena)

    return roots_map, arena


# ----------------------------
# Normalized (token-specific) NFA construction and equivalence
# ----------------------------

# The alphabet for normalized graphs: labels are pairs (k, sid)
Label = Tuple[int, int]  # (k, state_id)
NormEdge = Tuple[Label, TrieNodeIndex]  # (label, dest_node_index)


class EdgeProvider:
    """
    Abstract thin adapter around a graph that supports:
      - iter_edges(node, token) -> iterator of (pop_count, state_id or None, dest_node)
      - is_end(node) -> bool
    """
    def iter_edges(self, node: TrieNodeIndex, token: int) -> Iterator[Tuple[int, Optional[int], TrieNodeIndex]]:
        raise NotImplementedError

    def is_end(self, node: TrieNodeIndex) -> bool:
        raise NotImplementedError


class RefEdgeProvider(EdgeProvider):
    """
    EdgeProvider for the reference precompute2 arena.
    Filters edges by checking the LLM token membership in per-edge RangeSet.
    """
    def __init__(self, arena: Arena):
        self.arena = arena

    def iter_edges(self, node: TrieNodeIndex, token: int) -> Iterator[Tuple[int, Optional[int], TrieNodeIndex]]:
        n = self.arena.get(node)
        if not n:
            return
        for (pop_count, sid_opt), dest_map in n.get("children", []) or []:
            for dest_idx, edge_bv in dest_map:
                edge_bv_rs: RangeSet = edge_bv
                if edge_bv_rs.contains(token):
                    yield (int(pop_count), int(sid_opt) if sid_opt is not None else None, int(dest_idx))

    def is_end(self, node: TrieNodeIndex) -> bool:
        n = self.arena.get(node)
        if not n:
            return False
        return bool((n.get("value", {}) or {}).get("end", False))


class PluginEdgeProvider(EdgeProvider):
    """
    EdgeProvider that delegates to a plugin's API functions.
    """
    def __init__(
        self,
        structure: Any,
        iter_edges_func: Callable[..., Iterable[Tuple[int, Optional[int], TrieNodeIndex]]],
        is_end_func: Callable[..., bool],
    ):
        self.structure = structure
        self._iter_edges_func = iter_edges_func
        self._is_end_func = is_end_func

    def iter_edges(self, node: TrieNodeIndex, token: int) -> Iterator[Tuple[int, Optional[int], TrieNodeIndex]]:
        # Allow flexible signatures: (structure, node, token) | (node, token)
        try:
            out = self._iter_edges_func(self.structure, node, token)
        except TypeError:
            out = self._iter_edges_func(node, token)
        for pop_count, sid_opt, dest in out:
            yield (int(pop_count), int(sid_opt) if sid_opt is not None else None, int(dest))

    def is_end(self, node: TrieNodeIndex) -> bool:
        # Allow flexible signatures: (structure, node) | (node)
        try:
            return bool(self._is_end_func(self.structure, node))
        except TypeError:
            return bool(self._is_end_func(node))


class TokenNormalizer:
    """
    Builds token-specific normalized edges (epsilon-free) on demand.
    - Edges: from a node u, explore all 0+ (None) edges whose BVs contain 'token'.
             For each path to a node v where we see a (Some state_id) edge e=(p, sid, w),
             we add a normalized edge from u to w labeled (k_sum + p, sid),
             where k_sum is the sum of pop_counts along the None-only path.
    - Accepting: a node u is accepting if there exists a path through only (None) edges
                 (filtered by token) from u to any node with end=True.
    """
    def __init__(self, provider: EdgeProvider, token: int, max_closure_expansions: int = 200000):
        self.provider = provider
        self.token = int(token)
        self._accept_cache: Dict[TrieNodeIndex, bool] = {}
        self._edges_cache: Dict[TrieNodeIndex, List[NormEdge]] = {}
        self._max_closure_expansions = max(1, int(max_closure_expansions))

    def accepting(self, node: TrieNodeIndex) -> bool:
        if node in self._accept_cache:
            return self._accept_cache[node]
        # BFS over None-edges only
        q: Deque[TrieNodeIndex] = collections.deque([node])
        seen: Set[TrieNodeIndex] = set()
        accepts = False
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            if self.provider.is_end(u):
                accepts = True
                break
            for pop, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is None:
                    # pop_count is ignored for acceptance test at end-of-path
                    q.append(v)
        self._accept_cache[node] = accepts
        return accepts

    def out_edges(self, node: TrieNodeIndex) -> List[NormEdge]:
        if node in self._edges_cache:
            return self._edges_cache[node]

        # Enumerate all None-only paths from node; collect (k_sum, u) pairs
        # Then for each u, add normalized edges via its Some-edges.
        out: List[NormEdge] = []

        # Track expansions to avoid runaway in pathological graphs
        expansions = 0

        # Each state in the stack: (u, k_sum)
        stack: List[Tuple[TrieNodeIndex, int]] = [(node, 0)]
        # We must allow visiting the same u with different k_sum (they yield distinct labels).
        seen: Set[Tuple[TrieNodeIndex, int]] = set()

        while stack:
            u, ksum = stack.pop()
            key = (u, ksum)
            if key in seen:
                continue
            seen.add(key)

            # Add edges for Some transitions at u
            for p, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is not None:
                    label: Label = (ksum + p, sid_opt)
                    out.append((label, v))

            # Continue exploring None transitions at u
            for p, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is None:
                    # accumulate pop_count into ksum
                    new_ksum = ksum + p
                    stack.append((v, new_ksum))
                    expansions += 1
                    if expansions > self._max_closure_expansions:
                        raise RuntimeError(
                            "Exceeded maximum None-closure expansions while normalizing for token "
                            f"{self.token}. Detected potentially unbounded None-edge cycles with positive pop. "
                            "Normalization aborted to avoid infinite enumeration."
                        )

        # Deduplicate edges (there can be duplicates by design)
        # Use a dict keyed by (k, sid, dest) to remove duplicates but preserve determinism.
        dedup: Dict[Tuple[int, int, int], None] = {}
        norm_edges: List[NormEdge] = []
        for (label, dest) in out:
            k, sid = label
            key = (k, sid, dest)
            if key not in dedup:
                dedup[key] = None
                norm_edges.append((label, dest))

        self._edges_cache[node] = norm_edges
        return norm_edges


def nfa_equivalence_on_labels(
    normA: TokenNormalizer,
    rootA: TrieNodeIndex,
    normB: TokenNormalizer,
    rootB: TrieNodeIndex,
    max_product_states: int = 200000,
) -> Tuple[bool, Optional[List[Label]]]:
    """
    Language equivalence check for two epsilon-free NFAs over alphabet of (k, sid) labels,
    with an added structural sanity check:
      - If at any reachable pair of subset-states, the sets of outgoing labels differ,
        we immediately report a mismatch and return a witness label sequence that exposes
        the divergence.

    Uses on-the-fly subset construction product and BFS.
    Returns (equivalent, counterexample_label_sequence_if_not).
    """
    # Helper to get acceptance for subset
    def subset_accepts(nodes: Set[TrieNodeIndex], norm: TokenNormalizer) -> bool:
        for n in nodes:
            if norm.accepting(n):
                return True
        return False

    # Helper to get next subset under a label from a subset
    def next_subset(nodes: Set[TrieNodeIndex], norm: TokenNormalizer, label: Label) -> Set[TrieNodeIndex]:
        k, sid = label
        out: Set[TrieNodeIndex] = set()
        for n in nodes:
            for (lbl, dest) in norm.out_edges(n):
                if lbl == label:
                    out.add(dest)
        return out

    # Helper to collect all outgoing labels from a subset
    def labels_from_subset(nodes: Set[TrieNodeIndex], norm: TokenNormalizer) -> Set[Label]:
        lab: Set[Label] = set()
        for n in nodes:
            for (lbl, _dest) in norm.out_edges(n):
                lab.add(lbl)
        return lab

    # BFS over pairs of subsets
    startA: frozenset = frozenset({rootA})
    startB: frozenset = frozenset({rootB})

    # Parent map to reconstruct counterexample
    ParentKey = Tuple[frozenset, frozenset]
    parent: Dict[ParentKey, Tuple[Optional[ParentKey], Optional[Label]]] = {}

    def _reconstruct_path(to_key: ParentKey) -> List[Label]:
        """
        Reconstruct label sequence from start pair to 'to_key' (exclusive of any extra
        step taken afterwards). The start pair has (None, None).
        """
        seq: List[Label] = []
        cur: Optional[ParentKey] = to_key
        while cur is not None:
            par, edge_lab = parent[cur]
            if edge_lab is not None:
                seq.append(edge_lab)
            cur = par
        seq.reverse()
        return seq
    parent[(startA, startB)] = (None, None)

    # Visited pairs
    visited: Set[ParentKey] = set()
    q: Deque[ParentKey] = collections.deque()
    q.append((startA, startB))

    # Early acceptance mismatch on empty sequence
    if subset_accepts(set(startA), normA) != subset_accepts(set(startB), normB):
        return (False, [])  # empty label sequence witnesses mismatch

    explored = 0
    while q:
        SA, SB = q.popleft()
        if (SA, SB) in visited:
            continue
        visited.add((SA, SB))
        explored += 1
        if explored > max_product_states:
            raise RuntimeError("Equivalence check exceeded product state limit; graph too large or highly nondeterministic.")

        # Collect all labels from either side
        labels_A = labels_from_subset(set(SA), normA)
        labels_B = labels_from_subset(set(SB), normB)
        labels = labels_A | labels_B

        # Structural sanity: if outgoing label sets differ at this reachable product state,
        # consider it a mismatch and return a short witness (path to here + a differing label).
        if labels_A != labels_B:
            # Choose any differing label as the final step of the witness
            diff_labels = (labels_A - labels_B) or (labels_B - labels_A)
            witness_step = next(iter(diff_labels))
            # Reconstruct path to this state and append the differing label
            seq = _reconstruct_path((SA, SB))
            seq.append(witness_step)
            return (False, seq)

        for lab in labels:
            NSA = frozenset(next_subset(set(SA), normA, lab))
            NSB = frozenset(next_subset(set(SB), normB, lab))

            key = (NSA, NSB)
            if key not in parent:
                parent[key] = ((SA, SB), lab)

            # Check acceptance mismatch at this DFA state (i.e., empty suffix)
            if subset_accepts(set(NSA), normA) != subset_accepts(set(NSB), normB):
                # Reconstruct counterexample: the path of labels leading to key
                seq: List[Label] = []
                cur: Optional[ParentKey] = key
                while cur is not None:
                    par, edge_lab = parent[cur]
                    if edge_lab is not None:
                        seq.append(edge_lab)
                    cur = par
                seq.reverse()
                return (False, seq)

            if key not in visited:
                q.append(key)

    return (True, None)


# ----------------------------
# Scoring utilities (competition) (legacy score preserved)
# ----------------------------

def collect_all_state_ids(arena: Arena) -> List[StateID]:
    """
    Collect a pool of observed StateIDs from the arena edges (with Some(state_id)).
    """
    pool: Set[int] = set()
    for node in arena.values():
        for (pop_count, state_id_opt), _dest_map in node.get("children", []) or []:
            if state_id_opt is not None:
                pool.add(int(state_id_opt))
    return list(pool) or [0]


def _try_call_plugin(func: Callable, structure: Any, *args) -> Any:
    """
    Call a plugin function supporting flexible signatures by trying:
      1) func(structure, *args)
      2) func(*args)
    """
    try:
        return func(structure, *args)
    except TypeError:
        return func(*args)


def _try_call_stat(func: Callable, structure: Any) -> Any:
    """
    Call a plugin stat function by trying:
      1) func(structure)
      2) func()
    """
    try:
        return func(structure)
    except TypeError:
        return func()


def _load_plugin(module_path_or_name: str):
    """
    Load a competitor plugin module either by file path or module name.
    """
    p = Path(module_path_or_name)
    if p.exists() and p.is_file():
        spec = importlib.util.spec_from_file_location("competitor_plugin", str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from path: {module_path_or_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module
    return importlib.import_module(module_path_or_name)


def _build_competitor_structure(
    plugin_module: Any,
    precompute_path: Path,
    roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]],
    arena: Arena,
) -> Any:
    """
    Build competitor structure by trying, in order:
      - build_from_precompute2_path(path)
      - build(roots_map, arena)
      - init(roots_map, arena)
    Returns the structure object (or None if not provided).
    """
    if hasattr(plugin_module, "build_from_precompute2_path"):
        return plugin_module.build_from_precompute2_path(str(precompute_path))
    if hasattr(plugin_module, "build"):
        return plugin_module.build(roots_map, arena)
    if hasattr(plugin_module, "init"):
        return plugin_module.init(roots_map, arena)
    return None


def _get_competitor_size_functions(plugin_module: Any) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
    """
    Return potential functions: (nodes_func, edges_func, stats_func)
    Where:
      - nodes_func: returns node count (int)
      - edges_func: returns edge count (int)
      - stats_func: returns dict-like with keys 'nodes' and/or 'edges'
    Any or all can be None.
    """
    nodes_candidates = ["count_nodes", "num_nodes", "node_count", "nodes", "get_node_count"]
    edges_candidates = ["count_edges", "num_edges", "edge_count", "edges", "get_edge_count"]
    stats_candidates = ["stats", "get_stats", "summary", "info"]

    nodes_func = None
    edges_func = None
    stats_func = None

    for name in nodes_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            nodes_func = getattr(plugin_module, name)
            break
    for name in edges_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            edges_func = getattr(plugin_module, name)
            break
    for name in stats_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            stats_func = getattr(plugin_module, name)
            break

    return nodes_func, edges_func, stats_func


def score_competitor_on_file(
    precompute_path: Path,
    plugin_module_path: str,
    state_id_filter: Optional[int] = None,
    tokens: Optional[str] = None,
    sample_tokens: Optional[int] = 256,
    seed: Optional[int] = 0,
    max_closure_expansions: int = 200000,
    max_product_states: int = 200000,
) -> None:
    """
    End-to-end scorer:
      - loads reference precompute2
      - imports competitor plugin module
      - builds competitor structure
      - runs a deterministic equivalence check against the reference.
      - if equivalent, the score is based on the plugin-reported edge count.
    Prints a summary report with correctness and size metrics.
    """
    t0 = time.time()

    # Load reference and plugin
    print(f"Loading reference file: {precompute_path}")
    roots_map, arena = load_precompute2(precompute_path)
    print(f"Loading plugin: {plugin_module_path}")
    plugin_module = _load_plugin(plugin_module_path)
    print("Building plugin structure...")
    structure = _build_competitor_structure(plugin_module, precompute_path, roots_map, arena)

    # Check for graph API, which is required for scoring.
    iter_edges_func, is_end_func, get_root_func = _get_plugin_graph_api(plugin_module)
    if iter_edges_func is None or is_end_func is None or get_root_func is None:
        print("\nERROR: Plugin does not expose the required graph API for equivalence checking.")
        print("Please implement: get_root(...), iter_edges(...), and is_end(...)")
        print("Scoring aborted.")
        return

    # Determine which state IDs to test
    state_ids_to_test: List[int]
    if state_id_filter is not None:
        if state_id_filter not in dict(roots_map):
            raise ValueError(f"state-id {state_id_filter} not found in roots map")
        state_ids_to_test = [state_id_filter]
    else:
        state_ids_to_test = sorted([sid for sid, _ in roots_map])

    # Token selection logic
    tokens_list: Optional[List[int]] = None
    sample_n: Optional[int] = sample_tokens
    if tokens is not None:
        if tokens.strip() == "@all":
            tokens_list = None
            sample_n = None
        else:
            tokens_list = [int(x) for x in tokens.split(",") if x.strip() != ""]
            sample_n = None

    # Run equivalence check
    overall_ok = True
    failed_states: List[int] = []

    for i, state_id in enumerate(state_ids_to_test, 1):
        print("-" * 40)
        print(f"[{i}/{len(state_ids_to_test)}] Scoring State ID: {state_id}")
        ok = run_equivalence_check_for_state(
            roots_map=roots_map,
            arena=arena,
            plugin_module=plugin_module,
            structure=structure,
            state_id=state_id,
            tokens=tokens_list,
            sample_tokens=sample_n,
            seed=seed,
            max_closure_expansions=max_closure_expansions,
            max_product_states=max_product_states,
        )
        if not ok:
            overall_ok = False
            failed_states.append(state_id)

    dur = time.time() - t0

    # Get size stats
    nodes_func, edges_func, stats_func = _get_competitor_size_functions(plugin_module)

    def plugin_counts() -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
        """
        Attempt to obtain (nodes, edges, raw_stats_dict) from plugin.
        Returns (nodes, edges, stats_dict). Any element may be None if unavailable.
        """
        stats: Dict[str, Any] = {}
        nodes: Optional[int] = None
        edges: Optional[int] = None

        # Prefer stats() if available
        if stats_func is not None:
            try:
                s = _try_call_stat(stats_func, structure)
                if isinstance(s, dict):
                    stats = dict(s)
                    n = s.get("nodes")
                    e = s.get("edges")
                    if isinstance(n, int) and n >= 0:
                        nodes = n
                    if isinstance(e, int) and e >= 0:
                        edges = e
            except Exception:
                pass

        # Direct node/edge funcs override or fill gaps
        if nodes is None and nodes_func is not None:
            try:
                n = _try_call_stat(nodes_func, structure)
                if isinstance(n, int) and n >= 0:
                    nodes = n
            except Exception:
                pass
        if edges is None and edges_func is not None:
            try:
                e = _try_call_stat(edges_func, structure)
                if isinstance(e, int) and e >= 0:
                    edges = e
            except Exception:
                pass

        # As a very last resort, allow module-level constants
        if nodes is None:
            mn = getattr(plugin_module, "NODES", None)
            if isinstance(mn, int) and mn >= 0:
                nodes = mn
        if edges is None:
            me = getattr(plugin_module, "EDGES", None)
            if isinstance(me, int) and me >= 0:
                edges = me

        return nodes, edges, stats

    nodes_reported, edges_reported, stats_raw = plugin_counts()

    print("\n" + "=" * 40)
    print("Overall Scoring Summary")
    print(f"File: {precompute_path}")
    print(f"Plugin: {plugin_module_path}")
    print(f"Total states tested: {len(state_ids_to_test)}")
    print(f"Correctness: {'PASS' if overall_ok else 'FAIL'}")
    if not overall_ok:
        print(f"  Mismatches found in states: {failed_states}")

    print("\n--- Contestant-reported size ---")
    if nodes_reported is not None:
        print(f"Nodes: {nodes_reported}")
    if edges_reported is not None:
        print(f"Edges: {edges_reported}  [Score is based on this if Correctness=PASS]")
    else:
        print("Edges: Not reported. [This is the primary scoring metric]")

    if stats_raw:
        extra = {k: v for k, v in stats_raw.items() if k not in ("nodes", "edges")}
        if extra:
            print(f"Other stats: {extra}")

    print(f"\nDone in {dur:.2f}s.")


# New: plugin graph API helpers
def _get_plugin_graph_api(plugin_module: Any) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
    """
    Returns (iter_edges_func, is_end_func, get_root_func) from the plugin if present.
    - iter_edges(structure, node, token) -> iterable of (pop_count, state_id or None, dest)
    - is_end(structure, node) -> bool
    - get_root(structure, state_id) -> node
    """
    iter_edges_candidates = ["iter_edges", "edges_for_token", "edges"]
    is_end_candidates = ["is_end", "is_accepting", "end"]
    get_root_candidates = ["get_root", "root_for_state", "get_root_for_state", "root"]

    iter_edges_func = None
    is_end_func = None
    get_root_func = None

    for name in iter_edges_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            iter_edges_func = getattr(plugin_module, name)
            break
    for name in is_end_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            is_end_func = getattr(plugin_module, name)
            break
    for name in get_root_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            get_root_func = getattr(plugin_module, name)
            break

    return iter_edges_func, is_end_func, get_root_func


# ----------------------------
# Token utilities: collect/iterate tokens from arena
# ----------------------------

def collect_interesting_tokens_from_arena(arena: Arena) -> List[int]:
    """
    Collects tokens that are likely to expose behavioral differences.
    These are the start/end points of ranges, and the points adjacent to them.
    """
    points: Set[int] = set()
    # Add 0 as a baseline token to test, as it's often special.
    points.add(0)
    for node in arena.values():
        for _ek, dest_map in node.get("children", []) or []:
            for _dest, bv in dest_map:
                # The loader ensures bv is a RangeSet
                rs: RangeSet = bv
                for s, e in rs.intervals:
                    if s > 0:
                        points.add(s - 1)
                    points.add(s)
                    points.add(e)
                    # Python handles large integers, so overflow isn't an issue.
                    points.add(e + 1)
    return sorted(list(points))


# New: collect tokens only from the subgraph reachable from a specific root
def collect_interesting_tokens_from_root(arena: Arena, root_index: TrieNodeIndex) -> List[int]:
    """
    Collects interesting tokens (range boundaries +/- 1) that appear on edges reachable
    from the provided root node. This focuses token sampling on what the chosen state
    can actually see, making equivalence checks harder to spoof by returning unrelated edges.
    """
    if root_index not in arena:
        return []

    points: Set[int] = set()
    points.add(0)  # baseline token

    visited: Set[TrieNodeIndex] = set()
    q: Deque[TrieNodeIndex] = collections.deque([root_index])

    while q:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        node = arena.get(u)
        if not node:
            continue
        for _ek, dest_map in node.get("children", []) or []:
            for dest, bv in dest_map:
                rs: RangeSet = bv
                for s, e in rs.intervals:
                    if s > 0:
                        points.add(s - 1)
                    points.add(s)
                    points.add(e)
                    points.add(e + 1)
                q.append(int(dest))
    return sorted(list(points))

# ----------------------------
# New equivalence runner (plugin vs reference), per-token
# ----------------------------

def check_equiv_for_state_and_token(
    roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]],
    arena: Arena,
    plugin_module: Any,
    structure: Any,
    state_id: int,
    token: int,
    max_closure_expansions: int = 200000,
    max_product_states: int = 200000,
) -> Tuple[bool, Optional[List[Label]]]:
    """
    For a given tokenizer state and token, build token-specific normalized NFAs
    for both the reference arena and the plugin and test language equivalence.
    Returns (equivalent, counterexample_label_sequence_if_not).
    """
    # Resolve reference root
    roots_map_dict = dict(roots_map)
    if state_id not in roots_map_dict:
        raise ValueError(f"State-id {state_id} not found in reference roots map.")
    ref_root = int(roots_map_dict[state_id])

    # Resolve plugin root
    iter_edges_func, is_end_func, get_root_func = _get_plugin_graph_api(plugin_module)
    if iter_edges_func is None or is_end_func is None or get_root_func is None:
        raise ValueError(
            "Plugin must provide graph API: iter_edges(...), is_end(...), get_root(...). "
            "See example_plugin.py for the required signatures."
        )
    try:
        plugin_root = int(_try_call_plugin(get_root_func, structure, int(state_id)))
    except Exception as e:
        raise RuntimeError(f"Error obtaining plugin root for state {state_id}: {e}") from e

    # Build providers and normalizers
    ref_provider = RefEdgeProvider(arena)
    plugin_provider = PluginEdgeProvider(structure, iter_edges_func, is_end_func)

    ref_norm = TokenNormalizer(ref_provider, token, max_closure_expansions=max_closure_expansions)
    plugin_norm = TokenNormalizer(plugin_provider, token, max_closure_expansions=max_closure_expansions)

    # Check equivalence
    eq, witness = nfa_equivalence_on_labels(
        ref_norm, ref_root, plugin_norm, plugin_root, max_product_states=max_product_states
    )
    return (eq, witness)


def run_equivalence_check_for_state(
    roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]],
    arena: Arena,
    plugin_module: Any,
    structure: Any,
    state_id: int,
    tokens: Optional[List[int]] = None,
    sample_tokens: Optional[int] = 256,
    seed: Optional[int] = 0,
    max_closure_expansions: int = 200000,
    max_product_states: int = 200000,
) -> bool:
    """
    High-level runner for a single state_id:
      - Selects tokens (provided or sampled from reference)
      - For each token, compares plugin vs reference by token-normalized NFA equivalence
    Prints summary and returns True if all tested tokens are equivalent for this state.
    Assumes reference and plugin structures are already loaded.
    """
    # Prepare tokens
    roots_map_dict = dict(roots_map)
    if state_id not in roots_map_dict:
        print(f"Warning: Tokenizer state ID {state_id} not found in reference roots map. Skipping.")
        return True
    ref_root = int(roots_map_dict[state_id])

    rng = random.Random(seed)
    tested_tokens: List[int] = []
    if tokens:
        tested_tokens = [int(t) for t in tokens]
    else:
        # Prefer tokens that are reachable from the chosen state's root
        interesting_tokens = collect_interesting_tokens_from_root(arena, ref_root)
        if not interesting_tokens:
            # Fallback: use global arena tokens (should be rare; e.g., missing root)
            interesting_tokens = collect_interesting_tokens_from_arena(arena)

        # If still empty, nothing to compare for this state
        if not interesting_tokens:
            print("Reference contains no tokens reachable from the specified state. Nothing to compare.")
            return True
        if not interesting_tokens:
            print("Reference contains no tokens in any edge. Nothing to compare.")
            return True

        if sample_tokens is None:
            # Exhaustive: use all interesting tokens
            tested_tokens = interesting_tokens
        else:
            num_to_sample = int(sample_tokens)
            if len(interesting_tokens) <= num_to_sample:
                tested_tokens = interesting_tokens
            else:
                tested_tokens = rng.sample(interesting_tokens, num_to_sample)
    all_ok = True
    start_time = time.time()
    print(f"Testing equivalence for state-id={state_id} over {len(tested_tokens)} token(s).")
    for idx, tok in enumerate(tested_tokens, 1):
        try:
            ok, witness = check_equiv_for_state_and_token(
                roots_map,
                arena,
                plugin_module,
                structure,
                int(state_id),
                int(tok),
                max_closure_expansions=max_closure_expansions,
                max_product_states=max_product_states,
            )
        except RuntimeError as e:
            print(f"[{idx}/{len(tested_tokens)}] token={tok}: ERROR during normalization/equivalence: {e}")
            all_ok = False
            continue

        if not ok:
            all_ok = False
            print(f"[{idx}/{len(tested_tokens)}] token={tok}: NOT EQUIVALENT.")
            if witness is not None:
                # Pretty print the label sequence
                seq_str = " -> ".join(f"(k={k}, sid={sid})" for (k, sid) in witness) if witness else "(empty)"
                print(f"  Counterexample label sequence: {seq_str}")
        else:
            print(f"[{idx}/{len(tested_tokens)}] token={tok}: OK")

    dur = time.time() - start_time
    print(f"\nDone in {dur:.2f}s. Result: {'EQUIVALENT' if all_ok else 'NOT EQUIVALENT'} over tested tokens.")
    return all_ok


# ----------------------------
# CLI: Inspect / Score / Equiv
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and test precompute2 tries (gzipped JSON). Also acts as a scoring harness for contestant implementations."
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Sub-commands")

    # Inspect command (default behavior)
    inspect_p = subparsers.add_parser("inspect", help="Inspect/print trie structure")
    inspect_p.add_argument(
        "file",
        nargs="?",
        default="/Users/isaacbreen/Projects2/grammars2024/.cache/test_precompute2/precomputed2_js_gpt2_small.json.gz",
        help="Path to a precompute2 gzipped JSON file. Defaults to a known test path."
    )
    inspect_p.add_argument("--state-id", type=int, default=None, help="Print only the trie for this tokenizer state ID.")
    inspect_p.add_argument("--root-index", type=int, default=None, help="Print from this root node index.")
    inspect_p.add_argument("--max-roots", type=int, default=None, help="Limit the number of roots to print.")

    # Score command
    score_p = subparsers.add_parser("score", help="Score a contestant module by correctness (via equivalence check) and size.")
    score_p.add_argument("file", help="Path to a precompute2 gzipped JSON file.")
    score_p.add_argument("plugin", help="Path or module name for the contestant plugin.")
    score_p.add_argument("--state-id", type=int, default=None, help="If provided, score only this tokenizer state ID.")
    group = score_p.add_mutually_exclusive_group()
    group.add_argument("--tokens", type=str, default=None, help="Comma-separated list of token IDs to test, or '@all' to test all tokens present in reference ranges.")
    group.add_argument("--sample", type=int, default=256, help="Number of tokens to sample for equivalence checks (default: 256).")
    score_p.add_argument("--seed", type=int, default=0, help="Random seed for token sampling.")
    score_p.add_argument("--max-closure", type=int, default=200000, help="Limit for None-closure expansions during normalization.")
    score_p.add_argument("--max-states", type=int, default=200000, help="Limit for product DFA states during equivalence.")

    # Equivalence command (NEW): compare plugin vs reference by per-token normalized NFA equivalence
    equiv_p = subparsers.add_parser("equiv", help="Equivalence check: plugin vs reference by token-normalized NFA equivalence")
    equiv_p.add_argument("file", help="Path to a precompute2 gzipped JSON file.")
    equiv_p.add_argument("plugin", help="Path or module name for the contestant plugin.")
    equiv_p.add_argument("--state-id", type=int, default=None, help="Tokenizer state ID to compare. If not provided, all states are tested.")
    group = equiv_p.add_mutually_exclusive_group()
    group.add_argument("--tokens", type=str, default=None,
                       help="Comma-separated list of token IDs to test, or '@all' to test all tokens present in reference ranges.")
    group.add_argument("--sample", type=int, default=256, help="Number of tokens to sample from reference ranges (default: 256).")
    equiv_p.add_argument("--seed", type=int, default=0, help="Random seed for token sampling.")
    equiv_p.add_argument("--max-closure", type=int, default=200000, help="Limit for None-closure expansions during normalization.")
    equiv_p.add_argument("--max-states", type=int, default=200000, help="Limit for product DFA states during equivalence.")

    # Equivalence command (legacy files vs files checker)
    equiv_files_p = subparsers.add_parser("equiv-files", help="Stochastic equivalence check between two precompute2 tries (legacy)")
    equiv_files_p.add_argument("file_a", help="Path to first precompute2 gzipped JSON file.")
    equiv_files_p.add_argument("file_b", help="Path to second precompute2 gzipped JSON file.")
    equiv_files_p.add_argument("--state-id", type=int, required=True, help="Tokenizer state ID to compare.")
    equiv_files_p.add_argument("--root-a", type=int, default=None, help="Override root node index for A.")
    equiv_files_p.add_argument("--root-b", type=int, default=None, help="Override root node index for B.")

    args = parser.parse_args()

    if args.cmd is None or args.cmd == "inspect":
        file_path = Path(args.file)
        print(f"Loading precomputed trie from: {file_path}")
        try:
            roots_map, arena = load_precompute2(file_path)
        except Exception as e:
            print(str(e))
            return

        print("\nSuccessfully loaded trie data.")
        print(f"Arena contains {len(arena)} nodes.")
        print(f"Found {len(roots_map)} root entries for different tokenizer states.")
        print("-" * 40)

        to_print: List[Tuple[TokenizerStateID, TrieNodeIndex]] = []
        if getattr(args, "root_index", None) is not None:
            if getattr(args, "state_id", None) is not None:
                to_print.append((args.state_id, args.root_index))
            else:
                to_print.append((-1, args.root_index))
        elif getattr(args, "state_id", None) is not None:
            to_print = [(sid, ridx) for sid, ridx in roots_map if sid == args.state_id]
            if not to_print:
                print(f"No root found for tokenizer state ID {args.state_id}")
                return
        else:
            to_print = roots_map

        if getattr(args, "max_roots", None) is not None:
            to_print = to_print[: max(0, args.max_roots)]

        for tokenizer_state_id, root_node_index in to_print:
            print(f"\n\n=== Trie for Tokenizer State ID: {tokenizer_state_id} (Root Node: {root_node_index}) ===")
            print_trie_recursive(root_node_index, arena)
            print("-" * 40)
        return

    if args.cmd == "score":
        file_path = Path(args.file)
        score_competitor_on_file(
            precompute_path=file_path,
            plugin_module_path=args.plugin,
            state_id_filter=args.state_id,
            tokens=args.tokens,
            sample_tokens=args.sample,
            seed=args.seed,
            max_closure_expansions=args.max_closure,
            max_product_states=args.max_states,
        )
        return

    if args.cmd == "equiv":
        file_path = Path(args.file)

        # Load reference and plugin once
        print(f"Loading reference file: {file_path}")
        roots_map, arena = load_precompute2(file_path)
        print(f"Loading plugin: {args.plugin}")
        plugin_module = _load_plugin(args.plugin)
        print("Building plugin structure...")
        structure = _build_competitor_structure(plugin_module, file_path, roots_map, arena)
        if structure is None:
            print("Error: Plugin did not provide a builder (build/init/build_from_precompute2_path).")
            return

        # Determine which state IDs to test
        state_ids_to_test: List[int]
        if args.state_id is not None:
            state_ids_to_test = [int(args.state_id)]
        else:
            state_ids_to_test = sorted([sid for sid, _ in roots_map])
            print(f"\nNo --state-id provided. Testing all {len(state_ids_to_test)} states found in the file.")

        # Common token selection logic
        tokens_list: Optional[List[int]] = None
        sample_n: Optional[int] = args.sample
        if args.tokens is not None:
            if args.tokens.strip() == "@all":
                tokens_list = None
                sample_n = None
            else:
                tokens_list = [int(x) for x in args.tokens.split(",") if x.strip() != ""]
                sample_n = None

        overall_ok = True
        failed_states: List[int] = []
        start_time = time.time()

        for i, state_id in enumerate(state_ids_to_test, 1):
            print("-" * 40)
            print(f"[{i}/{len(state_ids_to_test)}] Testing State ID: {state_id}")
            ok = run_equivalence_check_for_state(
                roots_map=roots_map,
                arena=arena,
                plugin_module=plugin_module,
                structure=structure,
                state_id=state_id,
                tokens=tokens_list,
                sample_tokens=sample_n,
                seed=args.seed,
                max_closure_expansions=args.max_closure,
                max_product_states=args.max_states,
            )
            if not ok:
                overall_ok = False
                failed_states.append(state_id)

        dur = time.time() - start_time
        print("\n" + "=" * 40)
        print("Overall Equivalence Check Summary")
        print(f"Total states tested: {len(state_ids_to_test)}")
        print(f"Total time: {dur:.2f}s")
        if overall_ok:
            print("Result: PASS. All tested states are equivalent.")
        else:
            print(f"Result: FAIL. The following states had mismatches: {failed_states}")
        return

    if args.cmd == "equiv-files":
        file_a = Path(args.file_a)
        file_b = Path(args.file_b)
        sid = args.state_id
        roots_a, arena_a = load_precompute2(file_a)
        roots_b, arena_b = load_precompute2(file_b)
        roots_a_map = dict(roots_a)
        roots_b_map = dict(roots_b)
        root_a_idx = args.root_a if args.root_a is not None else roots_a_map.get(sid)
        root_b_idx = args.root_b if args.root_b is not None else roots_b_map.get(sid)
        if root_a_idx is None or root_b_idx is None:
            print("Could not resolve root indices for provided state-id.")
            return
        ok = are_precompute2_trees_equivalent(root_a_idx, arena_a, root_b_idx, arena_b)
        print("Equivalent." if ok else "Not equivalent.")
        return


if __name__ == "__main__":
    main()
