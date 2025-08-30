import argparse
import collections
import gzip
import importlib
import importlib.util
import inspect
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
# JSON encoding for BVs: List of [start, end] inclusive ranges
LLMTokenBVJSON = List[List[int]]


# ----------------------------
# Efficient RangeSet for token bitvectors
# ----------------------------

@dataclass(frozen=True)
class RangeSet:
    """
    Efficient, normalized (sorted, disjoint, inclusive) intervals for large, sparse token sets.
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

    # ---------- Set operations (all return new RangeSet) ----------

    def union(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty():
            return other
        if other.is_empty():
            return self
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
            start = max(s1, s2)
            end = min(e1, e2)
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
                ce = max(ce, ne)
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
# Core Equivalence / Traversal Logic
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

    for _ in range(max_len):
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
                print("\n--- Precompute2 Equivalence Mismatch ---")
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
                print("\n--- Precompute2 Equivalence Mismatch ---")
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
    for node_idx, node in arena.items():
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
            new_children.append((tuple(edge_key_json), new_dest_map))
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
# Scoring utilities (competition)
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


def mutate_path(path: NormalizedPath, rng: random.Random, state_pool: List[StateID]) -> NormalizedPath:
    """
    Create a mutated variant of a path. Mutations may create invalid paths (by design).
    """
    if not path:
        # If empty, likely mutations are insert or keep empty
        ops = ["insert", "noop"]
    else:
        ops = ["insert", "delete", "modify_k", "modify_sid", "dup", "swap"]
        if len(path) >= 2:
            ops.extend(["merge"])

    op = rng.choice(ops)
    p = list(path)

    if op == "noop":
        return p

    if op == "insert":
        k = max(0, int(rng.gauss(1, 2)))
        sid = rng.choice(state_pool) if rng.random() < 0.8 else rng.randint(-10, 10_000_000)
        pos = rng.randint(0, len(p))
        p.insert(pos, (k, sid))
        return p

    if op == "delete" and p:
        pos = rng.randrange(len(p))
        del p[pos]
        return p

    if op == "modify_k" and p:
        pos = rng.randrange(len(p))
        k, sid = p[pos]
        delta = rng.randint(-2, 3)
        p[pos] = (max(0, k + delta), sid)
        return p

    if op == "modify_sid" and p:
        pos = rng.randrange(len(p))
        k, _sid = p[pos]
        # 70% from pool, 30% random possibly invalid
        sid = rng.choice(state_pool) if rng.random() < 0.7 else rng.randint(-10, 10_000_000)
        p[pos] = (k, sid)
        return p

    if op == "dup" and p:
        pos = rng.randrange(len(p))
        p.insert(pos, p[pos])
        return p

    if op == "swap" and len(p) >= 2:
        i, j = rng.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]
        return p

    if op == "merge" and len(p) >= 2:
        idx = rng.randrange(len(p) - 1)
        k1, sid1 = p[idx]
        k2, sid2 = p[idx + 1]
        # Merge by accumulating k and keeping next sid
        p[idx:idx + 2] = [(k1 + k2, sid2)]
        return p

    return p


def sample_paths_with_mutations(
    root_index: TrieNodeIndex,
    arena: Arena,
    base_samples: int,
    max_len: int,
    mutations_per_base: int,
    rng: random.Random,
    state_pool: List[StateID],
) -> List[NormalizedPath]:
    """
    Sample base paths from the reference trie, then create mutated variants.
    Ensures deduplication of resulting test paths.
    """
    base: List[NormalizedPath] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()

    # Attempt to include the empty path if it's valid to terminate at root
    empty_candidate = []
    if tuple(empty_candidate) not in seen:
        seen.add(tuple(empty_candidate))
        base.append(empty_candidate)

    attempts = 0
    while len(base) < base_samples and attempts < base_samples * 20:
        attempts += 1
        p = sample_normalized_path(root_index, max_len, arena, rng)
        if p is None:
            continue
        key = tuple((int(k), int(s)) for k, s in p)
        if key in seen:
            continue
        seen.add(key)
        base.append(p)

    # Mutate
    out_paths: Set[Tuple[Tuple[int, int], ...]] = set(seen)
    for p in base:
        for _ in range(mutations_per_base):
            mp = mutate_path(p, rng, state_pool)
            out_paths.add(tuple((int(k), int(s)) for k, s in mp))

    return [list(p) for p in out_paths]


def _coerce_rangeset(obj: Any) -> Optional[RangeSet]:
    """
    Try to coerce a competitor-returned object into a RangeSet. Supports:
      - RangeSet
      - JSON-like [[s,e], ...]
      - None (returns None)
    """
    if obj is None:
        return None
    if isinstance(obj, RangeSet):
        return obj
    if isinstance(obj, list) and all(isinstance(x, list) and len(x) == 2 for x in obj):
        try:
            return RangeSet.from_json(obj)
        except Exception:
            return None
    return None


def _try_call_plugin(func: Callable, structure: Any, state_id: int, path: NormalizedPath) -> Any:
    """
    Call a plugin function supporting flexible signatures by trying:
      1) func(structure, state_id, path)
      2) func(state_id, path)
      3) func(path)
    """
    try:
        return func(structure, state_id, path)
    except TypeError:
        pass
    try:
        return func(state_id, path)
    except TypeError:
        pass
    return func(path)


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


def _get_competitor_functions(plugin_module: Any) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Return a pair (get_bv_func, is_member_func) from the plugin, if present.
    get_bv_func should return a RangeSet (or JSON), is_member_func returns bool.
    """
    get_bv_candidates = ["get_bv", "get_bitvector", "tokens_for_path", "bv_for_path"]
    is_member_candidates = ["is_member", "membership", "is_path_member", "contains"]

    get_bv_func = None
    is_member_func = None

    for name in get_bv_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            get_bv_func = getattr(plugin_module, name)
            break
    for name in is_member_candidates:
        if hasattr(plugin_module, name) and callable(getattr(plugin_module, name)):
            is_member_func = getattr(plugin_module, name)
            break

    return get_bv_func, is_member_func


def score_competitor_on_file(
    precompute_path: Path,
    plugin_module_path: str,
    base_samples: int = 200,
    max_len: int = 32,
    mutations_per_base: int = 3,
    seed: Optional[int] = None,
    state_id_filter: Optional[int] = None,
    max_roots: Optional[int] = None,
    verbose_mismatches: int = 20,
) -> None:
    """
    End-to-end scorer:
      - loads reference precompute2
      - imports competitor plugin module
      - builds competitor structure (if builder provided)
      - generates test paths from reference + mutated variants
      - evaluates membership (and BV, if provided by plugin) against reference
    Prints a summary report with accuracy metrics and sample mismatches.
    """
    rng = random.Random(seed)

    t0 = time.time()
    roots_map, arena = load_precompute2(precompute_path)
    state_ids = [sid for sid, _ in roots_map]
    state_to_root: Dict[int, int] = dict(roots_map)

    if state_id_filter is not None:
        if state_id_filter not in state_to_root:
            raise ValueError(f"state-id {state_id_filter} not found in roots map")
        roots_to_test = [(state_id_filter, state_to_root[state_id_filter])]
    else:
        roots_to_test = roots_map

    if max_roots is not None:
        roots_to_test = roots_to_test[: max(0, max_roots)]

    # Load plugin and build competitor structure
    plugin = _load_plugin(plugin_module_path)
    structure = _build_competitor_structure(plugin, precompute_path, roots_map, arena)
    get_bv_func, is_member_func = _get_competitor_functions(plugin)

    if get_bv_func is None and is_member_func is None:
        raise ValueError(
            "Plugin must define at least one of: get_bv(...), is_member(...). "
            "See scorer help for supported names/signatures."
        )

    # Unified wrappers
    def plugin_get_bv(sid: int, path: NormalizedPath) -> Optional[RangeSet]:
        if get_bv_func is None:
            return None
        try:
            out = _try_call_plugin(get_bv_func, structure, sid, path)
        except TypeError as e:
            raise TypeError(
                f"get_bv incompatible signature: expected one of (structure, sid, path) | (sid, path) | (path). {e}"
            )
        rs = _coerce_rangeset(out)
        return rs

    def plugin_is_member(sid: int, path: NormalizedPath) -> Optional[bool]:
        if is_member_func is None:
            return None
        try:
            out = _try_call_plugin(is_member_func, structure, sid, path)
        except TypeError as e:
            raise TypeError(
                f"is_member incompatible signature: expected one of (structure, sid, path) | (sid, path) | (path). {e}"
            )
        return bool(out)

    # Prepare mutation pool
    state_pool = collect_all_state_ids(arena)

    # Run
    total_paths = 0
    membership_agree = 0

    bv_attempted = 0
    bv_exact_equal = 0

    mismatches: List[Dict[str, Any]] = []

    for sid, root_idx in roots_to_test:
        test_paths = sample_paths_with_mutations(
            root_idx,
            arena,
            base_samples=base_samples,
            max_len=max_len,
            mutations_per_base=mutations_per_base,
            rng=rng,
            state_pool=state_pool,
        )

        for path in test_paths:
            total_paths += 1

            # Reference
            ref_bv = get_bv_for_normalized_path(root_idx, path, arena)
            ref_member = not ref_bv.is_empty()

            # Competitor membership
            comp_bv = plugin_get_bv(sid, path)
            comp_member_from_bv = (comp_bv is not None) and (not comp_bv.is_empty())
            comp_member = comp_member_from_bv
            comp_member_src = "get_bv"

            if comp_bv is None:
                im = plugin_is_member(sid, path)
                if im is None:
                    raise RuntimeError("Plugin returned neither BV nor membership result.")
                comp_member = bool(im)
                comp_member_src = "is_member"

            if comp_member == ref_member:
                membership_agree += 1
            else:
                if len(mismatches) < verbose_mismatches:
                    mismatches.append(
                        {
                            "sid": sid,
                            "path": path,
                            "ref_member": ref_member,
                            "comp_member": comp_member,
                            "src": comp_member_src,
                            "ref_bv": ref_bv.to_json(),
                            "comp_bv": comp_bv.to_json() if comp_bv is not None else None,
                        }
                    )

            # If competitor returned BV, compare exact equality
            if comp_bv is not None:
                bv_attempted += 1
                if comp_bv == ref_bv:
                    bv_exact_equal += 1
                else:
                    if len(mismatches) < verbose_mismatches:
                        mismatches.append(
                            {
                                "sid": sid,
                                "path": path,
                                "ref_member": ref_member,
                                "comp_member": comp_member,
                                "src": "get_bv",
                                "ref_bv": ref_bv.to_json(),
                                "comp_bv": comp_bv.to_json(),
                            }
                        )

    dur = time.time() - t0
    print("\n=== Scoring Summary ===")
    print(f"File: {precompute_path}")
    print(f"Plugin: {plugin_module_path}")
    print(f"Roots tested: {len(roots_to_test)} (of {len(state_ids)})")
    print(f"Base samples per root: {base_samples}, mutations per base: {mutations_per_base}, max path len: {max_len}")
    print(f"Random seed: {seed}")
    print(f"Total test paths: {total_paths}")
    print(f"Membership agreement: {membership_agree}/{total_paths} ({(membership_agree/total_paths*100):.2f}%)")

    if bv_attempted > 0:
        print(f"BV exact equality (subset where plugin returned BV): {bv_exact_equal}/{bv_attempted} ({(bv_exact_equal/bv_attempted*100):.2f}%)")
    else:
        print("BV exact equality: plugin did not provide get_bv; skipped.")

    if mismatches:
        print("\n--- Sample mismatches ---")
        for i, m in enumerate(mismatches, 1):
            print(f"[{i}] sid={m['sid']} path={m['path']}")
            print(f"    ref_member={m['ref_member']} comp_member={m['comp_member']} (via {m['src']})")
            if m.get("ref_bv") is not None:
                print(f"    ref_bv={m['ref_bv']}")
            if m.get("comp_bv") is not None:
                print(f"    comp_bv={m['comp_bv']}")
    print(f"\nDone in {dur:.2f}s.")


# ----------------------------
# CLI: Inspect / Score
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
    score_p = subparsers.add_parser("score", help="Score a contestant module by membership/BV accuracy")
    score_p.add_argument("file", help="Path to a precompute2 gzipped JSON file.")
    score_p.add_argument("plugin", help="Path or module name for the contestant plugin.")
    score_p.add_argument("--samples", type=int, default=200, help="Base samples per root.")
    score_p.add_argument("--mutations", type=int, default=3, help="Mutations per base sample.")
    score_p.add_argument("--max-len", type=int, default=32, help="Maximum path length when sampling.")
    score_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    score_p.add_argument("--state-id", type=int, default=None, help="If provided, score only this tokenizer state ID.")
    score_p.add_argument("--max-roots", type=int, default=None, help="Limit number of roots to score (for large files).")
    score_p.add_argument("--verbose-mismatches", type=int, default=20, help="Print up to this many mismatches.")

    # Equivalence command (compare two precompute2 files on a given state)
    equiv_p = subparsers.add_parser("equiv", help="Stochastic equivalence check between two precompute2 tries")
    equiv_p.add_argument("file_a", help="Path to first precompute2 gzipped JSON file.")
    equiv_p.add_argument("file_b", help="Path to second precompute2 gzipped JSON file.")
    equiv_p.add_argument("--state-id", type=int, required=True, help="Tokenizer state ID to compare.")
    equiv_p.add_argument("--root-a", type=int, default=None, help="Override root node index for A.")
    equiv_p.add_argument("--root-b", type=int, default=None, help="Override root node index for B.")

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
            base_samples=args.samples,
            max_len=args.max_len,
            mutations_per_base=args.mutations,
            seed=args.seed,
            state_id_filter=args.state_id,
            max_roots=args.max_roots,
            verbose_mismatches=args.verbose_mismatches,
        )
        return

    if args.cmd == "equiv":
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
