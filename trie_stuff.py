import argparse
import collections
import gzip
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Deque

# Type aliases for clarity, matching the Rust/JSON structure
TrieNodeIndex = int
TokenizerStateID = int
LLMTokenBV = List[List[int]]  # List of [start, end] inclusive ranges
StateID = int
EdgeKey = Tuple[int, Optional[StateID]]  # (pop_count, state_id or None)
TrieNode = Dict[str, Any]
Arena = Dict[TrieNodeIndex, TrieNode]
NormalizedPath = List[Tuple[int, StateID]]  # List of (k, state_id)


# ----------------------------
# BitVector (Range List) Utils
# ----------------------------

def _merge_ranges(ranges: LLMTokenBV) -> LLMTokenBV:
    """Pure merge: returns a normalized (sorted, non-overlapping) copy of ranges."""
    if not ranges:
        return []

    # Sort without mutating the input
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged: LLMTokenBV = []
    cur_start, cur_end = sorted_ranges[0]

    for nxt_start, nxt_end in sorted_ranges[1:]:
        if nxt_start <= cur_end + 1:
            cur_end = max(cur_end, nxt_end)
        else:
            merged.append([cur_start, cur_end])
            cur_start, cur_end = nxt_start, nxt_end

    merged.append([cur_start, cur_end])
    return merged


def bv_union(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """Computes the union of two bitvectors. Returns a normalized BV."""
    if not bv1:
        return _merge_ranges(bv2)
    if not bv2:
        return _merge_ranges(bv1)
    return _merge_ranges(bv1 + bv2)


def bv_intersection(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """
    Computes the intersection of two bitvectors.
    Assumes each input BV is already sorted & normalized (as in precompute2 data).
    """
    if not bv1 or not bv2:
        return []

    result: LLMTokenBV = []
    i, j = 0, 0
    while i < len(bv1) and j < len(bv2):
        start1, end1 = bv1[i]
        start2, end2 = bv2[j]

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start <= overlap_end:
            result.append([overlap_start, overlap_end])

        if end1 < end2:
            i += 1
        else:
            j += 1
    return result


def bv_difference(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """
    Computes the set difference of two bitvectors (bv1 - bv2).
    Assumes each input BV is already sorted & normalized (as in precompute2 data).
    """
    if not bv1:
        return []
    if not bv2:
        return _merge_ranges(bv1)  # ensure normalized

    out: LLMTokenBV = []
    q: Deque[List[int]] = collections.deque(bv1)
    j = 0

    while q and j < len(bv2):
        start1, end1 = q.popleft()
        start2, end2 = bv2[j]

        # No overlap: range1 completely before range2
        if end1 < start2:
            out.append([start1, end1])
            continue

        # No overlap: range2 completely before range1
        if end2 < start1:
            j += 1
            q.appendleft([start1, end1])  # Re-evaluate range1 against next range2
            continue

        # Overlap exists
        if start1 < start2:
            out.append([start1, start2 - 1])

        if end1 > end2:
            # Residual of range1 after the overlap needs to be checked against next bv2 range
            q.appendleft([end2 + 1, end1])
            j += 1
        # else: range1 fully consumed; continue

    # Any remaining ranges from bv1 that were not processed
    out.extend(list(q))
    return _merge_ranges(out)


def bv_symmetric_difference(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """Symmetric difference of two bitvectors. Returns a normalized BV."""
    return bv_union(bv_difference(bv1, bv2), bv_difference(bv2, bv1))


def bv_is_empty(bv: LLMTokenBV) -> bool:
    """True if the bitvector is empty."""
    return not bv


def format_bitvector(bv: LLMTokenBV) -> str:
    """Formats a bitvector (list of ranges) into a readable string."""
    if not bv:
        return "{}"
    parts = []
    for start, end in bv:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return f"{{{', '.join(parts)}}}"


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
    live_tokens: LLMTokenBV = node_value.get("live_tokens", []) or []
    print(f"{indent}Node {node_index} [{end_str}] (live_tokens: {format_bitvector(_merge_ranges(live_tokens))})")

    children = node.get("children", []) or []
    if not children:
        return

    for edge_key_json, dest_map_json in children:
        pop_count, state_id_opt = edge_key_json
        edge_key_str = f"pop={pop_count}, state_id={state_id_opt if state_id_opt is not None else 'Any'}"
        print(f"{indent}  - Edge({edge_key_str}):")

        for dest_index, edge_bv in dest_map_json:
            print(f"{indent}    -> Dest: {dest_index} (tokens: {format_bitvector(_merge_ranges(edge_bv))})")
            # Use a copy of visited to ensure full exploration in DAG-like graphs
            print_trie_recursive(dest_index, arena, indent + "       | ", visited.copy())


# ----------------------------------
# Core Equivalence / Traversal Logic
# ----------------------------------

def _update_visited_bv(store: Dict[Any, LLMTokenBV], key: Any, incoming_bv: LLMTokenBV) -> Optional[LLMTokenBV]:
    """
    Maintains a map of key -> accumulated BV.
    Returns the 'new portion' (difference) to propagate if any; otherwise None.
    """
    if key in store:
        existing = store[key]
        diff = bv_difference(incoming_bv, existing)
        if bv_is_empty(diff):
            return None
        store[key] = bv_union(existing, diff)
        return diff
    else:
        store[key] = _merge_ranges(incoming_bv)
        return store[key]


def find_end_bv_from_node_via_none_edges(
    start_node_index: TrieNodeIndex,
    initial_bv: LLMTokenBV,
    arena: Arena
) -> LLMTokenBV:
    """
    Finds the union of BVs for all paths from a start node to any `end` node
    that consist solely of `(k, None)` edges. Intersects edge BVs along the way.
    """
    if bv_is_empty(initial_bv):
        return []

    total_end_bv: LLMTokenBV = []
    q: Deque[Tuple[TrieNodeIndex, LLMTokenBV]] = collections.deque()
    q.append((start_node_index, _merge_ranges(initial_bv)))

    visited: Dict[TrieNodeIndex, LLMTokenBV] = {}

    while q:
        node_idx, current_bv = q.popleft()
        node = arena.get(node_idx)
        if not node:
            continue

        node_val = node.get("value", {}) or {}
        if node_val.get("end"):
            total_end_bv = bv_union(total_end_bv, current_bv)

        for edge_key_json, dest_map_json in node.get("children", []) or []:
            _pop_count, state_id_opt = edge_key_json
            if state_id_opt is not None:
                continue  # Only traverse (k, None) edges here

            for dest_idx, edge_bv in dest_map_json:
                new_bv = bv_intersection(current_bv, edge_bv)
                if bv_is_empty(new_bv):
                    continue

                diff = _update_visited_bv(visited, dest_idx, new_bv)
                if diff is not None and not bv_is_empty(diff):
                    q.append((dest_idx, diff))

    return total_end_bv


def get_bv_for_normalized_path(
    root_index: TrieNodeIndex,
    path: NormalizedPath,
    arena: Arena
) -> LLMTokenBV:
    """
    For a given normalized path, computes the union of LLM token bitvectors for all
    possible ways to traverse that path in the trie.
    A normalized path is a list of (k, state_id) segments where k accumulates pop_counts
    from (k, None) edges until a (k', Some(state_id)) matches, then k resets to 0.
    """
    root_node = arena.get(root_index)
    if not root_node:
        return []

    initial_bv: LLMTokenBV = root_node.get("value", {}).get("live_tokens", []) or []
    if bv_is_empty(initial_bv) and path:
        return []

    final_bv: LLMTokenBV = []
    q: Deque[Tuple[TrieNodeIndex, int, int, LLMTokenBV]] = collections.deque()
    q.append((root_index, 0, 0, _merge_ranges(initial_bv)))

    # Visited key: (node_idx, path_idx, accumulated_k)
    visited: Dict[Tuple[TrieNodeIndex, int, int], LLMTokenBV] = { (root_index, 0, 0): _merge_ranges(initial_bv) }

    while q:
        node_idx, path_idx, k_so_far, current_bv = q.popleft()

        # If we've matched the full path, gather end BVs along None edges
        if path_idx == len(path):
            end_bv = find_end_bv_from_node_via_none_edges(node_idx, current_bv, arena)
            final_bv = bv_union(final_bv, end_bv)
            continue

        target_k, target_sid = path[path_idx]
        node = arena.get(node_idx)
        if not node:
            continue

        for edge_key_json, dest_map_json in node.get("children", []) or []:
            pop_count, state_id_opt = edge_key_json
            new_k = k_so_far + pop_count

            for dest_idx, edge_bv in dest_map_json:
                new_bv = bv_intersection(current_bv, edge_bv)
                if bv_is_empty(new_bv):
                    continue

                if state_id_opt is not None:
                    # Advance only if this edge completes the current (k, sid) segment
                    if new_k == target_k and state_id_opt == target_sid:
                        next_key = (dest_idx, path_idx + 1, 0)
                        diff = _update_visited_bv(visited, next_key, new_bv)
                        if diff is not None and not bv_is_empty(diff):
                            q.append((dest_idx, path_idx + 1, 0, diff))
                else:
                    # Accumulate k along None edges
                    if new_k <= target_k:
                        cont_key = (dest_idx, path_idx, new_k)
                        diff = _update_visited_bv(visited, cont_key, new_bv)
                        if diff is not None and not bv_is_empty(diff):
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

    current_bv: LLMTokenBV = _merge_ranges(root_node.get("value", {}).get("live_tokens", []) or [])

    for _ in range(max_len):
        node = arena.get(current_node_idx)
        if not node:
            return None

        can_terminate = bool(node.get("value", {}).get("end", False))
        edges: List[Tuple[EdgeKey, TrieNodeIndex, LLMTokenBV]] = []
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
        current_bv = bv_intersection(current_bv, edge_bv)
        if bv_is_empty(current_bv):
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
            if bv_is_empty(bv_a) and i > 0:
                continue
            bv_b = get_bv_for_normalized_path(root_b, path, arena_b)
            if bv_a != bv_b:
                print("\n--- Precompute2 Equivalence Mismatch ---")
                print("Path sampled from Tree A:")
                print(f"  Path: {path}")
                print(f"  BV from A: {format_bitvector(bv_a)}")
                print(f"  BV from B: {format_bitvector(bv_b)}")
                print(f"  Difference (A ^ B): {format_bitvector(bv_symmetric_difference(bv_a, bv_b))}")
                return False

    # Sample from B, check in A
    for i in range(NUM_SAMPLES):
        path = sample_normalized_path(root_b, MAX_PATH_LEN, arena_b)
        if path is not None:
            bv_b = get_bv_for_normalized_path(root_b, path, arena_b)
            if bv_is_empty(bv_b) and i > 0:
                continue
            bv_a = get_bv_for_normalized_path(root_a, path, arena_a)
            if bv_a != bv_b:
                print("\n--- Precompute2 Equivalence Mismatch ---")
                print("Path sampled from Tree B:")
                print(f"  Path: {path}")
                print(f"  BV from A: {format_bitvector(bv_a)}")
                print(f"  BV from B: {format_bitvector(bv_b)}")
                print(f"  Difference (A ^ B): {format_bitvector(bv_symmetric_difference(bv_a, bv_b))}")
                return False

    return True


# ----------------------------
# I/O and CLI
# ----------------------------

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

    return roots_map, arena


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and test precompute2 tries (gzipped JSON)."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="/Users/isaacbreen/Projects2/grammars2024/.cache/test_precompute2/precomputed2_js_gpt2_small.json.gz",
        help="Path to a precompute2 gzipped JSON file. Defaults to a known test path."
    )
    parser.add_argument(
        "--state-id",
        type=int,
        default=None,
        help="If provided, print only the trie for this tokenizer state ID."
    )
    parser.add_argument(
        "--root-index",
        type=int,
        default=None,
        help="If provided, start printing from this root node index instead of looking it up from state-id."
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=None,
        help="Limit the number of roots to print (useful for large files)."
    )
    args = parser.parse_args()

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

    # Determine which roots to print
    to_print: List[Tuple[TokenizerStateID, TrieNodeIndex]] = []
    if args.root_index is not None:
        # Print from an arbitrary root index, if present
        if args.state_id is not None:
            to_print.append((args.state_id, args.root_index))
        else:
            # Use -1 as placeholder tokenizer state ID if not specified
            to_print.append((-1, args.root_index))
    elif args.state_id is not None:
        to_print = [(sid, ridx) for sid, ridx in roots_map if sid == args.state_id]
        if not to_print:
            print(f"No root found for tokenizer state ID {args.state_id}")
            return
    else:
        to_print = roots_map

    if args.max_roots is not None:
        to_print = to_print[: max(0, args.max_roots)]

    # Print the trie structure for each selected root
    for tokenizer_state_id, root_node_index in to_print:
        print(f"\n\n=== Trie for Tokenizer State ID: {tokenizer_state_id} (Root Node: {root_node_index}) ===")
        print_trie_recursive(root_node_index, arena)
        print("-" * 40)


if __name__ == "__main__":
    main()
