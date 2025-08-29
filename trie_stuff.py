import gzip
import json
from typing import Dict, List, Any, Set, Tuple, Optional
import collections
import random

# Type aliases for clarity, matching the Rust/JSON structure
TrieNodeIndex = int
TokenizerStateID = int
LLMTokenBV = List[List[int]]
StateID = int
EdgeKey = Tuple[int, Optional[StateID]]
TrieNode = Dict[str, Any]
Arena = Dict[TrieNodeIndex, TrieNode]
NormalizedPath = List[Tuple[int, StateID]]


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

def print_trie_recursive(
    node_index: TrieNodeIndex,
    arena: Arena,
    indent: str = "",
    visited: Optional[Set[TrieNodeIndex]] = None
):
    """Recursively prints the structure of a Trie from a given node index."""
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

    # Print node value
    node_value = node.get("value", {})
    end_str = "END" if node_value.get("end") else "internal"
    live_tokens_str = format_bitvector(node_value.get("live_tokens", []))
    print(f"{indent}Node {node_index} [{end_str}] (live_tokens: {live_tokens_str})")

    # Print children
    children = node.get("children", [])
    if not children:
        # It's a leaf if it has no children. The 'end' flag marks valid termination points.
        return

    for edge_key_json, dest_map_json in children:
        pop_count, state_id_opt = edge_key_json
        edge_key_str = f"pop={pop_count}, state_id={state_id_opt if state_id_opt is not None else 'Any'}"
        
        print(f"{indent}  - Edge({edge_key_str}):")
        
        for dest_index, edge_bv in dest_map_json:
            edge_bv_str = format_bitvector(edge_bv)
            print(f"{indent}    -> Dest: {dest_index} (tokens: {edge_bv_str})")
            # Pass a copy of visited to handle non-tree graphs correctly
            print_trie_recursive(dest_index, arena, indent + "       | ", visited.copy())


def main():
    """
    Loads a precomputed trie from a gzipped JSON file and prints its structure.
    """
    # The user requested loading a precompute3 tree, but the file path and schema
    # indicate it's a precompute2 tree. This script is written for the precompute2 format.
    file_path = "/Users/isaacbreen/Projects2/grammars2024/.cache/test_precompute2/precomputed2_js_gpt2_small.json.gz"

    print(f"Loading precomputed trie from: {file_path}")

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure the path is correct and the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading or parsing the file: {e}")
        return

    if not isinstance(data, list) or len(data) != 2:
        print("Error: Expected top-level JSON to be a list of two elements: [roots_map, arena_data]")
        return

    roots_map_json, arena_data = data
    
    # Parse the arena into a dictionary for easy lookup by integer index
    try:
        arena: Arena = {
            # JSON keys must be strings, so we convert back to int
            int(index): node_data for index, node_data in arena_data.get("values", [])
        }
    except (ValueError, TypeError) as e:
        print(f"Error parsing arena data: {e}")
        return
    
    print(f"\nSuccessfully loaded trie data.")
    print(f"Arena contains {len(arena)} nodes.")
    print(f"Found {len(roots_map_json)} root entries for different tokenizer states.")
    print("-" * 40)

    # Print the trie structure for each root
    for tokenizer_state_id, root_node_index in roots_map_json:
        print(f"\n\n=== Trie for Tokenizer State ID: {tokenizer_state_id} (Root Node: {root_node_index}) ===")
        print_trie_recursive(root_node_index, arena)
        print("-" * 40)

# --- New Code for Trie Equivalence Checking ---

# --- BitVector (Range List) Helper Functions ---

def merge_ranges(ranges: LLMTokenBV) -> LLMTokenBV:
    """Merges overlapping or adjacent ranges in a list."""
    if not ranges:
        return []
    
    # Sort ranges based on the start value
    ranges.sort(key=lambda x: x[0])
    
    merged = []
    if not ranges:
        return merged
        
    current_start, current_end = ranges[0]
    
    for next_start, next_end in ranges[1:]:
        if next_start <= current_end + 1:
            # Overlapping or adjacent range, merge it
            current_end = max(current_end, next_end)
        else:
            # Disjoint range, finalize the current one and start a new one
            merged.append([current_start, current_end])
            current_start, current_end = next_start, next_end
            
    merged.append([current_start, current_end])
    return merged

def bv_union(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """Computes the union of two bitvectors."""
    return merge_ranges(bv1 + bv2)

def bv_intersection(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """Computes the intersection of two bitvectors."""
    result = []
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
    """Computes the difference of two bitvectors (bv1 - bv2)."""
    if not bv1:
        return []
    if not bv2:
        return bv1
        
    result = []
    i = 0
    j = 0
    
    current_ranges1 = collections.deque(bv1)
    
    while current_ranges1 and j < len(bv2):
        start1, end1 = current_ranges1.popleft()
        start2, end2 = bv2[j]

        # No overlap: range1 is completely before range2
        if end1 < start2:
            result.append([start1, end1])
            continue
        
        # No overlap: range2 is completely before range1
        if end2 < start1:
            j += 1
            current_ranges1.appendleft([start1, end1]) # Re-evaluate range1 with next range2
            continue

        # Overlap exists
        # Part of range1 before the overlap
        if start1 < start2:
            result.append([start1, start2 - 1])
        
        # Part of range1 after the overlap
        if end1 > end2:
            # This remaining part of range1 needs to be checked against the next range2
            current_ranges1.appendleft([end2 + 1, end1])
            j += 1
        # If end1 <= end2, range1 is fully consumed by this subtraction
    
    # Add any remaining ranges from bv1 that were not processed
    result.extend(list(current_ranges1))
    
    return merge_ranges(result)

def bv_symmetric_difference(bv1: LLMTokenBV, bv2: LLMTokenBV) -> LLMTokenBV:
    """Computes the symmetric difference of two bitvectors."""
    a_minus_b = bv_difference(bv1, bv2)
    b_minus_a = bv_difference(bv2, bv1)
    return bv_union(a_minus_b, b_minus_a)

def bv_is_empty(bv: LLMTokenBV) -> bool:
    """Checks if a bitvector is empty."""
    return not bv

# --- Core Equivalence Logic ---

def find_end_bv_from_node_via_none_edges(
    start_node_index: TrieNodeIndex,
    initial_bv: LLMTokenBV,
    arena: Arena
) -> LLMTokenBV:
    """
    Finds the union of BVs for all paths from a start node to any `end` node
    that consist solely of `(k, None)` edges.
    """
    total_end_bv = []
    q = collections.deque([(start_node_index, initial_bv)])
    visited = {start_node_index: initial_bv}

    while q:
        node_idx, current_bv = q.popleft()
        node = arena.get(node_idx)
        if not node: continue

        if node.get("value", {}).get("end"):
            total_end_bv = bv_union(total_end_bv, current_bv)

        for edge_key_json, dest_map_json in node.get("children", []):
            _pop_count, state_id_opt = edge_key_json
            if state_id_opt is None:  # Only (k, None) edges
                for dest_idx, edge_bv in dest_map_json:
                    new_bv = bv_intersection(current_bv, edge_bv)
                    if bv_is_empty(new_bv):
                        continue

                    if dest_idx in visited:
                        existing_bv = visited[dest_idx]
                        diff = bv_difference(new_bv, existing_bv)
                        if not bv_is_empty(diff):
                            visited[dest_idx] = bv_union(existing_bv, diff)
                            q.append((dest_idx, diff))
                    else:
                        visited[dest_idx] = new_bv
                        q.append((dest_idx, new_bv))
    
    return total_end_bv

def get_bv_for_normalized_path(
    root_index: TrieNodeIndex,
    path: NormalizedPath,
    arena: Arena
) -> LLMTokenBV:
    """
    For a given normalized path, computes the union of LLM token bitvectors for all
    possible ways to traverse that path in the trie.
    """
    q = collections.deque()
    final_bv = []
    
    root_node = arena.get(root_index)
    if not root_node: return []
    
    initial_bv = root_node.get("value", {}).get("live_tokens", [])
    q.append((root_index, 0, 0, initial_bv))
    
    # Visited key: (node_idx, path_idx, k_so_far)
    visited = {(root_index, 0, 0): initial_bv}

    while q:
        node_idx, path_idx, k_so_far, current_bv = q.popleft()
        
        if path_idx == len(path):
            end_bv = find_end_bv_from_node_via_none_edges(node_idx, current_bv, arena)
            final_bv = bv_union(final_bv, end_bv)
            continue

        target_k, target_sid = path[path_idx]
        
        node = arena.get(node_idx)
        if not node: continue

        for edge_key_json, dest_map_json in node.get("children", []):
            for dest_idx, edge_bv in dest_map_json:
                new_bv = bv_intersection(current_bv, edge_bv)
                if bv_is_empty(new_bv):
                    continue

                pop_count, state_id_opt = edge_key_json
                new_k = k_so_far + pop_count

                if state_id_opt is not None:
                    if new_k == target_k and state_id_opt == target_sid:
                        # Matched a path segment, advance to next segment
                        visited_key = (dest_idx, path_idx + 1, 0)
                        if visited_key in visited:
                            diff = bv_difference(new_bv, visited[visited_key])
                            if not bv_is_empty(diff):
                                visited[visited_key] = bv_union(visited[visited_key], diff)
                                q.append((dest_idx, path_idx + 1, 0, diff))
                        else:
                            visited[visited_key] = new_bv
                            q.append((dest_idx, path_idx + 1, 0, new_bv))
                else: # state_id_opt is None
                    if new_k <= target_k:
                        # Continue accumulating k for the current segment
                        visited_key = (dest_idx, path_idx, new_k)
                        if visited_key in visited:
                            diff = bv_difference(new_bv, visited[visited_key])
                            if not bv_is_empty(diff):
                                visited[visited_key] = bv_union(visited[visited_key], diff)
                                q.append((dest_idx, path_idx, new_k, diff))
                        else:
                            visited[visited_key] = new_bv
                            q.append((dest_idx, path_idx, new_k, new_bv))
    
    return final_bv

def sample_normalized_path(
    root_index: TrieNodeIndex,
    max_len: int,
    arena: Arena
) -> Optional[NormalizedPath]:
    """Samples a single normalized path by performing a random walk from the root."""
    current_node_idx = root_index
    path = []
    current_k = 0
    
    root_node = arena.get(root_index)
    if not root_node: return None
    current_bv = root_node.get("value", {}).get("live_tokens", [])

    while len(path) < max_len:
        node = arena.get(current_node_idx)
        if not node: return None

        can_terminate = node.get("value", {}).get("end", False)
        all_outgoing_edges = []
        for ek, d_map in node.get("children", []):
            for d_idx, e_bv in d_map:
                all_outgoing_edges.append((ek, d_idx, e_bv))
        
        can_continue = bool(all_outgoing_edges)

        if not can_continue:
            return path if can_terminate else None
        
        if can_terminate and random.random() < 0.2: # 20% chance to terminate
            return path

        edge_key_json, dest_idx, edge_bv = random.choice(all_outgoing_edges)
        
        current_bv = bv_intersection(current_bv, edge_bv)
        if bv_is_empty(current_bv):
            return None # Path became invalid

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


if __name__ == "__main__":
    main()
