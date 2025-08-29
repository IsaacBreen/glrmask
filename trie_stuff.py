import gzip
import json
from typing import Dict, List, Any, Set, Tuple

# Type aliases for clarity, matching the Rust/JSON structure
TrieNodeIndex = int
TokenizerStateID = int
LLMTokenBV = List[List[int]]
StateID = int
EdgeKey = Tuple[int, StateID | None]
TrieNode = Dict[str, Any]
Arena = Dict[TrieNodeIndex, TrieNode]

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
    visited: Set[TrieNodeIndex] = None
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


if __name__ == "__main__":
    main()
