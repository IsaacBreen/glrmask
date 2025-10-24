import sys
from pathlib import Path
import collections
import gzip
import json

# Add project root to sys.path to resolve local imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import the visualizer, which is a standalone script
from python.aug25.visualize_constraint import visualize_constraint

# Global variable to hold the inspector instance for easy REPL access
inspector = None

# --- Helper Functions ---

def format_ranges(ranges: list[list[int]], max_len: int = 40) -> str:
    """
    Converts a list of [start, end] ranges into a compact, readable string.
    Truncates the string if it exceeds max_len.
    """
    if not ranges:
        return "{}"

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}..{end}")

    result = ", ".join(parts)
    if len(result) > max_len:
        return result[:max_len - 3] + "..."
    return result

def token_in_ranges(token_id: int, ranges: list[list[int]]) -> bool:
    """Checks if a token ID is present in a list of [start, end] ranges."""
    for start, end in ranges:
        if start <= token_id <= end:
            return True
    return False

class Inspector:
    """
    A REPL-friendly wrapper around a loaded grammar constraint file
    for interactive inspection and analysis.
    """
    def __init__(self, constraint_data: dict, constraint_path: Path):
        self.data = constraint_data
        self.constraint_path = constraint_path
        
        trie_data = self.data.get("trie3_god", {})
        values_list = trie_data.get("values", [])
        self.values_dict: Dict[int, Dict] = {int(k): v for k, v in values_list}
        
        roots_list = self.data.get("precomputed3", [])
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_list}

        self._adj = None
        self._reverse_adj = None
        
        print(f"Inspector created for '{constraint_path.name}'.")
        print(f"Loaded {len(self.values_dict)} nodes.")
        print("Type 'inspector.help()' for a list of commands.")

    def help(self):
        """Prints a list of available inspection commands."""
        print("\n--- Inspector Commands ---")
        print("  load(path): Loads a new constraint file.")
        print("  inspector.help(): Show this help message.")
        print("  inspector.stats(): Print detailed statistics about the loaded arena.")
        print("  inspector.node(node_id): Show detailed info for a specific node.")
        print("  inspector.roots(): List all root node IDs.")
        print("  inspector.ends(): List all end node IDs.")
        print("  inspector.path(start_id, end_id, max_depth=10): Find a path between two nodes.")
        print("  inspector.find_pop(pop_value): Find nodes with edges having a specific pop value.")
        print("  inspector.find_token(token_id): Find nodes with edges accepting a specific LLM token.")
        print("  inspector.visualize_node(node_id, depth=3, output='node.png', rankdir='TB'):")
        print("    Visualize a subgraph starting from a specific node.")
        print("------------------------\n")

    def stats(self):
        """Prints detailed statistics about the loaded model's arena."""
        num_nodes = len(self.values_dict)
        num_roots = len(set(self.roots_map.values()))
        num_ends = 0
        num_edges = 0
        num_dests = 0
        pop_counts = collections.Counter()

        for node in self.values_dict.values():
            children = node.get("children", [])
            if not children:
                num_ends += 1
            
            for edge in children:
                num_edges += 1
                (pop, _llm_bv), dests = edge
                pop_counts[int(pop)] += 1
                num_dests += len(dests)

        print("\n--- Arena Statistics ---")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Unique Roots: {num_roots:,}")
        print(f"  End Nodes (no children): {num_ends:,}")
        print(f"  Total Edges (child groups): {num_edges:,}")
        print(f"  Total Destinations: {num_dests:,}")
        
        print("\n  Pop Counts:")
        if pop_counts:
            for pop, count in sorted(pop_counts.items()):
                print(f"    - Pop {pop}: {count:,} edges")
        else:
            print("    (No edges with pop counts found)")
        print("------------------------\n")

    @property
    def adj(self):
        """Builds and caches the forward adjacency map (parent -> set(dests))."""
        if self._adj is None:
            print("Building forward adjacency map (one-time operation)...")
            self._adj = collections.defaultdict(set)
            for u, node in self.values_dict.items():
                for edge in node.get("children", []):
                    _key, dests = edge
                    for dest_id, _state_bv in dests:
                        self._adj[int(u)].add(int(dest_id))
            print("Done.")
        return self._adj

    @property
    def reverse_adj(self):
        """Builds and caches the reverse adjacency map (dest -> set(parents))."""
        if self._reverse_adj is None:
            print("Building reverse adjacency map (one-time operation)...")
            self._reverse_adj = collections.defaultdict(set)
            for u, node in self.values_dict.items():
                for edge in node.get("children", []):
                    _key, dests = edge
                    for dest_id, _state_bv in dests:
                        self._reverse_adj[int(dest_id)].add(int(u))
            print("Done.")
        return self._reverse_adj

    def node(self, node_id: int):
        """Prints detailed information about a specific node."""
        node = self.values_dict.get(node_id)
        if not node:
            print(f"Node {node_id} not found in arena.")
            return

        print(f"\n--- Node {node_id} ---")
        is_root = any(r == node_id for r in self.roots_map.values())
        children = node.get("children", [])
        is_end = not children

        print(f"  Is Root: {'Yes' if is_root else 'No'}")
        print(f"  Is End (no children): {'Yes' if is_end else 'No'}")
        print(f"  Max Depth (heuristic): {node.get('max_depth', 'N/A')}")

        parents = self.reverse_adj.get(node_id)
        if parents:
            print(f"  Parents ({len(parents)}): {sorted(list(parents))}")
        else:
            print("  Parents: None")

        print(f"  Children ({len(children)} edges):")
        if not children:
            print("    None")
        else:
            for i, edge in enumerate(children):
                (pop, llm_bv_json), dests = edge
                dest_ids = sorted([int(d[0]) for d in dests])
                print(f"    - Edge {i}: pop={pop}")
                print(f"      - LLM Tokens: {format_ranges(llm_bv_json)}")
                print(f"      - Destinations ({len(dest_ids)}): {dest_ids}")
        print("-" * 15)

    def roots(self):
        """Lists all root nodes."""
        root_ids = sorted(list(set(self.roots_map.values())))
        print(f"Found {len(root_ids)} unique root nodes:")
        print(root_ids)
        return root_ids

    def ends(self):
        """Lists all end nodes (nodes with no children)."""
        end_ids = sorted([nid for nid, node in self.values_dict.items() if not node.get("children")])
        print(f"Found {len(end_ids)} end nodes:")
        print(end_ids)
        return end_ids

    def path(self, start_id: int, end_id: int, max_depth: int = 10):
        """Finds and prints a path from a start node to an end node using BFS."""
        if start_id not in self.values_dict or end_id not in self.values_dict:
            print("Error: One or both nodes not in arena.")
            return

        q = collections.deque([(start_id, [start_id])])
        visited = {start_id}

        print(f"Searching for path from {start_id} to {end_id} (max depth {max_depth})...")
        while q:
            current_node, path = q.popleft()

            if current_node == end_id:
                print("Path found:")
                print(" -> ".join(map(str, path)))
                return path

            if len(path) > max_depth:
                continue

            for neighbor in self.adj.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    q.append((neighbor, new_path))

        print("No path found within the depth limit.")
        return None

    def find_pop(self, pop_value: int):
        """Finds all nodes with at least one edge having the specified pop value."""
        found_nodes = []
        for nid, node in self.values_dict.items():
            for edge in node.get("children", []):
                (pop, _llm_bv), _dests = edge
                if int(pop) == pop_value:
                    found_nodes.append(nid)
                    break
        print(f"Found {len(found_nodes)} nodes with edges where pop={pop_value}:")
        print(sorted(found_nodes))
        return sorted(found_nodes)

    def find_token(self, token_id: int):
        """Finds all nodes with at least one edge that accepts the given LLM token ID."""
        found_nodes = []
        for nid, node in self.values_dict.items():
            for edge in node.get("children", []):
                (_pop, llm_bv_json), _dests = edge
                if token_in_ranges(token_id, llm_bv_json):
                    found_nodes.append(nid)
                    break
        print(f"Found {len(found_nodes)} nodes with edges accepting token {token_id}:")
        print(sorted(found_nodes))
        return sorted(found_nodes)

    def visualize_node(self, node_id: int, depth: int = 3, output: str = 'node_viz.png', rankdir: str = 'TB'):
        """
        Visualizes a subgraph starting from a specific node.
        Saves the output to the specified file.
        """
        output_path = Path(output)
        file_format = output_path.suffix[1:] if output_path.suffix else 'png'

        print(f"Visualizing subgraph from root {node_id} with depth {depth}...")
        print(f"Output will be saved to '{output_path}' as format '{file_format}'.")

        try:
            visualize_constraint(
                constraint_path=self.constraint_path,
                output_path=output_path,
                max_depth=depth,
                file_format=file_format,
                rankdir=rankdir,
                splines='curved',
                output_mode='render',
                selected_roots=[node_id],
            )
        except Exception as e:
            print(f"An error occurred during visualization: {e}")
            print("Please ensure Graphviz is installed and in your system's PATH.")


def load(constraint_path: str) -> Inspector:
    """
    Loads a constraint file and returns an Inspector instance.
    This also assigns the instance to the global 'inspector' variable.
    """
    global inspector
    path = Path(constraint_path)
    if not path.exists():
        print(f"Error: File not found at '{path}'")
        return

    print(f"Loading constraint data from '{path}'...")
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return

    inspector = Inspector(json_data, path)
    return inspector


if __name__ == "__main__":
    print("--- Grammar Constraint Inspector REPL ---")
    print("This script is designed for interactive use in a Python REPL (like IPython).")
    print("\nTo get started:")
    print("1. Run: ipython -i python/aug25/repl_inspector.py")
    print("2. In the IPython shell, load a constraint file:")
    print("   >>> load('path/to/your/constraint.json.gz')")
    print("3. Use the global 'inspector' object to run commands:")
    print("   >>> inspector.help()")
    print("   >>> inspector.node(123)")
    print("   >>> inspector.stats()")
    print("-" * 40)

    # Example of how to auto-load if a path is provided via command line
    if len(sys.argv) > 1:
        constraint_file = sys.argv[1]
        print(f"\nAttempting to auto-load constraint file from command line: {constraint_file}")
        load(constraint_file)
