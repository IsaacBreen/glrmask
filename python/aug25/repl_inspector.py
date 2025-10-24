import sys
from pathlib import Path
import collections
import gzip
import json

# Add project root to sys.path to resolve local imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import the model we want to inspect and the visualizer
from python.aug25.models.precompute3_model_pure_python_opt4 import Model
from python.aug25.visualize_constraint import visualize_constraint
from python.aug25.range_set import FFIRangeSet as RangeSet

# Global variable to hold the inspector instance for easy REPL access
inspector = None

class Inspector:
    """
    A REPL-friendly wrapper around a loaded grammar constraint Model
    for interactive inspection and analysis.
    """
    def __init__(self, model: Model, constraint_path: Path):
        self.model = model
        self.constraint_path = constraint_path
        self._reverse_adj = None
        print(f"Inspector created for '{constraint_path.name}'.")
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
        # The model already has a great stats printer
        self.model._compute_and_print_stats()

    @property
    def reverse_adj(self):
        """Builds and caches the reverse adjacency map (dest -> set(parents))."""
        if self._reverse_adj is None:
            print("Building reverse adjacency map (one-time operation)...")
            self._reverse_adj = collections.defaultdict(set)
            for u, node in self.model.arena.items():
                for edge in node.children:
                    for dest in edge.dests:
                        self._reverse_adj[dest.dest_idx].add(u)
            print("Done.")
        return self._reverse_adj

    def node(self, node_id: int):
        """Prints detailed information about a specific node."""
        node = self.model.arena.get(node_id)
        if not node:
            print(f"Node {node_id} not found in arena.")
            return

        print(f"\n--- Node {node_id} ---")
        is_root = any(r == node_id for r in self.model.roots_map.values())
        print(f"  Is Root: {'Yes' if is_root else 'No'}")
        print(f"  Is End (clean_end): {'Yes' if node.clean_end else 'No'}")
        print(f"  Max Depth (heuristic): {self.model.max_depth.get(node_id, 'N/A')}")

        parents = self.reverse_adj.get(node_id)
        if parents:
            print(f"  Parents ({len(parents)}): {sorted(list(parents))}")
        else:
            print("  Parents: None")

        print(f"  Children ({len(node.children)} edges):")
        if not node.children:
            print("    None")
        else:
            for i, edge in enumerate(node.children):
                dest_ids = sorted([d.dest_idx for d in edge.dests])
                print(f"    - Edge {i}: pop={edge.pop}")
                print(f"      - LLM Tokens: {edge.llm_bv.to_ranges()}")
                print(f"      - State Union: {edge.dest_states_union.to_ranges()}")
                print(f"      - Destinations ({len(dest_ids)}): {dest_ids}")
        print("-" * 15)

    def roots(self):
        """Lists all root nodes."""
        root_ids = sorted(list(set(self.model.roots_map.values())))
        print(f"Found {len(root_ids)} unique root nodes:")
        print(root_ids)
        return root_ids

    def ends(self):
        """Lists all end nodes (nodes with clean_end=True)."""
        end_ids = sorted([nid for nid, node in self.model.arena.items() if node.clean_end])
        print(f"Found {len(end_ids)} clean_end nodes:")
        print(end_ids)
        return end_ids

    def path(self, start_id: int, end_id: int, max_depth: int = 10):
        """Finds and prints a path from a start node to an end node using BFS."""
        if start_id not in self.model.arena or end_id not in self.model.arena:
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

            node_data = self.model.arena.get(current_node)
            if not node_data:
                continue

            for edge in node_data.children:
                for dest in edge.dests:
                    if dest.dest_idx not in visited:
                        visited.add(dest.dest_idx)
                        new_path = path + [dest.dest_idx]
                        q.append((dest.dest_idx, new_path))

        print("No path found within the depth limit.")
        return None

    def find_pop(self, pop_value: int):
        """Finds all nodes with at least one edge having the specified pop value."""
        found_nodes = []
        for nid, node in self.model.arena.items():
            for edge in node.children:
                if edge.pop == pop_value:
                    found_nodes.append(nid)
                    break
        print(f"Found {len(found_nodes)} nodes with edges where pop={pop_value}:")
        print(sorted(found_nodes))
        return sorted(found_nodes)

    def find_token(self, token_id: int):
        """Finds all nodes with at least one edge that accepts the given LLM token ID."""
        found_nodes = []
        for nid, node in self.model.arena.items():
            for edge in node.children:
                if edge.llm_bv.contains(token_id):
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

    print(f"Loading model from '{path}'...")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            json_str = f.read()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            json_str = f.read()

    model = Model.from_json_string(json_str)
    inspector = Inspector(model, path)
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
