#!/usr/bin/env python
"""
A script to visualize a grammar constraint file using Graphviz.

This tool loads a pre-compiled grammar constraint, traverses its internal graph
structure (trie), and generates a visual representation as an image or DOT file.
It helps in debugging and understanding the structure of the constraint model.

Example Usage:
  python python/aug25/visualize_constraint.py \\
    --constraint-file /path/to/your/constraint.json.gz \\
    --output my_graph.png \\
    --max-depth 5 \\
    --rankdir LR

Requires the 'graphviz' Python package:
  pip install graphviz
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

try:
    import graphviz
    from tqdm import tqdm
except ImportError:
    print(
        "Error: Missing required packages 'graphviz' and 'tqdm'.\n"
        "Please install them by running: pip install graphviz tqdm",
        file=sys.stderr
    )
    sys.exit(1)


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


def visualize_constraint(
    constraint_path: Path,
    output_path: Path,
    max_depth: int,
    file_format: str,
    rankdir: str,
    selected_roots: Optional[List[int]] = None,
):
    """
    Loads a constraint file and generates a Graphviz visualization.
    """
    print(f"Loading constraint file: {constraint_path}")
    try:
        if constraint_path.suffix == ".gz":
            with gzip.open(constraint_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(constraint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}", file=sys.stderr)
        return

    trie_data = data.get("trie3_god", {})
    values_list = trie_data.get("values", [])
    if not values_list:
        print("Error: No 'trie3_god.values' found in the constraint file.", file=sys.stderr)
        return

    values_dict: Dict[int, Dict[str, Any]] = {int(k): v for k, v in values_list}
    print(f"Loaded {len(values_dict)} nodes from trie.")

    # Identify root and end nodes
    roots_map = data.get("precomputed3", [])
    all_root_ids: Set[int] = {int(r) for _s, r in roots_map}
    end_ids: Set[int] = {
        nid for nid, node in values_dict.items()
        if node.get("value", {}).get("clean_end", False)
    }
    print(f"Found {len(all_root_ids)} total root nodes and {len(end_ids)} end nodes.")

    # Initialize Graphviz Digraph
    dot = graphviz.Digraph(
        'GrammarConstraintTrie',
        comment='Visualization of the grammar constraint graph'
    )
    dot.attr(
        'graph',
        rankdir=rankdir,
        splines='ortho',
        nodesep='0.6',
        ranksep='1.2',
        label=f"Trie from {constraint_path.name}\\n(max_depth={max_depth})",
        labelloc='t',
        fontsize='20',
    )
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # BFS traversal to build the graph
    q = collections.deque()
    seen_nodes: Set[int] = set()
    seen_edges: Set[str] = set()

    start_roots = selected_roots if selected_roots is not None else sorted(list(all_root_ids))
    if selected_roots is not None:
        print(f"Displaying selected roots: {start_roots}")

    for root_id in start_roots:
        if root_id in values_dict and root_id not in seen_nodes:
            q.append((root_id, 0))
            seen_nodes.add(root_id)

    print("Traversing graph to build visualization...")
    pbar = tqdm(total=len(q), desc="Nodes processed")

    while q:
        node_id, depth = q.popleft()
        pbar.update(1)

        # Determine node style
        is_root = node_id in all_root_ids
        is_end = node_id in end_ids
        label = f"Node {node_id}"
        fillcolor = 'lightblue'
        shape = 'box'

        if is_root and is_end:
            fillcolor = 'gold'
            shape = 'doubleoctagon'
            label = f"Root & End {node_id}"
        elif is_root:
            fillcolor = 'lightgreen'
            label = f"Root {node_id}"
        elif is_end:
            fillcolor = 'lightpink'
            shape = 'doubleoctagon'
            label = f"End {node_id}"

        dot.node(str(node_id), label=label, fillcolor=fillcolor, shape=shape)

        if max_depth is not None and depth >= max_depth:
            # Add a special node to indicate truncation
            trunc_node_id = f"trunc_{node_id}"
            dot.node(trunc_node_id, label="...", shape='plaintext')
            dot.edge(str(node_id), trunc_node_id, style='dashed', arrowhead='none')
            continue

        node_data = values_dict.get(node_id)
        if not node_data:
            continue

        # Process edges
        for child_group in node_data.get("children", []):
            (pop, llm_bv_json), dests = child_group
            for dest_id_int, state_bv_json in dests:
                dest_id = int(dest_id_int)
                
                edge_key = f"{node_id}->{dest_id}|{pop}|{json.dumps(llm_bv_json)}|{json.dumps(state_bv_json)}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                llm_summary = format_ranges(llm_bv_json)
                state_summary = format_ranges(state_bv_json)
                edge_label = f" pop={pop}\nLLM: {llm_summary}\nStates: {state_summary} "

                dot.edge(str(node_id), str(dest_id), label=edge_label)

                if dest_id in values_dict and dest_id not in seen_nodes:
                    q.append((dest_id, depth + 1))
                    seen_nodes.add(dest_id)
                    pbar.total = len(seen_nodes) + len(q)
    
    pbar.close()

    # Render the graph
    print(f"\nRendering graph to {output_path} (format: {file_format})...")
    try:
        dot.render(
            output_path.with_suffix(''),
            format=file_format,
            view=False,
            cleanup=True
        )
        print("Visualization saved successfully.")
    except graphviz.backend.execute.ExecutableNotFound:
        print(
            "\nError: Graphviz executable not found.",
            "Please install Graphviz on your system.",
            "  - On macOS (using Homebrew): brew install graphviz",
            "  - On Ubuntu/Debian: sudo apt-get install graphviz",
            "  - For other systems, see: https://graphviz.org/download/",
            file=sys.stderr
        )
    except Exception as e:
        print(f"An error occurred during rendering: {e}", file=sys.stderr)


def main():
    """
    Command-line interface for the visualization script.
    """
    parser = argparse.ArgumentParser(
        description="Visualize a grammar constraint trie using Graphviz.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python %(prog)s --constraint-file path/to/constraint.json.gz "
            "--output graph.png --max-depth 5"
        )
    )
    parser.add_argument(
        "-f", "--constraint-file",
        type=Path,
        required=True,
        help="Path to the pre-compiled .json.gz or .json grammar constraint file."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path for the graph. The format is determined by the extension "
             "or --format. (default: <constraint_name>.png)"
    )
    parser.add_argument(
        "-d", "--max-depth",
        type=int,
        default=5,
        help="Maximum depth to traverse from root nodes. (default: 5)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default='png',
        choices=['png', 'svg', 'pdf', 'dot'],
        help="Output file format. (default: png)"
    )
    parser.add_argument(
        "--rankdir",
        type=str,
        default='TB',
        choices=['TB', 'LR'],
        help="Direction of graph layout (Top-to-Bottom or Left-to-Right). (default: TB)"
    )
    parser.add_argument(
        "--roots",
        type=str,
        default=None,
        help="Comma-separated list of root node IDs to display. If not set, all roots are shown."
    )
    args = parser.parse_args()

    if not args.constraint_file.exists():
        parser.error(f"Constraint file not found: {args.constraint_file}")

    selected_roots = None
    if args.roots:
        try:
            selected_roots = [int(r.strip()) for r in args.roots.split(',')]
        except ValueError:
            parser.error("Invalid value for --roots. Must be a comma-separated list of integers.")

    output_path = args.output
    if output_path is None:
        base_name = args.constraint_file.name.replace('.json.gz', '').replace('.json', '')
        output_path = Path(f"{base_name}.{args.format}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_constraint(
        constraint_path=args.constraint_file,
        output_path=output_path,
        max_depth=args.max_depth,
        file_format=args.format,
        rankdir=args.rankdir,
        selected_roots=selected_roots,
    )


if __name__ == "__main__":
    main()
