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
from typing import Dict, Any, List, Set, Optional, Tuple
import subprocess
import tempfile

from tqdm import tqdm

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


def prune_ranges(ranges: List[List[int]], filter_range: Optional[Tuple[int, int]]) -> List[List[int]]:
    """Prunes a list of [start, end] ranges to only include parts within the filter_range."""
    if not filter_range:
        return ranges

    min_val, max_val = filter_range
    pruned = []
    for start, end in ranges:
        # Find the intersection of [start, end] and [min_val, max_val]
        overlap_start = max(start, min_val)
        overlap_end = min(end, max_val)

        if overlap_start <= overlap_end:
            pruned.append([overlap_start, overlap_end])

    return pruned


def visualize_constraint(
    constraint_path: Path,
    output_path: Path,
    max_depth: int,
    file_format: str,
    rankdir: str,
    splines: str,
    output_mode: str,
    max_edges_per_node: Optional[int] = None,
    selected_roots: Optional[List[int]] = None,
    llm_token_range: Optional[Tuple[int, int]] = None,
    state_bv_range: Optional[Tuple[int, int]] = None,
):
    """
    Loads a constraint file, generates Graphviz DOT source, and handles output.
    """
    print(f"Loading constraint file: {constraint_path}", file=sys.stderr)
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
    print(f"Loaded {len(values_dict)} nodes from trie.", file=sys.stderr)

    # Identify root and end nodes
    roots_map = data.get("precomputed3", [])
    all_root_ids: Set[int] = {int(r) for _s, r in roots_map}
    end_ids: Set[int] = {
        nid for nid, node in values_dict.items()
        if node.get("value", {}).get("clean_end", False)
    }
    print(f"Found {len(all_root_ids)} total root nodes and {len(end_ids)} end nodes.", file=sys.stderr)

    # --- Build DOT source manually ---
    dot_lines = ['digraph GrammarConstraintTrie {']
    
    # Graph attributes
    graph_attrs = [
        f'rankdir="{rankdir}"',
        f'splines="{splines}"',
        'nodesep="0.6"',
        'ranksep="1.2"',
        f'label="{constraint_path.name}\\n(max_depth={max_depth})"',
        'labelloc="t"',
        'fontsize="20"',
    ]
    dot_lines.append(f"  graph [{', '.join(graph_attrs)}];")
    dot_lines.append('  node [shape="box", style="rounded,filled", fontname="Helvetica"];')
    dot_lines.append('  edge [fontname="Helvetica", fontsize="10"];')

    # BFS traversal to build the graph
    q = collections.deque()
    seen_nodes: Set[int] = set()
    seen_edges: Set[str] = set()

    start_roots = selected_roots if selected_roots is not None else sorted(list(all_root_ids))
    if selected_roots is not None:
        print(f"Displaying selected roots: {start_roots}", file=sys.stderr)

    for root_id in start_roots:
        if root_id in values_dict and root_id not in seen_nodes:
            q.append((root_id, 0))
            seen_nodes.add(root_id)

    print("Traversing graph to build visualization...", file=sys.stderr)
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
            fillcolor, shape, label = 'gold', 'doubleoctagon', f"Root & End {node_id}"
        elif is_root:
            fillcolor, label = 'lightgreen', f"Root {node_id}"
        elif is_end:
            fillcolor, shape, label = 'lightpink', 'doubleoctagon', f"End {node_id}"

        dot_lines.append(f'  "{node_id}" [label={json.dumps(label)}, fillcolor="{fillcolor}", shape="{shape}"];')

        if max_depth is not None and depth >= max_depth:
            # Add a special node to indicate truncation
            trunc_node_id = f"trunc_{node_id}"
            dot_lines.append(f'  "{trunc_node_id}" [label="...", shape="plaintext"];')
            dot_lines.append(f'  "{node_id}" -> "{trunc_node_id}" [style="dashed", arrowhead="none"];')
            continue

        node_data = values_dict.get(node_id)
        if not node_data:
            continue

        # Process edges
        # Collect all potential outgoing edges from this node
        all_edges = []
        for child_group in node_data.get("children", []):
            (pop, llm_bv_json), dests = child_group
            for dest_id_int, state_bv_json in dests:
                all_edges.append({
                    'pop': pop,
                    'llm_bv_json': llm_bv_json,
                    'dest_id': int(dest_id_int),
                    'state_bv_json': state_bv_json,
                })

        edges_to_process = all_edges
        truncated_edges = False
        if max_edges_per_node is not None and len(all_edges) > max_edges_per_node:
            # Sort by pop value (ascending) to keep the lowest ones
            all_edges.sort(key=lambda e: e['pop'])
            edges_to_process = all_edges[:max_edges_per_node]
            truncated_edges = True

        # Process edges (the potentially truncated list)
        for edge_info in edges_to_process:
            pop = edge_info['pop']
            llm_bv_json = edge_info['llm_bv_json']
            dest_id = edge_info['dest_id']
            state_bv_json = edge_info['state_bv_json']
            
            # Prune BVs based on specified ranges
            pruned_llm_bv = prune_ranges(llm_bv_json, llm_token_range)
            pruned_state_bv = prune_ranges(state_bv_json, state_bv_range)

            # Skip edges where either BV becomes empty after pruning
            if not pruned_llm_bv or not pruned_state_bv:
                continue

            edge_key = f"{node_id}->{dest_id}|{pop}|{json.dumps(llm_bv_json)}|{json.dumps(state_bv_json)}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            llm_summary = format_ranges(pruned_llm_bv)
            state_summary = format_ranges(pruned_state_bv)
            edge_label = f" pop={pop}\\nLLM: {llm_summary}\\nStates: {state_summary} "
            dot_lines.append(f'  "{node_id}" -> "{dest_id}" [label="{edge_label}"];')

            if dest_id in values_dict and dest_id not in seen_nodes:
                q.append((dest_id, depth + 1))
                seen_nodes.add(dest_id)
                pbar.total = len(seen_nodes) + len(q)
        
        if truncated_edges:
            # Add a truncation indicator node for edges
            trunc_node_id = f"trunc_edges_{node_id}"
            num_omitted = len(all_edges) - len(edges_to_process)
            dot_lines.append(f'  "{trunc_node_id}" [label="... ({num_omitted} more edges)", shape="plaintext"];')
            dot_lines.append(f'  "{node_id}" -> "{trunc_node_id}" [style="dotted", arrowhead="none"];')
    
    pbar.close()
    dot_lines.append('}')
    dot_source = "\n".join(dot_lines)

    # --- Handle output ---
    if output_mode == 'clipboard':
        try:
            import pyperclip
            pyperclip.copy(dot_source)
            print("DOT source copied to clipboard.", file=sys.stderr)
        except ImportError:
            print(
                "Error: 'pyperclip' package not found. Cannot copy to clipboard.\n"
                "Please install it by running: pip install pyperclip",
                file=sys.stderr
            )
            print("\n--- DOT Source Fallback ---\n", file=sys.stderr)
            print(dot_source)
    elif output_mode == 'source':
        if output_path:
            output_path.write_text(dot_source, encoding='utf-8')
            print(f"DOT source saved to {output_path}", file=sys.stderr)
        else:
            print(dot_source)
    elif output_mode == 'render':
        if not output_path:
            print("Error: Output path is required for rendering.", file=sys.stderr)
            return

        print(f"\nRendering graph to {output_path} (format: {file_format})...", file=sys.stderr)
        tmp_dot_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False, encoding='utf-8') as f:
                f.write(dot_source)
                tmp_dot_path = f.name
            
            cmd = ['dot', f'-T{file_format}', '-o', str(output_path), tmp_dot_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"An error occurred during rendering with 'dot':", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            else:
                print("Visualization saved successfully.", file=sys.stderr)

        except FileNotFoundError:
            print(
                "\nError: Graphviz 'dot' command not found.",
                "Please install Graphviz on your system.",
                "  - On macOS (using Homebrew): brew install graphviz",
                "  - On Ubuntu/Debian: sudo apt-get install graphviz",
                "  - For other systems, see: https://graphviz.org/download/",
                file=sys.stderr
            )
        finally:
            if tmp_dot_path:
                Path(tmp_dot_path).unlink()


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
        help="Output file path. For render mode, format is determined by extension. "
             "For source mode, content is DOT source. Not used with --clipboard."
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
        help="Output file format for rendering. (default: png)"
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
    parser.add_argument(
        "--splines",
        type=str,
        default='curved',
        choices=['curved', 'line', 'ortho', 'polyline', 'spline'],
        help="Graphviz spline type for edge routing. 'curved' is more stable. (default: curved)"
    )
    parser.add_argument(
        "--source-only",
        action='store_true',
        help="Output the DOT source code to stdout or the --output file instead of rendering an image."
    )
    parser.add_argument(
        "--clipboard",
            action='store_true',
            help="Copy the DOT source code to the clipboard. Requires 'pyperclip'."
        )
    parser.add_argument(
        "--max-edges-per-node",
        type=int,
        default=None,
        help="Maximum number of edges to draw from a single node. Keeps lowest-pop edges. (default: no limit)"
    )
    parser.add_argument(
        "--llm-token-range",
        type=str,
        default=None,
        help="Only show LLM tokens in this range. Format: 'min,max'. Example: '0,1000'."
    )
    parser.add_argument(
        "--state-bv-range",
        type=str,
        default=None,
        help="Only show state bitvector values in this range. Format: 'min,max'. Example: '0,50'."
    )
    args = parser.parse_args()

    if not args.constraint_file.exists():
        parser.error(f"Constraint file not found: {args.constraint_file}")

    llm_token_range = None
    if args.llm_token_range:
        try:
            min_val, max_val = map(int, args.llm_token_range.split(','))
            if min_val > max_val:
                parser.error(f"Invalid range for --llm-token-range: min ({min_val}) cannot be greater than max ({max_val}).")
            llm_token_range = (min_val, max_val)
        except (ValueError, TypeError):
            parser.error(f"Invalid format for --llm-token-range. Expected 'min,max', got '{args.llm_token_range}'.")

    state_bv_range = None
    if args.state_bv_range:
        try:
            min_val, max_val = map(int, args.state_bv_range.split(','))
            if min_val > max_val:
                parser.error(f"Invalid range for --state-bv-range: min ({min_val}) cannot be greater than max ({max_val}).")
            state_bv_range = (min_val, max_val)
        except (ValueError, TypeError):
            parser.error(f"Invalid format for --state-bv-range. Expected 'min,max', got '{args.state_bv_range}'.")

    output_mode = 'render'
    if args.clipboard:
        output_mode = 'clipboard'
    elif args.source_only:
        output_mode = 'source'

    selected_roots = None
    if args.roots:
        try:
            selected_roots = [int(r.strip()) for r in args.roots.split(',')]
        except ValueError:
            parser.error("Invalid value for --roots. Must be a comma-separated list of integers.")

    output_path = args.output
    if output_mode == 'render':
        if output_path is None:
            base_name = args.constraint_file.name.replace('.json.gz', '').replace('.json', '')
            output_path = Path(f"{base_name}.{args.format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_mode == 'source' and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_constraint(
        constraint_path=args.constraint_file,
        output_path=output_path,
        max_depth=args.max_depth,
        file_format=args.format,
            rankdir=args.rankdir,
            splines=args.splines,
            output_mode=output_mode,
            max_edges_per_node=args.max_edges_per_node,
            selected_roots=selected_roots,
            llm_token_range=llm_token_range,
            state_bv_range=state_bv_range,
        )



if __name__ == "__main__":
    main()
