import json
import sys
import os
import time
import random
import multiprocessing
from collections import Counter

# We use rustfst for high-performance FST operations.
try:
    from rustfst import VectorFst, Tr
except ImportError:
    print("Error: 'rustfst' not found. Please install it.", file=sys.stderr)
    sys.exit(1)

# Use networkx for graph analysis
try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: 'networkx' or 'matplotlib' not found.", file=sys.stderr)
    print("Please install them using: pip install networkx matplotlib", file=sys.stderr)
    sys.exit(1)


# --- Core Determinization Function (unchanged) ---
def determinize_worker(
        num_states: int, start_state: int, final_states: set, transitions: set, result_queue: multiprocessing.Queue
):
    try:
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(num_states)}
        fst.set_start(state_map[start_state])
        for state_id in final_states:
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest in transitions:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))
        _ = fst.determinize()
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def time_determinization_with_timeout(
        num_states: int, start_state: int, final_states: set, transitions: set, timeout: float
) -> bool:
    if not transitions: return False
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=determinize_worker, args=(num_states, start_state, final_states, transitions, result_queue)
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return True
    try:
        return not result_queue.get_nowait()
    except Exception:
        return False


# --- MODIFIED: Reusable Graph Analysis Function ---
def analyze_graph_structure(G: nx.DiGraph, graph_name: str):
    """
    Performs and prints a detailed structural analysis of a given NetworkX graph.
    Focuses on non-trivial cycles (SCCs of size > 1).
    """
    print(f"\n--- Analyzing {graph_name} Graph Structure ---")

    if G.number_of_nodes() == 0:
        print("Graph is empty. No analysis to perform.")
        return

    print(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Degree Analysis
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 States by In-Degree (Hubs for Fan-In):")
    for node, degree in in_degrees[:10]:
        print(f"  - State {node}: {degree} incoming edges")

    print("\nTop 10 States by Out-Degree (Hubs for Fan-Out):")
    for node, degree in out_degrees[:10]:
        print(f"  - State {node}: {degree} outgoing edges")

    # Weakly Connected Components (WCC)
    wccs = list(nx.weakly_connected_components(G))
    print(f"\nFound {len(wccs)} Weakly Connected Component(s).")
    if wccs:
        largest_wcc = max(wccs, key=len)
        print(f"  - Largest WCC contains {len(largest_wcc)} nodes.")

    # --- MODIFIED SCC ANALYSIS ---
    # Find all SCCs, then filter to focus on actual cycles (size > 1)
    all_sccs = list(nx.strongly_connected_components(G))
    cycles_sccs = [scc for scc in all_sccs if len(scc) > 1]

    print(f"\nFound {len(all_sccs)} Strongly Connected Component(s) (SCCs).")
    print(f"  - {len(all_sccs) - len(cycles_sccs)} are trivial (size 1 nodes not in a cycle).")
    print(f"  - {len(cycles_sccs)} are non-trivial cycles (size > 1).")

    # Only proceed with detailed cycle analysis if there are any
    if cycles_sccs:
        cycles_sccs.sort(key=len) # Sort by size for analysis

        scc_size_counts = Counter(len(c) for c in cycles_sccs)
        print("\n  - Cycle Size Distribution (Top 5 most common):")
        for size, count in scc_size_counts.most_common(5):
            print(f"    - {count} cycles of size {size}")

        print("\n  - Examples of Smallest Cycles (SCCs > 1):")
        for i, scc in enumerate(cycles_sccs[:5]):
            nodes = list(scc)
            nodes_str = str(nodes[:10]) + ('...' if len(nodes) > 10 else '')
            print(f"    - Cycle #{i+1} (size {len(scc)}): {nodes_str}")

        print("\n  - Summary of Largest Cycles (SCCs > 1):")
        for i, scc in enumerate(cycles_sccs[-5:]):
            print(f"    - Largest Cycle #{5-i} has size {len(scc)}")
    else:
        # If there are no cycles, the graph is a DAG
        print("  - The graph is a Directed Acyclic Graph (DAG).")
    # --- END OF MODIFIED SCC ANALYSIS ---


# --- Main Analysis Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    DETERMINIZE_TIMEOUT_S = 1.0

    print(f"Loading original NWA from: {filepath}")
    with open(filepath, 'r') as f:
        nwa_data = json.load(f)

    num_states = len(nwa_data['states'])
    start_state = nwa_data['body']['start_state']
    final_states = {i for i, s in enumerate(nwa_data['states']) if s.get('final_weight') is not None}

    all_transitions = set()
    for i, state in enumerate(nwa_data['states']):
        for target_id, _ in state.get('epsilons', []):
            all_transitions.add((i, 0, target_id))
        for label_str, targets in state.get('transitions', {}).items():
            label = int(label_str)
            for target_id, _ in targets:
                all_transitions.add((i, label, target_id))

    print(f"Original NWA has {num_states} states and {len(all_transitions)} unique transitions.")

    # --- Initial Graph Analysis ---
    G_initial = nx.DiGraph()
    G_initial.add_nodes_from(range(num_states))
    G_initial.add_edges_from([(source, dest) for source, label, dest in all_transitions])
    analyze_graph_structure(G_initial, "Initial")

    print("\n--- Establishing Baseline Behavior ---")
    if not time_determinization_with_timeout(num_states, start_state, final_states, all_transitions, DETERMINIZE_TIMEOUT_S):
        print(f"Baseline determinization finished within {DETERMINIZE_TIMEOUT_S}s. This script is for hanging cases.")
        sys.exit(0)
    else:
        print(f"Baseline determinization timed out after {DETERMINIZE_TIMEOUT_S}s (as expected).")

    # --- Iterative Pruning with Adaptive Chunk Sizes ---
    print("\n--- Iterative Pruning ---")

    essential_transitions = all_transitions.copy()

    # Define the schedule of chunk sizes to try, from large to small
    total_trans_count = len(essential_transitions)
    chunk_sizes = [
        total_trans_count // 5,
        total_trans_count // 10,
        total_trans_count // 50,
        1000,
        100,
        10,
        1
    ]

    for chunk_size in chunk_sizes:
        if chunk_size <= 0: continue

        print(f"\n--- PASS with Chunk Size: {chunk_size} ---")

        untested = list(essential_transitions)
        random.shuffle(untested)

        potentially_essential = set()

        chunks = [untested[i:i + chunk_size] for i in range(0, len(untested), chunk_size)]

        for i, chunk in enumerate(chunks):
            candidate_transitions = essential_transitions - set(chunk)
            progress = f"[Chunk {i + 1}/{len(chunks)}]"

            if time_determinization_with_timeout(num_states, start_state, final_states, candidate_transitions, DETERMINIZE_TIMEOUT_S):
                essential_transitions = candidate_transitions
                print(f"{progress} ✅ Chunk removed. New count: {len(essential_transitions)}")
            else:
                potentially_essential.update(chunk)
                print(f"{progress} ❌ Chunk is essential. Keeping for next pass.")

        essential_transitions = potentially_essential
        print(f"--- End of Pass. {len(essential_transitions)} candidates remain. ---")
        if chunk_size == 1 and len(essential_transitions) == len(untested):
            print("No further reduction possible. Halting.")
            break

    print(f"\n--- Pruning Complete ---")
    print(f"Found a core of {len(essential_transitions)} transitions that causes the hang.")

    # --- Analyze the Final Core Graph using NetworkX ---
    G_final = nx.DiGraph()
    G_final.add_nodes_from(range(num_states))
    G_final.add_edges_from([(source, dest) for source, label, dest in essential_transitions])

    analyze_graph_structure(G_final, "Final Core")

    if G_final.number_of_nodes() > 0:
        degree_sequence = sorted([d for n, d in G_final.degree()], reverse=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(degree_sequence, 'b-', marker='o')
        ax1.set_title("Degree Rank Plot (Final Core Graph)")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")
        ax2.loglog(degree_sequence, 'b-', marker='o')
        ax2.set_title("Degree Rank Plot (log-log scale)")
        ax2.set_ylabel("Degree")
        ax2.set_xlabel("Rank")
        fig.tight_layout()
        plot_filename = "degree_distribution_final.png"
        plt.savefig(plot_filename)
        print(f"\nSaved final degree distribution plot to '{plot_filename}'.")