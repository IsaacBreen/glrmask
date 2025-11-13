#!/usr/bin/env python
import json
import sys
import os
import time
import random
import multiprocessing
import argparse
from collections import Counter, defaultdict

# (Import statements for rustfst and networkx remain the same)
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


# --- CORE UTILITIES (unchanged) ---

def load_nwa_data(filepath: str) -> dict:
    """Loads and parses the NWA JSON file into a structured dictionary."""
    print(f"Loading NWA from: {filepath}")
    with open(filepath, 'r') as f:
        nwa_data = json.load(f)

    num_states = len(nwa_data['states'])
    start_state = nwa_data['body']['start_state']
    final_states = {i for i, s in enumerate(nwa_data['states']) if s.get('final_weight') is not None}

    all_transitions = set()
    for i, state in enumerate(nwa_data['states']):
        for target_id, _ in state.get('epsilons', []):
            all_transitions.add((i, 0, target_id))  # Epsilon is label 0
        for label_str, targets in state.get('transitions', {}).items():
            label = int(label_str)
            for target_id, _ in targets:
                all_transitions.add((i, label, target_id))

    print(f"Original NWA has {num_states} states and {len(all_transitions)} unique transitions.")
    return {
        "num_states": num_states,
        "start_state": start_state,
        "final_states": final_states,
        "transitions": all_transitions,
        "raw_data": nwa_data
    }


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
        return True  # Timed out
    try:
        return not result_queue.get_nowait()  # False if determinization succeeded
    except Exception:
        return False  # Also false if it crashed but finished quickly


# --- ANALYSIS FUNCTIONS (unchanged) ---

def print_graph_stats(G: nx.DiGraph):
    """Prints high-level stats for a graph."""
    print(f"\nThe graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 States by In-Degree (Hubs for Fan-In):")
    for node, degree in in_degrees[:5]:
        print(f"  - State {node}: {degree} incoming edges")
    print("\nTop 5 States by Out-Degree (Hubs for Fan-Out):")
    for node, degree in out_degrees[:5]:
        print(f"  - State {node}: {degree} outgoing edges")
    wccs = list(nx.weakly_connected_components(G))
    print(f"\nFound {len(wccs)} Weakly Connected Component(s).")
    if wccs:
        print(f"  - Largest WCC contains {len(max(wccs, key=len))} nodes.")


def print_scc_analysis(G: nx.DiGraph, transitions: set):
    """Prints detailed analysis of cycles (SCCs > 1)."""
    all_sccs = list(nx.strongly_connected_components(G))
    cycles_sccs = [scc for scc in all_sccs if len(scc) > 1]
    print(f"\nFound {len(all_sccs)} Strongly Connected Component(s) (SCCs).")
    print(f"  - {len(all_sccs) - len(cycles_sccs)} are trivial (size 1).")
    print(f"  - {len(cycles_sccs)} are non-trivial cycles (size > 1).")

    if not cycles_sccs:
        print("  - The graph is a Directed Acyclic Graph (DAG).")
        return

    source_to_transitions = defaultdict(list)
    for source, label, dest in transitions:
        source_to_transitions[source].append((label, dest))
    cycles_sccs.sort(key=len)
    print("\nExamples of Smallest Cycles with internal edges:")
    MAX_EDGES_TO_SHOW = 10
    for i, scc in enumerate(cycles_sccs[:5]):
        scc_nodes = set(scc)
        print(f"    - Cycle #{i + 1} (size {len(scc_nodes)}): Nodes {list(scc_nodes)}")
        internal_edges = [
            f"{src} --({lbl})--> {dst}"
            for src in scc_nodes if src in source_to_transitions
            for lbl, dst in source_to_transitions[src] if dst in scc_nodes
        ]
        for edge_str in internal_edges[:MAX_EDGES_TO_SHOW]:
            print(f"      - {edge_str}")
        if len(internal_edges) > MAX_EDGES_TO_SHOW:
            print(f"      - ... and {len(internal_edges) - MAX_EDGES_TO_SHOW} more internal edges")


# --- PASS FUNCTIONS ---

def run_stats_pass(args):
    """Executes the 'stats' pass."""
    print("--- Running Stats Pass ---")
    nwa = load_nwa_data(args.filepath)
    G = nx.DiGraph()
    G.add_nodes_from(range(nwa["num_states"]))
    G.add_edges_from([(s, d) for s, l, d in nwa["transitions"]])
    print_graph_stats(G)


def run_scc_pass(args):
    """Executes the 'scc' pass."""
    print("--- Running SCC/Cycle Analysis Pass ---")
    nwa = load_nwa_data(args.filepath)
    G = nx.DiGraph()
    G.add_nodes_from(range(nwa["num_states"]))
    G.add_edges_from([(s, d) for s, l, d in nwa["transitions"]])
    print_scc_analysis(G, nwa["transitions"])


def run_determinize_pass(args):
    """Executes the 'determinize' pass."""
    print(f"--- Running Determinization Pass (Timeout: {args.timeout}s) ---")
    nwa = load_nwa_data(args.filepath)
    print("\nAttempting to determinize the full NWA...")
    timed_out = time_determinization_with_timeout(
        nwa["num_states"], nwa["start_state"], nwa["final_states"], nwa["transitions"], args.timeout
    )
    if timed_out:
        print(f"RESULT: ❌ Timed out after {args.timeout} seconds.")
    else:
        print("RESULT: ✅ Determinization finished within the time limit.")


# --- NEW AND MODIFIED PASSES FOR STATE INSPECTION ---

def run_inspect_states_pass(args):
    """Executes the 'inspect-states' pass for pretty-printing."""
    print("--- Running Inspect States Pass ---")
    nwa = load_nwa_data(args.filepath)
    print(f"\nInspecting data for states: {args.states}")
    for state_id in args.states:
        print(f"\n{'=' * 10} State {state_id} {'=' * 10}")
        if not (0 <= state_id < nwa["num_states"]):
            print(f"Error: State ID {state_id} is out of bounds (0-{nwa['num_states'] - 1}).")
            continue

        state_data = nwa["raw_data"]["states"][state_id]

        final_weight = state_data.get('final_weight')
        print(f"Final State: {'Yes (weight: ' + str(final_weight) + ')' if final_weight is not None else 'No'}")

        print("\nEpsilon Transitions (ε):")
        epsilons = state_data.get('epsilons', [])
        if epsilons:
            for target_id, weight in epsilons:
                print(f"  - ε --> {target_id}  (weight: {weight})")
        else:
            print("  - None")

        print("\nSymbol Transitions:")
        transitions = state_data.get('transitions', {})
        if transitions:
            for label_str, targets in sorted(transitions.items(), key=lambda item: int(item[0])):
                for target_id, weight in targets:
                    print(f"  - On symbol '{label_str}' --> {target_id}  (weight: {weight})")
        else:
            print("  - None")


def run_dump_states_pass(args):
    """Executes the 'dump-states' pass for raw JSON output."""
    # Note: No "--- Running Pass ---" print to keep output clean for piping
    nwa = load_nwa_data(args.filepath)
    for state_id in args.states:
        if 0 <= state_id < nwa["num_states"]:
            # Print compact, single-line JSON
            print(json.dumps(nwa["raw_data"]["states"][state_id], separators=(',', ':')))
        else:
            # Print error to stderr to not pollute stdout
            print(f"Error: State ID {state_id} is out of bounds (0-{nwa['num_states'] - 1}).", file=sys.stderr)


def run_prune_pass(args):
    """Executes the 'prune' pass."""
    print(f"--- Running Pruning Pass (Timeout: {args.timeout}s) ---")
    nwa = load_nwa_data(args.filepath)

    print("\n--- Establishing Baseline Behavior ---")
    if not time_determinization_with_timeout(nwa["num_states"], nwa["start_state"], nwa["final_states"], nwa["transitions"], args.timeout):
        print(f"Baseline determinization finished within {args.timeout}s. Pruning is not needed.")
        return
    else:
        print(f"Baseline determinization timed out after {args.timeout}s (as expected).")

    print("\n--- Iterative Pruning ---")
    essential_transitions = nwa["transitions"].copy()
    total_trans_count = len(essential_transitions)
    chunk_sizes = [
        total_trans_count // 5, total_trans_count // 10, total_trans_count // 50,
        1000, 100, 10, 1
    ]

    for chunk_size in [c for c in chunk_sizes if c > 0]:
        print(f"\n--- PASS with Chunk Size: {chunk_size} ---")
        untested = list(essential_transitions)
        random.shuffle(untested)
        potentially_essential = set()
        chunks = [untested[i:i + chunk_size] for i in range(0, len(untested), chunk_size)]

        for i, chunk in enumerate(chunks):
            candidate_transitions = essential_transitions - set(chunk)
            progress = f"[Chunk {i + 1}/{len(chunks)}]"
            if time_determinization_with_timeout(
                    nwa["num_states"],
                    nwa["start_state"],
                    nwa["final_states"],
                    candidate_transitions,
                    args.timeout
                    ):
                essential_transitions = candidate_transitions
                print(f"{progress} ✅ Chunk removed. New count: {len(essential_transitions)}")
            else:
                potentially_essential.update(chunk)
                print(f"{progress} ❌ Chunk is essential. Keeping for next pass.")

        if len(potentially_essential) == len(essential_transitions) and chunk_size == 1:
            print("No further reduction possible. Halting.")
            break
        essential_transitions = potentially_essential
        print(f"--- End of Pass. {len(essential_transitions)} candidates remain. ---")

    print(f"\n--- Pruning Complete ---")
    print(f"Found a core of {len(essential_transitions)} transitions that causes the hang.")

    print("\n--- Analysis of Final Core Graph ---")
    G_final = nx.DiGraph()
    G_final.add_nodes_from(range(nwa["num_states"]))
    G_final.add_edges_from([(s, d) for s, l, d in essential_transitions])
    print_graph_stats(G_final)
    print_scc_analysis(G_final, essential_transitions)


# --- MAIN CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An analysis tool for NWA (Nested Word Automata) JSON dumps.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Stats Pass ---
    parser_stats = subparsers.add_parser("stats", help="Show high-level graph statistics (degrees, components).")
    parser_stats.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_stats.set_defaults(func=run_stats_pass)

    # --- SCC Pass ---
    parser_scc = subparsers.add_parser("scc", help="Show detailed analysis of cycles (SCCs > 1) with edge labels.")
    parser_scc.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_scc.set_defaults(func=run_scc_pass)

    # --- Determinize Pass ---
    parser_det = subparsers.add_parser("determinize", help="Attempt to determinize the full NWA with a timeout.")
    parser_det.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_det.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds for the determinization attempt.")
    parser_det.set_defaults(func=run_determinize_pass)

    # --- MODIFIED: Inspect States Pass (Pretty) ---
    parser_inspect = subparsers.add_parser("inspect-states", help="Pretty-print a human-readable summary for specific state IDs.")
    parser_inspect.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_inspect.add_argument("states", type=int, nargs='+', help="One or more state IDs to display.")
    parser_inspect.set_defaults(func=run_inspect_states_pass)

    # --- NEW: Dump States Pass (Raw) ---
    parser_dump = subparsers.add_parser("dump-states", help="Print the raw, compact JSON data for specific state IDs.")
    parser_dump.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_dump.add_argument("states", type=int, nargs='+', help="One or more state IDs to display.")
    parser_dump.set_defaults(func=run_dump_states_pass)

    # --- Prune Pass ---
    parser_prune = subparsers.add_parser(
        "prune",
        help="Iteratively prune transitions to find the core set causing determinization to hang."
        )
    parser_prune.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_prune.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for each determinization check during pruning."
        )
    parser_prune.set_defaults(func=run_prune_pass)

    args = parser.parse_args()
    args.func(args)