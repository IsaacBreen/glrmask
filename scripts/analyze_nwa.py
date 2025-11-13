#!/usr/bin/env python
import json
import sys
import os
import time
import random
import multiprocessing
import argparse
from collections import Counter, defaultdict
import portion as P

from rustfst.algorithms.determinize import DeterminizeConfig, DeterminizeType
from rustfst.algorithms.minimize import MinimizeConfig

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


# --- NEW: Weight Parsing Utilities ---

# Special object for the "ALL" weight, assuming non-negative integers like Rust's usize
WEIGHT_ALL = P.closed(0, P.inf)

def parse_weight(data):
    """Parses NWA weight JSON into a portion.Interval object."""
    if data is None:
        return P.empty()
    if isinstance(data, str) and data == "ALL":
        return WEIGHT_ALL
    
    interval = P.empty()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, int):
                interval |= P.singleton(item)
            elif isinstance(item, list) and len(item) == 2:
                interval |= P.closed(item[0], item[1])
    return interval

# --- CORE UTILITIES (unchanged) ---

def load_nwa_data(filepath: str) -> dict:
    """Loads and parses the NWA JSON file into a structured dictionary."""
    print(f"Loading NWA from: {filepath}")
    with open(filepath, 'r') as f:
        nwa_data = json.load(f)

    num_states = len(nwa_data['states'])
    start_state = nwa_data['body']['start_state']
    final_states = {
        i: parse_weight(s.get('final_weight'))
        for i, s in enumerate(nwa_data['states'])
        if s.get('final_weight') is not None and not parse_weight(s.get('final_weight')).empty
    }

    all_transitions = set()
    for i, state in enumerate(nwa_data['states']):
        for target_id, weight_data in state.get('epsilons', []):
            weight = parse_weight(weight_data)
            if not weight.empty:
                all_transitions.add((i, 0, target_id, weight))  # Epsilon is label 0
        for label_str, targets in state.get('transitions', {}).items():
            label = int(label_str)
            for target_id, weight_data in targets:
                weight = parse_weight(weight_data)
                if not weight.empty:
                    all_transitions.add((i, label, target_id, weight))

    print(f"Original NWA has {num_states} states and {len(all_transitions)} unique transitions.")
    return {
        "num_states": num_states,
        "start_state": start_state,
        "final_states": final_states,
        "transitions": all_transitions,
        "raw_data": nwa_data
    }


def prune_to_final_state_transitions(transitions: set, final_state_ids: set) -> set:
    """
    For each (source, label) pair, if there are transitions to final states,
    prune the weights of transitions to non-final states by subtracting the
    union of weights of transitions to final states.
    """
    source_label_to_transitions = defaultdict(list)
    for t in transitions:
        source, label, _, _ = t
        source_label_to_transitions[(source, label)].append(t)

    new_transitions = set()
    for (source, label), trans_group in source_label_to_transitions.items():
        final_trans = []
        non_final_trans = []
        for t in trans_group:
            if t[2] in final_state_ids:
                final_trans.append(t)
            else:
                non_final_trans.append(t)

        if not final_trans:
            # No transitions to a final state for this (source, label), so keep all non-final ones.
            new_transitions.update(non_final_trans)
        else:
            # Keep all transitions to final states as they are.
            new_transitions.update(final_trans)

            # Calculate the union of weights for all transitions to final states.
            final_weight_union = P.empty()
            for _, _, _, weight in final_trans:
                final_weight_union |= weight

            # For non-final transitions, subtract the final_weight_union and keep if not empty.
            for s, l, d, w in non_final_trans:
                new_weight = w - final_weight_union
                if not new_weight.empty:
                    new_transitions.add((s, l, d, new_weight))

    return new_transitions

def determinize_worker(
        num_states: int, start_state: int, final_states: dict, transitions: set, result_queue: multiprocessing.Queue
):
    try:
        transitions = prune_to_final_state_transitions(transitions, final_states.keys())
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(num_states)}
        fst.set_start(state_map[start_state])
        for state_id, weight in final_states.items():
            # FIXME: Using 0.0 for final weight as rustfst-python only supports float weights.
            # The actual weight is `weight`.
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in transitions:
            # FIXME: Using 0.0 for transition weight as rustfst-python only supports float weights.
            # The actual weight is `weight`.
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))
        print("Minimizing")
        fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
        print("Determinizing")
        fst = fst.determinize()
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def time_determinization_with_timeout(
        num_states: int, start_state: int, final_states: dict, transitions: set, timeout: float
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

def print_graph_stats(G: nx.DiGraph, nwa_data: dict):
    """Prints high-level stats for a graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()  # Unique (src, dst) pairs

    num_transitions = len(nwa_data['transitions'])
    epsilon_transitions = sum(1 for _, l, _, _ in nwa_data['transitions'] if l == 0)
    symbol_transitions = num_transitions - epsilon_transitions

    print("\n--- NWA Structure ---")
    print(f"Total States: {num_nodes}")
    print(f"Start State: {nwa_data['start_state']}")
    print(f"Final States: {len(nwa_data['final_states'])}")

    print("\n--- Transitions ---")
    print(f"Total Unique Transitions (src, label, dst): {num_transitions}")
    print(f"  - Epsilon (ε) transitions: {epsilon_transitions}")
    print(f"  - Symbol transitions: {symbol_transitions}")

    print(f"\n--- Graph Connectivity ---")
    print(f"The graph has {num_nodes} nodes and {num_edges} unique source-destination edges.")
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
    for source, label, dest, weight in transitions:
        source_to_transitions[source].append((label, dest))
    cycles_sccs.sort(key=len)
    print("\nExamples of Smallest Cycles with internal edges:")
    MAX_EDGES_TO_SHOW = 10
    for i, scc in enumerate(cycles_sccs[:5]):
        scc_nodes = set(scc)
        print(f"    - Cycle #{i + 1} (size {len(scc_nodes)}): Nodes {list(scc_nodes)}")
        internal_edges = [
            f"{src} --({lbl})--> {dst}" # Note: weight is ignored for brevity
            for src in scc_nodes if src in source_to_transitions
            for lbl, dst in source_to_transitions[src] if dst in scc_nodes
        ]
        for edge_str in internal_edges[:MAX_EDGES_TO_SHOW]:
            print(f"      - {edge_str}")
        if len(internal_edges) > MAX_EDGES_TO_SHOW:
            print(f"      - ... and {len(internal_edges) - MAX_EDGES_TO_SHOW} more internal edges")

def print_fst_stats(fst: VectorFst, message: str):
    """Prints basic stats for a rustfst FST."""
    num_states = fst.num_states()
    num_arcs = sum(fst.num_trs(s) for s in fst.states())
    print(f"  - {message}: {num_states} states, {num_arcs} arcs.")


# --- PASS FUNCTIONS ---

def run_stats_pass(args):
    """Executes the 'stats' pass."""
    print("--- Running Stats Pass ---")
    nwa = load_nwa_data(args.filepath)
    G = nx.DiGraph()
    G.add_nodes_from(range(nwa["num_states"]))
    G.add_edges_from([(s, d) for s, l, d, w in nwa["transitions"]])
    print_graph_stats(G, nwa)


def run_scc_pass(args):
    """Executes the 'scc' pass."""
    print("--- Running SCC/Cycle Analysis Pass ---")
    nwa = load_nwa_data(args.filepath)
    G = nx.DiGraph()
    G.add_nodes_from(range(nwa["num_states"]))
    G.add_edges_from([(s, d) for s, l, d, w in nwa["transitions"]])
    print_scc_analysis(G, nwa["transitions"])


def run_determinize_pass(args):
    """Executes the 'determinize' pass."""
    print("--- Running Determinization Pass ---")
    nwa = load_nwa_data(args.filepath)
    print("\nAttempting to determinize the full NWA...")
    try:
        transitions = prune_to_final_state_transitions(nwa["transitions"], nwa["final_states"].keys())
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(nwa["num_states"])}
        fst.set_start(state_map[nwa["start_state"]])
        for state_id, weight in nwa["final_states"].items():
            # FIXME: Using 0.0 for final weight as rustfst-python only supports float weights.
            # The actual weight is `weight`.
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in transitions:
            # FIXME: Using 0.0 for transition weight as rustfst-python only supports float weights.
            # The actual weight is `weight`.
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        print("\nFST Statistics:")
        print_fst_stats(fst, "Initial FST")

        print("\nRemoving epsilons...")
        fst = fst.rm_epsilon()
        print_fst_stats(fst, "After removing epsilons")

        print("\nConnecting...")
        fst = fst.connect()
        print_fst_stats(fst, "After connecting")

        print("\ntr_unique...")
        fst = fst.tr_unique()
        print_fst_stats(fst, "After tr_unique")

        print("\nMinimizing...")
        fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
        print_fst_stats(fst, "After minimizing")

        print("\nPruning again before determinization...")
        num_states_after_min = fst.num_states()
        start_state_after_min = fst.start()
        final_states_after_min = {s for s in fst.states() if fst.is_final(s)}
        transitions_after_min = set()
        for s in fst.states():
            for tr in fst.trs(s):
                # FIXME: Weight is lost here, using a placeholder.
                transitions_after_min.add((s, tr.ilabel, tr.next_state, WEIGHT_ALL))

        pruned_transitions = prune_to_final_state_transitions(transitions_after_min, final_states_after_min)

        # Rebuild FST from pruned transitions
        pruned_fst = VectorFst()
        state_map_after_min = {i: pruned_fst.add_state() for i in range(num_states_after_min)}
        if start_state_after_min is not None:
            pruned_fst.set_start(state_map_after_min[start_state_after_min])
        for state_id in final_states_after_min:
            # FIXME: Weight is lost here.
            pruned_fst.set_final(state_map_after_min[state_id], 0.0)
        for source, label, dest, weight in pruned_transitions:
            if source in state_map_after_min and dest in state_map_after_min:
                # FIXME: Using 0.0 for transition weight.
                pruned_fst.add_tr(state_map_after_min[source], Tr(label, label, 0.0, state_map_after_min[dest]))

        fst = pruned_fst
        print_fst_stats(fst, "After second pruning")

        print("\nDeterminizing...")
        fst = fst.determinize()
        print_fst_stats(fst, "After determinizing")

        print("\nRESULT: ✅ Determinization finished successfully.")
    except Exception as e:
        print(f"RESULT: ❌ Determinization failed with an error: {e}")
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

        final_weight_data = state_data.get('final_weight')
        final_weight = parse_weight(final_weight_data)
        print(f"Final State: {'Yes' if not final_weight.empty else 'No'}")
        if not final_weight.empty:
            print(f"  - Final Weight: {final_weight}")

        print("\nEpsilon Transitions (ε):")
        epsilons = state_data.get('epsilons', [])
        if epsilons:
            for target_id, weight in epsilons:
                print(f"  - ε --> {target_id}  (weight: {parse_weight(weight)})")
        else:
            print("  - None")

        print("\nSymbol Transitions:")
        transitions = state_data.get('transitions', {})
        if transitions:
            for label_str, targets in sorted(transitions.items(), key=lambda item: int(item[0])):
                for target_id, weight in targets:
                    print(f"  - On symbol '{label_str}' --> {target_id}  (weight: {parse_weight(weight)})")
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
    pruned_nwa_data = {
        "start_state": nwa["start_state"],
        "final_states": nwa["final_states"],
        "transitions": essential_transitions,
    }
    print_graph_stats(G_final, pruned_nwa_data)
    print_scc_analysis(G_final, essential_transitions)


def run_chunked_determinize_pass(args):
    """Executes the 'chunked-determinize' pass."""
    print(f"--- Running Chunked Determinization Pass (Chunks: {args.chunks}) ---")
    nwa = load_nwa_data(args.filepath)

    # Pre-prune transitions
    print("\nPruning transitions that cannot lead to a final state...")
    transitions_pruned = prune_to_final_state_transitions(nwa["transitions"], nwa["final_states"].keys())
    print(f"Pruned {len(nwa['transitions']) - len(transitions_pruned)} transitions.")

    # 1. Partition transitions
    all_transitions = list(transitions_pruned)
    random.shuffle(all_transitions) # Shuffle to get reasonably balanced chunks
    if not all_transitions:
        print("No transitions left after pruning. Nothing to do.")
        return

    chunk_size = (len(all_transitions) + args.chunks - 1) // args.chunks # Ceiling division
    transition_chunks = [all_transitions[i:i + chunk_size] for i in range(0, len(all_transitions), chunk_size)]
    print(f"\nPartitioned {len(all_transitions)} transitions into {len(transition_chunks)} chunks of up to {chunk_size} transitions each.")

    optimized_chunks = []
    for i, chunk in enumerate(transition_chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(transition_chunks)} ---")
        if not chunk:
            print("Skipping empty chunk.")
            continue
        try:
            # 2. Create FST for the chunk
            fst = VectorFst()
            # We need all states in each FST so transitions are valid
            state_map = {j: fst.add_state() for j in range(nwa["num_states"])}
            fst.set_start(state_map[nwa["start_state"]])
            for state_id, weight in nwa["final_states"].items():
                # FIXME: Using 0.0 for final weight.
                fst.set_final(state_map[state_id], 0.0)
            for source, label, dest, weight in chunk:
                # FIXME: Using 0.0 for transition weight.
                fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

            print_fst_stats(fst, f"Chunk {i+1} Initial FST")

            # 3. Optimize the chunk
            fst = fst.rm_epsilon()
            fst = fst.connect()
            fst = fst.tr_unique()
            fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
            fst = fst.determinize()
            print_fst_stats(fst, f"Chunk {i+1} Optimized FST")
            optimized_chunks.append(fst)
        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}. Skipping this chunk.")

    if not optimized_chunks:
        print("\nRESULT: ❌ No chunks were successfully optimized. Halting.")
        return

    # 4. Union the optimized chunks
    print("\n--- Combining Optimized Chunks ---")

    # The `|` operator on VectorFst creates a copy, then unions. This is safe.
    final_fst = optimized_chunks[0]
    print_fst_stats(final_fst, "Combined FST (1 chunk)")
    for i, next_fst in enumerate(optimized_chunks[1:], 2):
        final_fst = final_fst | next_fst
        print_fst_stats(final_fst, f"Combined FST ({i} chunks)")

    # 5. Final optimization pass
    print("\n--- Final Optimization Pass on Combined FST ---")
    try:
        print_fst_stats(final_fst, "Combined FST before final optimization")

        # The full pipeline from the original determinize pass
        final_fst = final_fst.rm_epsilon()
        print_fst_stats(final_fst, "After rm_epsilon")
        final_fst = final_fst.connect()
        print_fst_stats(final_fst, "After connecting")
        final_fst = final_fst.tr_unique()
        print_fst_stats(final_fst, "After tr_unique")
        final_fst = final_fst.minimize(config=MinimizeConfig(allow_nondet=True))
        print_fst_stats(final_fst, "After minimizing")
        final_fst = final_fst.determinize()
        print_fst_stats(final_fst, "After determinizing")

        print("\nRESULT: ✅ Chunked determinization finished successfully.")
    except Exception as e:
        print(f"\nRESULT: ❌ Final optimization failed with an error: {e}")


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
    parser_det = subparsers.add_parser("determinize", help="Attempt to determinize the full NWA.")
    parser_det.add_argument("filepath", help="Path to the nwa_dump.json file.")
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

    # --- NEW: Chunked Determinize Pass ---
    parser_chunked_det = subparsers.add_parser(
        "chunked-determinize",
        help="Determinize the NWA by breaking it into chunks, optimizing them, and re-combining."
    )
    parser_chunked_det.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_chunked_det.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Number of chunks to split the transitions into (default: 10)."
    )
    parser_chunked_det.set_defaults(func=run_chunked_determinize_pass)

    args = parser.parse_args()
    args.func(args)
