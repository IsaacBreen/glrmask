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

# We use rustfst for high-performance FST operations.
try:
    from rustfst import VectorFst, Tr
    from rustfst.algorithms.minimize import MinimizeConfig
    from rustfst.algorithms.determinize import DeterminizeConfig, DeterminizeType
except ImportError:
    print("Error: 'rustfst' not found. Please install it (e.g., 'pip install rustfst').", file=sys.stderr)
    sys.exit(1)

# Use networkx for graph analysis
try:
    import networkx as nx
except ImportError:
    print("Error: 'networkx' not found. Please install it (e.g., 'pip install networkx').", file=sys.stderr)
    sys.exit(1)

# Use tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' not found. Please install it (e.g., 'pip install tqdm').", file=sys.stderr)
    # Define a dummy tqdm class if it's not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- Weight Parsing Utilities ---
WEIGHT_ALL = P.closed(0, P.inf)
EPS_LABEL = 0


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


# --- CORE UTILITIES ---

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
                all_transitions.add((i, EPS_LABEL, target_id, weight))
        for label_str, targets in state.get('transitions', {}).items():
            # Add 1 to label to avoid conflict with epsilon=0
            label = int(label_str) + 1
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
            new_transitions.update(non_final_trans)
        else:
            new_transitions.update(final_trans)
            final_weight_union = P.empty()
            for _, _, _, weight in final_trans:
                final_weight_union |= weight
            for s, l, d, w in non_final_trans:
                new_weight = w - final_weight_union
                if not new_weight.empty:
                    new_transitions.add((s, l, d, new_weight))
    return new_transitions


def determinize_worker(
        num_states: int, start_state: int, final_states: dict, transitions: set, result_queue: multiprocessing.Queue
):
    """Worker process to build and determinize an FST."""
    try:
        # This is a simplified, unweighted determinization check.
        # We don't need the custom BitsetWeight here, just the structure.
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(num_states)}

        if start_state not in state_map:
            # If the start state isn't in the (sub)graph, there's nothing to do.
            result_queue.put(True) # Success (empty graph is determinized)
            return

        fst.set_start(state_map[start_state])

        for state_id in final_states:
            if state_id in state_map:
                fst.set_final(state_map[state_id], 0.0)

        for source, label, dest, _ in transitions:
            if source in state_map and dest in state_map:
                fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        # The full optimization pipeline
        fst = fst.rm_epsilon()
        fst.compute_and_update_properties_all()
        fst = fst.determinize()
        fst = fst.minimize()

        result_queue.put(True) # Success
    except Exception as e:
        # Put the exception in the queue to signal failure
        result_queue.put(e)


def time_determinization_with_timeout(
        num_states: int, start_state: int, final_states: dict, transitions: set, timeout: float
) -> bool:
    """
    Runs determinize_worker in a separate process with a timeout.
    Returns True if it hangs/fails, False if it succeeds.
    """
    if not transitions and not final_states:
        return False # Empty graph is trivially determinized.

    result_queue = multiprocessing.Queue(1)
    process = multiprocessing.Process(
        target=determinize_worker, args=(num_states, start_state, final_states, transitions, result_queue)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return True  # Timed out, so it's problematic.

    try:
        result = result_queue.get_nowait()
        # If result is an exception, it failed. If it's True, it succeeded.
        return isinstance(result, Exception)
    except Exception:
        # Queue was empty, process likely crashed without putting anything.
        return True


# --- ANALYSIS FUNCTIONS ---

def print_graph_stats(G: nx.DiGraph, nwa_data: dict):
    """Prints high-level stats for a graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_transitions = len(nwa_data['transitions'])
    epsilon_transitions = sum(1 for _, l, _, _ in nwa_data['transitions'] if l == EPS_LABEL)
    symbol_transitions = num_transitions - epsilon_transitions

    print("\n--- NWA Structure ---")
    print(f"Total States: {num_nodes}")
    if "start_state" in nwa_data and nwa_data["start_state"] is not None:
        print(f"Start State: {nwa_data['start_state']}")
    else:
        print("Start State: None")
    print(f"Final States: {len(nwa_data['final_states'])}")
    print("\n--- Transitions ---")
    print(f"Total Unique Transitions (src, label, dst, weight): {num_transitions}")
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
            f"{src} --({lbl if lbl != EPS_LABEL else 'ε'})--> {dst}"
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
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(nwa["num_states"])}
        fst.set_start(state_map[nwa["start_state"]])
        for state_id, weight in nwa["final_states"].items():
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in nwa["transitions"]:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        print("\nFST Statistics:")
        print_fst_stats(fst, "Initial FST")
        fst = fst.connect()
        print_fst_stats(fst, "After connecting")
        fst = fst.rm_epsilon()
        print_fst_stats(fst, "After removing epsilons")
        fst.compute_and_update_properties_all()
        print_fst_stats(fst, "After property computation")
        fst = fst.determinize()
        print_fst_stats(fst, "After determinizing")
        fst = fst.minimize()
        print_fst_stats(fst, "After minimizing")
        print("\nRESULT: ✅ Determinization finished successfully.")
    except Exception as e:
        print(f"RESULT: ❌ Determinization failed with an error: {e}")


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
        final_weight = parse_weight(state_data.get('final_weight'))
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
    nwa = load_nwa_data(args.filepath)
    for state_id in args.states:
        if 0 <= state_id < nwa["num_states"]:
            print(json.dumps(nwa["raw_data"]["states"][state_id], separators=(',', ':')))
        else:
            print(f"Error: State ID {state_id} is out of bounds (0-{nwa['num_states'] - 1}).", file=sys.stderr)


def create_subgraph_nwa(nwa_data: dict, states_to_keep: set) -> dict | None:
    """
    Creates a new NWA dictionary containing only the states and transitions
    within the 'states_to_keep' set.
    """
    if not states_to_keep:
        return None

    new_start_state = nwa_data["start_state"]
    if new_start_state not in states_to_keep:
        new_start_state = next(iter(states_to_keep), None)
        if new_start_state is None:
            return None

    subgraph_transitions = {
        (s, l, d, w) for s, l, d, w in nwa_data["transitions"]
        if s in states_to_keep and d in states_to_keep
    }
    subgraph_finals = {
        s: w for s, w in nwa_data["final_states"].items() if s in states_to_keep
    }

    return {
        "num_states": nwa_data["num_states"],
        "start_state": new_start_state,
        "final_states": subgraph_finals,
        "transitions": subgraph_transitions,
    }


def run_bisect_pass(args):
    """Executes the 'bisect' pass to find a minimal failing example."""
    print(f"--- Running Bisect Pass (Timeout: {args.timeout}s) ---")
    nwa_data = load_nwa_data(args.filepath)
    all_states = set(range(nwa_data["num_states"]))

    print("\n--- Establishing Baseline ---")
    if not time_determinization_with_timeout(
        nwa_data["num_states"], nwa_data["start_state"], nwa_data["final_states"],
        nwa_data["transitions"], args.timeout
    ):
        print("✅ Baseline determinization finished within the timeout. Nothing to bisect.")
        return
    else:
        print(f"✅ Baseline determinization timed out after {args.timeout}s. Starting bisection.")

    problematic_states = list(all_states)
    smallest_failing_set = set(problematic_states)

    iteration = 0
    while len(problematic_states) > 1:
        iteration += 1
        print(f"\n--- Bisection Iteration {iteration}: {len(problematic_states)} states remaining ---")
        
        problematic_states.sort() # Sort for deterministic splitting
        mid = len(problematic_states) // 2
        s1 = set(problematic_states[:mid])
        s2 = set(problematic_states[mid:])
        
        print(f"Splitting into two sets of sizes: {len(s1)} and {len(s2)}")

        subgraph1_data = create_subgraph_nwa(nwa_data, s1)
        subgraph2_data = create_subgraph_nwa(nwa_data, s2)

        hangs1 = time_determinization_with_timeout(
            subgraph1_data["num_states"], subgraph1_data["start_state"],
            subgraph1_data["final_states"], subgraph1_data["transitions"], args.timeout
        ) if subgraph1_data else False
        
        hangs2 = time_determinization_with_timeout(
            subgraph2_data["num_states"], subgraph2_data["start_state"],
            subgraph2_data["final_states"], subgraph2_data["transitions"], args.timeout
        ) if subgraph2_data else False

        if hangs1 and not hangs2:
            print("-> Problem is in the first half. Discarding second half.")
            problematic_states = list(s1)
            smallest_failing_set = s1
        elif not hangs1 and hangs2:
            print("-> Problem is in the second half. Discarding first half.")
            problematic_states = list(s2)
            smallest_failing_set = s2
        else:
            print("-> Interaction detected or both halves are problematic. Cannot reduce further with this split.")
            break
            
    print("\n--- Bisection Complete ---")
    print(f"Found a minimal problematic set of {len(smallest_failing_set)} states:")
    print(sorted(list(smallest_failing_set)))

    print("\n--- Analysis of Minimal Failing Subgraph ---")
    minimal_nwa_data = create_subgraph_nwa(nwa_data, smallest_failing_set)
    G_minimal = nx.DiGraph()
    G_minimal.add_nodes_from(minimal_nwa_data["start_state"] if minimal_nwa_data["start_state"] is not None else [])
    G_minimal.add_edges_from([(s, d) for s, l, d, w in minimal_nwa_data["transitions"]])
    
    print_graph_stats(G_minimal, minimal_nwa_data)
    print_scc_analysis(G_minimal, minimal_nwa_data["transitions"])

    output_path = "minimal_failing_nwa.json"
    print(f"\nSaving minimal failing NWA to {output_path}...")
    
    state_remapping = {old_id: new_id for new_id, old_id in enumerate(sorted(list(smallest_failing_set)))}
    
    remapped_states = []
    for old_id in sorted(list(smallest_failing_set)):
        state_data = nwa_data["raw_data"]["states"][old_id]
        new_state_data = {}
        if 'final_weight' in state_data:
            new_state_data['final_weight'] = state_data['final_weight']
        if 'epsilons' in state_data:
            new_state_data['epsilons'] = [[state_remapping[t[0]], t[1]] for t in state_data['epsilons'] if t[0] in state_remapping]
        if 'transitions' in state_data:
            new_transitions = {}
            for label, targets in state_data['transitions'].items():
                new_targets = [[state_remapping[t[0]], t[1]] for t in targets if t[0] in state_remapping]
                if new_targets:
                    new_transitions[label] = new_targets
            new_state_data['transitions'] = new_transitions
        remapped_states.append(new_state_data)

    minimal_json = {
        "states": remapped_states,
        "body": {
            "start_state": state_remapping.get(minimal_nwa_data["start_state"])
        }
    }
    with open(output_path, 'w') as f:
        json.dump(minimal_json, f, indent=2)
    print("Save complete.")


# --- MAIN CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An analysis and determinization tool for NWA JSON dumps with interval weights.",
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
    parser_det = subparsers.add_parser("determinize", help="Attempt to determinize the full NWA using a single FST (may fail).")
    parser_det.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_det.set_defaults(func=run_determinize_pass)

    # --- Inspect States Pass (Pretty) ---
    parser_inspect = subparsers.add_parser("inspect-states", help="Pretty-print a human-readable summary for specific state IDs.")
    parser_inspect.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_inspect.add_argument("states", type=int, nargs='+', help="One or more state IDs to display.")
    parser_inspect.set_defaults(func=run_inspect_states_pass)

    # --- Dump States Pass (Raw) ---
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
        "--timeout", type=float, default=1.0,
        help="Timeout in seconds for each determinization check during pruning."
    )
    parser_prune.set_defaults(func=run_prune_pass)

    # --- Bisect Pass ---
    parser_bisect = subparsers.add_parser(
        "bisect",
        help="Use binary search to find a minimal subgraph that fails to determinize."
    )
    parser_bisect.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_bisect.add_argument(
        "--timeout", type=float, default=1.0,
        help="Timeout in seconds for each determinization check."
    )
    parser_bisect.set_defaults(func=run_bisect_pass)

    # --- Chunked Determinize Pass ---
    parser_chunked_det = subparsers.add_parser(
        "chunked-determinize",
        help="Determinize the NWA by breaking it into chunks, optimizing them, and re-combining."
    )
    parser_chunked_det.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_chunked_det.add_argument(
        "--chunks", type=int, default=10,
        help="Number of chunks to split the transitions into (default: 10)."
    )
    parser_chunked_det.set_defaults(func=run_chunked_determinize_pass)
    
    # --- Recompose Pass ---
    parser_recompose = subparsers.add_parser(
        "recompose",
        help="Determinize an interval-weighted NWA by decomposing it into multiple standard NFAs,\n"
             "determinizing each, and recomposing the results into a DWA."
    )
    parser_recompose.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_recompose.set_defaults(func=run_recompose_pass)

    args = parser.parse_args()
    args.func(args)