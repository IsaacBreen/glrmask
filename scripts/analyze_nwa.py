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
from tqdm import tqdm

# We use rustfst for high-performance FST operations.
try:
    from rustfst import VectorFst, Tr
    from rustfst.algorithms.minimize import MinimizeConfig
except ImportError:
    print("Error: 'rustfst' not found. Please install it.", file=sys.stderr)
    sys.exit(1)

# Use networkx for graph analysis
try:
    import networkx as nx
except ImportError:
    print("Error: 'networkx' not found. Please install it.", file=sys.stderr)
    sys.exit(1)

# --- Weight Parsing Utilities ---
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
    try:
        transitions = prune_to_final_state_transitions(transitions, final_states.keys())
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(num_states)}
        fst.set_start(state_map[start_state])
        for state_id, weight in final_states.items():
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in transitions:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
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
        return True
    try:
        return not result_queue.get_nowait()
    except Exception:
        return False


# --- ANALYSIS FUNCTIONS ---

def print_graph_stats(G: nx.DiGraph, nwa_data: dict):
    """Prints high-level stats for a graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
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
            f"{src} --({lbl})--> {dst}"
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


# --- Recomposition Pass Functions ---

def get_atomic_intervals(nwa_data: dict) -> list[P.Interval]:
    """
    Finds all unique interval boundaries and returns a list of "atomic" intervals
    where the set of active transitions is constant.
    """
    print("Discretizing weight intervals...")
    boundary_points = {0}
    for _, _, _, weight in nwa_data["transitions"]:
        if not weight.empty:
            boundary_points.add(weight.lower)
            if weight.upper != P.inf:
                boundary_points.add(weight.upper + 1)
    for _, weight in nwa_data["final_states"].items():
        if not weight.empty:
            boundary_points.add(weight.lower)
            if weight.upper != P.inf:
                boundary_points.add(weight.upper + 1)

    sorted_points = sorted(list(boundary_points))

    atomic_intervals = []
    for i in range(len(sorted_points) - 1):
        lower, upper = sorted_points[i], sorted_points[i + 1]
        if lower < upper:
            atomic_intervals.append(P.closed(lower, upper - 1))

    if sorted_points and sorted_points[-1] != P.inf:
        last_point = sorted_points[-1]
        if any(w.upper == P.inf for _, _, _, w in nwa_data["transitions"]) or \
                any(w.upper == P.inf for _, w in nwa_data["final_states"].items()):
            atomic_intervals.append(P.closed(last_point, P.inf))

    print(f"Found {len(atomic_intervals)} atomic weight intervals to process.")
    return atomic_intervals


def build_and_determinize_nfa(
        interval: P.Interval,
        simplified_nwa_structure: dict
) -> VectorFst | None:
    """
    Builds a standard NFA from a pre-simplified structure for a given weight interval
    and determinizes it. Returns the determinized FST (DFA) or None on failure.
    """
    test_weight = interval.lower

    active_transitions = {
        (s, l, d) for s, l, d, w in simplified_nwa_structure["transitions"] if test_weight in w
    }
    active_finals = {
        s for s, w in simplified_nwa_structure["final_states"].items() if test_weight in w
    }

    if not active_transitions and not active_finals:
        return None

    try:
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in simplified_nwa_structure["states"]}

        if simplified_nwa_structure["start_state"] in state_map:
            fst.set_start(state_map[simplified_nwa_structure["start_state"]])
        else:
            return None

        for state_id in active_finals:
            if state_id in state_map:
                fst.set_final(state_map[state_id], 0.0)

        for source, label, dest in active_transitions:
            if source in state_map and dest in state_map:
                fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        fst = fst.connect()
        if fst.num_states() == 0:
            return None

        fst = fst.determinize()
        return fst
    except Exception:
        return None


def recompose_dwa(determinized_dfas: dict[P.Interval, VectorFst], original_start_state: int) -> dict:
    """
    Merges a dictionary of DFAs (keyed by their valid weight interval)
    into a single Deterministic Weighted Automaton (DWA).
    """
    print("\nRecomposing DFAs into a single DWA...")

    state_key_to_id = {}
    final_states = defaultdict(P.empty)
    transitions = defaultdict(P.empty)
    dfa_list = list(determinized_dfas.items())

    for dfa_idx, (_, dfa) in enumerate(dfa_list):
        for dfa_state_id in dfa.states():
            state_key = (dfa_idx, dfa_state_id)
            if state_key not in state_key_to_id:
                state_key_to_id[state_key] = len(state_key_to_id)

    pbar = tqdm(total=len(dfa_list), desc="Recomposing")
    for dfa_idx, (interval, dfa) in enumerate(dfa_list):
        for dfa_source_id in dfa.states():
            global_source_id = state_key_to_id[(dfa_idx, dfa_source_id)]
            if dfa.is_final(dfa_source_id):
                final_states[global_source_id] |= interval
            for tr in dfa.trs(dfa_source_id):
                dfa_dest_id = tr.next_state
                global_dest_id = state_key_to_id[(dfa_idx, dfa_dest_id)]
                label = tr.ilabel
                transitions[(global_source_id, label, global_dest_id)] |= interval
        pbar.update(1)
    pbar.close()

    new_start_state = None
    if dfa_list:
        first_dfa_idx = 0
        first_dfa = dfa_list[0][1]
        if first_dfa.start() is not None:
            new_start_state = state_key_to_id.get((first_dfa_idx, first_dfa.start()))

    return {
        "num_states": len(state_key_to_id),
        "start_state": new_start_state,
        "final_states": dict(final_states),
        "transitions": {
            (s, l, d): w for (s, l, d), w in transitions.items() if not w.empty
        }
    }


def print_dwa_stats(dwa_data: dict):
    """Prints high-level stats for the recomposed DWA."""
    num_states = dwa_data["num_states"]
    num_transitions = len(dwa_data["transitions"])
    num_final = len(dwa_data["final_states"])

    print("\n--- Recomposed DWA Statistics ---")
    print(f"Total States: {num_states}")
    print(f"Start State: {dwa_data['start_state']}")
    print(f"Final States: {num_final}")
    print(f"Total Transitions: {num_transitions}")

    if num_transitions > 0 or num_final > 0:
        weight_intervals = list(dwa_data["transitions"].values()) + list(dwa_data["final_states"].values())

        # DEFINITIVE FIX: Iterate directly over the interval object `w` to get its atomic components.
        num_atomic_intervals = sum(len(list(w)) for w in weight_intervals)
        print(f"Complexity: The weights are described by {num_atomic_intervals} distinct atomic intervals.")


# --- PASS FUNCTIONS ---

def create_subgraph_nwa(nwa_data: dict, states_to_keep: set) -> dict | None:
    """
    Creates a new NWA dictionary containing only the states and transitions
    within the 'states_to_keep' set.
    """
    if not states_to_keep:
        return None

    # Find a valid start state within the subset, preferring the original
    new_start_state = nwa_data["start_state"]
    if new_start_state not in states_to_keep:
        # If original start is gone, just pick one to make the FST valid
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
        "num_states": nwa_data["num_states"], # Keep original state IDs for mapping
        "start_state": new_start_state,
        "final_states": subgraph_finals,
        "transitions": subgraph_transitions,
    }

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
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in transitions:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        print("\nFST Statistics:")
        print_fst_stats(fst, "Initial FST")
        fst = fst.connect()
        print_fst_stats(fst, "After connecting")
        fst = fst.rm_epsilon()
        print_fst_stats(fst, "After removing epsilons")
        fst = fst.tr_unique()
        print_fst_stats(fst, "After tr_unique")
        fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
        print_fst_stats(fst, "After minimizing")
        fst = fst.determinize(config=DeterminizeConfig(det_type=DeterminizeType.DETERMINIZE_FUNCTIONAL))
        print_fst_stats(fst, "After determinizing")
        print("\nRESULT: ✅ Determinization finished successfully.")
    except Exception as e:
        print(f"RESULT: ❌ Determinization failed with an error: {e}")


def run_determinize_acyclic_pass(args):
    """Executes the 'determinize' pass after breaking all cycles."""
    print("--- Running Acyclic Determinization Pass ---")
    nwa = load_nwa_data(args.filepath)

    print("\nBuilding graph to find cycles...")
    G = nx.DiGraph()
    G.add_nodes_from(range(nwa["num_states"]))
    G.add_edges_from([(s, d) for s, l, d, w in nwa["transitions"]])

    all_sccs = list(nx.strongly_connected_components(G))
    transitions_to_remove = set()

    # A map for quick lookup
    source_to_trans = defaultdict(list)
    for t in nwa["transitions"]:
        source_to_trans[t[0]].append(t)

    for scc in all_sccs:
        scc_nodes = set(scc)
        if len(scc_nodes) > 1:
            # Non-trivial SCC, find an internal edge to break the cycle
            edge_to_break = None
            for source_node in scc_nodes:
                for s, l, d, w in source_to_trans.get(source_node, []):
                    if d in scc_nodes:
                        edge_to_break = (s, l, d, w)
                        break
                if edge_to_break:
                    break
            if edge_to_break:
                print(f"Breaking cycle in SCC of size {len(scc_nodes)} by removing transition: {edge_to_break[:3]}")
                transitions_to_remove.add(edge_to_break)
        else:
            # Trivial SCC, check for self-loop
            node = list(scc_nodes)[0]
            for s, l, d, w in source_to_trans.get(node, []):
                if d == node:  # self-loop
                    self_loop_edge = (s, l, d, w)
                    print(f"Breaking self-loop on state {node} by removing transition: {self_loop_edge[:3]}")
                    transitions_to_remove.add(self_loop_edge)

    acyclic_transitions = nwa["transitions"] - transitions_to_remove
    print(f"\nRemoved {len(transitions_to_remove)} transitions to make the graph acyclic.")
    print(f"Proceeding with {len(acyclic_transitions)} transitions.")

    print("\nAttempting to determinize the acyclic NWA...")
    try:
        transitions = prune_to_final_state_transitions(acyclic_transitions, nwa["final_states"].keys())
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(nwa["num_states"])}
        fst.set_start(state_map[nwa["start_state"]])
        for state_id, weight in nwa["final_states"].items():
            fst.set_final(state_map[state_id], 0.0)
        for source, label, dest, weight in transitions:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        print("\nFST Statistics:")
        print_fst_stats(fst, "Initial FST")
        fst = fst.connect()
        print_fst_stats(fst, "After connecting")
        fst = fst.rm_epsilon()
        print_fst_stats(fst, "After removing epsilons")
        fst = fst.tr_unique()
        print_fst_stats(fst, "After tr_unique")
        fst = fst.minimize(config=MinimizeConfig(allow_nondet=True))
        print_fst_stats(fst, "After minimizing")
        fst = fst.determinize(config=DeterminizeConfig(det_type=DeterminizeType.DETERMINIZE_FUNCTIONAL))
        print_fst_stats(fst, "After determinizing")
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


def run_prune_pass(args):
    """Executes the 'prune' pass by finding a maximal subset of transitions that does NOT cause a timeout."""
    print(f"--- Running Pruning Pass (Timeout: {args.timeout}s) ---")
    nwa = load_nwa_data(args.filepath)
    all_transitions = nwa["transitions"]

    print("\n--- Establishing Baseline Behavior ---")
    if not time_determinization_with_timeout(nwa["num_states"], nwa["start_state"], nwa["final_states"], all_transitions, args.timeout):
        print(f"Baseline determinization finished within {args.timeout}s. Pruning is not needed.")
        return
    else:
        print(f"Baseline determinization timed out after {args.timeout}s (as expected).")

    print("\n--- Finding maximal non-problematic subset (additive approach) ---")
    good_transitions = set()
    candidates = list(all_transitions)
    random.shuffle(candidates)
    chunk_sizes = sorted([10000, 1000, 100, 10, 1], reverse=True)

    for chunk_size in chunk_sizes:
        if not candidates: break
        print(f"\n--- PASS with Chunk Size: {chunk_size} ---")
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        next_round_candidates = []
        for i, chunk in enumerate(chunks):
            candidate_set = good_transitions | set(chunk)
            if not time_determinization_with_timeout(
                    nwa["num_states"],
                    nwa["start_state"],
                    nwa["final_states"],
                    candidate_set,
                    args.timeout
                    ):
                good_transitions.update(chunk)
                print(f"[{i + 1}/{len(chunks)}] ✅ Chunk added. Good set size: {len(good_transitions)}")
            else:
                next_round_candidates.extend(chunk)
                print(f"[{i + 1}/{len(chunks)}] ❌ Chunk is problematic. Deferring {len(chunk)} transitions.")
        candidates = next_round_candidates
        random.shuffle(candidates)

    problematic_transitions = set(candidates)
    print(f"\n--- Pruning Complete ---")
    print(f"Original transition count: {len(all_transitions)}")
    print(f"Maximal good subset size: {len(good_transitions)}")
    print(f"Found a core of {len(problematic_transitions)} problematic transitions.")

    if not problematic_transitions:
        print("\nWarning: Could not isolate any problematic transitions.")
        return

    print("\n--- Analysis of Final Core Graph ---")
    G_final = nx.DiGraph()
    G_final.add_nodes_from(range(nwa["num_states"]))
    G_final.add_edges_from([(s, d) for s, l, d, w in problematic_transitions])
    pruned_nwa_data = {"num_states": nwa["num_states"], "start_state": nwa["start_state"], "final_states": nwa["final_states"],
                       "transitions": problematic_transitions}
    print_graph_stats(G_final, pruned_nwa_data)
    print_scc_analysis(G_final, problematic_transitions)


def run_chunked_determinize_pass(args):
    """Executes the 'chunked-determinize' pass."""
    print(f"--- Running Chunked Determinization Pass (Chunks: {args.chunks}) ---")
    nwa = load_nwa_data(args.filepath)
    print("\nPruning transitions that cannot lead to a final state...")
    transitions_pruned = prune_to_final_state_transitions(nwa["transitions"], nwa["final_states"].keys())
    print(f"Pruned {len(nwa['transitions']) - len(transitions_pruned)} transitions.")
    all_transitions = list(transitions_pruned)
    random.shuffle(all_transitions)
    if not all_transitions:
        print("No transitions left after pruning. Nothing to do.")
        return

    chunk_size = (len(all_transitions) + args.chunks - 1) // args.chunks
    transition_chunks = [all_transitions[i:i + chunk_size] for i in range(0, len(all_transitions), chunk_size)]
    print(f"\nPartitioned {len(all_transitions)} transitions into {len(transition_chunks)} chunks of up to {chunk_size} transitions each.")

    optimized_chunks = []
    for i, chunk in enumerate(transition_chunks):
        print(f"\n--- Processing Chunk {i + 1}/{len(transition_chunks)} ---")
        if not chunk: continue
        try:
            fst = VectorFst()
            state_map = {j: fst.add_state() for j in range(nwa["num_states"])}
            fst.set_start(state_map[nwa["start_state"]])
            for state_id, weight in nwa["final_states"].items():
                fst.set_final(state_map[state_id], 0.0)
            for source, label, dest, weight in chunk:
                fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))
            print_fst_stats(fst, f"Chunk {i + 1} Initial FST")
            fst = fst.rm_epsilon().connect().tr_unique().minimize(config=MinimizeConfig(allow_nondet=True)).determinize()
            print_fst_stats(fst, f"Chunk {i + 1} Optimized FST")
            optimized_chunks.append(fst)
        except Exception as e:
            print(f"❌ Error processing chunk {i + 1}: {e}. Skipping this chunk.")

    if not optimized_chunks:
        print("\nRESULT: ❌ No chunks were successfully optimized. Halting.")
        return

    print("\n--- Combining Optimized Chunks ---")
    final_fst = optimized_chunks[0]
    print_fst_stats(final_fst, "Combined FST (1 chunk)")
    for i, next_fst in enumerate(optimized_chunks[1:], 2):
        final_fst = final_fst | next_fst
        print_fst_stats(final_fst, f"Combined FST ({i} chunks)")

    print("\n--- Final Optimization Pass on Combined FST ---")
    try:
        print_fst_stats(final_fst, "Combined FST before final optimization")
        final_fst = final_fst.rm_epsilon().connect().tr_unique().minimize(config=MinimizeConfig(allow_nondet=True)).determinize()
        print_fst_stats(final_fst, "After final optimization")
        print("\nRESULT: ✅ Chunked determinization finished successfully.")
    except Exception as e:
        print(f"\nRESULT: ❌ Final optimization failed with an error: {e}")


def run_recompose_pass(args):
    """Executes the full decomposition, determinization, and recomposition pass."""
    print("--- Running Recomposition Pass (Optimized Workflow) ---")
    nwa_data = load_nwa_data(args.filepath)

    print("\nStep 1: Building and simplifying the global NWA structure...")
    global_fst = VectorFst()
    state_map = {i: global_fst.add_state() for i in range(nwa_data["num_states"])}
    global_fst.set_start(state_map[nwa_data["start_state"]])
    for state_id in nwa_data["final_states"]:
        global_fst.set_final(state_map[state_id], 0.0)
    for s, l, d, _ in nwa_data["transitions"]:
        global_fst.add_tr(state_map[s], Tr(l, l, 0.0, state_map[d]))

    print_fst_stats(global_fst, "Initial global FST")

    global_fst = global_fst.rm_epsilon()
    print_fst_stats(global_fst, "After rm_epsilon")
    global_fst = global_fst.connect()
    print_fst_stats(global_fst, "After connect")
    global_fst = global_fst.tr_unique()
    print_fst_stats(global_fst, "After tr_unique")
    global_fst = global_fst.minimize(config=MinimizeConfig(allow_nondet=True))
    print_fst_stats(global_fst, "After minimize (still an NFA)")

    print("\nStep 2: Extracting simplified structure and re-applying weights...")
    simplified_states = list(global_fst.states())
    if not simplified_states:
        print("RESULT: ❌ The automaton is empty after initial simplification. Halting.")
        return

    surviving_states = set(simplified_states)
    filtered_original_transitions = {
        (s, l, d, w) for s, l, d, w in nwa_data["transitions"]
        if s in surviving_states and d in surviving_states
    }

    simplified_nwa_structure = {
        "states": surviving_states,
        "start_state": global_fst.start(),
        "final_states": {
            s: w for s, w in nwa_data["final_states"].items() if s in surviving_states
        },
        "transitions": filtered_original_transitions
    }

    print(f"Simplified structure has {len(surviving_states)} states and {len(filtered_original_transitions)} transitions.")

    print("\nStep 3: Decomposing simplified structure into atomic intervals...")
    atomic_intervals = get_atomic_intervals(simplified_nwa_structure)
    if not atomic_intervals:
        print("No weight intervals found. Nothing to process.")
        return

    print("\nStep 4: Determinizing an NFA for each interval...")
    determinized_dfas = {}
    pbar = tqdm(atomic_intervals, desc="Determinizing NFAs")
    for interval in pbar:
        dfa = build_and_determinize_nfa(interval, simplified_nwa_structure)
        if dfa and dfa.num_states() > 0:
            determinized_dfas[interval] = dfa
    pbar.close()

    successful_count = len(determinized_dfas)
    print(f"\nSuccessfully determinized {successful_count} / {len(atomic_intervals)} NFAs.")
    if successful_count == 0:
        print("RESULT: ❌ No NFAs could be determinized. Halting.")
        return

    print("\nStep 5: Recomposing DFAs into a single DWA...")
    final_dwa = recompose_dwa(determinized_dfas, simplified_nwa_structure["start_state"])

    print_dwa_stats(final_dwa)
    print("\nRESULT: ✅ Recomposition finished successfully.")

    output_path = "dwa_recomposed.json"
    print(f"Saving result to {output_path}...")
    serializable_dwa = {
        "num_states": final_dwa["num_states"],
        "start_state": final_dwa["start_state"],
        "final_states": {s: str(w) for s, w in final_dwa["final_states"].items()},
        "transitions": [(s, l, d, str(w)) for (s, l, d), w in final_dwa["transitions"].items()]
    }
    with open(output_path, 'w') as f:
        json.dump(serializable_dwa, f, indent=2)
    print("Save complete.")


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

    # We'll keep track of the smallest known failing set of states
    smallest_failing_set = set(problematic_states)

    iteration = 0
    while len(problematic_states) > 1:
        iteration += 1
        print(f"\n--- Bisection Iteration {iteration}: {len(problematic_states)} states remaining ---")

        # Split the current set of problematic states
        mid = len(problematic_states) // 2
        s1 = set(problematic_states[:mid])
        s2 = set(problematic_states[mid:])

        print(f"Splitting into two sets of sizes: {len(s1)} and {len(s2)}")

        # Create and test the two subgraphs
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
            # This is the complex case: the interaction is the problem, or both halves are problematic.
            # We can't simply discard one half.
            print("-> Interaction detected or both halves are problematic. Cannot reduce further with this split.")
            # A simple strategy is to just stop here. A more advanced one would be to try a different split.
            # For now, we'll break and report the last known smallest failing set.
            break

    print("\n--- Bisection Complete ---")
    print(f"Found a minimal problematic set of {len(smallest_failing_set)} states:")
    print(sorted(list(smallest_failing_set)))

    print("\n--- Analysis of Minimal Failing Subgraph ---")
    minimal_nwa_data = create_subgraph_nwa(nwa_data, smallest_failing_set)
    G_minimal = nx.DiGraph()
    G_minimal.add_nodes_from(smallest_failing_set)
    G_minimal.add_edges_from([(s, d) for s, l, d, w in minimal_nwa_data["transitions"]])

    print_graph_stats(G_minimal, minimal_nwa_data)
    print_scc_analysis(G_minimal, minimal_nwa_data["transitions"])

    # Save the minimal failing example to a file for inspection
    output_path = "minimal_failing_nwa.json"
    print(f"\nSaving minimal failing NWA to {output_path}...")

    # We need to remap state IDs to be contiguous from 0 for the output file
    state_remapping = {old_id: new_id for new_id, old_id in enumerate(sorted(list(smallest_failing_set)))}

    remapped_states = [nwa_data["raw_data"]["states"][i] for i in sorted(list(smallest_failing_set))]
    for state_data in remapped_states:
        if 'epsilons' in state_data:
            state_data['epsilons'] = [[state_remapping[t[0]], t[1]] for t in state_data['epsilons'] if t[0] in state_remapping]
        if 'transitions' in state_data:
            new_transitions = {}
            for label, targets in state_data['transitions'].items():
                new_targets = [[state_remapping[t[0]], t[1]] for t in targets if t[0] in state_remapping]
                if new_targets:
                    new_transitions[label] = new_targets
            state_data['transitions'] = new_transitions

    minimal_json = {
        "states": remapped_states,
        "body": {
            "start_state": state_remapping[minimal_nwa_data["start_state"]]
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

    # --- Recompose Pass ---
    parser_recompose = subparsers.add_parser(
        "recompose",
        help="Determinize an interval-weighted NWA by decomposing it into multiple standard NFAs,\n"
             "determinizing each, and recomposing the results into a DWA."
    )
    parser_recompose.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_recompose.set_defaults(func=run_recompose_pass)

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

    # --- Acyclic Determinize Pass ---
    parser_det_acyclic = subparsers.add_parser(
        "determinize-acyclic",
        help="Breaks all cycles (SCCs) in the NWA and then attempts to determinize it."
    )
    parser_det_acyclic.add_argument("filepath", help="Path to the nwa_dump.json file.")
    parser_det_acyclic.set_defaults(func=run_determinize_acyclic_pass)

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

    args = parser.parse_args()
    args.func(args)