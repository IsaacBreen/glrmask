import json
import sys
import glob
import os
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple, Set, Union

# We use automata-lib for NFA/DFA operations
try:
    from automata.fa.nfa import NFA
    from automata.fa.dfa import DFA
except ImportError:
    print("Error: 'automata-lib' not found. Please install it using 'pip install automata-lib'", file=sys.stderr)
    sys.exit(1)

# We use NetworkX for graph analysis, specifically for finding strongly connected components.
try:
    import networkx as nx
except ImportError:
    print("Error: 'networkx' not found. Please install it using 'pip install networkx'", file=sys.stderr)
    sys.exit(1)

# A dedicated symbol for default transitions when converting to rustfst.
# We keep this definition to correctly parse the NWA JSON structure.
DEFAULT_TRANSITION_SYMBOL = 0xFFFE

# --- NWA Data Structures (Kept for JSON parsing) ---

# Type aliases for clarity based on Rust types
Weight = Any
NWAStateID = int
Symbol = int  # i16 in Rust


@dataclass
class NWAState:
    """Represents a single state in an NWA."""
    final_weight: Optional[Weight]
    # In JSON, the keys of the transitions map are strings.
    transitions: Dict[str, List[Tuple[NWAStateID, Weight]]]
    epsilons: List[Tuple[NWAStateID, Weight]]


@dataclass
class NWABody:
    """Represents the body of an NWA, containing the start state."""
    start_state: NWAStateID


@dataclass
class NWA:
    """
    Represents a Non-deterministic Weighted Automaton (NWA), loaded from a JSON dump.
    """
    body: NWABody
    states: List['NWAState']

    @staticmethod
    def from_dict(data: dict) -> 'NWA':
        """Creates an NWA instance from a dictionary (parsed from JSON)."""
        body = NWABody(**data['body'])
        states_data = data['states']
        states = []
        for s_data in states_data:
            state = NWAState(
                final_weight=s_data.get('final_weight'),
                transitions=s_data.get('transitions', {}),
                epsilons=[tuple(e) for e in s_data.get('epsilons', [])]
            )
            states.append(state)
        return NWA(body=body, states=states)

    def num_states(self) -> int:
        return len(self.states)


# --- Analysis Functions (Simplified) ---

def get_nwa_stats(nwa: NWA) -> dict:
    """
    Computes various statistics about the NWA.
    """
    stats = {
        'num_states': nwa.num_states(),
        'start_state': nwa.body.start_state,
        'num_final_states': 0,
        'num_epsilon_transitions': 0,
        'num_labeled_transitions': 0,
        'num_default_transitions': 0,
    }

    for i, state in enumerate(nwa.states):
        if state.final_weight is not None:
            stats['num_final_states'] += 1
        stats['num_epsilon_transitions'] += len(state.epsilons)
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            if label == 0:  # Count transitions with label 0 as epsilons
                stats['num_epsilon_transitions'] += len(targets)
            elif label == DEFAULT_TRANSITION_SYMBOL:
                stats['num_default_transitions'] += len(targets)
            else:
                stats['num_labeled_transitions'] += len(targets)

    total_transitions = stats['num_epsilon_transitions'] + stats['num_labeled_transitions'] + stats['num_default_transitions']
    stats['total_transitions'] = total_transitions
    return stats


def print_nwa_stats(stats: dict):
    """
    Prints a summary of the NWA statistics to the console.
    """
    print("--- NWA Statistics ---")
    print(f"Number of states: {stats['num_states']}")
    print(f"Start state: {stats['start_state']}")
    print(f"Number of final states: {stats['num_final_states']}")
    print("\n--- Transitions ---")
    print(f"Total epsilon transitions: {stats['num_epsilon_transitions']}")
    print(f"Total labeled transitions (explicit): {stats['num_labeled_transitions']}")
    print(f"Total default transitions: {stats['num_default_transitions']}")
    total_transitions = stats['total_transitions']
    print(f"Total transitions: {total_transitions}")
    if stats['num_states'] > 0:
        print(f"Average outgoing degree: {total_transitions / stats['num_states']:.2f}")


def load_nwa(filepath: str) -> NWA:
    """
    Loads an NWA from a JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA.from_dict(data)


# --- NFA Conversion and Optimization ---

def convert_nwa_to_nfa_automata(nwa: NWA) -> NFA:
    """
    Converts the NWA structure into a standard NFA (ignoring weights).
    """
    states = set(range(nwa.num_states()))
    input_symbols = set()
    initial_state = nwa.body.start_state
    final_states = {i for i, state in enumerate(nwa.states) if state.final_weight is not None}
    transitions: Dict[int, Dict[Union[int, str], Set[int]]] = defaultdict(lambda: defaultdict(set))

    for i, state in enumerate(nwa.states):
        # Epsilon transitions from the dedicated list
        if state.epsilons:
            eps_targets = {t[0] for t in state.epsilons}
            transitions[i][''].update(eps_targets)

        # Labeled transitions from the map
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            target_ids = {t[0] for t in targets}

            # --- FIX IS HERE ---
            # Treat label 0 as an epsilon transition for automata-lib
            if label == 0:
                transitions[i][''].update(target_ids)
                continue
            # --- END FIX ---

            if label == DEFAULT_TRANSITION_SYMBOL:
                # Ignoring default transitions for simplicity and speed.
                continue

            transitions[i][label].update(target_ids)
            input_symbols.add(label)

    return NFA(
        states=states,
        input_symbols=input_symbols,
        transitions={k: dict(v) for k, v in transitions.items()},
        initial_state=initial_state,
        final_states=final_states
    )


def prune_unreachable_states(nfa: NFA) -> NFA:
    """Removes states not reachable from the initial state."""
    reachable_states = {nfa.initial_state}
    queue = [nfa.initial_state]

    head = 0
    while head < len(queue):
        current_state = queue[head]
        head += 1
        if current_state in nfa.transitions:
            for symbol, targets in nfa.transitions[current_state].items():
                for target_state in targets:
                    if target_state not in reachable_states:
                        reachable_states.add(target_state)
                        queue.append(target_state)

    # Filter transitions to only include those from reachable states to reachable states
    filtered_transitions = {}
    for state, trans in nfa.transitions.items():
        if state in reachable_states:
            filtered_transitions[state] = {
                symbol: {target for target in targets if target in reachable_states}
                for symbol, targets in trans.items()
            }

    return NFA(
        states=reachable_states,
        input_symbols=nfa.input_symbols,
        transitions=filtered_transitions,
        initial_state=nfa.initial_state,
        final_states=nfa.final_states & reachable_states
    )


def remove_epsilon_transitions(nfa: NFA) -> NFA:
    """Creates an equivalent NFA with no epsilon transitions."""
    epsilon_closures = {}
    for state in nfa.states:
        closure = {state}
        queue = [state]
        head = 0
        while head < len(queue):
            q_state = queue[head]
            head += 1
            epsilon_targets = nfa.transitions.get(q_state, {}).get('', set())
            for target in epsilon_targets:
                if target not in closure:
                    closure.add(target)
                    queue.append(target)
        epsilon_closures[state] = closure

    new_transitions = defaultdict(lambda: defaultdict(set))
    for u_state in nfa.states:
        for v_state in epsilon_closures[u_state]:
            for symbol, targets in nfa.transitions.get(v_state, {}).items():
                if symbol == '':
                    continue
                for w_state in targets:
                    new_transitions[u_state][symbol].update(epsilon_closures[w_state])

    new_final_states = {
        state for state in nfa.states
        if not nfa.final_states.isdisjoint(epsilon_closures.get(state, set()))
    }

    return NFA(
        states=nfa.states,
        input_symbols=nfa.input_symbols,
        transitions={k: dict(v) for k, v in new_transitions.items()},
        initial_state=nfa.initial_state,
        final_states=new_final_states
    )


# --- Main Execution ---

if __name__ == "__main__":
    filepath = None
    if len(sys.argv) > 2:
        print(f"Usage: python {sys.argv[0]} [<path_to_nwa_dump.json>]")
        sys.exit(1)

    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else:
        try:
            search_paths = ['./nwa_dump_*.json', '../nwa_dump_*.json']
            dump_files = [f for path in search_paths for f in glob.glob(path)]
            if not dump_files:
                print("No NWA dump file provided and no 'nwa_dump_*.json' files found.")
                sys.exit(1)
            filepath = max(dump_files, key=os.path.getmtime)
            print(f"No path provided. Using most recent dump file: {filepath}")
        except Exception as e:
            print(f"Error finding dump file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        nwa = load_nwa(filepath)
        stats = get_nwa_stats(nwa)
        print_nwa_stats(stats)

        print("\n--- Converting NWA to standard NFA (ignoring weights) ---")
        nfa = convert_nwa_to_nfa_automata(nwa)
        print(f"Initial NFA States: {len(nfa.states)}")
        nfa_arcs = sum(len(t) for s in nfa.transitions.values() for t in s.values())
        print(f"Initial NFA Transitions: {nfa_arcs}")

        # --- ACCELERATION STRATEGY: PRE-PROCESSING ---
        print("\n--- NFA Pre-processing for Acceleration ---")

        # Step 1: Prune unreachable states
        start_prune_time = time.monotonic()
        nfa_pruned = prune_unreachable_states(nfa)
        end_prune_time = time.monotonic()
        print(f"1. Pruning unreachable states completed in {end_prune_time - start_prune_time:.4f} seconds.")
        print(f"   - States reduced from {len(nfa.states)} to {len(nfa_pruned.states)}")

        # Step 2: Remove all epsilon transitions
        start_eps_time = time.monotonic()
        nfa_optimized = remove_epsilon_transitions(nfa_pruned)
        end_eps_time = time.monotonic()
        print(f"2. Epsilon transition removal completed in {end_eps_time - start_eps_time:.4f} seconds.")

        opt_nfa_arcs = sum(len(t) for s in nfa_optimized.transitions.values() for t in s.values())
        print(f"   - Optimized NFA has {len(nfa_optimized.states)} states and {opt_nfa_arcs} transitions (all non-epsilon).")

        # --- TIMED DETERMINIZATION ---
        print("\n--- Determinizing OPTIMIZED NFA using automata-lib ---")

        start_det_time = time.monotonic()
        dfa = DFA.from_nfa(nfa_optimized)  # Determinize the simplified NFA
        end_det_time = time.monotonic()

        duration = end_det_time - start_det_time
        print(f"Determinization successful in {duration:.4f} seconds.")

        print("\n--- DFA Summary ---")
        print(f"DFA States: {len(dfa.states)}")
        dfa_arcs = sum(len(symbol_map) for symbol_map in dfa.transitions.values())
        print(f"DFA Transitions: {dfa_arcs}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)