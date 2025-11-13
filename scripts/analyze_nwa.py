# scripts/analyze_nwa.py
import json
import sys
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple, Set, Union

# We use automata-lib for NFA/DFA operations
try:
    from automata.fa.nfa import NFA
    from automata.fa.dfa import DFA
except ImportError:
    print("Error: 'automata-lib' not found. Please install it using 'pip install automata-lib'", file=sys.stderr)
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
        # The `NWAStates` newtype in Rust serializes transparently as its inner Vec,
        # so `data['states']` is the list of state objects.
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

def analyze_weight(weight: Weight, weight_stats: dict):
    """Updates weight statistics based on a given weight object."""
    if weight == "ALL":
        weight_stats['all_count'] += 1
    elif isinstance(weight, list) and not weight:
        weight_stats['empty_count'] += 1
    else:
        weight_stats['complex_count'] += 1


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
        'weight_stats': {
            'all_count': 0,
            'empty_count': 0,
            'complex_count': 0,
        }
    }

    for i, state in enumerate(nwa.states):
        if state.final_weight is not None:
            stats['num_final_states'] += 1
            analyze_weight(state.final_weight, stats['weight_stats'])

        # Epsilon transitions
        eps_count = len(state.epsilons)
        stats['num_epsilon_transitions'] += eps_count
        for target, weight in state.epsilons:
            analyze_weight(weight, stats['weight_stats'])

        # Labeled transitions
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            if label == DEFAULT_TRANSITION_SYMBOL:
                stats['num_default_transitions'] += len(targets)
            else:
                stats['num_labeled_transitions'] += len(targets)

            for target, weight in targets:
                analyze_weight(weight, stats['weight_stats'])

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

    print("\n--- Weight Statistics (Ignored for NFA conversion) ---")
    print(f"  'ALL' weights: {stats['weight_stats']['all_count']}")
    print(f"  Empty weights: {stats['weight_stats']['empty_count']}")
    print(f"  Complex weights: {stats['weight_stats']['complex_count']}")


def load_nwa(filepath: str) -> NWA:
    """
    Loads an NWA from a JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA.from_dict(data)


# --- NFA Conversion using automata-lib ---

def convert_nwa_to_nfa_automata(nwa: NWA) -> NFA:
    """
    Converts the NWA structure into a standard NFA (ignoring weights).
    Default transitions (0xFFFE) are expanded to cover all non-explicitly
    handled symbols.
    """

    # 1. Collect all symbols, excluding 0 which is reserved for epsilon.
    all_symbols: Set[int] = set()
    for state in nwa.states:
        for label_str in state.transitions:
            label = int(label_str)
            if label != 0 and label != DEFAULT_TRANSITION_SYMBOL:
                all_symbols.add(label)

    # 2. Prepare NFA components
    states = set(range(nwa.num_states()))
    input_symbols = all_symbols
    initial_state = nwa.body.start_state

    # Final states are any state with a non-None final_weight
    final_states = {i for i, state in enumerate(nwa.states) if state.final_weight is not None}

    # Transitions: {state: {symbol: {target_states}}}
    # The symbol can be an int or the empty string '' for epsilon
    transitions: Dict[int, Dict[Union[int, str], Set[int]]] = defaultdict(lambda: defaultdict(set))

    # 3. Process transitions
    for i, state in enumerate(nwa.states):

        explicit_symbols: Set[int] = set()
        default_targets: Optional[Set[int]] = None

        # First pass: Collect explicit transitions and default targets
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            target_ids = {t[0] for t in targets}  # Ignore weights

            if label == DEFAULT_TRANSITION_SYMBOL:
                default_targets = target_ids
                continue

            if label == 0:
                # This is an epsilon transition, map to empty string ''
                transitions[i][''].update(target_ids)
                continue

            # Standard labeled transition
            transitions[i][label].update(target_ids)
            explicit_symbols.add(label)

        # Epsilon transitions from the dedicated `epsilons` list
        if state.epsilons:
            eps_targets = {t[0] for t in state.epsilons}
            transitions[i][''].update(eps_targets)

        # Second pass: Expand default transitions
        if default_targets:
            default_symbols = input_symbols - explicit_symbols
            for symbol in default_symbols:
                transitions[i][symbol].update(default_targets)

    # Convert transitions to the format required by automata-lib (no defaultdict)
    nfa_transitions = {
        state: {
            symbol: targets
            for symbol, targets in symbol_map.items()
        }
        for state, symbol_map in transitions.items()
    }

    return NFA(
        states=states,
        input_symbols=input_symbols,
        transitions=nfa_transitions,
        initial_state=initial_state,
        final_states=final_states
    )


# --- Main Execution ---

if __name__ == "__main__":
    filepath = None
    if len(sys.argv) > 2:
        print(f"Usage: python {sys.argv[0]} [<path_to_nwa_dump.json>]")
        sys.exit(1)

    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else:  # len(sys.argv) == 1
        try:
            search_paths = ['./nwa_dump_*.json', '../nwa_dump_*.json']
            dump_files = []
            for path in search_paths:
                dump_files.extend(glob.glob(path))

            if not dump_files:
                print("No NWA dump file provided and no 'nwa_dump_*.json' files found in current or parent directory.")
                sys.exit(1)

            filepath = max(dump_files, key=os.path.getmtime)
            print(f"No path provided. Using most recent dump file: {filepath}")
        except Exception as e:
            print(f"Error finding most recent dump file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        nwa = load_nwa(filepath)
        stats = get_nwa_stats(nwa)
        print_nwa_stats(stats)

        print("\n--- Converting NWA to standard NFA (ignoring weights) ---")
        nfa = convert_nwa_to_nfa_automata(nwa)

        print("NFA created successfully.")
        print(f"NFA States: {len(nfa.states)}")
        print(f"NFA Input Symbols: {len(nfa.input_symbols)}")

        # Calculate NFA transitions count
        nfa_arcs = sum(len(targets) for symbol_map in nfa.transitions.values() for targets in symbol_map.values())
        print(f"NFA Transitions (after default expansion): {nfa_arcs}")

        print("\n--- Determinizing NFA using automata-lib ---")

        # Correct way to determinize: use the DFA class method
        dfa = DFA.from_nfa(nfa)

        print("Determinization successful.")
        print("\n--- DFA Summary ---")
        print(f"DFA States: {len(dfa.states)}")

        # Calculate DFA transitions count
        dfa_arcs = sum(len(targets) for symbol_map in dfa.transitions.values() for targets in symbol_map.values())
        print(f"DFA Transitions: {dfa_arcs}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)