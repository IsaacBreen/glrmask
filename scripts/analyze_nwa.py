# scripts/analyze_nwa.py
import json
import sys
import os
import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple

# Try to import rustfst, but don't fail if it's not there.
try:
    from rustfst import VectorFst, Tr
    from rustfst.weight import weight_one
    RUSTFST_AVAILABLE = True
except ImportError:
    RUSTFST_AVAILABLE = False
    VectorFst = None # for type hinting


# Type aliases for clarity based on Rust types
Weight = Any
NWAStateID = int
Symbol = int  # i16 in Rust

@dataclass
class NWADefaultTransition:
    """Represents a default transition in an NWA."""
    target: NWAStateID
    weight: Weight
    exceptions: List[Symbol]

@dataclass
class NWAState:
    """Represents a single state in an NWA."""
    final_weight: Optional[Weight]
    # In JSON, the keys of the transitions map are strings.
    transitions: Dict[str, List[Tuple[NWAStateID, Weight]]]
    epsilons: List[Tuple[NWAStateID, Weight]]
    default: List[NWADefaultTransition]

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
            default_trans = [NWADefaultTransition(**d) for d in s_data.get('default', [])]
            state = NWAState(
                final_weight=s_data.get('final_weight'),
                transitions=s_data.get('transitions', {}),
                epsilons=[tuple(e) for e in s_data.get('epsilons', [])],
                default=default_trans
            )
            states.append(state)
        return NWA(body=body, states=states)

    def num_states(self) -> int:
        return len(self.states)

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
        'labeled_transitions_per_state': defaultdict(int),
        'epsilon_transitions_per_state': defaultdict(int),
        'default_transitions_per_state': defaultdict(int),
        'outgoing_degree_per_state': defaultdict(int),
        'incoming_degree_per_state': defaultdict(int),
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
        stats['epsilon_transitions_per_state'][i] = eps_count
        stats['outgoing_degree_per_state'][i] += eps_count
        for target, weight in state.epsilons:
            stats['incoming_degree_per_state'][target] += 1
            analyze_weight(weight, stats['weight_stats'])

        # Labeled transitions
        labeled_count = 0
        for _label, targets in state.transitions.items():
            labeled_count += len(targets)
            for target, weight in targets:
                stats['incoming_degree_per_state'][target] += 1
                analyze_weight(weight, stats['weight_stats'])
        stats['num_labeled_transitions'] += labeled_count
        stats['labeled_transitions_per_state'][i] = labeled_count
        stats['outgoing_degree_per_state'][i] += labeled_count

        # Default transitions
        default_count = len(state.default)
        stats['num_default_transitions'] += default_count
        stats['default_transitions_per_state'][i] = default_count
        stats['outgoing_degree_per_state'][i] += default_count
        for default_trans in state.default:
            target = default_trans.target
            weight = default_trans.weight
            stats['incoming_degree_per_state'][target] += 1
            analyze_weight(weight, stats['weight_stats'])

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
    print(f"Total labeled transitions: {stats['num_labeled_transitions']}")
    print(f"Total default transitions: {stats['num_default_transitions']}")
    
    total_transitions = stats['num_epsilon_transitions'] + stats['num_labeled_transitions'] + stats['num_default_transitions']
    print(f"Total transitions: {total_transitions}")
    if stats['num_states'] > 0:
        print(f"Average outgoing degree: {total_transitions / stats['num_states']:.2f}")

    print("\n--- Weight Statistics ---")
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

def create_rustfst_from_nwa(nwa: NWA) -> Optional['VectorFst']:
    """
    Creates a rustfst.VectorFst from an NWA object.

    Note: This is a partial conversion.
    - Weights (SimpleBitset) are not converted; a default weight is used.
    - Default transitions are ignored.
    """
    if not RUSTFST_AVAILABLE:
        print("rustfst library not found. Cannot create FST.", file=sys.stderr)
        return None

    print("\n--- Creating rustfst.VectorFst from NWA ---")
    print("WARNING: This is a partial conversion. Weights and default transitions are not fully supported.")

    fst = VectorFst()
    state_map = {}  # NWAStateID -> FST state ID

    # Create states
    for i in range(nwa.num_states()):
        state_map[i] = fst.add_state()

    # Set start state
    if nwa.body.start_state < nwa.num_states():
        fst.set_start(state_map[nwa.body.start_state])

    # Add transitions and final states
    for i, state in enumerate(nwa.states):
        from_id = state_map[i]

        # Final state
        if state.final_weight is not None:
            # Using default weight as we can't represent SimpleBitset
            fst.set_final(from_id, weight_one())

        # Epsilon transitions
        for target, _weight in state.epsilons:
            to_id = state_map[target]
            # Using default weight
            fst.add_tr(from_id, Tr(0, 0, weight_one(), to_id))

        # Labeled transitions
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            for target, _weight in targets:
                to_id = state_map[target]
                # Using default weight
                fst.add_tr(from_id, Tr(label, label, weight_one(), to_id))
        
        # Default transitions are ignored
        if state.default:
            print(f"WARNING: Ignoring {len(state.default)} default transition(s) from state {i}", file=sys.stderr)

    print("FST created successfully (partially).")
    return fst

if __name__ == "__main__":
    filepath = None
    if len(sys.argv) > 2:
        print(f"Usage: python {sys.argv[0]} [<path_to_nwa_dump.json>]")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else: # len(sys.argv) == 1
        try:
            # Search for dump files in the current directory and parent directory
            # to handle being run from project root or from scripts/
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

        fst = create_rustfst_from_nwa(nwa)
        if fst:
            print("\n--- rustfst.VectorFst Summary ---")
            print(f"Number of states: {fst.num_states()}")
            if fst.start() is not None:
                print(f"Start state: {fst.start()}")
                num_arcs = 0
                for s in fst.states():
                    num_arcs += fst.num_trs(s)
                print(f"Number of arcs: {num_arcs}")
            else:
                print("No start state.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}", file=sys.stderr)
        sys.exit(1)
