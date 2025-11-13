import json
import sys
import glob
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple

# We use rustfst for high-performance FST operations.
# This assumes you have the rustfst-python library installed.
try:
    from rustfst import VectorFst, Tr
except ImportError:
    print("Error: 'rustfst' not found. Please install it.", file=sys.stderr)
    print("See: https://github.com/Garvys/rustfst/tree/master/rustfst-python", file=sys.stderr)
    sys.exit(1)

# --- NWA Data Structures (Kept for JSON parsing) ---

# Type aliases for clarity
Weight = Any
NWAStateID = int


@dataclass
class NWAState:
    """Represents a single state in an NWA, loaded from JSON."""
    final_weight: Optional[Weight]
    transitions: Dict[str, List[Tuple[NWAStateID, Weight]]]
    epsilons: List[Tuple[NWAStateID, Weight]]


@dataclass
class NWABody:
    """Represents the body of an NWA, containing the start state."""
    start_state: NWAStateID


@dataclass
class NWA:
    """Represents a Non-deterministic Weighted Automaton (NWA)."""
    body: NWABody
    states: List[NWAState]

    @staticmethod
    def from_dict(data: dict) -> 'NWA':
        """Creates an NWA instance from a dictionary (parsed from JSON)."""
        body = NWABody(**data['body'])
        states_data = data['states']
        states = [
            NWAState(
                final_weight=s_data.get('final_weight'),
                transitions=s_data.get('transitions', {}),
                epsilons=[tuple(e) for e in s_data.get('epsilons', [])]
            ) for s_data in states_data
        ]
        return NWA(body=body, states=states)

    def num_states(self) -> int:
        return len(self.states)


def load_nwa(filepath: str) -> NWA:
    """Loads an NWA from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA.from_dict(data)


# --- RustFST Conversion and Determinization ---

def convert_nwa_to_rustfst(nwa: NWA) -> VectorFst:
    """
    Converts the NWA structure into a rustfst.VectorFst object.
    """
    print("--- Building rustfst.VectorFst from NWA data ---")
    fst = VectorFst()

    # Add all states first. The state ID will be its index.
    for _ in range(nwa.num_states()):
        fst.add_state()

    # Set the start state
    fst.set_start(nwa.body.start_state)

    # Add transitions and final states
    for i, state in enumerate(nwa.states):
        # Set final weight if the state is final.
        # For determinization, the actual weight value doesn't matter, only its presence.
        # We use 0.0, which is the identity weight in the Tropical Semiring.
        if state.final_weight is not None:
            fst.set_final(i, 0.0)

        # Add labeled transitions. In an acceptor, ilabel == olabel.
        # The label '0' is treated as an epsilon transition.
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            for target_id, _ in targets:
                # The weight is ignored for determinization, so we use 0.0
                tr = Tr(ilabel=label, olabel=label, weight=0.0, nextstate=target_id)
                fst.add_tr(i, tr)

        # Add explicit epsilon transitions (label 0)
        for target_id, _ in state.epsilons:
            tr = Tr(ilabel=0, olabel=0, weight=0.0, nextstate=target_id)
            fst.add_tr(i, tr)

    print(f"Successfully created VectorFst with {fst.num_states()} states.")
    return fst


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
        # 1. Load the NWA from the JSON file
        print(f"Loading NWA from {filepath}...")
        nwa = load_nwa(filepath)
        print(f"NWA loaded with {nwa.num_states()} states.")

        # 2. Convert the NWA to a RustFST VectorFst object
        nfa = convert_nwa_to_rustfst(nwa)

        # 3. Time the determinization process
        print("\n--- Determinizing NFA using RustFST ---")
        start_time = time.monotonic()

        # This is the core operation, executed in Rust
        dfa = nfa.determinize()

        end_time = time.monotonic()
        duration = end_time - start_time

        print(f"Determinization successful in {duration:.4f} seconds.")

        # 4. Print the results
        print("\n--- Summary ---")
        print(f"Initial NFA States: {nfa.num_states()}")
        print(f"Determinized DFA States: {dfa.num_states()}")

        if duration < 0.100:
            print("\nGoal achieved: Determinization took less than 100 ms.")
        else:
            print("\nNote: Determinization took longer than the 100 ms goal.")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)