import json
import sys
import os
import time
import random
import copy
from dataclasses import dataclass, asdict
from typing import Any, Optional, List, Dict, Tuple

# We use rustfst for high-performance FST operations.
try:
    from rustfst import VectorFst, Tr
except ImportError:
    print("Error: 'rustfst' not found. Please install it.", file=sys.stderr)
    sys.exit(1)

# --- NWA Data Structures ---
Weight = Any  # Placeholder for weight type
NWAStateID = int

@dataclass
class NWAState:
    final_weight: Optional[Weight]
    transitions: Dict[str, List[Tuple[NWAStateID, Weight]]]
    epsilons: List[Tuple[NWAStateID, Weight]]

@dataclass
class NWABody:
    start_state: NWAStateID

@dataclass
class NWA:
    body: NWABody
    states: List[NWAState]

    @staticmethod
    def from_dict(data: dict) -> 'NWA':
        body = NWABody(**data['body'])
        states = [
            NWAState(
                final_weight=s_data.get('final_weight'),
                transitions=s_data.get('transitions', {}),
                epsilons=[tuple(e) for e in s_data.get('epsilons', [])]
            ) for s_data in data['states']
        ]
        return NWA(body=body, states=states)

    def to_dict(self) -> dict:
        """Converts the NWA object back to a serializable dictionary."""
        return asdict(self)

    def num_transitions(self) -> int:
        """Counts the total number of transitions in the NWA."""
        count = 0
        for state in self.states:
            count += len(state.epsilons)
            for targets in state.transitions.values():
                count += len(targets)
        return count

def load_nwa(filepath: str) -> NWA:
    """Loads an NWA from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA.from_dict(data)

# --- Core Timed Determinization Function ---

def time_determinization(nwa: NWA) -> float:
    """Converts an NWA to a RustFST object and times its determinization."""
    fst = VectorFst()
    for _ in range(len(nwa.states)):
        fst.add_state()
    fst.set_start(nwa.body.start_state)

    for i, state in enumerate(nwa.states):
        if state.final_weight is not None:
            fst.set_final(i, 0.0)
        for label_str, targets in state.transitions.items():
            label = int(label_str)
            for target_id, _ in targets:
                fst.add_tr(i, Tr(label, label, 0.0, target_id))
        for target_id, _ in state.epsilons:
            fst.add_tr(i, Tr(0, 0, 0.0, target_id))

    start_time = time.monotonic()
    _ = fst.determinize()
    end_time = time.monotonic()
    return end_time - start_time

# --- Main Pruning Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filepath = "minimized_nwa_dump.json"
    DETERMINIZE_THRESHOLD_S = 0.100  # 100 ms

    print(f"Loading original NWA from: {filepath}")
    original_nwa = load_nwa(filepath)
    original_transitions = original_nwa.num_transitions()
    print(f"Original NWA has {len(original_nwa.states)} states and {original_transitions} transitions.")

    print("\n--- Establishing Baseline Performance ---")
    baseline_duration = time_determinization(original_nwa)
    print(f"Initial determinization time: {baseline_duration:.4f} seconds.")

    if baseline_duration < DETERMINIZE_THRESHOLD_S:
        print("Baseline is already faster than the threshold. Nothing to do.")
        sys.exit(0)

    # This will be our working copy that we shrink over time
    minimized_nwa = copy.deepcopy(original_nwa)

    # --- Create a master list of all transitions to try removing ---
    # Format: (state_idx, 'epsilon'/'labeled', transition_key, target_idx)
    all_transitions_to_try = []
    for i, state in enumerate(minimized_nwa.states):
        for j in range(len(state.epsilons)):
            all_transitions_to_try.append((i, 'epsilon', j))
        for label_str, targets in state.transitions.items():
            for j in range(len(targets)):
                all_transitions_to_try.append((i, 'labeled', label_str, j))

    random.shuffle(all_transitions_to_try)
    print(f"\n--- Starting Pruning Process: {len(all_transitions_to_try)} candidates to check ---")

    # We need to iterate by index because the list of transitions inside the NWA will shrink
    for idx, (state_idx, trans_type, key) in enumerate(all_transitions_to_try):

        # Create a temporary copy to test the removal
        candidate_nwa = copy.deepcopy(minimized_nwa)

        # Find and remove the specific transition in the candidate
        removed = False
        if trans_type == 'epsilon':
            # Find the epsilon transition by its original index
            original_epsilon = original_nwa.states[state_idx].epsilons[key]
            if original_epsilon in candidate_nwa.states[state_idx].epsilons:
                candidate_nwa.states[state_idx].epsilons.remove(original_epsilon)
                removed = True
        else: # 'labeled'
            label_str = key
            target_idx = all_transitions_to_try[idx][3]
            original_labeled = original_nwa.states[state_idx].transitions[label_str][target_idx]
            if label_str in candidate_nwa.states[state_idx].transitions and \
               original_labeled in candidate_nwa.states[state_idx].transitions[label_str]:
                candidate_nwa.states[state_idx].transitions[label_str].remove(original_labeled)
                # Clean up empty list
                if not candidate_nwa.states[state_idx].transitions[label_str]:
                    del candidate_nwa.states[state_idx].transitions[label_str]
                removed = True

        if not removed:
            # This transition was already removed as part of a previous successful prune
            continue

        duration = time_determinization(candidate_nwa)

        progress = f"[{idx + 1}/{len(all_transitions_to_try)}]"
        if duration >= DETERMINIZE_THRESHOLD_S:
            # SUCCESS: It's still slow, so the removal was safe.
            minimized_nwa = candidate_nwa
            print(f"{progress} ✅ Prune SUCCESSFUL. Time: {duration:.4f}s. New transition count: {minimized_nwa.num_transitions()}")
        else:
            # FAILURE: It's fast now, so this transition was critical.
            print(f"{progress} ❌ Prune FAILED. Time: {duration:.4f}s. Reverting.")

    # --- Final Report ---
    print("\n--- Pruning Complete ---")
    final_transitions = minimized_nwa.num_transitions()
    print(f"Original transitions: {original_transitions}")
    print(f"Minimized transitions: {final_transitions}")
    reduction = original_transitions - final_transitions
    reduction_pct = (reduction / original_transitions * 100) if original_transitions > 0 else 0
    print(f"Removed {reduction} transitions ({reduction_pct:.2f}% reduction).")

    print(f"\nSaving minimized NWA to: {output_filepath}")
    with open(output_filepath, 'w') as f:
        json.dump(minimized_nwa.to_dict(), f, indent=2)

    print("\nProcess finished. You can now inspect the minimized NWA.")