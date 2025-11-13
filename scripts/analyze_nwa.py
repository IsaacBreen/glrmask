import json
import sys
import os
import time
import random
import copy
import multiprocessing
from dataclasses import dataclass, asdict
from typing import Any, Optional, List, Dict, Tuple

# We use rustfst for high-performance FST operations.
try:
    from rustfst import VectorFst, Tr
except ImportError:
    print("Error: 'rustfst' not found. Please install it.", file=sys.stderr)
    sys.exit(1)


# --- NWA Data Structures (Identical to previous script) ---

@dataclass
class NWAState:
    final_weight: Optional[Any]
    transitions: Dict[str, List[Tuple[int, Any]]]
    epsilons: List[Tuple[int, Any]]


@dataclass
class NWABody:
    start_state: int


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
        return asdict(self)

    def num_transitions(self) -> int:
        count = 0
        for state in self.states:
            count += len(state.epsilons)
            for targets in state.transitions.values():
                count += len(targets)
        return count


def load_nwa(filepath: str) -> NWA:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA.from_dict(data)


# --- Core Determinization Function (to be run in a separate process) ---

def determinize_worker(nwa_dict: dict, result_queue: multiprocessing.Queue):
    """This function runs in a separate process to avoid blocking."""
    try:
        nwa = NWA.from_dict(nwa_dict)
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(len(nwa.states))}
        fst.set_start(state_map[nwa.body.start_state])

        for i, state in enumerate(nwa.states):
            if state.final_weight is not None:
                fst.set_final(state_map[i], 0.0)
            for label_str, targets in state.transitions.items():
                label = int(label_str)
                for target_id, _ in targets:
                    fst.add_tr(state_map[i], Tr(label, label, 0.0, state_map[target_id]))
            for target_id, _ in state.epsilons:
                fst.add_tr(state_map[i], Tr(0, 0, 0.0, state_map[target_id]))

        _ = fst.determinize()
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def time_determinization_with_timeout(nwa: NWA, timeout: float) -> bool:
    """
    Runs determinization in a subprocess.
    Returns True if it HANGS (times out), False if it completes.
    """
    if nwa.num_transitions() == 0: return False

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=determinize_worker, args=(nwa.to_dict(), result_queue))

    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return True  # Timed out -> HANGS

    try:
        # If it finished, it doesn't hang
        return not result_queue.get_nowait()
    except Exception:
        # Crashed or queue empty, treat as not hanging
        return False


def remove_transitions_from_nwa(nwa: NWA, transitions_to_remove: set) -> NWA:
    """Creates a new NWA with a set of transitions removed."""
    candidate_nwa = copy.deepcopy(nwa)
    for trans in transitions_to_remove:
        trans_type, state_idx, *rest = trans
        if trans_type == 'epsilon':
            target_to_remove = rest[0]
            if target_to_remove in candidate_nwa.states[state_idx].epsilons:
                candidate_nwa.states[state_idx].epsilons.remove(target_to_remove)
        else:
            label_str, target_to_remove = rest
            if label_str in candidate_nwa.states[state_idx].transitions and \
                    target_to_remove in candidate_nwa.states[state_idx].transitions[label_str]:
                candidate_nwa.states[state_idx].transitions[label_str].remove(target_to_remove)
                if not candidate_nwa.states[state_idx].transitions[label_str]:
                    del candidate_nwa.states[state_idx].transitions[label_str]
    return candidate_nwa


# --- Main Pruning Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filepath = "minimized_nwa_adaptive.json"
    DETERMINIZE_TIMEOUT_S = 1.0

    print(f"Loading original NWA from: {filepath}")
    original_nwa = load_nwa(filepath)
    original_transitions_count = original_nwa.num_transitions()
    print(f"Original NWA has {len(original_nwa.states)} states and {original_transitions_count} transitions.")

    print("\n--- Establishing Baseline Behavior ---")
    if not time_determinization_with_timeout(original_nwa, DETERMINIZE_TIMEOUT_S):
        print(f"Baseline determinization finished within {DETERMINIZE_TIMEOUT_S}s. This script is for hanging cases.")
        sys.exit(0)
    else:
        print(f"Baseline determinization timed out after {DETERMINIZE_TIMEOUT_S}s (as expected).")

    # --- PHASE 1: Adaptive Chunk Pruning ---
    print("\n--- PHASE 1: Adaptive Pruning ---")

    # Get all transitions as a list of unique, hashable tuples
    all_transitions = set()
    for i, state in enumerate(original_nwa.states):
        for epsilon_target in state.epsilons:
            all_transitions.add(('epsilon', i, epsilon_target))
        for label_str, targets in state.transitions.items():
            for labeled_target in targets:
                all_transitions.add(('labeled', i, label_str, labeled_target))

    # `good_transitions` are ones we know are essential
    # `untested_transitions` are ones we haven't checked yet
    good_transitions = set()
    untested_transitions = list(all_transitions)
    random.shuffle(untested_transitions)

    while len(untested_transitions) > 1:
        # Try to remove half of the remaining untested transitions
        chunk_size = len(untested_transitions) // 2
        if chunk_size == 0: break

        chunk_to_test = set(untested_transitions[:chunk_size])

        # The candidate NWA contains only the good transitions and the other half of untested ones
        transitions_to_keep = good_transitions.union(set(untested_transitions[chunk_size:]))
        transitions_to_remove = all_transitions - transitions_to_keep
        candidate_nwa = remove_transitions_from_nwa(original_nwa, transitions_to_remove)

        print(f"Testing removal of {len(chunk_to_test)} transitions... (Remaining untested: {len(untested_transitions)})")

        if time_determinization_with_timeout(candidate_nwa, DETERMINIZE_TIMEOUT_S):
            # SUCCESS: It still hangs. The chunk we removed was irrelevant.
            # The new set of untested transitions is the half we kept.
            untested_transitions = untested_transitions[chunk_size:]
            print(f"  ✅ SUCCESS. Chunk was not essential. {len(untested_transitions)} candidates remain.")
        else:
            # FAILURE: It became fast. A critical transition is in the chunk we removed.
            # The chunk becomes the new set of untested transitions.
            good_transitions.update(set(untested_transitions[chunk_size:]))
            untested_transitions = untested_transitions[:chunk_size]
            print(f"  ❌ FAILURE. Chunk is essential. Narrowing search to {len(untested_transitions)} candidates.")

    # At this point, `untested_transitions` contains the minimal set from phase 1
    minimized_nwa = remove_transitions_from_nwa(original_nwa, all_transitions - good_transitions - set(untested_transitions))
    print(f"\n--- PHASE 1 COMPLETE ---")
    print(f"Reduced to {minimized_nwa.num_transitions()} candidate transitions.")

    # --- PHASE 2: Fine Pruning (One-by-one) ---
    print(f"\n--- PHASE 2: Fine Pruning ---")

    essential_transitions = list(minimized_nwa.to_dict()['states'])  # A bit of a hack to get current transitions

    # Convert minimized_nwa to a list of transitions to test one-by-one
    transitions_to_refine = []
    for i, state in enumerate(minimized_nwa.states):
        for epsilon_target in state.epsilons:
            transitions_to_refine.append(('epsilon', i, epsilon_target))
        for label_str, targets in state.transitions.items():
            for labeled_target in targets:
                transitions_to_refine.append(('labeled', i, label_str, labeled_target))

    random.shuffle(transitions_to_refine)

    for idx, trans_to_remove in enumerate(transitions_to_refine):
        # Don't test removing a transition if it's the last one
        if minimized_nwa.num_transitions() <= 1: break

        candidate_nwa = remove_transitions_from_nwa(minimized_nwa, {trans_to_remove})
        progress = f"[{idx + 1}/{len(transitions_to_refine)}]"

        if time_determinization_with_timeout(candidate_nwa, DETERMINIZE_TIMEOUT_S):
            minimized_nwa = candidate_nwa
            print(f"{progress} ✅ Prune SUCCESSFUL. Still hangs. New transition count: {minimized_nwa.num_transitions()}")
        else:
            print(f"{progress} ❌ Prune FAILED. Reverting.")

    # --- Final Report ---
    print("\n--- Pruning Complete ---")
    final_transitions_count = minimized_nwa.num_transitions()
    print(f"Original transitions: {original_transitions_count}")
    print(f"Minimized transitions: {final_transitions_count}")
    reduction = original_transitions_count - final_transitions_count
    reduction_pct = (reduction / original_transitions_count * 100) if original_transitions_count > 0 else 0
    print(f"Removed {reduction} transitions ({reduction_pct:.2f}% reduction).")

    print(f"\nSaving minimized NWA to: {output_filepath}")
    with open(output_filepath, 'w') as f:
        json.dump(minimized_nwa.to_dict(), f, indent=2)

    print("\nProcess finished. You can now inspect the minimized NWA.")