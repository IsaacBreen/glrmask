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


# --- NWA Data Structures ---
# We only need these for the initial load. The pruning logic will use simpler structures.

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


# --- Core Determinization Function (to be run in a separate process) ---

def determinize_worker(
        num_states: int, start_state: int, final_states: set, transitions: set, result_queue: multiprocessing.Queue
):
    """
    This worker function builds an FST from simple, pre-processed data structures.
    It completely ignores weights.
    """
    try:
        fst = VectorFst()
        state_map = {i: fst.add_state() for i in range(num_states)}
        fst.set_start(state_map[start_state])

        for state_id in final_states:
            fst.set_final(state_map[state_id], 0.0)

        for source, label, dest in transitions:
            fst.add_tr(state_map[source], Tr(label, label, 0.0, state_map[dest]))

        _ = fst.determinize()
        result_queue.put(True)  # Finished successfully
    except Exception:
        result_queue.put(False)  # An error occurred


def time_determinization_with_timeout(
        num_states: int, start_state: int, final_states: set, transitions: set, timeout: float
) -> bool:
    """
    Runs determinization in a subprocess.
    Returns True if it HANGS (times out), False if it completes or fails.
    """
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
        return True  # Timed out -> HANGS

    try:
        # If it finished, it doesn't hang.
        # We check the result queue to see if it finished cleanly.
        return not result_queue.get_nowait()
    except Exception:
        # Crashed or queue empty, treat as not hanging
        return False


# --- Main Pruning Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filepath = "minimized_nwa_adaptive.json"
    DETERMINIZE_TIMEOUT_S = 1.0

    print(f"Loading original NWA from: {filepath}")
    with open(filepath, 'r') as f:
        nwa_data = json.load(f)

    # --- Extract a simple, hashable representation of the NWA graph ---
    num_states = len(nwa_data['states'])
    start_state = nwa_data['body']['start_state']
    final_states = {i for i, s in enumerate(nwa_data['states']) if s.get('final_weight') is not None}

    all_transitions = set()
    for i, state in enumerate(nwa_data['states']):
        for target_id, _ in state.get('epsilons', []):
            # Epsilon is label 0
            all_transitions.add((i, 0, target_id))
        for label_str, targets in state.get('transitions', {}).items():
            label = int(label_str)
            for target_id, _ in targets:
                all_transitions.add((i, label, target_id))

    original_transitions_count = len(all_transitions)
    print(f"Original NWA has {num_states} states and {original_transitions_count} unique transitions.")

    print("\n--- Establishing Baseline Behavior ---")
    if not time_determinization_with_timeout(num_states, start_state, final_states, all_transitions, DETERMINIZE_TIMEOUT_S):
        print(f"Baseline determinization finished within {DETERMINIZE_TIMEOUT_S}s. This script is for hanging cases.")
        sys.exit(0)
    else:
        print(f"Baseline determinization timed out after {DETERMINIZE_TIMEOUT_S}s (as expected).")

    # --- PHASE 1: Adaptive Chunk Pruning ---
    print("\n--- PHASE 1: Adaptive Pruning ---")

    good_transitions = set()
    untested_transitions = list(all_transitions)
    random.shuffle(untested_transitions)

    while len(untested_transitions) > 1:
        chunk_size = len(untested_transitions) // 2
        if chunk_size == 0: break

        chunk_to_test = set(untested_transitions[:chunk_size])
        transitions_to_keep = good_transitions.union(set(untested_transitions[chunk_size:]))

        print(f"Testing removal of {len(chunk_to_test)} transitions... (Remaining untested: {len(untested_transitions)})")

        if time_determinization_with_timeout(num_states, start_state, final_states, transitions_to_keep, DETERMINIZE_TIMEOUT_S):
            untested_transitions = untested_transitions[chunk_size:]
            print(f"  ✅ SUCCESS. Chunk was not essential. {len(untested_transitions)} candidates remain.")
        else:
            good_transitions.update(set(untested_transitions[chunk_size:]))
            untested_transitions = untested_transitions[:chunk_size]
            print(f"  ❌ FAILURE. Chunk is essential. Narrowing search to {len(untested_transitions)} candidates.")

    essential_transitions = good_transitions.union(set(untested_transitions))
    print(f"\n--- PHASE 1 COMPLETE ---")
    print(f"Reduced to {len(essential_transitions)} candidate transitions.")

    # --- PHASE 2: Fine Pruning (One-by-one) ---
    print(f"\n--- PHASE 2: Fine Pruning ---")

    transitions_to_refine = list(essential_transitions)
    random.shuffle(transitions_to_refine)

    for idx, trans_to_remove in enumerate(transitions_to_refine):
        if len(essential_transitions) <= 1: break

        candidate_transitions = essential_transitions - {trans_to_remove}
        progress = f"[{idx + 1}/{len(transitions_to_refine)}]"

        if time_determinization_with_timeout(num_states, start_state, final_states, candidate_transitions, DETERMINIZE_TIMEOUT_S):
            essential_transitions = candidate_transitions
            print(f"{progress} ✅ Prune SUCCESSFUL. Still hangs. New transition count: {len(essential_transitions)}")
        else:
            print(f"{progress} ❌ Prune FAILED. Reverting.")

    # --- Final Report and Save ---
    print("\n--- Pruning Complete ---")
    final_transitions_count = len(essential_transitions)
    print(f"Original transitions: {original_transitions_count}")
    print(f"Minimized transitions: {final_transitions_count}")
    reduction = original_transitions_count - final_transitions_count
    reduction_pct = (reduction / original_transitions_count * 100) if original_transitions_count > 0 else 0
    print(f"Removed {reduction} transitions ({reduction_pct:.2f}% reduction).")

    # Reconstruct the NWA JSON from the minimal set of transitions
    minimized_nwa_dict = {
        "body": {"start_state": start_state},
        "states": [{"transitions": {}, "epsilons": []} for _ in range(num_states)]
    }
    for i in final_states:
        minimized_nwa_dict["states"][i]["final_weight"] = "ALL"  # Use a placeholder

    for source, label, dest in essential_transitions:
        # We add a dummy weight back in for format compatibility
        if label == 0:
            minimized_nwa_dict["states"][source]["epsilons"].append([dest, []])
        else:
            label_str = str(label)
            if label_str not in minimized_nwa_dict["states"][source]["transitions"]:
                minimized_nwa_dict["states"][source]["transitions"][label_str] = []
            minimized_nwa_dict["states"][source]["transitions"][label_str].append([dest, []])

    print(f"\nSaving minimized NWA to: {output_filepath}")
    with open(output_filepath, 'w') as f:
        json.dump(minimized_nwa_dict, f, indent=2)

    print("\nProcess finished. You can now inspect the minimized NWA.")