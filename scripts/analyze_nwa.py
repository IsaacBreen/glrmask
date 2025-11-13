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

        # The actual determinization call
        _ = fst.determinize()

        result_queue.put(True)  # Signal success
    except Exception as e:
        # If any error occurs in the subprocess, signal failure
        result_queue.put(False)


def time_determinization_with_timeout(nwa: NWA, timeout: float) -> Optional[float]:
    """
    Runs determinization in a subprocess with a timeout.
    Returns duration if it completes, None if it times out or fails.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=determinize_worker, args=(nwa.to_dict(), result_queue))

    start_time = time.monotonic()
    process.start()
    process.join(timeout)
    end_time = time.monotonic()

    if process.is_alive():
        # Process is still running, so it timed out
        process.terminate()
        process.join()
        return None

    # Process finished, check if it was successful
    try:
        success = result_queue.get_nowait()
        if success:
            return end_time - start_time
        else:
            # Worker process had an error
            return None
    except Exception:
        # Queue was empty or process crashed without putting a result
        return None


# --- Main Pruning Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filepath = "minimized_hanging_nwa.json"
    # Set a reasonable timeout. If it takes longer than this, we assume it's hanging.
    DETERMINIZE_TIMEOUT_S = 0.1

    print(f"Loading original NWA from: {filepath}")
    original_nwa = load_nwa(filepath)
    original_transitions = original_nwa.num_transitions()
    print(f"Original NWA has {len(original_nwa.states)} states and {original_transitions} transitions.")

    print("\n--- Establishing Baseline Behavior ---")
    baseline_duration = time_determinization_with_timeout(original_nwa, DETERMINIZE_TIMEOUT_S)

    if baseline_duration is not None:
        print(f"Baseline determinization finished in {baseline_duration:.4f} seconds.")
        if baseline_duration < DETERMINIZE_TIMEOUT_S:
            print("Baseline is not hanging. This script is for finding hanging cases.")
            sys.exit(0)
    else:
        print(f"Baseline determinization timed out after {DETERMINIZE_TIMEOUT_S} seconds (as expected).")

    minimized_nwa = copy.deepcopy(original_nwa)

    # --- Create a master list of all transitions to try removing ---
    all_transitions_to_try = []
    for i, state in enumerate(minimized_nwa.states):
        for j, epsilon_target in enumerate(state.epsilons):
            all_transitions_to_try.append(('epsilon', i, epsilon_target))
        for label_str, targets in state.transitions.items():
            for j, labeled_target in enumerate(targets):
                all_transitions_to_try.append(('labeled', i, label_str, labeled_target))

    random.shuffle(all_transitions_to_try)
    print(f"\n--- Starting Pruning Process: {len(all_transitions_to_try)} candidates to check ---")

    for idx, trans_info in enumerate(all_transitions_to_try):
        candidate_nwa = copy.deepcopy(minimized_nwa)

        # Remove the specific transition in the candidate
        trans_type = trans_info[0]
        state_idx = trans_info[1]

        removed = False
        if trans_type == 'epsilon':
            target_to_remove = trans_info[2]
            if target_to_remove in candidate_nwa.states[state_idx].epsilons:
                candidate_nwa.states[state_idx].epsilons.remove(target_to_remove)
                removed = True
        else:  # 'labeled'
            label_str = trans_info[2]
            target_to_remove = trans_info[3]
            if label_str in candidate_nwa.states[state_idx].transitions and \
                    target_to_remove in candidate_nwa.states[state_idx].transitions[label_str]:
                candidate_nwa.states[state_idx].transitions[label_str].remove(target_to_remove)
                if not candidate_nwa.states[state_idx].transitions[label_str]:
                    del candidate_nwa.states[state_idx].transitions[label_str]
                removed = True

        if not removed:
            continue

        duration = time_determinization_with_timeout(candidate_nwa, DETERMINIZE_TIMEOUT_S)

        progress = f"[{idx + 1}/{len(all_transitions_to_try)}]"
        if duration is None:
            # SUCCESS: It's still hanging/timing out, so the removal was safe.
            minimized_nwa = candidate_nwa
            print(f"{progress} ✅ Prune SUCCESSFUL. Still hangs. New transition count: {minimized_nwa.num_transitions()}")
        else:
            # FAILURE: It's fast now, so this transition was critical.
            print(f"{progress} ❌ Prune FAILED. Finished in {duration:.4f}s. Reverting.")

    # --- Final Report ---
    print("\n--- Pruning Complete ---")
    final_transitions = minimized_nwa.num_transitions()
    print(f"Original transitions: {original_transitions}")
    print(f"Minimized transitions: {final_transitions}")
    reduction = original_transitions - final_transitions
    reduction_pct = (reduction / original_transitions * 100) if original_transitions > 0 else 0
    print(f"Removed {reduction} transitions ({reduction_pct:.2f}% reduction).")

    print(f"\nSaving minimized hanging NWA to: {output_filepath}")
    with open(output_filepath, 'w') as f:
        json.dump(minimized_nwa.to_dict(), f, indent=2)

    print("\nProcess finished. You can now inspect the minimized NWA.")