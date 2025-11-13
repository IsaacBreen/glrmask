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
        # Pre-allocate all states
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


def time_determinization_with_timeout(nwa: NWA, timeout: float) -> Optional[float]:
    """Runs determinization in a subprocess. Returns duration or None if it times out."""
    if not nwa.states: return 0.0  # Empty NWA is instant

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=determinize_worker, args=(nwa.to_dict(), result_queue))

    start_time = time.monotonic()
    process.start()
    process.join(timeout)
    end_time = time.monotonic()

    if process.is_alive():
        process.terminate()
        process.join()
        return None

    try:
        if result_queue.get_nowait():
            return end_time - start_time
    except Exception:
        pass
    return None


# --- Main Pruning Logic ---

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_nwa_dump.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    output_filepath = "minimized_nwa.json"

    # --- CONFIGURATION ---
    DETERMINIZE_TIMEOUT_S = 1.0  # Assume it's hanging if it takes longer than this
    CHUNK_SIZE = 5000  # How many transitions to try removing at once in Phase 1

    print(f"Loading original NWA from: {filepath}")
    original_nwa = load_nwa(filepath)
    original_transitions = original_nwa.num_transitions()
    print(f"Original NWA has {len(original_nwa.states)} states and {original_transitions} transitions.")

    print("\n--- Establishing Baseline Behavior ---")
    baseline_duration = time_determinization_with_timeout(original_nwa, DETERMINIZE_TIMEOUT_S)
    if baseline_duration is not None:
        print(f"Baseline determinization finished in {baseline_duration:.4f}s. This script is for hanging cases.")
        sys.exit(0)
    else:
        print(f"Baseline determinization timed out after {DETERMINIZE_TIMEOUT_S}s (as expected).")

    minimized_nwa = copy.deepcopy(original_nwa)

    # --- PHASE 1: Coarse Pruning by Chunk ---
    print(f"\n--- PHASE 1: Coarse Pruning (Chunk Size: {CHUNK_SIZE}) ---")

    all_transitions = []
    for i, state in enumerate(minimized_nwa.states):
        for epsilon_target in state.epsilons:
            all_transitions.append(('epsilon', i, epsilon_target))
        for label_str, targets in state.transitions.items():
            for labeled_target in targets:
                all_transitions.append(('labeled', i, label_str, labeled_target))

    random.shuffle(all_transitions)

    chunks = [all_transitions[i:i + CHUNK_SIZE] for i in range(0, len(all_transitions), CHUNK_SIZE)]

    for i, chunk in enumerate(chunks):
        candidate_nwa = copy.deepcopy(minimized_nwa)

        # Remove the entire chunk from the candidate
        for trans_type, state_idx, *rest in chunk:
            if trans_type == 'epsilon':
                target_to_remove = rest[0]
                if target_to_remove in candidate_nwa.states[state_idx].epsilons:
                    candidate_nwa.states[state_idx].epsilons.remove(target_to_remove)
            else:  # 'labeled'
                label_str, target_to_remove = rest
                if label_str in candidate_nwa.states[state_idx].transitions and \
                        target_to_remove in candidate_nwa.states[state_idx].transitions[label_str]:
                    candidate_nwa.states[state_idx].transitions[label_str].remove(target_to_remove)
                    if not candidate_nwa.states[state_idx].transitions[label_str]:
                        del candidate_nwa.states[state_idx].transitions[label_str]

        duration = time_determinization_with_timeout(candidate_nwa, DETERMINIZE_TIMEOUT_S)
        progress = f"[Chunk {i + 1}/{len(chunks)}]"

        if duration is None:
            minimized_nwa = candidate_nwa
            print(f"{progress} ✅ Chunk removed successfully. Still hangs. New transition count: {minimized_nwa.num_transitions()}")
        else:
            print(f"{progress} ❌ Chunk contained critical transition(s). Finished in {duration:.4f}s. Reverting.")

    print(f"\n--- PHASE 1 COMPLETE ---")
    print(f"Reduced to {minimized_nwa.num_transitions()} transitions.")

    # --- PHASE 2: Fine Pruning (One-by-one) ---
    print(f"\n--- PHASE 2: Fine Pruning (One-by-one) ---")

    remaining_transitions = []
    for i, state in enumerate(minimized_nwa.states):
        for epsilon_target in state.epsilons:
            remaining_transitions.append(('epsilon', i, epsilon_target))
        for label_str, targets in state.transitions.items():
            for labeled_target in targets:
                remaining_transitions.append(('labeled', i, label_str, labeled_target))

    random.shuffle(remaining_transitions)

    for idx, trans_info in enumerate(remaining_transitions):
        candidate_nwa = copy.deepcopy(minimized_nwa)
        trans_type, state_idx, *rest = trans_info

        removed = False
        if trans_type == 'epsilon':
            target_to_remove = rest[0]
            if target_to_remove in candidate_nwa.states[state_idx].epsilons:
                candidate_nwa.states[state_idx].epsilons.remove(target_to_remove)
                removed = True
        else:  # 'labeled'
            label_str, target_to_remove = rest
            if label_str in candidate_nwa.states[state_idx].transitions and \
                    target_to_remove in candidate_nwa.states[state_idx].transitions[label_str]:
                candidate_nwa.states[state_idx].transitions[label_str].remove(target_to_remove)
                if not candidate_nwa.states[state_idx].transitions[label_str]:
                    del candidate_nwa.states[state_idx].transitions[label_str]
                removed = True

        if not removed: continue

        duration = time_determinization_with_timeout(candidate_nwa, DETERMINIZE_TIMEOUT_S)
        progress = f"[{idx + 1}/{len(remaining_transitions)}]"

        if duration is None:
            minimized_nwa = candidate_nwa
            print(f"{progress} ✅ Prune SUCCESSFUL. Still hangs. New transition count: {minimized_nwa.num_transitions()}")
        else:
            print(f"{progress} ❌ Prune FAILED. Finished in {duration:.4f}s. Reverting.")

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