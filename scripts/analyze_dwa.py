#!/usr/bin/env python3
"""
Analyze exported DWA JSON files for epsilon transition explosion investigation.

Usage:
    python scripts/analyze_dwa.py temp/terminal_dwa_original.json temp/terminal_dwa_modified.json
    
Or just one:
    python scripts/analyze_dwa.py temp/terminal_dwa_original.json
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set


def load_dwa(path: str) -> dict:
    """Load a DWA from a JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_weight(w: dict) -> Tuple[bool, bool, List[Tuple[int, int]], int]:
    """
    Parse a weight object from JSON.
    Returns: (is_all, is_empty, ranges, length)
    """
    if w.get("is_all"):
        return True, False, [], float('inf')
    if w.get("is_empty"):
        return False, True, [], 0
    ranges = [tuple(r) for r in w.get("ranges", [])]
    length = w.get("len", sum(r[1] - r[0] + 1 for r in ranges))
    return False, False, ranges, length


def weight_signature(w: dict) -> str:
    """Create a hashable signature for a weight."""
    is_all, is_empty, ranges, _ = parse_weight(w)
    if is_all:
        return "ALL"
    if is_empty:
        return "EMPTY"
    return str(sorted(ranges))


class DWAAnalyzer:
    def __init__(self, dwa: dict, name: str = "DWA"):
        self.dwa = dwa
        self.name = name
        self.states = {s["id"]: s for s in dwa["states"]}
        self.start_state = dwa["start_state"]
        self.num_states = dwa["num_states"]
        self.num_transitions = dwa["num_transitions"]
        
    def summary(self):
        """Print a summary of the DWA."""
        print(f"\n=== {self.name} Summary ===")
        print(f"Start state: {self.start_state}")
        print(f"States: {self.num_states}")
        print(f"Transitions: {self.num_transitions}")
        
        # Count final states
        final_states = [sid for sid, s in self.states.items() if "final_weight" in s]
        print(f"Final states: {len(final_states)}")
        
        # Transition weight analysis
        all_weights = 0
        non_all_weights = 0
        for state in self.dwa["states"]:
            for trans in state["transitions"]:
                w = trans["weight"]
                if w.get("is_all"):
                    all_weights += 1
                else:
                    non_all_weights += 1
        print(f"Transitions with ALL weight: {all_weights}")
        print(f"Transitions with non-ALL weight: {non_all_weights}")
        
    def analyze_state(self, state_id: int, show_weights: bool = True):
        """Analyze a single state in detail."""
        if state_id not in self.states:
            print(f"State {state_id} not found")
            return
            
        state = self.states[state_id]
        print(f"\n=== State {state_id} ===")
        
        if "final_weight" in state:
            fw = state["final_weight"]
            is_all, is_empty, ranges, length = parse_weight(fw)
            if is_all:
                print(f"  Final weight: ALL")
            elif is_empty:
                print(f"  Final weight: EMPTY")
            else:
                print(f"  Final weight: len={length}, ranges={ranges[:5]}{'...' if len(ranges) > 5 else ''}")
        else:
            print(f"  Not a final state")
            
        print(f"  Outgoing transitions: {len(state['transitions'])}")
        
        # Group transitions by target
        by_target = defaultdict(list)
        for trans in state["transitions"]:
            by_target[trans["target"]].append(trans)
            
        print(f"  Unique targets: {len(by_target)}")
        
        if show_weights:
            # Group by weight signature
            by_weight = defaultdict(list)
            for trans in state["transitions"]:
                sig = weight_signature(trans["weight"])
                by_weight[sig].append(trans)
                
            print(f"  Unique weight signatures: {len(by_weight)}")
            for sig, trans_list in sorted(by_weight.items(), key=lambda x: -len(x[1]))[:5]:
                targets = [t["target"] for t in trans_list]
                if sig == "ALL":
                    print(f"    ALL weight: {len(trans_list)} transitions -> {targets[:5]}{'...' if len(targets) > 5 else ''}")
                else:
                    # Parse the first weight to show
                    w = trans_list[0]["weight"]
                    _, _, ranges, length = parse_weight(w)
                    print(f"    len={length}: {len(trans_list)} transitions -> {targets[:3]}{'...' if len(targets) > 3 else ''}")
                    
    def find_states_with_uniform_outgoing_weights(self) -> List[int]:
        """Find states where all outgoing transitions have the same weight."""
        uniform_states = []
        for state_id, state in self.states.items():
            trans = state["transitions"]
            if len(trans) <= 1:
                continue
                
            # Check if all trans_weights are the same
            sigs = set(weight_signature(t["weight"]) for t in trans)
            if len(sigs) == 1:
                uniform_states.append(state_id)
                
        return uniform_states
        
    def find_states_with_same_final_weight(self) -> Dict[str, List[int]]:
        """Group states by their final weight signature."""
        by_final = defaultdict(list)
        for state_id, state in self.states.items():
            if "final_weight" in state:
                sig = weight_signature(state["final_weight"])
                by_final[sig].append(state_id)
        return dict(by_final)
        
    def compare_states(self, state_ids: List[int]):
        """Compare multiple states side by side."""
        print(f"\n=== Comparing states {state_ids} ===")
        
        states = [self.states.get(sid) for sid in state_ids]
        if None in states:
            missing = [sid for sid, s in zip(state_ids, states) if s is None]
            print(f"States not found: {missing}")
            return
            
        # Compare final weights
        print("\nFinal weights:")
        for sid, state in zip(state_ids, states):
            if "final_weight" in state:
                is_all, is_empty, ranges, length = parse_weight(state["final_weight"])
                if is_all:
                    print(f"  State {sid}: ALL")
                else:
                    print(f"  State {sid}: len={length}")
            else:
                print(f"  State {sid}: (not final)")
                
        # Compare transition counts
        print("\nTransition counts:")
        for sid, state in zip(state_ids, states):
            print(f"  State {sid}: {len(state['transitions'])} transitions")
            
        # Compare targets
        print("\nTargets:")
        all_targets = set()
        targets_by_state = {}
        for sid, state in zip(state_ids, states):
            targets = set(t["target"] for t in state["transitions"])
            targets_by_state[sid] = targets
            all_targets.update(targets)
            
        shared = set.intersection(*targets_by_state.values()) if targets_by_state else set()
        print(f"  Shared targets: {len(shared)}")
        print(f"  All unique targets: {len(all_targets)}")
        
        # Compare trans_weights for shared targets
        if shared:
            print("\nTrans weights for shared targets (sample):")
            for target in sorted(shared)[:5]:
                weights = []
                for sid, state in zip(state_ids, states):
                    for t in state["transitions"]:
                        if t["target"] == target:
                            is_all, _, _, length = parse_weight(t["weight"])
                            weights.append(f"{sid}:{'ALL' if is_all else length}")
                            break
                print(f"  -> {target}: {', '.join(weights)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dwa.py <dwa.json> [dwa2.json]")
        sys.exit(1)
        
    # Load first DWA
    dwa1_path = sys.argv[1]
    dwa1 = load_dwa(dwa1_path)
    analyzer1 = DWAAnalyzer(dwa1, dwa1_path.split("/")[-1])
    analyzer1.summary()
    
    # Find uniform states
    uniform = analyzer1.find_states_with_uniform_outgoing_weights()
    print(f"\nStates with uniform outgoing weights: {len(uniform)}")
    if uniform:
        print(f"  Examples: {uniform[:10]}")
        
    # Find states with same final weight  
    by_final = analyzer1.find_states_with_same_final_weight()
    print(f"\nStates grouped by final weight signature: {len(by_final)} groups")
    for sig, sids in sorted(by_final.items(), key=lambda x: -len(x[1]))[:5]:
        if len(sids) > 1:
            print(f"  {sig[:30]}{'...' if len(sig) > 30 else ''}: {len(sids)} states")
            
    # If two DWAs provided, compare them
    if len(sys.argv) >= 3:
        dwa2_path = sys.argv[2]
        dwa2 = load_dwa(dwa2_path)
        analyzer2 = DWAAnalyzer(dwa2, dwa2_path.split("/")[-1])
        analyzer2.summary()
        
        uniform2 = analyzer2.find_states_with_uniform_outgoing_weights()
        print(f"\nStates with uniform outgoing weights: {len(uniform2)}")
        
        by_final2 = analyzer2.find_states_with_same_final_weight()
        print(f"States grouped by final weight signature: {len(by_final2)} groups")
        for sig, sids in sorted(by_final2.items(), key=lambda x: -len(x[1]))[:5]:
            if len(sids) > 1:
                print(f"  {sig[:30]}{'...' if len(sig) > 30 else ''}: {len(sids)} states")
                
        # Compare state counts
        print(f"\n=== Comparison ===")
        print(f"State ratio: {analyzer2.num_states / analyzer1.num_states:.2f}x")
        print(f"Transition ratio: {analyzer2.num_transitions / analyzer1.num_transitions:.2f}x")


if __name__ == "__main__":
    main()
