# scripts/analyze_nwa.py
import json
import sys
import os
import glob
from collections import defaultdict

class NWA:
    """
    A class to load and analyze a Non-deterministic Weighted Automaton (NWA)
    from a JSON dump file.
    """
    def __init__(self, data):
        self.start_state = data['body']['start_state']
        # The states are wrapped in a newtype struct in Rust, which serializes
        # to a single-element list containing the actual list of states.
        self.states = data['states'][0]

    def num_states(self):
        return len(self.states)

    def get_stats(self):
        """
        Computes various statistics about the NWA.
        """
        stats = {
            'num_states': self.num_states(),
            'start_state': self.start_state,
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

        for i, state in enumerate(self.states):
            if state['final_weight'] is not None:
                stats['num_final_states'] += 1
                self._analyze_weight(state['final_weight'], stats['weight_stats'])

            # Epsilon transitions
            eps_count = len(state['epsilons'])
            stats['num_epsilon_transitions'] += eps_count
            stats['epsilon_transitions_per_state'][i] = eps_count
            stats['outgoing_degree_per_state'][i] += eps_count
            for target, weight in state['epsilons']:
                stats['incoming_degree_per_state'][target] += 1
                self._analyze_weight(weight, stats['weight_stats'])

            # Labeled transitions
            labeled_count = 0
            for _label, targets in state['transitions'].items():
                labeled_count += len(targets)
                for target, weight in targets:
                    stats['incoming_degree_per_state'][target] += 1
                    self._analyze_weight(weight, stats['weight_stats'])
            stats['num_labeled_transitions'] += labeled_count
            stats['labeled_transitions_per_state'][i] = labeled_count
            stats['outgoing_degree_per_state'][i] += labeled_count

            # Default transitions
            default_count = len(state['default'])
            stats['num_default_transitions'] += default_count
            stats['default_transitions_per_state'][i] = default_count
            stats['outgoing_degree_per_state'][i] += default_count
            for default_trans in state['default']:
                target = default_trans['target']
                weight = default_trans['weight']
                stats['incoming_degree_per_state'][target] += 1
                self._analyze_weight(weight, stats['weight_stats'])

        return stats

    def _analyze_weight(self, weight, weight_stats):
        if weight == "ALL":
            weight_stats['all_count'] += 1
        elif isinstance(weight, list) and not weight:
            weight_stats['empty_count'] += 1
        else:
            weight_stats['complex_count'] += 1

    def print_stats(self):
        """
        Prints a summary of the NWA statistics to the console.
        """
        stats = self.get_stats()
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

def load_nwa(filepath):
    """
    Loads an NWA from a JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NWA(data)

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
        nwa.print_stats()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
