import json
import sys

def analyze_nwa(file_path):
    with open(file_path, 'r') as f:
        nwa = json.load(f)

    states = nwa['states']
    num_states = len(states)
    
    num_final_states = 0
    total_epsilon_transitions = 0
    total_labeled_transitions = 0
    total_default_transitions = 0
    
    for state in states:
        if state['final_weight'] is not None:
            num_final_states += 1
        
        total_epsilon_transitions += len(state['epsilons'])
        
        if 'transitions' in state and state['transitions']:
            for symbol, transitions in state['transitions'].items():
                total_labeled_transitions += len(transitions)
            
        if 'default' in state and state['default']:
            total_default_transitions += len(state['default'])

    print("NWA Analysis:")
    print(f"  Number of states: {num_states}")
    print(f"  Number of final states: {num_final_states}")
    print(f"  Total epsilon transitions: {total_epsilon_transitions}")
    print(f"  Total labeled transitions: {total_labeled_transitions}")
    print(f"  Total default transitions: {total_default_transitions}")
    
    if num_states > 0:
        print(f"  Avg epsilon/state: {total_epsilon_transitions / num_states:.2f}")
        print(f"  Avg labeled/state: {total_labeled_transitions / num_states:.2f}")
        print(f"  Avg default/state: {total_default_transitions / num_states:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_nwa.py <path_to_nwa_dump.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_nwa(file_path)
