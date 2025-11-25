import json
import os
import base64
import math
from collections import defaultdict, Counter
import copy

# --- Configuration ---
# The path to your grammar constraint file.
FILE_PATH = "/Users/isaacbreen/Projects2/grammars2024/.cache/test_vocabs/constraint_js.json"
# FILE_PATH = "/Users/isaacbreen/Projects2/grammars2024/.cache/test_vocabs/example_diff_constraint.json"


# --- Generic Helper Functions ---

def set_to_ranges(s):
    """Converts a set of integers to a sorted list of [start, end] ranges."""
    if not s: return []
    sorted_tokens = sorted(list(s))
    ranges, start, end = [], sorted_tokens[0], sorted_tokens[0] + 1
    for token in sorted_tokens[1:]:
        if token == end: end += 1
        else:
            ranges.append([start, end]); start, end = token, token + 1
    ranges.append([start, end])
    return ranges

def set_to_u8set_obj(s):
    """Converts a set of integers (0-255) to a compact U8Set JSON object."""
    if not s: return []
    sorted_bytes = sorted(list(s))
    u8set, start, end = [], -1, -1
    for b in sorted_bytes:
        if start == -1: start, end = b, b
        elif b == end + 1: end = b
        else:
            u8set.append(start if start == end else [start, end]); start, end = b, b
    if start != -1: u8set.append(start if start == end else [start, end])
    return u8set

def calculate_bitset_json_size(s, max_value):
    """Calculates JSON size for a set encoded as a base64 bitmask."""
    if not s or max_value < 0: return len(json.dumps(""))
    num_bytes = math.ceil((max_value + 1) / 8)
    bitmask = bytearray(num_bytes)
    for item_id in s:
        byte_index, bit_index = divmod(item_id, 8)
        bitmask[byte_index] |= (1 << bit_index)
    return len(json.dumps(base64.b64encode(bitmask).decode('ascii')))

def calculate_pooled_size(instances):
    """Calculates the total size of a pool and its references."""
    if not instances: return 0
    counts = Counter(instances)
    num_unique = len(counts)
    # Size of the pool (the array of unique component objects)
    pool_size = sum(len(unique_obj) for unique_obj in counts.keys())
    json_array_overhead = len(json.dumps([0] * num_unique)) - num_unique
    pool_size += json_array_overhead
    # Size of all integer references
    obj_to_index = {obj: i for i, obj in enumerate(counts.keys())}
    references_size = sum(len(str(obj_to_index[inst])) for inst in instances)
    return pool_size + references_size

# --- Weight-Specific Helper ---

def parse_weight_to_set(data, format_type):
    """Converts a JSON weight object into a standard Python set of integers."""
    s = set()
    if not data: return frozenset()
    if format_type == 'SimpleBitset':
        for start, end in data: s.update(range(start, end))
    elif format_type == 'LLMTokenBV':
        for i in range(0, len(data), 2): s.update(range(data[i], data[i+1]))
    return frozenset(s)


# --- Analysis Core Functions ---

def analyze_weights(data):
    """Analyzes the space efficiency of all weights in the GrammarConstraint."""
    all_weights = []
    max_token_id = 0

    # Extract weights from all known locations
    if 'precomputed4' in data and 'states' in data['precomputed4']:
        for state in data['precomputed4']['states']:
            for key in ['final_weight', 'state_weight']:
                if state.get(key):
                    w_obj = state[key]
                    w_set = parse_weight_to_set(w_obj, 'SimpleBitset')
                    all_weights.append({'original_obj': w_obj, 'set': w_set, 'format': 'SimpleBitset'})
                    if w_set: max_token_id = max(max_token_id, max(w_set))
            for _, w_obj in state.get('trans_weights', []):
                w_set = parse_weight_to_set(w_obj, 'SimpleBitset')
                all_weights.append({'original_obj': w_obj, 'set': w_set, 'format': 'SimpleBitset'})
                if w_set: max_token_id = max(max_token_id, max(w_set))

    for _, terminal_map in data.get('possible_matches', []):
        for _, w_obj in terminal_map:
            w_set = parse_weight_to_set(w_obj, 'LLMTokenBV')
            all_weights.append({'original_obj': w_obj, 'set': w_set, 'format': 'LLMTokenBV'})
            if w_set: max_token_id = max(max_token_id, max(w_set))

    for _, w_obj in data.get('precompute4_vocab', []):
        w_set = parse_weight_to_set(w_obj, 'LLMTokenBV')
        all_weights.append({'original_obj': w_obj, 'set': w_set, 'format': 'LLMTokenBV'})
        if w_set: max_token_id = max(max_token_id, max(w_set))

    if not all_weights: return None

    original_total_size = sum(len(json.dumps(w['original_obj'])) for w in all_weights)

    # Use a dictionary to count occurrences of each unique weight set
    unique_weights = defaultdict(lambda: {'count': 0})
    for w in all_weights:
        unique_weights[w['set']]['count'] += 1
        # Store the format type, ensuring consistency if a set appears in multiple formats
        if 'format' not in unique_weights[w['set']]:
            unique_weights[w['set']]['format'] = w['format']

    # Pre-calculate the size of each unique weight in different encodings
    for s, info in unique_weights.items():
        is_simple = info['format'] == 'SimpleBitset'
        ranges = set_to_ranges(s)
        range_obj = ranges if is_simple else [i for r in ranges for i in r]
        size_range = len(json.dumps(range_obj))
        size_bitset = calculate_bitset_json_size(s, max_token_id)
        info['sizes'] = {'range': size_range, 'bitset': size_bitset, 'hybrid': min(size_range, size_bitset)}

    # Calculate totals for non-pooled strategies
    total_size_bitset = sum(w['sizes']['bitset'] * w['count'] for w in unique_weights.values())
    total_size_hybrid = sum(w['sizes']['hybrid'] * w['count'] for w in unique_weights.values())

    # For pooling, we need to get the pre-calculated size for each individual weight instance
    # and then pass the list of these sizes (as canonical strings) to the pooling calculator.

    # Create lists of the JSON-stringified *unique objects* that will be in the pool
    unique_range_objects = [json.dumps(unique_weights[s]['sizes']['range']) for s in unique_weights]
    unique_bitset_objects = [json.dumps(unique_weights[s]['sizes']['bitset']) for s in unique_weights]
    unique_hybrid_objects = [json.dumps(unique_weights[s]['sizes']['hybrid']) for s in unique_weights]

    # Create a map from a weight set to its future index in the pool
    set_to_index = {s: i for i, s in enumerate(unique_weights.keys())}

    # Calculate the size of all integer references
    references_size = sum(len(str(set_to_index[w['set']])) for w in all_weights)

    # Calculate the size of the pool itself (unique objects + JSON overhead)
    def get_pool_storage_size(unique_objects):
        if not unique_objects: return 0
        num_unique = len(unique_objects)
        json_array_overhead = len(json.dumps([0] * num_unique)) - num_unique
        return sum(len(obj) for obj in unique_objects) + json_array_overhead

    pooled_size_range = get_pool_storage_size(unique_range_objects) + references_size
    pooled_size_bitset = get_pool_storage_size(unique_bitset_objects) + references_size
    pooled_size_hybrid = get_pool_storage_size(unique_hybrid_objects) + references_size

    return {
        'component_name': 'Weights (SimpleBitset & LLMTokenBV)',
        'original_size': original_total_size,
        'stats': {
            'Total Instances': len(all_weights),
            'Unique Instances': len(unique_weights),
            'Max Token ID': max_token_id
        },
        'strategies': {
            'Bitset-based (No Pooling)': total_size_bitset,
            'Hybrid (No Pooling)': total_size_hybrid,
            'Pooled Range-based': pooled_size_range,
            'Pooled Bitset-based': pooled_size_bitset,
            'Pooled Hybrid (Best Method)': pooled_size_hybrid
        }
    }

def analyze_tokenizer_dfa(data):
    """Analyzes the space efficiency of the tokenizer DFA with multiple strategies."""
    tokenizer_data = data.get('tokenizer')
    if not tokenizer_data or 'dfa' not in tokenizer_data: return None

    original_tokenizer_size = len(json.dumps(tokenizer_data))
    dfa_states = tokenizer_data.get('dfa', {}).get('states', [])
    num_states = len(dfa_states)
    if num_states == 0: return None

    # --- Pre-computation Step: Re-encode all states and components ---
    max_group_id = -1
    for state in dfa_states:
        all_ids = set(state.get('finalizers', [])).union(set(state.get('possible_future_group_ids', [])))
        if all_ids: max_group_id = max(max_group_id, max(all_ids))

    state_component_data = []
    for state in dfa_states:
        components = {}
        # Transitions
        grouped_by_dest = defaultdict(set)
        for byte, dest in state.get('transitions', []): grouped_by_dest[dest].add(byte)
        components['transitions'] = sorted([[dest, set_to_u8set_obj(bs)] for dest, bs in grouped_by_dest.items()])
        # Sets (Finalizers, etc.)
        for key in ['finalizers', 'possible_future_group_ids']:
            s = set(state.get(key, []))
            size_range = len(json.dumps(set_to_ranges(s)))
            size_bitset = calculate_bitset_json_size(s, max_group_id)
            components[key] = set_to_ranges(s) if size_range <= size_bitset else f"bitset_placeholder_{key}"
        # group_id_to_u8set
        components['group_id_to_u8set'] = state.get('group_id_to_u8set', [])
        state_component_data.append(components)

    # --- Calculation Step: Evaluate each strategy ---
    strategies = {}
    dfa_shell_size = len(json.dumps({k: v for k, v in tokenizer_data['dfa'].items() if k != 'states'})) - 1

    # Strategy 1: Re-encoded Only (no pooling)
    size = sum(len(json.dumps(s)) for s in state_component_data)
    strategies['Re-encoded Only'] = size + dfa_shell_size

    # Strategy 2: State-level Pooling
    canonical_states = [json.dumps(s) for s in state_component_data]
    strategies['State-level Pooling'] = calculate_pooled_size(canonical_states) + dfa_shell_size

    # Strategy 3: Selective Pooling (Sets Only)
    size = 0
    pooled_keys = ['finalizers', 'possible_future_group_ids']
    inline_keys = ['transitions', 'group_id_to_u8set']
    for key in pooled_keys:
        instances = [json.dumps(s[key]) for s in state_component_data]
        size += calculate_pooled_size(instances)
    for key in inline_keys:
        size += sum(len(json.dumps(s[key])) for s in state_component_data)
    strategies['Selective Pooling (Sets Only)'] = size + dfa_shell_size

    # Strategy 4: Full Component Pooling
    size = 0
    component_stats = {}
    for key in state_component_data[0].keys():
        instances = [json.dumps(s[key]) for s in state_component_data]
        size += calculate_pooled_size(instances)
        component_stats[key] = {'total': len(instances), 'unique': len(set(instances))}
    strategies['Full Component Pooling (Best Method)'] = size + dfa_shell_size

    return {
        'component_name': 'Tokenizer DFA',
        'original_size': original_tokenizer_size,
        'stats': {
            'DFA States': num_states,
            'Max Group ID': max_group_id,
            'Component Uniqueness': component_stats
        },
        'strategies': strategies
    }


# --- Reporting ---

def print_analysis_report(results):
    """Prints a formatted report from an analysis result dictionary."""
    if not results: return
    print(f"--- Analysis for: {results['component_name']} ---")
    for key, value in results['stats'].items():
        if key == 'Component Uniqueness':
            print("Component Uniqueness:")
            for comp, counts in value.items():
                print(f"  - {comp:<25} {counts['unique']:,} unique of {counts['total']:,} total")
        else: print(f"{key}: {value:,}")

    print("\n" + f"{'Strategy':<45} | {'Total Size':>15} | {'Improvement vs Original':>32}")
    print("-" * 100)
    original_size = results['original_size']
    print(f"{'Original':<45} | {original_size:>15,} bytes |")

    # Sort strategies by size for clearer presentation
    sorted_strategies = sorted(results['strategies'].items(), key=lambda item: item[1])
    for name, new_size in sorted_strategies:
        reduction = original_size - new_size
        percent = (reduction / original_size * 100) if original_size > 0 else 0
        print(f"{name:<45} | {new_size:>15,} bytes | Reduction: {reduction:>12,} bytes ({percent:.2f}%)")
    print("\n")


# --- Main Execution ---

def main(file_path):
    """Main function to load data and run all analyses."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'"); return

    print(f"Analyzing file: {file_path}\n" + "=" * 100)
    try:
        with open(file_path, 'r') as f: data = json.load(f)
        original_total_file_size = os.path.getsize(file_path)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing JSON file: {e}"); return

    weights_results = analyze_weights(data)
    tokenizer_results = analyze_tokenizer_dfa(data)

    print_analysis_report(weights_results)
    print_analysis_report(tokenizer_results)

    print("=" * 100 + "\n--- Overall File Size Impact Summary ---\n" + "=" * 100)
    if not weights_results or not tokenizer_results:
        print("Could not generate a full summary due to missing data."); return

    original_weights_size = weights_results['original_size']
    original_tokenizer_size = tokenizer_results['original_size']
    best_weights_size = weights_results['strategies']['Pooled Hybrid (Best Method)']
    best_tokenizer_size = tokenizer_results['strategies']['Full Component Pooling (Best Method)']

    size_of_other_json_parts = original_total_file_size - original_weights_size - original_tokenizer_size
    new_total_file_size = size_of_other_json_parts + best_weights_size + best_tokenizer_size
    total_reduction = original_total_file_size - new_total_file_size
    total_percent = (total_reduction / original_total_file_size * 100) if original_total_file_size > 0 else 0

    print(f"{'Original Total File Size:':<35} {original_total_file_size:>15,} bytes")
    print(f"{'Estimated New File Size (Best Methods):':<35} {new_total_file_size:>15,} bytes")
    print("-" * 52)
    print(f"{'Estimated Total Reduction:':<35} {total_reduction:>15,} bytes ({total_percent:.2f}%)")
    print("=" * 100)

if __name__ == "__main__":
    main(FILE_PATH)