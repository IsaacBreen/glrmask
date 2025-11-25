import json
import os
import base64
import math
from collections import defaultdict, Counter
import copy

# --- Configuration ---
# The path to your grammar constraint file.
FILE_PATH = "/Users/isaacbreen/Projects2/grammars2024/.cache/test_vocabs/constraint_js.json"


# --- Generic Set/Encoding Helper Functions ---

def set_to_ranges(s):
    """Converts a set of integers to a sorted list of [start, end] ranges."""
    if not s:
        return []

    sorted_tokens = sorted(list(s))
    ranges = []
    start = sorted_tokens[0]
    end = start + 1

    for token in sorted_tokens[1:]:
        if token == end:
            end += 1
        else:
            ranges.append([start, end])
            start = token
            end = token + 1
    ranges.append([start, end])
    return ranges

def set_to_u8set_obj(s):
    """Converts a set of integers (0-255) to a compact U8Set JSON object."""
    if not s:
        return []

    sorted_bytes = sorted(list(s))
    u8set = []
    start = -1
    end = -1

    for b in sorted_bytes:
        if start == -1:
            start = b
            end = b
        elif b == end + 1:
            end = b
        else:
            if start == end:
                u8set.append(start)
            else:
                u8set.append([start, end])
            start = b
            end = b

    if start != -1:
        if start == end:
            u8set.append(start)
        else:
            u8set.append([start, end])

    return u8set

def calculate_bitset_json_size(s, max_value):
    """Calculates JSON size for a set encoded as a base64 bitmask."""
    if not s or max_value < 0:
        return len(json.dumps(""))

    num_bytes = math.ceil((max_value + 1) / 8)
    bitmask = bytearray(num_bytes)

    for item_id in s:
        byte_index = item_id // 8
        bit_index = item_id % 8
        bitmask[byte_index] |= (1 << bit_index)

    b64_encoded = base64.b64encode(bitmask).decode('ascii')
    return len(json.dumps(b64_encoded))


# --- Weight-Specific Helper ---

def parse_weight_to_set(data, format_type):
    """Converts a JSON weight object into a standard Python set of integers."""
    s = set()
    if not data:
        return frozenset()

    if format_type == 'SimpleBitset': # [[start, end], ...]
        for start, end in data:
            s.update(range(start, end))
    elif format_type == 'LLMTokenBV': # [start, end, start, end, ...]
        for i in range(0, len(data), 2):
            start, end = data[i], data[i+1]
            s.update(range(start, end))
    return frozenset(s)


# --- Analysis Core Functions ---

def analyze_weights(data):
    """Analyzes the space efficiency of all weights in the GrammarConstraint."""
    all_weights = []
    max_token_id = 0

    # Extract weights from all known locations
    locations = {
        'precomputed4.states': ('SimpleBitset', ['final_weight', 'state_weight', ('trans_weights', 1)]),
        'possible_matches': ('LLMTokenBV', [(1, 1)]),
        'precompute4_vocab': ('LLMTokenBV', [1])
    }

    # A bit of metaprogramming to traverse the structure based on the map above
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

    if not all_weights:
        return None

    original_total_size = sum(len(json.dumps(w['original_obj'])) for w in all_weights)

    unique_weights = {}
    for w in all_weights:
        if w['set'] not in unique_weights:
            unique_weights[w['set']] = {'count': 0, 'format': w['format']}
        unique_weights[w['set']]['count'] += 1

    for s, info in unique_weights.items():
        size_range = len(json.dumps(set_to_ranges(s) if info['format'] == 'SimpleBitset' else [i for r in set_to_ranges(s) for i in r]))
        size_bitset = calculate_bitset_json_size(s, max_token_id)
        info['sizes'] = {'range': size_range, 'bitset': size_bitset, 'hybrid': min(size_range, size_bitset)}

    # Calculate totals
    total_size_bitset = sum(w['sizes']['bitset'] * w['count'] for w in unique_weights.values())
    total_size_hybrid = sum(w['sizes']['hybrid'] * w['count'] for w in unique_weights.values())

    pool_size_range = sum(w['sizes']['range'] for w in unique_weights.values())
    pool_size_bitset = sum(w['sizes']['bitset'] for w in unique_weights.values())
    pool_size_hybrid = sum(w['sizes']['hybrid'] for w in unique_weights.values())

    num_unique = len(unique_weights)
    json_array_overhead = len(json.dumps([0] * num_unique)) - num_unique if num_unique > 0 else 0

    set_to_index = {s: i for i, s in enumerate(unique_weights.keys())}
    references_total_size = sum(len(str(set_to_index[w['set']])) for w in all_weights)

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
            'Pooled Range-based': pool_size_range + json_array_overhead + references_total_size,
            'Pooled Bitset-based': pool_size_bitset + json_array_overhead + references_total_size,
            'Pooled Hybrid (Best Method)': pool_size_hybrid + json_array_overhead + references_total_size
        }
    }

def analyze_tokenizer_dfa(data):
    """Analyzes the space efficiency of the tokenizer DFA."""
    tokenizer_data = data.get('tokenizer')
    if not tokenizer_data or 'dfa' not in tokenizer_data:
        return None

    original_tokenizer_size = len(json.dumps(tokenizer_data))
    dfa_states = tokenizer_data.get('dfa', {}).get('states', [])
    num_states = len(dfa_states)

    if num_states == 0:
        return None

    max_group_id = -1
    for state in dfa_states:
        all_ids = set(state.get('finalizers', [])).union(set(state.get('possible_future_group_ids', [])))
        if all_ids: max_group_id = max(max_group_id, max(all_ids))

    component_instances = defaultdict(list)
    total_size_reencoded = 0

    for state in dfa_states:
        reencoded_state_sizes = {}

        # Transitions
        grouped_by_dest = defaultdict(set)
        for byte, dest in state.get('transitions', []): grouped_by_dest[dest].add(byte)
        new_trans = sorted([[dest, set_to_u8set_obj(byte_set)] for dest, byte_set in grouped_by_dest.items()])
        reencoded_state_sizes['transitions'] = len(json.dumps(new_trans))
        component_instances['transitions'].append(json.dumps(new_trans))

        # Sets (Finalizers, etc.)
        for key in ['finalizers', 'possible_future_group_ids']:
            original_set = set(state.get(key, []))
            size_range = len(json.dumps(set_to_ranges(original_set)))
            size_bitset = calculate_bitset_json_size(original_set, max_group_id)
            reencoded_state_sizes[key] = min(size_range, size_bitset)
            component_instances[key].append(json.dumps(set_to_ranges(original_set)))

        # group_id_to_u8set
        gu_map = state.get('group_id_to_u8set', [])
        reencoded_state_sizes['group_id_to_u8set'] = len(json.dumps(gu_map))
        component_instances['group_id_to_u8set'].append(json.dumps(gu_map))

        total_size_reencoded += sum(reencoded_state_sizes.values())

    total_size_reencoded += len(json.dumps([0] * num_states)) - num_states

    pooled_total_size = 0
    component_stats = {}
    for key, instances in component_instances.items():
        counts = Counter(instances)
        component_stats[key] = {'total': len(instances), 'unique': len(counts)}
        pool_size = sum(len(obj) for obj in counts.keys())
        num_unique = len(counts)
        json_array_overhead = len(json.dumps([0] * num_unique)) - num_unique if num_unique > 0 else 0
        obj_to_index = {obj: i for i, obj in enumerate(counts.keys())}
        references_size = sum(len(str(obj_to_index[inst])) for inst in instances)
        pooled_total_size += pool_size + json_array_overhead + references_size

    dfa_shell_size = len(json.dumps({k: v for k, v in tokenizer_data['dfa'].items() if k != 'states'})) - 1

    return {
        'component_name': 'Tokenizer DFA',
        'original_size': original_tokenizer_size,
        'stats': {
            'DFA States': num_states,
            'Max Group ID': max_group_id,
            'Component Uniqueness': component_stats
        },
        'strategies': {
            'Re-encoded (Grouped Transitions, Hybrid Sets)': total_size_reencoded + dfa_shell_size,
            'Component Pooling (Best Method)': pooled_total_size + dfa_shell_size
        }
    }


# --- Reporting ---

def print_analysis_report(results):
    """Prints a formatted report from an analysis result dictionary."""
    if not results:
        return

    print(f"--- Analysis for: {results['component_name']} ---")

    # Print stats
    for key, value in results['stats'].items():
        if key == 'Component Uniqueness':
            print("Component Uniqueness:")
            for comp, counts in value.items():
                print(f"  - {comp:<25} {counts['unique']:,} unique of {counts['total']:,} total")
        else:
            print(f"{key}: {value:,}")

    print("\n" + f"{'Strategy':<45} | {'Total Size':>15} | {'Improvement vs Original':>32}")
    print("-" * 100)

    original_size = results['original_size']
    print(f"{'Original':<45} | {original_size:>15,} bytes |")

    for name, new_size in results['strategies'].items():
        reduction = original_size - new_size
        percent = (reduction / original_size * 100) if original_size > 0 else 0
        print(f"{name:<45} | {new_size:>15,} bytes | Reduction: {reduction:>12,} bytes ({percent:.2f}%)")
    print("\n")


# --- Main Execution ---

def main(file_path):
    """Main function to load data and run all analyses."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Analyzing file: {file_path}")
    print("=" * 100)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        original_total_file_size = os.path.getsize(file_path)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    # Run analyses
    weights_results = analyze_weights(data)
    tokenizer_results = analyze_tokenizer_dfa(data)

    # Print individual reports
    print_analysis_report(weights_results)
    print_analysis_report(tokenizer_results)

    # Print final summary
    print("=" * 100)
    print("--- Overall File Size Impact Summary ---")
    print("=" * 100)

    if not weights_results or not tokenizer_results:
        print("Could not generate a full summary due to missing data.")
        return

    original_weights_size = weights_results['original_size']
    original_tokenizer_size = tokenizer_results['original_size']

    best_weights_size = weights_results['strategies']['Pooled Hybrid (Best Method)']
    best_tokenizer_size = tokenizer_results['strategies']['Component Pooling (Best Method)']

    # Calculate the size of the file *excluding* the parts we analyzed
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