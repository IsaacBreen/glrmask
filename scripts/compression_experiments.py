import json
import os
import base64
import math
from collections import defaultdict, Counter
import copy

# --- Configuration ---
# The path to your grammar constraint file.
# The path to your grammar constraint file.
FILE_PATH = "json.json"
# FILE_PATH = "/Users/isaacbreen/Projects2/grammars2024/.cache/test_vocabs/constraint_js.json"
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

def parse_weight_to_set(data, format_type, pool=None):
    """Converts a JSON weight object into a standard Python set of integers."""
    s = set()
    if data is None: return frozenset()
    
    # Handle pooled indices
    if isinstance(data, int):
        if not pool: return frozenset()
        if format_type == 'SimpleBitset':
            # pool['weights'] maps id -> SimpleBitset data
            data = pool['weights'].get(str(data)) or pool['weights'].get(data)
        elif format_type == 'LLMTokenBV':
            # pool['hybrid_bitsets'] maps id -> HybridBitset data
            data = pool['hybrid_bitsets'].get(str(data)) or pool['hybrid_bitsets'].get(data)
        
        if data is None: return frozenset() # Should not happen if index is valid

    if format_type == 'SimpleBitset':
        # SimpleBitset is usually [start, end, start, end...] or similar?
        # Wait, SimpleBitset serialization in Rust might be different.
        # Let's assume it's list of ranges [start, end] based on previous code, 
        # BUT previous code said: "for start, end in data: s.update(range(start, end))"
        # This implies data is [[start, end], [start, end]]?
        # Let's check the data format.
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                 for start, end in data: s.update(range(start, end))
            else:
                 # Maybe flattened?
                 pass
    elif format_type == 'LLMTokenBV':
        # HybridBitset is [start, end, start, end...] flattened
        # Ranges are INCLUSIVE in Rust (start..=end)
        if isinstance(data, list):
            if len(data) % 2 != 0:
                # Malformed data, skip
                pass
            else:
                for i in range(0, len(data), 2): 
                    s.update(range(data[i], data[i+1] + 1)) # +1 because Rust ranges are inclusive
            
    return frozenset(s)

def load_pool(data):
    """Parses the 'pool' field from the JSON data."""
    pool = {'weights': {}, 'hybrid_bitsets': {}, 'dfa_states': {}}
    if 'pool' not in data: return None
    
    p = data['pool']
    
    # Helper to parse DedupValueMap
    def parse_map(map_data):
        res = {}
        if 'values' in map_data:
            for pair in map_data['values']:
                if len(pair) == 2:
                    id_val, val = pair
                    res[id_val] = val
                    res[str(id_val)] = val
        return res

    if 'weights' in p: pool['weights'] = parse_map(p['weights'])
    if 'hybrid_bitsets' in p: pool['hybrid_bitsets'] = parse_map(p['hybrid_bitsets'])
    if 'dfa_states' in p: pool['dfa_states'] = parse_map(p['dfa_states'])
    
    return pool


# --- Analysis Core Functions ---

def analyze_weights(data):
    """Analyzes the space efficiency of all weights in the GrammarConstraint."""
    pool = load_pool(data)
    all_weights = []
    max_token_id = 0

    # Extract weights from all known locations
    # 1. DWA (Pooled)
    if 'dwa' in data and 'states' in data['dwa']:
        # data['dwa']['states'] is a list of objects with indices
        for state in data['dwa']['states']:
            for key in ['final_weight', 'state_weight']:
                if state.get(key) is not None:
                    w_idx = state[key]
                    w_set = parse_weight_to_set(w_idx, 'SimpleBitset', pool)
                    all_weights.append({'is_pooled': True, 'index': w_idx, 'set': w_set, 'format': 'SimpleBitset'})
                    if w_set: max_token_id = max(max_token_id, max(w_set))
            
            # trans_weights is a map label -> index
            # It might be serialized as a dict (if keys are strings) or list of pairs (if keys are ints)
            tw = state.get('trans_weights', {})
            iterator = []
            if isinstance(tw, dict):
                iterator = tw.items()
            elif isinstance(tw, list):
                iterator = tw
            
            for _, w_idx in iterator:
                 w_set = parse_weight_to_set(w_idx, 'SimpleBitset', pool)
                 all_weights.append({'is_pooled': True, 'index': w_idx, 'set': w_set, 'format': 'SimpleBitset'})
                 if w_set: max_token_id = max(max_token_id, max(w_set))
    
    # 2. Possible Matches (Pooled)
    if 'possible_matches' in data:
        pm = data['possible_matches']
        iterator = []
        if isinstance(pm, dict):
            iterator = pm.items()
        elif isinstance(pm, list):
            iterator = pm
            
        for _, terminal_map in iterator:
            # terminal_map is BTreeMap<TerminalID, usize> -> list of pairs
            inner_iterator = []
            if isinstance(terminal_map, dict):
                inner_iterator = terminal_map.items()
            elif isinstance(terminal_map, list):
                inner_iterator = terminal_map
                
            for _, w_idx in inner_iterator:
                w_set = parse_weight_to_set(w_idx, 'LLMTokenBV', pool)
                all_weights.append({'is_pooled': True, 'index': w_idx, 'set': w_set, 'format': 'LLMTokenBV'})
                if w_set: max_token_id = max(max_token_id, max(w_set))

    # 3. Vocab (Not Pooled)
    # precompute4_vocab -> internal_to_original is a map index -> LLMTokenBV
    if 'vocab' in data and 'internal_to_original' in data['vocab']:
         # internal_to_original is a list of [k, v] pairs in JSON if it's a BTreeMap? 
         # Wait, Rust BTreeMap<usize, LLMTokenBV> serializes to JSON object if keys are strings, 
         # but keys are usize. JSON keys must be strings. 
         # Let's assume it's an object or check the file.
         # In the previous code it iterated over it.
         vocab_map = data['vocab']['internal_to_original']
         # If it's a dict
         if isinstance(vocab_map, dict):
             iterator = vocab_map.items()
         elif isinstance(vocab_map, list): # Array of pairs
             iterator = vocab_map
         else:
             iterator = []

         for _, w_obj in iterator:
            w_set = parse_weight_to_set(w_obj, 'LLMTokenBV', pool)
            all_weights.append({'is_pooled': False, 'original_obj': w_obj, 'set': w_set, 'format': 'LLMTokenBV'})
            if w_set: max_token_id = max(max_token_id, max(w_set))

    if not all_weights: return None

    # Calculate "Unpooled Size" (what it would be without pooling)
    unpooled_total_size = 0
    current_pooled_size = 0
    
    # For pooled items, unpooled size is size of the set serialized.
    # Current size is size of index.
    
    # For unpooled items (vocab), unpooled size is size of obj.
    # Current size is size of obj.

    for w in all_weights:
        # Re-construct the unpooled object to measure its size
        is_simple = w['format'] == 'SimpleBitset'
        ranges = set_to_ranges(w['set'])
        # SimpleBitset format: [[s,e], [s,e]]? Or flattened?
        # HybridBitset format: [s, e, s, e]
        
        if is_simple:
            # Approximation of SimpleBitset JSON
            obj = ranges 
        else:
            obj = [x for r in ranges for x in r]
            
        obj_size = len(json.dumps(obj))
        
        if w['is_pooled']:
            unpooled_total_size += obj_size
            current_pooled_size += len(str(w['index']))
        else:
            unpooled_total_size += len(json.dumps(w.get('original_obj', obj)))
            current_pooled_size += len(json.dumps(w.get('original_obj', obj)))

    # Add the size of the pool itself to the current size
    pool_size = 0
    if pool:
        if 'weights' in data.get('pool', {}):
            pool_size += len(json.dumps(data['pool']['weights']))
        if 'hybrid_bitsets' in data.get('pool', {}):
            pool_size += len(json.dumps(data['pool']['hybrid_bitsets']))

    current_total_size = current_pooled_size + pool_size

    return {
        'component_name': 'Weights (SimpleBitset & LLMTokenBV)',
        'original_size': unpooled_total_size, # This is the "Before Optimization" size
        'stats': {
            'Total Instances': len(all_weights),
            'Max Token ID': max_token_id,
            'Pool Size': pool_size
        },
        'strategies': {
            'Current Pooled Implementation': current_total_size
        }
    }

def analyze_tokenizer_dfa(data):
    """Analyzes the space efficiency of the tokenizer DFA with multiple strategies."""
    # Check for new pooled format first
    if 'tokenizer_dfa' in data:
        # New format
        # tokenizer_dfa has 'state_indices'
        indices = data['tokenizer_dfa']['state_indices']
        references_size = sum(len(str(i)) for i in indices)
        
        pool = load_pool(data)
        pool_size = 0
        if pool and 'dfa_states' in data.get('pool', {}):
             pool_size = len(json.dumps(data['pool']['dfa_states']))
             
        # Calculate current size (pooled)
        tokenizer_dfa_size = len(json.dumps(data['tokenizer_dfa']))
        current_size = pool_size + tokenizer_dfa_size
        
        # Estimate unpooled size
        # We need to reconstruct the states to estimate their unpooled size
        unpooled_size = 0
        if pool:
            for idx in indices:
                state = pool['dfa_states'].get(idx) or pool['dfa_states'].get(str(idx))
                if state:
                    unpooled_size += len(json.dumps(state))
        
        # Add shell overhead
        shell_size = len(json.dumps({k: v for k, v in data['tokenizer_dfa'].items() if k != 'state_indices'}))
        unpooled_size += shell_size
        
        strategies = {'Current Pooled Implementation': current_size}
        original_tokenizer_size = unpooled_size # Estimate
        
        return {
            'component_name': 'Tokenizer DFA',
            'original_size': original_tokenizer_size,
            'stats': {
                'DFA States': len(indices),
                'Pool Size (bytes)': pool_size,
            },
            'strategies': strategies
        }
    
    # Old format analysis
    tokenizer_data = data.get('tokenizer')
    if not tokenizer_data or 'dfa' not in tokenizer_data: 
        return None



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
    best_weights_size = weights_results['strategies']['Current Pooled Implementation']
    best_tokenizer_size = tokenizer_results['strategies']['Current Pooled Implementation']

    size_of_other_json_parts = original_total_file_size - original_weights_size - original_tokenizer_size
    new_total_file_size = size_of_other_json_parts + best_weights_size + best_tokenizer_size
    total_reduction = original_total_file_size - new_total_file_size
    total_percent = (total_reduction / original_total_file_size * 100) if original_total_file_size > 0 else 0

    print(f"{'Original Total File Size:':<35} {original_total_file_size:>15,} bytes")
    print(f"{'Estimated New File Size (Best Methods):':<35} {new_total_file_size:>15,} bytes")
    print("-" * 52)
    print(f"{'Estimated Total Reduction:':<35} {total_reduction:>15,} bytes ({total_percent:.2f}%)")
    print("=" * 100)

import sys
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(FILE_PATH)