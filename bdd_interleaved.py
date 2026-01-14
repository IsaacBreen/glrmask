
import json
import sys
import time
from dd import autoref as _bdd

# --- Configuration ---
NUM_TSIDS = 4476
TOKEN_BITS = 12  # Covers up to 4096
TSID_BITS = 13   # Covers up to 8192

# --- BDD Setup ---
bdd = _bdd.BDD()

# Define interleaved variables (MSB first)
# tsid_12, tok_11, tsid_11, tok_10, ...
ordered_vars = []
for i in range(12, -1, -1): # 12 down to 0
    # TSID bit i
    ordered_vars.append(f'tsid_{i}')
    # Token bit i (if exists)
    if i < TOKEN_BITS:
        ordered_vars.append(f'tok_{i}')

bdd.declare(*ordered_vars)
# bdd.configure(reordering=False) # Trust our ordering

print(f"Declared {len(ordered_vars)} variables with interleaved ordering.")
# print(f"Order: {ordered_vars}")

# --- Helper Functions ---

def build_interval(prefix, bits, low, high):
    """
    Build BDD for 'low <= var <= high'.
    Uses dd's add_expr with binary encoding logic is messy.
    Better: Logic using GEQ/LEQ.
    But dd doesn't builtin integer comparators easily.
    We'll implement a recursive 'range_bdd' helper.
    """
    # Logic: OR of all values? identifying ranges in bits?
    # Standard approach: Common prefix + suffix logic.
    # Recursively:
    # range(bit_idx, current_val, low, high)
    # If current_val's range [min, max] is widely inside [low, high], return TRUE.
    # If disjoint, return FALSE.
    # Else, split on bit.
    
    cache = {}
    
    def _recruit(bit_idx, current_val):
        state = (bit_idx, current_val)
        if state in cache: return cache[state]
        
        # Determine range covered by this path so far
        # Assuming remaining bits are all 0..all 1
        shift = bit_idx + 1
        min_v = current_val
        max_v = current_val + (1 << shift) - 1
        
        # 1. Provide range is fully inside [low, high] -> True
        if min_v >= low and max_v <= high:
            return bdd.true
            
        # 2. Provide range is fully outside -> False
        if max_v < low or min_v > high:
            return bdd.false
            
        # 3. Intersection -> Split on current bit
        # Next bit is bit_idx - 1
        # Variable name: f'{prefix}_{bit_idx}'
        var = f'{prefix}_{bit_idx}'
        
        # Case 0: bit is 0
        low_branch = _recruit(bit_idx - 1, current_val)
        
        # Case 1: bit is 1. Value increases by 2^bit_idx
        high_branch = _recruit(bit_idx - 1, current_val | (1 << bit_idx))
        
        node = bdd.ite(bdd.var(var), high_branch, low_branch)
        cache[state] = node
        return node

    return _recruit(bits - 1, 0)

def make_rectangle(tok_min, tok_max, tsid_min, tsid_max):
    """Build BDD for (tok in range) AND (tsid in range)."""
    # Since var ordering is interleaved, we can't just build separate BDDs 
    # and AND them strictly if we want to guide size, but bdd.apply('and', ...) handles it.
    # ANDing separate interval BDDs will enforce the constraints regardless of ordering.
    
    t_bdd = build_interval('tok', TOKEN_BITS, tok_min, tok_max)
    s_bdd = build_interval('tsid', TSID_BITS, tsid_min, tsid_max)
    return bdd.apply('&', t_bdd, s_bdd)

def ranges_to_rects(ranges):
    """Decompose 1D ranges into 2D rectangles."""
    rects = []
    for s, e in ranges:
        # Clip max
        if s > 18446744073709551614: continue # usize max
        if s > 10_000_000_000: continue # clipped earlier logic
        
        tok_s, tsid_s = divmod(s, NUM_TSIDS)
        tok_e, tsid_e = divmod(e, NUM_TSIDS)
        
        if tok_s == tok_e:
            rects.append((tok_s, tok_s, tsid_s, tsid_e))
        else:
            rects.append((tok_s, tok_s, tsid_s, NUM_TSIDS - 1))
            if tok_s + 1 <= tok_e - 1:
                rects.append((tok_s + 1, tok_e - 1, 0, NUM_TSIDS - 1))
            rects.append((tok_e, tok_e, 0, tsid_e))
    return rects

# --- Benchmarking ---

with open('range_weights_terminal_dwa.json') as f:
    weights = json.load(f)

test_indices = [977, 400] # Cartesian, Banded
results = {}

print("\n--- Testing Specific Weights ---")
for idx in test_indices:
    raw_ranges = weights[idx]
    if len(raw_ranges) == 0: continue
    
    rects = ranges_to_rects(raw_ranges)
    print(f"Weight {idx}: {len(raw_ranges)} ranges -> {len(rects)} rectangles")
    
    start = time.time()
    # Union all rectangles
    w_bdd = bdd.false
    for i, (t1, t2, s1, s2) in enumerate(rects):
        r_bdd = make_rectangle(t1, t2, s1, s2)
        w_bdd = bdd.apply('|', w_bdd, r_bdd)
        
        # Cleanup intermediate
        if i % 10 == 0:
             bdd.collect_garbage()
             
    size = len(w_bdd)
    dt = time.time() - start
    print(f"  -> BDD Size: {size} nodes")
    print(f"  -> Time: {dt:.4f}s")
    print(f"  -> Compression: {len(raw_ranges)/size:.2f}x")
    results[idx] = size

print("\n--- Extrapolating to Full Set (Sample 20) ---")
# Sample 20 random normal weights
import random
random.seed(42)
sample_indices = random.sample(range(len(weights)), 20)
# Ensure we include complex ones
sample_indices = list(set(sample_indices + [977, 400]))

total_bdd = 0
total_ranges = 0

for idx in sample_indices:
    raw_ranges = weights[idx]
    if not raw_ranges: continue
    
    # Clip huge weights > 10M range
    if any(s > 10_000_000 for s,e in raw_ranges):
        continue
        
    rects = ranges_to_rects(raw_ranges)
    w_bdd = bdd.false
    try:
        for t1, t2, s1, s2 in rects:
            r_bdd = make_rectangle(t1, t2, s1, s2)
            w_bdd = bdd.apply('|', w_bdd, r_bdd)
    except:
        print(f"Error building weight {idx}")
        continue
        
    sz = len(w_bdd)
    total_bdd += sz
    total_ranges += len(raw_ranges)
    print(f"W{idx}: {len(raw_ranges)}r -> {sz}n")

print(f"\nSample Total: {total_ranges} ranges -> {total_bdd} nodes")
print(f"Compression: {total_ranges/total_bdd:.2f}x")
