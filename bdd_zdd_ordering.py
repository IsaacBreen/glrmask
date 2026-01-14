
import json
import time
import sys

# Custom BDD and ZDD Implementation for Benchmarking
# (Simple recursive implementation with unique table)

class NodeManager:
    def __init__(self, mode='bdd', order=None):
        self.mode = mode # 'bdd' or 'zdd'
        self.order = order # list of var names
        self.var_to_level = {v: i for i, v in enumerate(order)}
        self.nodes = {0: (None, None, None), 1: (None, None, None)} # id -> (var, lo, hi)
        self.unique = {(None, None, None): 0, (None, None, None): 1} # (var, lo, hi) -> id
        self.next_id = 2
        self.op_cache = {}

    def get_node(self, var, lo, hi):
        # BDD Reduction: if lo == hi, return lo
        if self.mode == 'bdd' and lo == hi:
            return lo
        
        # ZDD Reduction: if hi == 0 (False), return lo
        # (Only if not terminal)
        if self.mode == 'zdd' and hi == 0:
            return lo
            
        key = (var, lo, hi)
        if key in self.unique:
            return self.unique[key]
        
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = key
        self.unique[key] = nid
        return nid

    def top_var(self, n1, n2):
        if n1 <= 1 and n2 <= 1: return None
        v1 = self.nodes[n1][0]
        v2 = self.nodes[n2][0]
        if n1 <= 1: return v2
        if n2 <= 1: return v1
        
        lvl1 = self.var_to_level[v1]
        lvl2 = self.var_to_level[v2]
        return v1 if lvl1 < lvl2 else v2

    def apply(self, op, n1, n2):
        # Base cases
        if op == 'or':
            if n1 == 1 or n2 == 1: return 1
            if n1 == 0: return n2
            if n2 == 0: return n1
            if n1 == n2: return n1
        elif op == 'and':
            if n1 == 0 or n2 == 0: return 0
            if n1 == 1: return n2
            if n2 == 1: return n1
            if n1 == n2: return n1
            
        key = (op, n1, n2) if n1 <= n2 else (op, n2, n1)
        if key in self.op_cache: return self.op_cache[key]
        
        top = self.top_var(n1, n2)
        
        # Co-factors
        def cofactor(n, v):
            if n <= 1: return n
            n_var, n_lo, n_hi = self.nodes[n]
            if n_var == v:
                return n_lo, n_hi
            # If n_var is below v, then n doesn't depend on v
            # Wait, verify ordering
            if self.var_to_level[n_var] > self.var_to_level[v]:
                return n, n # In BDD, n doesn't depend on v
            # If n_var is ABOVE v, shouldn't happen if we picked top
            return n, n # Default

        # ZDD Cofactor is different? 
        # "ZDD semantics for boolean operations are slightly different"
        # Actually for set operations (Union/Intersect), the recursion is same structure
        # provided we interpret nodes as subsets.
        # But we are doing Characteristic Function logic (0/1).
        # Standard Apply logic works for BDD/ZDD graph construction if using 0/1 terminals.
        
        l1, h1 = cofactor(n1, top)
        l2, h2 = cofactor(n2, top)
        
        res_lo = self.apply(op, l1, l2)
        res_hi = self.apply(op, h1, h2)
        
        res = self.get_node(top, res_lo, res_hi)
        self.op_cache[key] = res
        return res
        
    def build_interval(self, prefix, bit_count, low, high):
        """Builds formula for prefix <val> in [low, high]"""
        # We need to construct this carefully using APPLY.
        # Direct construction is hard with reduction rules.
        # Let's iterate: Or of all values? Slow.
        # Recursive construction is best.
        
        # Define variable indices for this prefix
        # We assume they are in self.order properly
        # We'll just generate the boolean formula: (val >= low) & (val <= high)
        
        cache = {}
        def rec(idx, current_val):
            # idx: current bit index (from bit_count-1 down to 0)
            state = (idx, current_val)
            if state in cache: return cache[state]
            
            # Helper to generate ranges
            bit = idx
            val_min = current_val
            val_max = current_val + (1 << (bit + 1)) - 1
            
            if val_min >= low and val_max <= high: return 1
            if val_max < low or val_min > high: return 0
            if bit < 0: return 0 # Should be covered above
            
            var = f"{prefix}_{bit}"
            
            # Hi branch (bit=1)
            hi_node = rec(idx - 1, current_val | (1 << bit))
            # Lo branch (bit=0)
            lo_node = rec(idx - 1, current_val)
            
            return self.get_node(var, lo_node, hi_node)
            
        return rec(bit_count - 1, 0)


# --- Benchmark Setup ---
NUM_TSIDS = 4476
TOKEN_BITS = 12
TSID_BITS = 13

def run_benchmark(weights, ordering_type='interleaved', mode='bdd'):
    # Define Order
    order = []
    if ordering_type == 'interleaved':
        for i in range(12, -1, -1):
            order.append(f'tsid_{i}')
            if i < TOKEN_BITS: order.append(f'tok_{i}')
    else: # concatenated: All Tokens, then All TSIDs
        for i in range(TOKEN_BITS-1, -1, -1): order.append(f'tok_{i}')
        for i in range(TSID_BITS-1, -1, -1): order.append(f'tsid_{i}')
            
    mgr = NodeManager(mode, order)
    
    # Process Weights
    total_nodes = 0
    start_time = time.time()
    
    root_node = 0 # Empty
    
    # Just do Union of ALL weights to see shared size?
    # Or sum of individual sizes?
    # User cares about "Total compressed size". Shared makes sense.
    
    # But doing union of all 1120 weights is slow in python.
    # Let's measure unique nodes across individual BDDs.
    
    roots = []
    
    for w_idx, ranges in enumerate(weights):
        if not ranges: continue
        
        # Decompose to rects
        rects = []
        for s, e in ranges:
             if s > 10000000: continue
             e = min(e, 10000000)
             t_s, p_s = divmod(s, NUM_TSIDS)
             t_e, p_e = divmod(e, NUM_TSIDS)
             # ... rect splitting logic ...
             if t_s == t_e:
                 rects.append((t_s, t_s, p_s, p_e))
             else:
                 rects.append((t_s, t_s, p_s, NUM_TSIDS-1))
                 if t_s+1 <= t_e-1: rects.append((t_s+1, t_e-1, 0, NUM_TSIDS-1))
                 rects.append((t_e, t_e, 0, p_e))

        # Build BDD for weight
        w_node = 0
        for t1, t2, s1, s2 in rects:
            # Build Term
            t_tree = mgr.build_interval('tok', TOKEN_BITS, t1, t2)
            s_tree = mgr.build_interval('tsid', TSID_BITS, s1, s2)
            term = mgr.apply('and', t_tree, s_tree)
            w_node = mgr.apply('or', w_node, term)
            
        roots.append(w_node)
        
        if w_idx % 20 == 0:
            print(f"  Processed {w_idx} weights...", flush=True)
            
    dt = time.time() - start_time
    count = len(mgr.nodes) - 2
    return count, dt

# Load Data
with open('range_weights_terminal_dwa.json') as f:
    all_weights = json.load(f)

# Select Test Weights (977 + 400 + randoms)
sample_indices = [977, 400, 451, 859, 501] + list(range(100, 110))
sample_weights = [all_weights[i] for i in sample_indices]

print(f"Benchmarking on {len(sample_weights)} sample weights...")

configs = [
    ('interleaved', 'bdd'),
    ('concatenated', 'bdd'),
    ('interleaved', 'zdd'),
    ('concatenated', 'zdd')
]

for order, mode in configs:
    print(f"\n--- Testing {mode.upper()} with {order} ordering ---")
    nodes, dt = run_benchmark(sample_weights, order, mode)
    print(f"Nodes: {nodes}")
    print(f"Time: {dt:.2f}s")
    
