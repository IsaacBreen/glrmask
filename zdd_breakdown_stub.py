
import json
import time

# --- Minimal Node Manager (Recap) ---
class NodeManager:
    def __init__(self, mode='bdd'):
        self.mode = mode 
        self.nodes = {0: (None, None, None), 1: (None, None, None)} 
        self.unique = {(None, None, None): 0, (None, None, None): 1}
        self.next_id = 2
        self.cache = {} # Operation cache

    def get_node(self, var, lo, hi):
        if self.mode == 'bdd' and lo == hi: return lo
        if self.mode == 'zdd' and hi == 0: return lo
        key = (var, lo, hi)
        if key in self.unique: return self.unique[key]
        
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = key
        self.unique[key] = nid
        return nid

    def apply_or(self, n1, n2, level):
        # Specific OR implementation for profiling
        # Assume variables are just integers 0..MAX_LEVEL
        # level: current variable index (from MAX down to 0)
        
        if n1 == 1 or n2 == 1: return 1
        if n1 == 0: return n2
        if n2 == 0: return n1
        if n1 == n2: return n1
        
        key = (n1, n2) if n1 < n2 else (n2, n1)
        if key in self.cache: return self.cache[key]
        
        # Determine top variable
        # But wait, to handle variable levels properly we need mapping.
        # Let's simplify: Assume ALL BDDs built on full variable set 0..24
        # Recursion depth = variable index.
        pass

# Simplified Approach: Use Build Interval logic which is cleaner
# We just want to measure size.
# Re-using the recursion from previous script but strictly per-weight.

def measure_weight(w, mode='bdd', shape='concatenated'):
    mgr = NodeManager(mode)
    NUM_TSIDS = 4476
    TOKEN_BITS = 12
    TSID_BITS = 13
    
    # 1. Setup Order
    if shape == 'concatenated':
        # Tok MSB..LSB, Tsid MSB..LSB
        # Vars 0..11 = Tok, 12..24 = Tsid
        pass
    
    # Actually, let's just use recursion depth.
    # Depth 24 down to 0.
    # Concatenated: 24..13=Tok, 12..0=Tsid
    
    memo = {}
    def build_interval(depth, low, high):
        state = (depth, low, high)
        if state in memo: return memo[state]
        
        # Total range covered by this depth
        span = 1 << (depth + 1)
        
        if low == 0 and high == span - 1:
            return 1 # True
        if low >= span or high < 0:
            return 0 # False
            
        # Split
        mid = 1 << depth
        
        # High branch (bit=1) -> range [mid, 2*mid - 1]
        # Map [low, high] to [low-mid, high-mid]
        hi_node = build_interval(depth - 1, max(0, low - mid), min(mid - 1, high - mid))
        
        # Low branch (bit=0) -> range [0, mid - 1]
        lo_node = build_interval(depth - 1, min(mid - 1, low), min(mid - 1, high))
        
        node = mgr.get_node(depth, lo_node, hi_node)
        memo[state] = node
        return node

    # For Concatenated, we treat (tok, tsid) as one 25-bit integer:
    # val = (tok << 13) | tsid
    
    w_node = 0
    
    # To Union properly without implementing full Apply:
    # Just build the full Set of 25-bit integers and run BuildInterval on the SET?
    # No, that's hard.
    # Better: Use the `apply` logic from previous script, applied PER WEIGHT.
    pass

# --- USING PREVIOUS LOGIC (Imported/Copied) ---
# To save context space, I will re-implement the core measure function
# and run it on specific weights.

def analyze_breakdown():
    # Load
    with open('range_weights_terminal_dwa.json') as f:
        weights = json.load(f)
        
    # Weights to analyze
    targets = {
        977: "Cartesian (Sparse)",
        400: "Banded (Dense)",
        0: "Mixed",
        100: "Tiny"
    }
    
    for wid, desc in targets.items():
        w = weights[wid]
        # Rect decomposition
        rects = []
        NUM_TSIDS = 4476
        for s, e in w:
             if s > 10000000: continue
             e = min(e, 10000000)
             t_s, p_s = divmod(s, NUM_TSIDS)
             t_e, p_e = divmod(e, NUM_TSIDS)
             if t_s == t_e: rects.append((t_s, t_s, p_s, p_e))
             else:
                 rects.append((t_s, t_s, p_s, NUM_TSIDS-1))
                 if t_s+1 <= t_e-1: rects.append((t_s+1, t_e-1, 0, NUM_TSIDS-1))
                 rects.append((t_e, t_e, 0, p_e))

        print(f"\n--- Weight {wid} ({desc}) ---")
        print(f"Ranges: {len(w)}, Rects: {len(rects)}")
        
        # Measure BDD vs ZDD (Concatenated)
        # Using simplified simulation: BDD nodes vs ZDD nodes
        
        # 1. Build Exact Boolean Function
        # Naive: Set of all valid (tok, tsid) points
        points = set()
        for t1, t2, s1, s2 in rects:
            for t in range(t1, t2 + 1):
                for s in range(s1, s2 + 1):
                    points.add((t, s))
        
        # 2. Convert to ZDD/BDD Nodes
        # Construct Trie
        # Concatenated Order: Tok (12 bits) then Tsid (13 bits)
        
        bdd_nodes = {(None, None, None)} # set of unique nodes
        zdd_nodes = {(None, None, None)}
        
        # Recursive builder
        memo_bdd = {}
        memo_zdd = {}
        
        def build(depth, current_points, mode):
            # depth 24..0
            state = (depth, frozenset(current_points))
            if mode == 'bdd' and state in memo_bdd: return memo_bdd[state]
            if mode == 'zdd' and state in memo_zdd: return memo_zdd[state]
            
            if not current_points: return 0 # False
            
            # Check if all points covered (Universe)
            # Universe size at this depth = 2^(depth+1)
            # Only valid for simple ranges, but here we have restricted points
            # If current_points == universe_of_depth?
            # Actually, standard terminal check:
            if depth == -1: return 1 # True
            
            # Split
            bit = depth
            hi_set = set() # points with bit=1
            lo_set = set() # points with bit=0
            
            # Since points are (tok, tsid), we need to extract the bit
            # Bit 24..13 are Tok, 12..0 are Tsid
            
            mask = 0
            shift = 0
            if bit >= 13: # Token bit
                shift_local = bit - 13
                # Check token bit
                for (t, s) in current_points:
                    if (t >> shift_local) & 1: hi_set.add((t, s))
                    else: lo_set.add((t, s))
            else: # Tsid bit
                shift_local = bit
                for (t, s) in current_points:
                    if (s >> shift_local) & 1: hi_set.add((t, s))
                    else: lo_set.add((t, s))
                    
            hi_res = build(depth - 1, frozenset(hi_set), mode)
            lo_res = build(depth - 1, frozenset(lo_set), mode)
            
            # Reduction
            if mode == 'bdd':
                if hi_res == lo_res: return hi_res
            elif mode == 'zdd':
                if hi_res == 0: return lo_res # Zero suppression
                
            node = (depth, lo_res, hi_res)
            # Hash consing (simulated just by getting ID)
            # We return the node STRUCTURE to count uniques later
            # (In real recursion we'd return ID, but here we construct sets)
            pass 
            # This naive simulation is too slow for Sets.
            # Must reuse the efficient Manager from previous script.

# --- Re-use logic from previous script but strictly compare BDD/ZDD counts ---
analyze_breakdown()

