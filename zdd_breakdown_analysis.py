
import json
import time

# --- Node Manager ---
class NodeManager:
    def __init__(self, mode='bdd', order=None):
        self.mode = mode 
        self.order = order
        self.var_to_level = {v: i for i, v in enumerate(order)}
        self.nodes = {0: (None, None, None), 1: (None, None, None)} 
        self.unique = {(None, None, None): 0, (None, None, None): 1}
        self.next_id = 2
        self.op_cache = {}

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
        
        def cofactor(n, v):
            if n <= 1: return n
            n_var, n_lo, n_hi = self.nodes[n]
            if n_var == v: return n_lo, n_hi
            if self.var_to_level[n_var] > self.var_to_level[v]: return n, n
            return n, n 
        
        l1, h1 = cofactor(n1, top)
        l2, h2 = cofactor(n2, top)
        res_lo = self.apply(op, l1, l2)
        res_hi = self.apply(op, h1, h2)
        res = self.get_node(top, res_lo, res_hi)
        self.op_cache[key] = res
        return res
        
    def build_interval(self, prefix, bit_count, low, high):
        cache = {}
        def rec(idx, current_val):
            state = (idx, current_val)
            if state in cache: return cache[state]
            bit = idx
            val_min = current_val
            val_max = current_val + (1 << (bit + 1)) - 1
            if val_min >= low and val_max <= high: return 1
            if val_max < low or val_min > high: return 0
            
            var = f"{prefix}_{bit}"
            hi = rec(idx - 1, current_val | (1 << bit))
            lo = rec(idx - 1, current_val)
            node = self.get_node(var, lo, hi)
            cache[state] = node
            return node
        return rec(bit_count - 1, 0)
        
    def count(self, root):
        if root <= 1: return 0
        visited = set()
        stack = [root]
        count = 0
        while stack:
            n = stack.pop()
            if n <= 1 or n in visited: continue
            visited.add(n)
            count += 1
            var, lo, hi = self.nodes[n]
            stack.append(lo)
            stack.append(hi)
        return count

# --- Analysis Logic ---
NUM_TSIDS = 4476
TOKEN_BITS = 12
TSID_BITS = 13

# Setup Concatenated Order
order = []
for i in range(TOKEN_BITS-1, -1, -1): order.append(f'tok_{i}')
for i in range(TSID_BITS-1, -1, -1): order.append(f'tsid_{i}')

# Load
with open('range_weights_terminal_dwa.json') as f:
    weights = json.load(f)

targets = {
    977: "Weight 977 (Cartesian)",
    400: "Weight 400 (Banded / Dense)",
    501: "Weight 501 (Sparse)"
}

print(f"{'Weight':<25} | {'Technique':<10} | {'Nodes':<8} | {'Factor'}")
print("-" * 60)

for wid, desc in targets.items():
    ranges = weights[wid]
    if not ranges: continue
    
    rects = []
    for s, e in ranges:
         if s > 10000000: continue
         e = min(e, 10000000)
         t_s, p_s = divmod(s, NUM_TSIDS)
         t_e, p_e = divmod(e, NUM_TSIDS)
         if t_s == t_e: rects.append((t_s, t_s, p_s, p_e))
         else:
             rects.append((t_s, t_s, p_s, NUM_TSIDS-1))
             if t_s+1 <= t_e-1: rects.append((t_s+1, t_e-1, 0, NUM_TSIDS-1))
             rects.append((t_e, t_e, 0, p_e))

    # Test BDD
    mgr_bdd = NodeManager('bdd', order)
    root_bdd = 0
    t_cache = {}
    for t1, t2, s1, s2 in rects:
        if (t1, t2) not in t_cache:
            t_cache[(t1,t2)] = mgr_bdd.build_interval('tok', TOKEN_BITS, t1, t2)
        t_tree = t_cache[(t1,t2)]
        s_tree = mgr_bdd.build_interval('tsid', TSID_BITS, s1, s2)
        term = mgr_bdd.apply('and', t_tree, s_tree)
        root_bdd = mgr_bdd.apply('or', root_bdd, term)
    bdd_count = mgr_bdd.count(root_bdd)
    
    # Test ZDD
    mgr_zdd = NodeManager('zdd', order)
    root_zdd = 0
    t_cache_z = {}
    for t1, t2, s1, s2 in rects:
        if (t1, t2) not in t_cache_z:
            t_cache_z[(t1,t2)] = mgr_zdd.build_interval('tok', TOKEN_BITS, t1, t2)
        t_tree = t_cache_z[(t1,t2)]
        s_tree = mgr_zdd.build_interval('tsid', TSID_BITS, s1, s2)
        term = mgr_zdd.apply('and', t_tree, s_tree)
        root_zdd = mgr_zdd.apply('or', root_zdd, term)
    zdd_count = mgr_zdd.count(root_zdd)
    
    ratio = zdd_count / bdd_count if bdd_count > 0 else 0
    print(f"{desc:<25} | {'BDD':<10} | {bdd_count:<8} | 1.0x")
    print(f"{'':<25} | {'ZDD':<10} | {zdd_count:<8} | {ratio:.2f}x")
    print("-" * 60)
