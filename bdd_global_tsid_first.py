
import json
import time
import sys

# BDD Manager (Minimal, shared)
class BDD:
    def __init__(self, order):
        self.order = order
        self.var_idx = {v: i for i, v in enumerate(order)}
        self.nodes = {0: None, 1: None}
        self.unique = {}
        self.next_id = 2
        self.cache = {}
        
    def mk(self, v, lo, hi):
        if lo == hi: return lo
        k = (v, lo, hi)
        if k in self.unique: return self.unique[k]
        n = self.next_id; self.next_id += 1
        self.nodes[n] = k
        self.unique[k] = n
        return n
    
    def apply_or(self, a, b):
        if a == 1 or b == 1: return 1
        if a == 0: return b
        if b == 0: return a
        if a == b: return a
        k = (a, b) if a < b else (b, a)
        if k in self.cache: return self.cache[k]
        
        va = self.nodes[a][0] if a > 1 else None
        vb = self.nodes[b][0] if b > 1 else None
        
        if va is None: top = vb
        elif vb is None: top = va
        else: top = va if self.var_idx[va] < self.var_idx[vb] else vb
        
        # Cofactor
        def cof(n, v):
            if n <= 1: return n, n
            nv, lo, hi = self.nodes[n]
            if nv == v: return lo, hi
            return n, n
            
        la, ha = cof(a, top)
        lb, hb = cof(b, top)
        res_lo = self.apply_or(la, lb)
        res_hi = self.apply_or(ha, hb)
        r = self.mk(top, res_lo, res_hi)
        self.cache[k] = r
        return r
    
    def rect(self, tok_lo, tok_hi, tsid_lo, tsid_hi):
        # 1. Build Token interval BDD
        # 2. Build TSID interval BDD
        # 3. AND them? No, we are building ONE BDD.
        # Since ordering is TSID then Token, we can build efficient tree.
        
        # Correct approach for TSID-first:
        # Top part: Range of TSID
        # Bottom part: Range of Token
        
        # Actually, standard APPLY(AND) is easiest and correct.
        
        # Build Token Logic
        tok_cache = {}
        def info_tok(bit_idx, curr): # bit_idx relative to Token part
            if bit_idx == 12: return 1 if tok_lo <= curr <= tok_hi else 0
            state = (bit_idx, curr)
            if state in tok_cache: return tok_cache[state]
            span = 1 << (12 - bit_idx - 1)
            lo_v, hi_v = curr, curr + 2*span - 1
            if hi_v < tok_lo or lo_v > tok_hi: return 0
            if lo_v >= tok_lo and hi_v <= tok_hi: return 1
            
            var = f"tok_{12 - 1 - bit_idx}"
            r = self.mk(var, info_tok(bit_idx+1, curr), info_tok(bit_idx+1, curr+span))
            tok_cache[state] = r
            return r
            
        # Build TSID Logic
        tsid_cache = {}
        def info_tsid(bit_idx, curr):
            if bit_idx == 13: return 1 if tsid_lo <= curr <= tsid_hi else 0
            state = (bit_idx, curr)
            if state in tsid_cache: return tsid_cache[state]
            span = 1 << (13 - bit_idx - 1)
            lo_v, hi_v = curr, curr + 2*span - 1
            if hi_v < tsid_lo or lo_v > tsid_hi: return 0
            if lo_v >= tsid_lo and hi_v <= tsid_hi: return 1
            
            var = f"tsid_{13 - 1 - bit_idx}"
            r = self.mk(var, info_tsid(bit_idx+1, curr), info_tsid(bit_idx+1, curr+span))
            tsid_cache[state] = r
            return r

        t_node = info_tok(0, 0)
        s_node = info_tsid(0, 0)
        
        # Since ordering is Tsid then Token, ANDing them is just:
        # "TSID checks" -> if true, point to "Token checks"
        
        # But we need general apply to be sure.
        # Actually, optimization:
        # s_node uses ONLY tsid vars (top)
        # t_node uses ONLY tok vars (bottom)
        # AND(s_node, t_node) -> simply replace '1' leaves of s_node with t_node!
        
        def replace_leaf(n, replacement):
            if n == 0: return 0
            if n == 1: return replacement
            v, lo, hi = self.nodes[n]
            # Since s_node ONLY has TSID vars, and replacement is Token vars (strictly lower),
            # we don't need to check var ordering.
            return self.mk(v, replace_leaf(lo, replacement), replace_leaf(hi, replacement))
            
        return replace_leaf(s_node, t_node)

    def count(self, roots):
        visited = set()
        stack = list(roots)
        c = 0
        while stack:
            n = stack.pop()
            if n <= 1 or n in visited: continue
            visited.add(n)
            c += 1
            stack.extend([self.nodes[n][1], self.nodes[n][2]])
        return c

# --- Main Benchmark ---
NUM_TSIDS = 4476
with open('range_weights_terminal_dwa.json') as f:
    weights = json.load(f)

# TSID First Order: tsid_12..0 followed by tok_11..0
order = [f'tsid_{i}' for i in range(12, -1, -1)] + [f'tok_{i}' for i in range(11, -1, -1)]
bdd = BDD(order)

print("Benchmarking Global BDD Size (TSID-First)...")
roots = []
start = time.time()

for idx, w in enumerate(weights):
    if not w: 
        roots.append(0)
        continue
    
    # Check max
    # Treat usize::MAX as "All Valid"
    if any(e > 18446744073700000000 for s, e in w):
        roots.append(1)
        continue
        
    root = 0
    
    # Convert ranges to rects
    # Clip to 10M for fairness with previous bench
    ranges = [(s, min(e, 10_000_000)) for s, e in w if s <= 10_000_000]
    
    rects = []
    for s, e in ranges:
        t_s, p_s = divmod(s, NUM_TSIDS)
        t_e, p_e = divmod(e, NUM_TSIDS)
        if t_s == t_e:
            rects.append((t_s, t_s, p_s, p_e))
        else:
            rects.append((t_s, t_s, p_s, NUM_TSIDS-1))
            if t_s+1 <= t_e-1: rects.append((t_s+1, t_e-1, 0, NUM_TSIDS-1))
            rects.append((t_e, t_e, 0, p_e))
            
    for t1, t2, s1, s2 in rects:
        term = bdd.rect(t1, t2, s1, s2)
        root = bdd.apply_or(root, term)
    
    roots.append(root)
    
    if idx % 100 == 0:
        c = bdd.count(roots)
        print(f"Processed {idx}/{len(weights)}, Unique Nodes: {c}", flush=True)

final_c = bdd.count(roots)
dt = time.time() - start
print(f"\n--- Results (TSID-First) ---")
print(f"Total Unique Nodes: {final_c}")
print(f"Time: {dt:.2f}s")
print(f"Compare to Tok-First (Prev): 167,346 nodes")
