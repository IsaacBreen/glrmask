import json
import heapq
import sys
from collections import defaultdict, Counter

class RangeSet:
    def __init__(self, ranges):
        # ranges is list of [start, end] inclusive
        if not ranges:
            self.ranges = tuple()
            return
            
        sorted_ranges = sorted([tuple(r) for r in ranges], key=lambda x: x[0])
        merged = []
        for start, end in sorted_ranges:
            if not merged:
                merged.append((start, end))
            else:
                last_s, last_e = merged[-1]
                if start <= last_e + 1: # Adjacent or overlapping
                    merged[-1] = (last_s, max(last_e, end))
                else:
                    merged.append((start, end))
        self.ranges = tuple(merged)

    def __repr__(self):
        return f"RangeSet({list(self.ranges)})"
    
    def __len__(self):
        return len(self.ranges)
    
    def __hash__(self):
        return hash(self.ranges)
    
    def __eq__(self, other):
        return self.ranges == other.ranges
        
    def union(self, other):
        return RangeSet(list(self.ranges) + list(other.ranges))

def load_weights(filename):
    with open(filename) as f:
        raw = json.load(f)
    return [RangeSet(w) for w in raw]

class GenericRePair:
    def __init__(self, vocab_rev, mapped_weights, initial_ranges, name="Generic"):
        self.vocab_rev = vocab_rev # List of RangeSets
        self.vocab_map = {rs: i for i, rs in enumerate(vocab_rev)}
        self.mapped_weights = mapped_weights # List of List of IDs
        self.initial_ranges = initial_ranges
        self.name = name

    def run(self, max_iterations=20000):
        pair_counts = Counter()
        for w in self.mapped_weights:
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i+1])] += 1
        
        heap = []
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (-count, pair))
            
        iterations = 0
        while heap and iterations < max_iterations:
            neg_count, pair = heapq.heappop(heap)
            count = -neg_count
            
            if pair_counts[pair] != count: continue
            if count < 2: break
                
            a, b = pair
            rs_c = self.vocab_rev[a].union(self.vocab_rev[b])
            
            # Savings = count - cost_added
            # cost_added = ranges in new basis symbol
            cost_c = len(rs_c)
            # We replace 'count' occurrences of (a,b) with c.
            # Refs saved = count.
            # Basis added = cost_c.
            net_savings = count - cost_c
            
            if net_savings <= 0: continue
            
            # Merit! Merge.
            if rs_c not in self.vocab_map:
                new_id = len(self.vocab_rev)
                self.vocab_rev.append(rs_c)
                self.vocab_map[rs_c] = new_id
            else:
                new_id = self.vocab_map[rs_c]
            
            del pair_counts[pair]
            
            # Update weights
            for w in self.mapped_weights:
                i = 0
                while i < len(w) - 1:
                    if w[i] == a and w[i+1] == b:
                        if i > 0:
                            prev_pair = (w[i-1], w[i])
                            pair_counts[prev_pair] -= 1
                            if pair_counts[prev_pair] == 0: del pair_counts[prev_pair]
                        if i + 2 < len(w):
                            next_pair = (w[i+1], w[i+2])
                            pair_counts[next_pair] -= 1
                            if pair_counts[next_pair] == 0: del pair_counts[next_pair]
                            
                        w[i] = new_id
                        w.pop(i+1)
                        
                        if i > 0:
                            p = (w[i-1], w[i])
                            pair_counts[p] += 1
                            heapq.heappush(heap, (-pair_counts[p], p))
                        if i + 1 < len(w):
                            n = (w[i], w[i+1])
                            pair_counts[n] += 1
                            heapq.heappush(heap, (-pair_counts[n], n))
                    else:
                        i += 1
            
            iterations += 1
            if iterations % 1000 == 0:
                print(f"  [{self.name}] Iter {iterations}: Merged {pair} (count {count}) -> Savings {net_savings}")

    def report(self):
        used_ids = set()
        for w in self.mapped_weights:
            used_ids.update(w)
            
        basis_cost = sum(len(self.vocab_rev[i]) for i in used_ids)
        ref_cost = sum(len(w) for w in self.mapped_weights)
        total = basis_cost + ref_cost
        print(f"[{self.name}] Final Report:")
        print(f"  Basis Elements: {len(used_ids)}")
        print(f"  Basis Ranges: {basis_cost}")
        print(f"  References: {ref_cost}")
        print(f"  Total Cost: {total}")
        print(f"  Ratio: {total / self.initial_ranges:.3f}")

def run_standard_repair(weights, initial_ranges):
    print("  Running Standard Re-Pair...")
    # Intern ranges
    vocab_rev = []
    vocab_map = {}
    mapped_weights = []
    
    for w in weights:
        ids = []
        for r in w.ranges:
            rs = RangeSet([r])
            if rs not in vocab_map:
                vocab_map[rs] = len(vocab_rev)
                vocab_rev.append(rs)
            ids.append(vocab_map[rs])
        mapped_weights.append(ids)

    opt = GenericRePair(vocab_rev, mapped_weights, initial_ranges, name="Standard")
    opt.run()
    opt.report()

def check_scalar_hypothesis(weights):
    scalar_count = 0
    hole_count = 0
    complex_count = 0
    
    threshold = 10**12
    
    for w in weights:
        # Check if weight is essentially a scalar or hole
        # Scalar: 1 range, len 1.
        # Hole: 1 or 2 ranges, large coverage.
        
        is_scalar = False
        is_hole = False
        
        if len(w) == 1:
            s, e = w.ranges[0]
            if e - s == 0:
                is_scalar = True
            elif (e - s) > threshold:
                is_hole = True
        elif len(w) == 2:
            # Check for hole: [0, k-1], [k+1, MAX]
            r1 = w.ranges[0]
            r2 = w.ranges[1]
            if r1[0] == 0 and r2[1] > threshold:
                is_hole = True
        
        if is_scalar: scalar_count += 1
        elif is_hole: hole_count += 1
        else: complex_count += 1
        
    print(f"\nHypothesis Check ({len(weights)} weights):")
    print(f"  Scalars [k,k]: {scalar_count}")
    print(f"  Holes ![k,k]: {hole_count}")
    print(f"  Complex: {complex_count}")
    
    # Num ranges frequency
    range_counts = Counter(len(w) for w in weights)
    print("\nRange Count Frequencies:")
    for count, freq in sorted(range_counts.items()):
        print(f"  {count} ranges: {freq} weights")

def process_file(filename):
    print(f"\nProcessing {filename}...")
    try:
        weights = load_weights(filename)
    except FileNotFoundError:
        print("File not found.")
        return

    initial_ranges = sum(len(w) for w in weights)
    print(f"Loaded {len(weights)} weights. Total Ranges: {initial_ranges}")
    
    check_scalar_hypothesis(weights)
    
    if "terminal" in filename:
        pass # Skip long optimization
    else:
        run_standard_repair(weights, initial_ranges)

if __name__ == "__main__":
    process_file("range_weights_terminal_dwa.json")
    process_file("range_weights_parser_dwa.json")
