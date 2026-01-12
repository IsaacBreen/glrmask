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
        # Return number of ranges (cost)
        return len(self.ranges)
    
    def __hash__(self):
        return hash(self.ranges)
    
    def __eq__(self, other):
        return self.ranges == other.ranges

def load_weights(filename):
    with open(filename) as f:
        raw = json.load(f)
    return [RangeSet(w) for w in raw]

class RePairOptimizer:
    def __init__(self, weights):
        self.original_weights = weights
        self.vocab = {} # RangeSet -> ID
        self.vocab_rev = [] # ID -> RangeSet
        self.mapped_weights = [] # List of list of IDs
        
        # Intern initial ranges (atoms/basis)
        next_id = 0
        for w in weights:
            ids = []
            for r in w.ranges:
                rs = RangeSet([r])
                if rs not in self.vocab:
                    self.vocab[rs] = next_id
                    self.vocab_rev.append(rs)
                    next_id += 1
                ids.append(self.vocab[rs])
            self.mapped_weights.append(ids)

        self.initial_ranges = sum(len(w) for w in weights)
        print(f"  Initialized: {len(self.vocab)} unique basis ranges. Original total ranges: {self.initial_ranges}")

    def run(self, max_iterations=50000):
        # Pair counts
        pair_counts = Counter()
        for w in self.mapped_weights:
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i+1])] += 1
        
        # Heap: (-count, pair)
        heap = []
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (-count, pair))
            
        iterations = 0
        while heap and iterations < max_iterations:
            neg_count, pair = heapq.heappop(heap)
            count = -neg_count
            
            # Lazy check
            if pair_counts[pair] != count:
                continue
            
            if count < 2:
                break
                
            a, b = pair
            
            # Construct merged range set
            rs_a = self.vocab_rev[a]
            rs_b = self.vocab_rev[b]
            
            new_ranges = []
            new_ranges.extend(rs_a.ranges)
            new_ranges.extend(rs_b.ranges)
            rs_c = RangeSet(new_ranges)
            
            cost_c = len(rs_c)
            # Savings = count - cost_c
            net_savings = count - cost_c
            
            if net_savings <= 0:
                continue
                
            # Perform Merge
            new_id = len(self.vocab_rev)
            self.vocab[rs_c] = new_id
            self.vocab_rev.append(rs_c)
            
            del pair_counts[pair]
            
            # Update weights - Brute force scan
            # Optimize: In a real implementation, keep track of pair locations.
            merge_occurred = False
            
            for w in self.mapped_weights:
                i = 0
                while i < len(w) - 1:
                    if w[i] == a and w[i+1] == b:
                        merge_occurred = True
                        
                        # Decrement neighbor counts
                        if i > 0:
                            prev_pair = (w[i-1], w[i])
                            pair_counts[prev_pair] -= 1
                            
                        if i + 2 < len(w):
                            next_pair = (w[i+1], w[i+2])
                            pair_counts[next_pair] -= 1
                        
                        # Replace
                        w[i] = new_id
                        w.pop(i+1)
                        
                        # Increment new neighbor counts
                        if i > 0:
                            new_prev_pair = (w[i-1], w[i])
                            pair_counts[new_prev_pair] += 1
                            heapq.heappush(heap, (-pair_counts[new_prev_pair], new_prev_pair))
                            
                        if i + 1 < len(w):
                            new_next_pair = (w[i], w[i+1])
                            pair_counts[new_next_pair] += 1
                            heapq.heappush(heap, (-pair_counts[new_next_pair], new_next_pair))
                        
                        # Continue from i (now new_id), check (new_id, next) next iter
                    else:
                        i += 1
            
            iterations += 1
            if iterations % 1000 == 0:
                print(f"  Iter {iterations}: Merged {pair} (count {count}) -> Savings {net_savings}")

    def report(self):
        # Calculate used basis
        used_ids = set()
        for w in self.mapped_weights:
            used_ids.update(w)
            
        basis_cost = sum(len(self.vocab_rev[i]) for i in used_ids)
        ref_cost = sum(len(w) for w in self.mapped_weights)
        total = basis_cost + ref_cost
        print("Final Report:")
        print(f"  Basis Elements: {len(used_ids)}")
        print(f"  Basis Ranges: {basis_cost}")
        print(f"  References: {ref_cost}")
        print(f"  Total Cost: {total}")
        print(f"  Ratio: {total / self.initial_ranges:.3f}")

def process_file(filename):
    print(f"\nProcessing {filename}...")
    try:
        weights = load_weights(filename)
    except FileNotFoundError:
        print("File not found.")
        return

    print(f"Loaded {len(weights)} weights.")
    opt = RePairOptimizer(weights)
    opt.run()
    opt.report()

if __name__ == "__main__":
    process_file("range_weights_terminal_dwa.json")
    process_file("range_weights_parser_dwa.json")
