"""
Submission for the Precompute2 Trie Competition by Gemini.

This implementation minimizes the trie by merging equivalent nodes using a
partition refinement algorithm, which is a standard technique for DFA minimization.

The core logic is as follows:
1.  **Initial Partition**: Nodes are initially grouped into two sets: accepting ('end')
    nodes and non-accepting nodes. These sets are the first approximation of
    equivalence classes.

2.  **Iterative Refinement**: The algorithm repeatedly refines these partitions. In each
    iteration, it calculates a signature for every node. This signature is based on
    the node's outgoing transitions, where each transition is identified by its
    (pop_count, state_id, token_rangeset) and the *current partition* of its
    destination node.

3.  **Splitting**: If nodes within the same partition have different signatures, that
    partition is split into smaller ones. This process continues until no more
    partitions can be split, meaning the partitions have stabilized into the final
    set of equivalence classes.

4.  **Graph Reconstruction**: A new, minimized graph is built from these final
    partitions. Each partition becomes a single node in the new graph. The edges
    for this new node are formed by merging all outgoing edges from the original
    nodes within that partition. Edges with the same (pop_count, state_id,
    destination_partition) have their token RangeSets combined via a union operation.

This process guarantees that the resulting graph is semantically equivalent to the
original while being minimal in the number of states (nodes) and transitions (edges)
with respect to bisimulation equivalence.
"""

import bisect
import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------

TrieNodeIndex = int
TokenizerStateID = int
StateID = int
LLMTokenBVJSON = List[List[int]]

# -----------------------------------------------------------------------------
# Self-Contained RangeSet Implementation
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RangeSet:
    """
    Efficient, normalized (sorted, disjoint, inclusive) intervals for large, sparse token sets.
    """
    intervals: Tuple[Tuple[int, int], ...]

    @staticmethod
    def empty() -> "RangeSet":
        return RangeSet(())

    @staticmethod
    def from_json(ranges_json: Optional[LLMTokenBVJSON]) -> "RangeSet":
        if not ranges_json:
            return RangeSet.empty()
        return RangeSet.from_ranges(ranges_json)

    @staticmethod
    def from_ranges(ranges: Iterable[Sequence[int]]) -> "RangeSet":
        normalized = RangeSet._merge_unsorted(ranges)
        return RangeSet(tuple(normalized))

    def is_empty(self) -> bool:
        return not self.intervals

    def contains(self, x: int) -> bool:
        a = self.intervals
        if not a:
            return False
        starts = [s for s, _ in a]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0:
            return False
        s, e = a[i]
        return s <= x <= e

    def union(self, other: "RangeSet") -> "RangeSet":
        if self.is_empty(): return other
        if other.is_empty(): return self
        merged: List[Tuple[int, int]] = []
        i, j = 0, 0
        a, b = self.intervals, other.intervals
        def append_or_merge(start: int, end: int) -> None:
            if not merged:
                merged.append((start, end)); return
            ps, pe = merged[-1]
            if start <= pe + 1: merged[-1] = (ps, max(pe, end))
            else: merged.append((start, end))
        while i < len(a) and j < len(b):
            if a[i][0] <= b[j][0]: append_or_merge(a[i][0], a[i][1]); i += 1
            else: append_or_merge(b[j][0], b[j][1]); j += 1
        while i < len(a): append_or_merge(a[i][0], a[i][1]); i += 1
        while j < len(b): append_or_merge(b[j][0], b[j][1]); j += 1
        return RangeSet(tuple(merged))

    @staticmethod
    def _merge_unsorted(ranges: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
        items = [(int(s), int(e)) for s, e in ranges if s is not None and e is not None]
        if not items: return []
        items.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1: ce = max(ce, ne)
            else: merged.append((cs, ce)); cs, ce = ns, ne
        merged.append((cs, ce))
        return merged

# -----------------------------------------------------------------------------
# Minimized Plugin Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MinimizedEdge:
    pop_count: int
    state_id_opt: Optional[int]
    dest_node: int
    tokens: RangeSet

@dataclass(frozen=True, slots=True)
class MinimizedNode:
    is_end: bool
    children: Tuple[MinimizedEdge, ...]

class MinimizedTrie:
    """A container for the minimized trie data."""
    def __init__(self, roots: Dict[TokenizerStateID, int], arena: Dict[int, MinimizedNode]):
        self.roots = roots
        self.arena = arena
        self.node_count = len(self.arena)
        self.edge_count = sum(len(node.children) for node in self.arena.values())

# -----------------------------------------------------------------------------
# Graph API implementation (new, preferred)
# -----------------------------------------------------------------------------

def build(roots_map: List[Tuple[TokenizerStateID, TrieNodeIndex]], arena: Dict) -> MinimizedTrie:
    """
    Builds a minimized trie structure using a partition refinement algorithm.
    """
    print("Gemini submission: Starting trie minimization...")

    # 1. Initial Partition
    all_node_indices = set(arena.keys())
    end_nodes = {idx for idx, node in arena.items() if (node.get("value", {}) or {}).get("end", False)}
    non_end_nodes = all_node_indices - end_nodes
    
    partitions: List[frozenset] = []
    if end_nodes:
        partitions.append(frozenset(end_nodes))
    if non_end_nodes:
        partitions.append(frozenset(non_end_nodes))

    # 2. Partition Refinement Loop
    iteration = 0
    while True:
        iteration += 1
        num_partitions_before = len(partitions)
        
        node_to_partition_id = {node_id: i for i, p in enumerate(partitions) for node_id in p}
        
        new_partitions = []
        for p_idx, partition in enumerate(partitions):
            # Split this partition based on transition signatures
            splitter: Dict[Any, List[TrieNodeIndex]] = collections.defaultdict(list)
            
            for node_id in partition:
                original_node = arena[node_id]
                is_end = (original_node.get("value", {}) or {}).get("end", False)
                
                # Signature is based on (is_end, canonical_outgoing_edges)
                edges_for_sig = []
                for (pop, sid), dest_map in original_node.get("children", []):
                    for dest_id, rangeset in dest_map:
                        dest_partition_id = node_to_partition_id[dest_id]
                        edges_for_sig.append((pop, sid, dest_partition_id, rangeset))
                
                signature = (is_end, frozenset(edges_for_sig))
                splitter[signature].append(node_id)
            
            for group in splitter.values():
                new_partitions.append(frozenset(group))

        partitions = new_partitions
        if len(partitions) == num_partitions_before:
            break # Converged

    print(f"Gemini submission: Minimization converged in {iteration} iterations.")
    print(f"Gemini submission: Original nodes = {len(all_node_indices)}, Minimized nodes = {len(partitions)}")

    # 3. Construct Minimized Graph
    canonical_map = {node_id: i for i, p in enumerate(partitions) for node_id in p}
    canonical_arena: Dict[int, MinimizedNode] = {}

    for i, partition in enumerate(partitions):
        # All nodes in a partition are equivalent, so pick one as representative for is_end
        rep_node_id = next(iter(partition))
        is_end_val = (arena[rep_node_id].get("value", {}) or {}).get("end", False)

        # Merge all edges from all original nodes in this partition
        merged_edges = collections.defaultdict(RangeSet.empty)
        for node_id in partition:
            original_node = arena[node_id]
            for (pop, sid), dest_map in original_node.get("children", []):
                for dest_id, rangeset in dest_map:
                    dest_canonical_id = canonical_map[dest_id]
                    key = (pop, sid, dest_canonical_id)
                    merged_edges[key] = merged_edges[key].union(rangeset)
        
        final_children = []
        for (pop, sid, dest_id), combined_rs in merged_edges.items():
            if not combined_rs.is_empty():
                final_children.append(MinimizedEdge(pop, sid, dest_id, combined_rs))
        
        canonical_arena[i] = MinimizedNode(is_end_val, tuple(sorted(final_children, key=lambda e: (e.dest_node, e.pop_count, e.state_id_opt is None, e.state_id_opt))))

    canonical_roots = {
        sid: canonical_map[root_id] for sid, root_id in roots_map if root_id in canonical_map
    }

    minimized_trie = MinimizedTrie(canonical_roots, canonical_arena)
    print(f"Gemini submission: Final edge count = {minimized_trie.edge_count}")
    return minimized_trie


def get_root(structure: MinimizedTrie, state_id: TokenizerStateID) -> TrieNodeIndex:
    """
    Returns the root node index for the given tokenizer state_id.
    """
    return structure.roots.get(int(state_id), -1)


def iter_edges(structure: MinimizedTrie, node: TrieNodeIndex, token: int) -> Iterator[Tuple[int, Optional[int], TrieNodeIndex]]:
    """
    Yields outgoing edges from a minimized node that are valid for the given token.
    """
    min_node = structure.arena.get(int(node))
    if not min_node:
        return
    
    for edge in min_node.children:
        if edge.tokens.contains(int(token)):
            yield (edge.pop_count, edge.state_id_opt, edge.dest_node)


def is_end(structure: MinimizedTrie, node: TrieNodeIndex) -> bool:
    """
    Returns whether the given minimized node is an accepting state.
    """
    min_node = structure.arena.get(int(node))
    if not min_node:
        return False
    return min_node.is_end


def stats(structure: MinimizedTrie) -> Dict[str, Any]:
    """
    Returns statistics about the minimized data structure.
    The 'nodes' and 'edges' keys are used for scoring.
    """
    return {
        "nodes": structure.node_count,
        "edges": structure.edge_count,
        "comment": "Graph minimized using a partition refinement algorithm."
    }
