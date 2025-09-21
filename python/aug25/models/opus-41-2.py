import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Precompute graph structure for faster traversal
        self._precompute_graph()

    def _precompute_graph(self):
        """Precompute an efficient graph representation and topological information."""
        # Build adjacency list and edge data
        self.edges = {}  # (node, dest) -> (pop, llm_bv, state_bv)
        self.adj_list = defaultdict(list)  # node -> [dests]
        self.in_degree = defaultdict(int)  # Track incoming edges

        for node_id, node_data in self.arena.items():
            children = node_data.get("children", [])
            for (pop, llm_bv), dests in children:
                for dest_idx, state_bv in dests:
                    dest_idx = int(dest_idx)
                    self.edges[(node_id, dest_idx)] = (pop, llm_bv, state_bv)
                    self.adj_list[node_id].append(dest_idx)
                    self.in_degree[dest_idx] += 1

        # Identify end nodes
        self.end_nodes = {n for n in self.arena if self.is_end(n)}

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    @profile
    def get_mask(self) -> RangeSet:
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset

        # Accumulate incoming states for each node
        incoming_states = defaultdict(list)  # node -> [(gss, mask)]

        # Initialize with roots
        for sid, gss in state_map.items():
            r = self.roots_map.get(int(sid))
            if r is not None:
                incoming_states[int(r)].append((gss, all_ones))

        # Process nodes using modified topological order with batching
        processed = {}  # node -> (merged_gss, merged_mask)
        remaining_in_degree = self.in_degree.copy()
        queue = deque()

        # Start with roots (nodes with no predecessors or that have incoming states)
        for node in incoming_states:
            if node not in remaining_in_degree or remaining_in_degree[node] == 0:
                queue.append(node)

        final_mask = ffi.Bitset.zeros()
        pmc = self.possible_matches_cache
        max_state = self.tokenizer_max_state

        def compute_terminal_mask(gss_node, llm_mask):
            """Compute allowed terminals for an end node."""
            acc = gss_node.reduce_acc()
            if acc is None:
                return llm_mask

            forbid = ffi.Bitset.zeros()
            disallowed = acc.terminals_union.complement()

            for (start, end), bv in disallowed.range_values():
                if bv.is_empty():
                    continue
                for tsid in range(start, min(end, max_state) + 1):
                    pm = pmc.get(tsid)
                    if not pm:
                        continue
                    for terminal_id_str, llm_tokens in pm.items():
                        if bv.contains(int(terminal_id_str)):
                            forbid = forbid.union(llm_tokens)

            return llm_mask.difference(forbid)

        while queue:
            node = queue.popleft()

            # Skip if already processed
            if node in processed:
                continue

            # Get all incoming states for this node
            states = incoming_states.get(node, [])
            if not states:
                continue

            # Batch merge all incoming states at once
            if len(states) == 1:
                merged_gss, merged_mask = states[0]
            else:
                gss_list = [s[0] for s in states]
                merged_gss = GSS.merge_many(gss_list)
                merged_mask = ffi.Bitset.zeros()
                for _, mask in states:
                    merged_mask = merged_mask.union(mask)

            # Store processed result
            processed[node] = (merged_gss, merged_mask)

            # Handle end nodes
            if node in self.end_nodes:
                allowed = compute_terminal_mask(merged_gss, merged_mask)
                if not allowed.is_empty():
                    final_mask = final_mask.union(allowed)

            # Skip propagation if mask is empty
            if merged_mask.is_empty():
                continue

            # Propagate to children
            for dest in self.adj_list[node]:
                edge_key = (node, dest)
                if edge_key not in self.edges:
                    continue

                pop, llm_bv, state_bv = self.edges[edge_key]

                # Apply edge transformations
                popped_gss = merged_gss.popn(pop)

                # Filter by state
                if not state_bv.is_empty():
                    peek_states = popped_gss.peek()
                    matched = [popped_gss.isolate(s) for s in peek_states if state_bv.contains(s)]
                    if not matched:
                        continue
                    child_gss = GSS.merge_many(matched) if len(matched) > 1 else matched[0]
                else:
                    continue

                # Apply mask intersection
                child_mask = merged_mask if llm_bv.is_empty() else merged_mask.intersection(llm_bv)

                # Add to incoming states for destination
                incoming_states[dest].append((child_gss, child_mask))

                # Decrement in-degree and add to queue if ready
                if dest in remaining_in_degree:
                    remaining_in_degree[dest] -= 1
                    if remaining_in_degree[dest] == 0:
                        queue.append(dest)
                else:
                    # Node has no tracked in-degree, can process immediately
                    queue.append(dest)

        # Convert to original mask
        original_mask = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])

        return RangeSet.from_ranges(original_mask.to_ranges())