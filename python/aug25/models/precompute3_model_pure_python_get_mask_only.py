import heapq
from typing import Dict, Tuple

import _sep1 as ffi

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    """A wrapper model focusing on get_mask for performance analysis."""

    def __init__(self, inner_model: InnerModel):
        self.inner_model = inner_model

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        """Creates the model from a JSON string via the inner model."""
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int):
        """Passes the commit operation to the inner model."""
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        """Provides access to the state from the inner model."""
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        """Checks if a trie node is a terminal node."""
        return bool((self.inner_model.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    @profile
    def get_mask(self) -> RangeSet:
        """Computes the final LLM token mask by traversing the precomputed trie."""
        im = self.inner_model
        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        todo: Dict[int, set[int]] = {}
        depth_heap: list[int] = []

        def enqueue(depth: int, node_idx: int):
            if depth not in todo:
                heapq.heappush(depth_heap, depth)
                todo[depth] = set()
            todo[depth].add(node_idx)

        # Seed the traversal from the current GSS state
        for sid, gss in self.state.items():
            if (root_idx := im.roots_map.get(int(sid))) is None:
                continue
            
            if root_idx in values:
                existing_gss, _ = values[root_idx]
                values[root_idx] = (existing_gss.merge(gss), im.all_internal_llm_tokens_bitset)
            else:
                values[root_idx] = (gss, im.all_internal_llm_tokens_bitset)
            enqueue(im.max_depth[root_idx], root_idx)

        # Main traversal loop
        while depth_heap:
            depth = heapq.heappop(depth_heap)
            if not (node_indices := todo.pop(depth, None)):
                continue

            for node_idx in node_indices:
                if (item := values.pop(node_idx, None)) is None:
                    continue
                gss_node, llm_mask = item

                # End-node handling
                if self.is_end(node_idx):
                    merged_acc = gss_node.reduce_acc()
                    disallowed = ffi.HybridL2Bitset.all() if not merged_acc else merged_acc.terminals_union.complement()
                    
                    forbidden = ffi.Bitset.zeros()
                    for (start, end), bv in disallowed.range_values():
                        if bv.is_empty(): continue
                        end = min(end, im.tokenizer_max_state)
                        for tsid in range(start, end + 1):
                            if not (matches := im.possible_matches_cache.get(tsid)): continue
                            for term_id, tokens in matches.items():
                                if bv.contains(int(term_id)):
                                    forbidden.union_inplace(tokens)
                    final_mask.union_inplace(llm_mask.difference(forbidden))

                if llm_mask.is_empty():
                    continue

                # Transitions
                for (pop, llm_bv), dests in (im.arena.get(node_idx, {}).get("children") or []):
                    popped = gss_node.popn(pop)
                    child_llm_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)

                    for dest_idx, state_bv in dests:
                        matched = [popped.isolate(sid) for sid in popped.peek() if state_bv.contains(sid)]
                        if not matched: continue

                        child_gss = GSS.merge_many(matched)
                        dest_idx = int(dest_idx)
                        if dest_idx in values:
                            ex_gss, ex_mask = values[dest_idx]
                            values[dest_idx] = (ex_gss.merge(child_gss), ex_mask.union(child_llm_mask))
                        else:
                            values[dest_idx] = (child_gss, child_llm_mask)
                        enqueue(im.max_depth[dest_idx], dest_idx)

        # Convert to original token IDs
        original_indices = [
            im.internal_to_original_map[i]
            for i in final_mask.to_indices()
            if i in im.internal_to_original_map
        ]
        return RangeSet.from_ranges(ffi.Bitset.from_indices(original_indices).to_ranges())
