import heapq
from typing import Dict, List, Set, Optional, Tuple
import collections

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel, PyAcc
from ..common_interface import GraphProvider, RangeSet


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, RangeSet]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[RangeSet] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map
        self.all_terminals_bitset: Optional[RangeSet] = im.all_terminals_bitset

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        node_data = self.arena.get(node)
        if node_data:
            value_data = node_data.get("value")
            if value_data:
                return value_data.get("clean_end", False)
        return False

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        This is an optimized version focusing on speed.
        """
        state_map: Dict[int, GSS] = self.state
        all_ones: RangeSet = self.all_internal_llm_tokens_bitset or RangeSet.empty()
        final_mask: RangeSet = RangeSet.empty()

        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = collections.defaultdict(set)
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map: Tuple[Tuple[int, RangeSet], ...] = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map:
                if tsid > max_state:
                    continue
                terminals_to_llm = pmc.get(tsid)
                if terminals_to_llm is None:
                    continue

                for terminal_id in disallowed_terminals.to_indices():
                    llm_set = terminals_to_llm.get(terminal_id)
                    if llm_set is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(llm_set)

            allowed_mask = all_ones.difference(disallowed_llm_mask)
            return PyAcc(terminals_union={}, llm_mask=allowed_mask)

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)

            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if not todo[d]:
                hp(depth_heap, d)
            todo[d].add(r)

        # Main traversal loop
        while depth_heap:
            depth: int = hpop(depth_heap)
            current_depth_nodes = todo.pop(depth)

            for node in current_depth_nodes:
                gss_node: GSS = values.pop(node)

                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc and reduced_acc.llm_mask:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                edges = arena.get(node, {}).get("children") or []
                for (pop, llm_bv), dests in edges:
                    if llm_bv.is_empty():
                        continue

                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    peeked = popped.peek()
                    if not peeked:
                        continue

                    peeked_rs = RangeSet.from_indices(peeked)

                    # Memoization for applying LLM mask, shared for all dests of this edge
                    acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}
                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        if acc in acc_memo:
                            return acc_memo[acc]

                        new_mask = acc.llm_mask.intersection(llm_bv)
                        if new_mask.is_empty():
                            result = None
                        else:
                            result = PyAcc(terminals_union={}, llm_mask=new_mask)
                        acc_memo[acc] = result
                        return result

                    for dest_idx, state_bv in dests:
                        kept_rs = peeked_rs.intersection(state_bv)
                        if kept_rs.is_empty():
                            continue

                        values_to_keep = kept_rs.to_indices()
                        if not values_to_keep:
                            continue

                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        if child_gss.is_empty():
                            continue

                        child_gss = child_gss.apply_and_prune(intersect_and_prune)
                        if child_gss.is_empty():
                            continue

                        d: int = int(dest_idx)
                        if d in values:
                            values[d] = values[d].merge(child_gss)
                        else:
                            values[d] = child_gss
                            dest_max_depth = max_depth[d]
                            if not todo[dest_max_depth]:
                                hp(depth_heap, dest_max_depth)
                            todo[dest_max_depth].add(d)

        # Convert internal mask back to original vocabulary token IDs
        original_indices: List[int] = []
        internal_to_original_map = self.internal_to_original_map
        for i in final_mask.to_indices():
            original_id = internal_to_original_map.get(i)
            if original_id is not None:
                original_indices.append(original_id)

        return RangeSet.from_indices(original_indices)
