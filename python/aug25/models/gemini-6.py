import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional

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
        self.possible_matches_cache: Optional[Dict[int, Dict[str, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Precomputation to accelerate `get_mask`.
        # We group terminals by the LLM token bitset they map to.
        self.forbid_precomputation: Dict[int, List[Tuple[ffi.Bitset, ffi.Bitset]]] = {}
        if self.possible_matches_cache:
            for tsid, pm in self.possible_matches_cache.items():
                # pm: Dict[terminal_id_str, llm_bitset]
                # Since ffi.Bitset is not hashable, we use its string representation as a temporary key.
                inverted_map: Dict[str, Tuple[ffi.Bitset, ffi.Bitset]] = {}
                for terminal_id_str, llm_bs in pm.items():
                    llm_bs_str = llm_bs.to_json_string()
                    if llm_bs_str not in inverted_map:
                        inverted_map[llm_bs_str] = (llm_bs, ffi.Bitset.zeros())
                    # Add the terminal to the bitset for this group.
                    inverted_map[llm_bs_str][1].insert(int(terminal_id_str))
                
                self.forbid_precomputation[tsid] = list(inverted_map.values())

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
        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        max_state: int = self.tokenizer_max_state

        for sid, gss in state_map.items():
            r: Optional[int] = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)
            d: int = max_depth[r]
            b: Optional[Set[int]] = todo.get(d)
            if b is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                b.add(r)

        def enqueue(d: int, n: int) -> None:
            b: Optional[Set[int]] = todo.get(d)
            if b is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                b.add(n)

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        while depth_heap:
            depth: int = hpop(depth_heap)
            nodes: Optional[Set[int]] = todo.pop(depth, None)
            if not nodes:
                continue
            for node in nodes:
                if node in stopped:
                    continue
                item: Optional[Tuple[GSS, ffi.Bitset]] = values.pop(node, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                if is_end(node):
                    forbid: ffi.Bitset = ffi.Bitset.zeros()
                    for (start, end), bv in disallowed_terminals(gss_node).range_values():
                        if bv.is_empty():
                            continue
                        for tsid in range(start, min(end, max_state) + 1):
                            precomputed_list = self.forbid_precomputation.get(tsid)
                            if not precomputed_list:
                                continue
                            for llm_bs, term_bs in precomputed_list:
                                if not bv.intersection(term_bs).is_empty():
                                    forbid = forbid.union(llm_bs)
                    
                    allowed: ffi.Bitset = llm_mask.difference(forbid)
                    if not allowed.is_empty():
                        final_mask = final_mask.union(allowed)

                if llm_mask.is_empty():
                    stopped.add(node)
                    continue

                for (pop, llm_bv), dests in (arena.get(node, {}).get("children") or []):
                    popped: GSS = gss_node.popn(pop)
                    for dest_idx, state_bv in dests:
                        if not state_bv.is_empty():
                            matched: List[GSS] = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                            if not matched:
                                continue
                        else:
                            continue
                        child_gss: GSS = GSS.merge_many(matched)
                        child_mask: ffi.Bitset = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                        d: int = int(dest_idx)
                        if d in values:
                            g0, m0 = values[d]
                            values[d] = (g0.merge(child_gss), m0.union(child_mask))
                        else:
                            values[d] = (child_gss, child_mask)
                        enqueue(max_depth[d], d)

        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])
        return RangeSet.from_ranges(original_mask.to_ranges())
