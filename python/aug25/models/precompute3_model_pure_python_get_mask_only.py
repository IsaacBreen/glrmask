import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel, PyAcc
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
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, ffi.Bitset]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        memo = {}
        for sid, gss in state_map.items():
            r: Optional[int] = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)

            def initialize_acc(acc: PyAcc) -> PyAcc:
                # Compute forbidden LLM tokens from disallowed terminals for this GSS
                forbid: ffi.Bitset = ffi.Bitset.zeros()
                disallowed_l2 = acc.terminals_union.complement()
                for (start, end), bv in disallowed_l2.range_values():
                    if bv.is_empty():
                        continue
                    for tsid in range(start, min(end, max_state) + 1):
                        pm: Optional[Dict[int, ffi.Bitset]] = pmc.get(tsid)
                        if not pm:
                            continue
                        for terminal_id_str, llm_tokens in pm.items():
                            if bv.contains(int(terminal_id_str)):
                                forbid = forbid.union(llm_tokens)
                allowed_mask: ffi.Bitset = all_ones.difference(forbid)  # type: ignore[union-attr]
                return PyAcc(
                    terminals_union=ffi.HybridL2Bitset.all(),  # consume
                    llm_mask=allowed_mask
                )

            gss_initialized: GSS = gss.apply(initialize_acc, memo)

            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

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

        # Main loop
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()
                gss_node: GSS = values.pop(node)

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges and propagate masks
                for (pop, llm_bv), dests in (arena.get(node, {}).get("children") or []):
                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():
                            continue

                        values_to_keep = [s for s in popped.peek() if state_bv.contains(s)]
                        if not values_to_keep:
                            continue

                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        if child_gss.is_empty():
                            continue

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                        if not llm_bv.is_empty():
                            def intersect_edge(acc: PyAcc) -> PyAcc:
                                return PyAcc(
                                    terminals_union=acc.terminals_union,
                                    llm_mask=acc.llm_mask.intersection(llm_bv)
                                )
                            child_gss = child_gss.apply(intersect_edge)

                        d: int = int(dest_idx)
                        if d in values:
                            values[d] = values[d].merge(child_gss)
                        else:
                            values[d] = child_gss
                        enqueue(max_depth[d], d)

                todo.pop(depth)

        # Convert internal mask back to original IDs
        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])
        return RangeSet.from_ranges(original_mask.to_ranges())
