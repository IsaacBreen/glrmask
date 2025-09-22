import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Optional

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
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from allowed terminals for this accumulator
            allowed_mask: ffi.Bitset = ffi.Bitset.zeros()
            allowed_l2 = acc.terminals_union
            for (start, end), bv in allowed_l2.range_values():
                if bv.is_empty():
                    continue
                for tsid in range(start, min(end, max_state) + 1):
                    pm: Optional[Dict[int, ffi.Bitset]] = pmc.get(tsid)
                    if not pm:
                        continue
                    for terminal_id_key, llm_tokens in pm.items():
                        if bv.contains(int(terminal_id_key)):
                            allowed_mask = allowed_mask.union(llm_tokens)
            return PyAcc(
                terminals_union=ffi.HybridL2Bitset.all(),  # consume
                llm_mask=allowed_mask
            )

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)

        def enqueue(d: int, n: int) -> None:
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main loop
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()
                gss_node: GSS = values.pop(node)

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
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
                            acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                            def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                                if acc in acc_memo:
                                    return acc_memo[acc]
                                new_mask = acc.llm_mask.intersection(llm_bv)
                                if new_mask.is_empty():
                                    result = None
                                else:
                                    result = PyAcc(
                                        terminals_union=acc.terminals_union,
                                        llm_mask=new_mask
                                    )
                                acc_memo[acc] = result
                                return result

                            child_gss = child_gss.apply_and_prune(intersect_and_prune)
                            if child_gss.is_empty():
                                continue

                        d: int = int(dest_idx)
                        if d in values:
                            existing_gss = values[d]
                            new_gss = child_gss
                            merged_gss = existing_gss.merge(new_gss)
                            values[d] = merged_gss
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
