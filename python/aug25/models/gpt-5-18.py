import heapq
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
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Optimized for speed:
        - No stats or printing.
        - Minimize Python-level overhead in inner loops.
        - Reuse memoization across destinations within the same edge-group for apply-and-prune.
        """
        state_map: Dict[int, GSS] = self.state

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union

            # Aggregate all disallowed LLM tokens
            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.to_indices():
                    bv = terminals_to_llm.get(terminal_id)
                    if bv is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(bv)

            allowed_mask = (all_ones if all_ones is not None else RangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
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
            bucket = todo.pop(depth)
            while bucket:
                node: int = bucket.pop()
                gss_node: GSS = values.pop(node, None)
                if gss_node is None:
                    continue

                # End-node handling: union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges and propagate masks
                edges = arena.get(node, {}).get("children") or ()
                for (pop, llm_bv), dests in edges:
                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    peeked = popped.peek()
                    if not peeked:
                        continue

                    # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv.
                    # Reuse the acc memo across all destinations in this edge group.
                    acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        cached = acc_memo.get(acc, None)
                        if acc in acc_memo:
                            return cached
                        new_mask = acc.llm_mask.intersection(llm_bv)
                        if new_mask.is_empty():
                            acc_memo[acc] = None
                            return None
                        res = PyAcc(
                            terminals_union=acc.terminals_union,
                            llm_mask=new_mask
                        )
                        acc_memo[acc] = res
                        return res

                    # Pre-fast-path for small peek sizes (very common).
                    if isinstance(peeked, list):
                        sids_list = peeked
                    else:
                        sids_list = list(peeked)
                    n_peek = len(sids_list)

                    for dest_idx, state_bv in dests:
                        state_contains = state_bv.contains

                        if n_peek == 1:
                            sid0 = sids_list[0]
                            if not state_contains(sid0):
                                continue
                            values_to_keep = [sid0]
                        elif n_peek == 2:
                            sid0, sid1 = sids_list
                            k0 = state_contains(sid0)
                            if k0:
                                if state_contains(sid1):
                                    values_to_keep = [sid0, sid1]
                                else:
                                    values_to_keep = [sid0]
                            else:
                                if state_contains(sid1):
                                    values_to_keep = [sid1]
                                else:
                                    continue
                        else:
                            # General path for n >= 3
                            # Build only if there is at least one match.
                            tmp = []
                            append = tmp.append
                            for sid in sids_list:
                                if state_contains(sid):
                                    append(sid)
                            if not tmp:
                                continue
                            values_to_keep = tmp

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
                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs
        original_indices: List[int] = []
        map_get = self.internal_to_original_map.get
        for i in final_mask.to_indices():
            v = map_get(i)
            if v is not None:
                original_indices.append(v)

        return RangeSet.from_indices(original_indices)
